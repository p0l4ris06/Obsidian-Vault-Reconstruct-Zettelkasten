import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from prepare import Tokenizer

# --- Architecture Configuration (Must match train.py) ---
DEPTH = 8
DIM = 512
HEADS = 16 # Reverting to 16 for stability unless 32 wins experiment 44
T = 512

@dataclass
class Config:
    vocab_size: int = 8192
    n_layer: int = DEPTH
    n_head: int = HEADS
    n_embd: int = DIM
    sequence_length: int = T

# --- Model Components ---

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * self.weight

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.wq = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.wk = nn.Linear(config.n_embd, self.head_dim, bias=False)
        self.wv = nn.Linear(config.n_embd, self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_embd, config.n_embd, bias=False)
        inv_freq = 1.0 / (10000**(torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def _apply_rope(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        x_rotated = torch.cat((-x[..., x.size(-1)//2:], x[..., :x.size(-1)//2]), dim=-1)
        return (x * cos) + (x_rotated * sin)

    def forward(self, x):
        B, T, C = x.size()
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, 1, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, 1, self.head_dim).transpose(1, 2)
        q = self._apply_rope(q, T)
        k = self._apply_rope(k, T)
        k = k.expand(-1, self.n_head, -1, -1)
        v = v.expand(-1, self.n_head, -1, -1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 3).contiguous().view(B, T, C)
        return self.wo(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.n_embd, int(config.n_embd * 8 / 3), bias=False)
        self.w2 = nn.Linear(config.n_embd, int(config.n_embd * 8 / 3), bias=False)
        self.w3 = nn.Linear(int(config.n_embd * 8 / 3), config.n_embd, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd)
        self.attn = Attention(config)
        self.ln2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        tok_emb = self.transformer.wte(idx)
        x = tok_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

def generate(model, tokenizer, prompt, max_new_tokens=256, temperature=0.8, top_k=20):
    model.eval()
    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= T else idx[:, -T:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        
        if idx_next.item() == 0: # Assuming 0 is EOS or pad
            break
            
    return tokenizer.decode(idx[0].tolist())

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    config = Config()
    model = Transformer(config).to(device)
    
    if os.path.exists("model.pt"):
        print("Loading weights from model.pt...")
        model.load_state_dict(torch.load("model.pt", map_location=device))
    else:
        print("Warning: model.pt not found. Running with random weights.")
        
    tokenizer = Tokenizer.from_directory()
    
    # Load vault path from .env (parent dir)
    vault_path = ""
    env_path = os.path.join("..", ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if line.startswith("VAULT_INPUT_PATH="):
                    vault_path = line.split("=")[1].strip().replace("\\\\", "\\")
                    break
    
    if not vault_path:
        vault_path = os.getcwd() # Fallback
        
    lit_folder = os.path.join(vault_path, "03_Literature")
    os.makedirs(lit_folder, exist_ok=True)
    
    print(f"Literature notes will be saved to: {lit_folder}")
    
    prompt = input("Enter a research prompt: ").strip()
    if not prompt:
        prompt = "Summarize the core knowledge found in this vault."
        
    print(f"Generating for: {prompt}")
    
    output = generate(model, tokenizer, prompt)
    
    filename = f"research_{int(time.time())}.md"
    filepath = os.path.join(lit_folder, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# Synthetic Research Note\n\nPrompt: {prompt}\n\n---\n\n{output}")
        
    print(f"Note saved to {filepath}")
    print("\n--- Generated Content ---\n")
    print(output)
