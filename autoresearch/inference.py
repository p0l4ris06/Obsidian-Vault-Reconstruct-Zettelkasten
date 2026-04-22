import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# --- IMPORT CONFIG FROM TRAIN.PY ---
@dataclass
class Config:
    vocab_size: int = 8192
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    sequence_length: int = 512

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * self.weight

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.n_embd, int(config.n_embd * 8 / 3), bias=False)
        self.w2 = nn.Linear(config.n_embd, int(config.n_embd * 8 / 3), bias=False)
        self.w3 = nn.Linear(int(config.n_embd * 8 / 3), config.n_embd, bias=False)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

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

class AutoresearchInference:
    def __init__(self, model_path, tokenizer_dir):
        from prepare import Tokenizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = Config()
        self.model = Transformer(self.config).to(self.device)
        
        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        # Handle cases where training script used DDP or different wrapping
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        self.tokenizer = Tokenizer.from_directory(tokenizer_dir)

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=256, temperature=0.8, top_k=40):
        # Encode prompt (without BOS prepended — we track prompt length to strip it later)
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_len = len(prompt_ids)
        idx = torch.tensor(prompt_ids).unsqueeze(0).to(self.device)
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.sequence_length:]
            logits = self.model(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Stop if model generates BOS/EOS token after a few steps
            bos = self.tokenizer.get_bos_token_id()
            if idx_next.item() == bos and _ > 10:
                break
        
        # Return only the newly generated tokens (strip the prompt)
        generated_ids = idx[0, prompt_len:].tolist()
        
        # Strip any BOS/EOS tokens from the output
        bos = self.tokenizer.get_bos_token_id()
        generated_ids = [t for t in generated_ids if t != bos]
        
        return self.tokenizer.decode(generated_ids)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="The Zettelkasten method is")
    parser.add_argument("--model", type=str, default="autoresearch/model.pt")
    # Default cache dir used in prepare.py
    cache_tok = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", "tokenizer")
    parser.add_argument("--tokenizer", type=str, default=cache_tok)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found.")
        sys.exit(1)
        
    inf = AutoresearchInference(args.model, args.tokenizer)
    print(f"--- PROMPT ---\n{args.prompt}\n--- GENERATED ---")
    print(inf.generate(args.prompt))
