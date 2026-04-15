import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import gc
import math
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Import from your exact prepare.py ---
from prepare import Tokenizer, make_dataloader, evaluate_bpb

# --- FA2 Implementation & Hardware Check ---
try:
    from flash_attn import flash_attn_func
    print("Flash Attention 2 library loaded successfully.")
except ImportError:
    flash_attn_func = None
    print("Flash Attention 2 not found. Relying on PyTorch optimized SDPA (Memory-Efficient Attention).")

# --- 4GB VRAM SURVIVAL CONFIG (GTX 1650 Super) ---
TOTAL_BATCH_SIZE = 4096 
BATCH_SIZE = 2           
DEPTH = 8                
DIM = 512                
HEADS = 8                
T = 256                  

@dataclass
class Config:
    vocab_size: int = 8192 # Must match VOCAB_SIZE in prepare.py
    n_layer: int = DEPTH
    n_head: int = HEADS
    n_embd: int = DIM
    sequence_length: int = T

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.wqkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.wo = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.wqkv(x).reshape(B, T, 3, self.n_head, C // self.n_head).transpose(1, 3)
        q, k, v = qkv.unbind(2)

        if flash_attn_func is not None and x.is_cuda:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            y = flash_attn_func(q, k, v, causal=True)
            y = y.reshape(B, T, C)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 3).contiguous().view(B, T, C)
            
        return self.wo(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.w2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * self.weight

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
            wpe = nn.Embedding(config.sequence_length, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None, reduction='mean'):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction=reduction)
            if reduction == 'none':
                return loss
                
        return logits, loss

# --- Training Loop Setup ---
print("Initializing device...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to: {device}")

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
gc.collect()

print("Building model...")
config = Config()
model = Transformer(config).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.1)

print("Loading tokenizer and dataloader from cache...")
tokenizer = Tokenizer.from_directory()
train_loader = make_dataloader(tokenizer, BATCH_SIZE, T, "train")

print("Starting 5-minute training run with Cosine LR Decay...")
start_time = time.time()
step = 0
base_lr = 2e-3
min_lr = 2e-4
budget = 300

while True:
    elapsed = time.time() - start_time
    if elapsed >= budget:
        break
        
    model.train()
    
    # Cosine LR Schedule based on time
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * elapsed / budget))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    x, y, epoch = next(train_loader)
    x, y = x.to(device), y.to(device)
    
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    
    if step < 5 or step % 50 == 0:
        print(f"step {step} | lr {lr:.2e} | loss {loss.item():.4f} | time {elapsed:.1f}s")
    step += 1

training_time = time.time() - start_time
print("Training loop finished! Running evaluation...")
model.eval()

# Calculates the final BPB score
val_bpb = evaluate_bpb(model, tokenizer, BATCH_SIZE)

peak_vram = torch.cuda.max_memory_allocated(device) / 1024 / 1024 if torch.cuda.is_available() else 0
num_params = sum(p.numel() for p in model.parameters()) / 1e6

print(f"---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {training_time:.1f}")
print(f"peak_vram_mb:     {peak_vram:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params:.1f}")
print(f"depth:            {DEPTH}")
print(f"dim:              {DIM}")