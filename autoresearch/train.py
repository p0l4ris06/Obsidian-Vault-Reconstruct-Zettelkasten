import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import gc
import math
import time
import subprocess
import argparse
from dataclasses import dataclass
from pathlib import Path
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
BATCH_SIZE = 4           
DEPTH = 8                
DIM = 512                
<<<<<<< HEAD
HEADS = 8                
=======
HEADS = 16                
>>>>>>> ce713fd35b3faae3112424d12481776de75fec1c
T = 512                  

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
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None, reduction='mean'):
        tok_emb = self.transformer.wte(idx)
        x = tok_emb
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

def save_and_log(model, tokenizer, device, repo_root, epoch_desc):
    model.eval()
    val_bpb = evaluate_bpb(model, tokenizer, BATCH_SIZE)
    peak_vram = torch.cuda.max_memory_allocated(device) / 1024 / 1024 if device == "cuda" else 0
    
    # Save model
    torch.save(model.state_dict(), 'autoresearch/model.pt')
    
    # Log to TSV
    results_file = repo_root / 'autoresearch' / 'results.tsv'
    file_exists = results_file.exists()
    with open(results_file, 'a', encoding='utf-8') as f:
        if not file_exists:
            f.write("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")
        f.write(f"auto\t{val_bpb:.6f}\t{peak_vram/1024:.1f}\tkeep\t{epoch_desc}\n")
    
    # Refresh Dashboard
    try:
        subprocess.run(["uv", "run", "python", "autoresearch/plot_results.py"], check=False)
    except:
        pass
    
    print(f"[{time.strftime('%H:%M:%S')}] Checkpoint saved: val_bpb={val_bpb:.4f}")
    model.train()
<<<<<<< HEAD

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=300, help="Training budget in seconds")
    args = parser.parse_args()

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

    config = Config()
    model = Transformer(config).to(device)
    repo_root = Path(__file__).resolve().parent.parent

    # Load weights
    weights_path = 'autoresearch/model.pt'
    if os.path.exists(weights_path):
        print(f"Loading existing weights for refinement...")
        model.load_state_dict(torch.load(weights_path, map_location=device))
        base_lr = 1.2e-3
=======
    
    # Cosine LR Schedule with Warmup
    warmup_steps = 100
    if step < warmup_steps:
        lr = min_lr + (base_lr - min_lr) * (step / warmup_steps)
>>>>>>> ce713fd35b3faae3112424d12481776de75fec1c
    else:
        base_lr = 4e-3

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.05, betas=(0.8, 0.98))
    tokenizer = Tokenizer.from_directory()
    train_loader = make_dataloader(tokenizer, BATCH_SIZE, T, "train")

    print(f"Starting Final Sprint: {args.budget}s budget.")
    start_time = time.time()
    last_checkpoint_time = start_time
    checkpoint_interval = 3600 # 1 hour
    step = 0
    min_lr = 1e-4

    while True:
        elapsed = time.time() - start_time
        if elapsed >= args.budget:
            break
            
        model.train()
        
        # Hourly Checkpoint
        if elapsed - (last_checkpoint_time - start_time) >= checkpoint_interval:
            save_and_log(model, tokenizer, device, repo_root, f"Final Sprint Checkpoint (Step {step})")
            last_checkpoint_time = time.time()

        # Cosine LR Schedule
        warmup_steps = 500
        if step < warmup_steps:
            lr = min_lr + (base_lr - min_lr) * (step / warmup_steps)
        else:
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * elapsed / args.budget))
            
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        x, y, epoch = next(train_loader)
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step < 5 or step % 100 == 0:
            print(f"step {step} | lr {lr:.2e} | loss {loss.item():.4f} | time {elapsed:.1f}s")
        step += 1

    # Final Save
    save_and_log(model, tokenizer, device, repo_root, "Final Sprint Complete")
    print("Final Sprint finished!")

if __name__ == "__main__":
    main()