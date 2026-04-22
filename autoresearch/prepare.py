import os
import sys
import time
import math
import argparse
import pickle
from multiprocessing import Pool
import requests
import pyarrow.parquet as pq
import rustbpe
import tiktoken
import torch
try:
    import reconstruct_rust
except ImportError:
    reconstruct_rust = None

try:
    import ctypes
    from ctypes import c_char_p, c_int, POINTER
    # Path to the Go shared library
    _DLL_PATH = os.path.join(os.path.dirname(__file__), "dataloader.dll")
    if os.path.exists(_DLL_PATH):
        go_lib = ctypes.CDLL(_DLL_PATH)
        go_lib.LoadShard.argtypes = [c_char_p]
        go_lib.LoadShard.restype = c_int
        go_lib.GetBatch.argtypes = [c_int]
        go_lib.GetBatch.restype = POINTER(c_char_p)
        go_lib.FreeBatch.argtypes = [POINTER(c_char_p), c_int]
    else:
        go_lib = None
except Exception:
    go_lib = None

# This MUST match the 'T' in your train.py
MAX_SEQ_LEN = 512        
TIME_BUDGET = 300        
EVAL_TOKENS = 10 * 524288  

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542 
VAL_SHARD = MAX_SHARD  
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"
VOCAB_SIZE = 8192

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

def download_single_shard(index):
    filename = f"shard_{index:05d}.parquet"
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath): return True
    url = f"{BASE_URL}/{filename}"
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024): f.write(chunk)
        return True
    except: return False

def download_data(num_shards, download_workers=8):
    os.makedirs(DATA_DIR, exist_ok=True)
    ids = list(range(min(num_shards, MAX_SHARD))) + [VAL_SHARD]
    with Pool(processes=download_workers) as pool: pool.map(download_single_shard, ids)

def list_parquet_files():
    return [os.path.join(DATA_DIR, f) for f in sorted(os.listdir(DATA_DIR)) if f.endswith(".parquet")]

class Tokenizer:
    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)
    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            return cls(pickle.load(f))
    def get_vocab_size(self): return self.enc.n_vocab
    def get_bos_token_id(self): return self.bos_token_id
    def encode(self, text, prepend=None):
        ids = self.enc.encode_ordinary(text)
        if prepend: ids.insert(0, prepend)
        return ids
    def decode(self, ids): return self.enc.decode(ids)

def get_token_bytes():
    return torch.load(os.path.join(TOKENIZER_DIR, "token_bytes.pt"), map_location="cpu")

def _document_batches(split):
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    paths = [val_path] if split == "val" else [p for p in list_parquet_files() if p != val_path]
    while True:
        for p in paths:
            try:
                pf = pq.ParquetFile(p)
                rg = pf.read_row_group(0)
                yield rg.column('text').to_pylist()
            except Exception:
                continue

def make_dataloader(tokenizer, B, T, split):
    # Use Fast Go DataLoader if available
    if go_lib is not None:
        val_shards = [os.path.join(DATA_DIR, VAL_FILENAME)]
        train_shards = [p for p in list_parquet_files() if p != val_shards[0]]
        shards = val_shards if split == "val" else train_shards
        
        bos = tokenizer.get_bos_token_id()
        
        while True:
            for s_path in shards:
                rows_in_shard = go_lib.LoadShard(s_path.encode('utf-8'))
                if rows_in_shard <= 0: continue
                
                for _ in range(0, rows_in_shard, B):
                    batch_ptr = go_lib.GetBatch(B)
                    if not batch_ptr: break
                    
                    x = torch.zeros((B, T), dtype=torch.long)
                    y = torch.zeros((B, T), dtype=torch.long)
                    
                    for j in range(B):
                        if batch_ptr[j]:
                            s = batch_ptr[j].decode('utf-8')
                            tokens = tokenizer.encode(s, prepend=bos)[:T+1]
                            if len(tokens) < T+1: tokens += [0] * (T+1-len(tokens))
                            x[j] = torch.tensor(tokens[:T])
                            y[j] = torch.tensor(tokens[1:T+1])
                    
                    go_lib.FreeBatch(batch_ptr, B)
                    yield x, y, 0
        return

    # Fallback to pure Python implementation
    docs = _document_batches(split)
    bos = tokenizer.get_bos_token_id()
    while True:
        batch = next(docs)
        for i in range(0, len(batch) - B, B):
            x = torch.zeros((B, T), dtype=torch.long)
            y = torch.zeros((B, T), dtype=torch.long)
            for j in range(B):
                tokens = tokenizer.encode(batch[i+j], prepend=bos)[:T+1]
                if len(tokens) < T+1: tokens += [0] * (T+1-len(tokens))
                x[j] = torch.tensor(tokens[:T])
                y[j] = torch.tensor(tokens[1:T+1])
            yield x, y, 0

@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    token_bytes = get_token_bytes().to("cpu")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = 20 
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        # FIXED: Added the third variable (_) to catch the dummy epoch value
        x, y, _ = next(val_loader)
        loss_flat = model(x.to("cuda"), y.to("cuda"), reduction='none').view(-1).to("cpu")
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)