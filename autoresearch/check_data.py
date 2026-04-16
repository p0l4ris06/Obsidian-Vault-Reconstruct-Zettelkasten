import os
import torch
from prepare import Tokenizer, list_parquet_files, pq

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", "data")

print(f"Checking data in {CACHE_DIR}")
val_file = os.path.join(CACHE_DIR, "shard_06542.parquet")

if not os.path.exists(val_file):
    print("Validation file not found!")
    exit()

pf = pq.ParquetFile(val_file)
rg = pf.read_row_group(0)
texts = rg.column("text").to_pylist()

tokenizer = Tokenizer.from_directory()

max_token_found = 0
for text in texts[:1000]: # Check the first 1000 documents
    tokens = tokenizer.encode(text)
    if len(tokens) > 0:
        max_id = max(tokens)
        if max_id > max_token_found:
            max_token_found = max_id

print(f"--- RESULTS ---")
print(f"Tokenizer official vocab size: {tokenizer.get_vocab_size()}")
print(f"Max token ID found in the dataset: {max_token_found}")
print(f"If the 'Max token ID' is greater than or equal to the 'official vocab size', your model config is too small.")