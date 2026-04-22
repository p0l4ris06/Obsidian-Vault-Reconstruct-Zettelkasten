import json
import numpy as np
import os
import warnings
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pickle
import logging

# Silence specific BERT and Hub warnings
warnings.filterwarnings("ignore", message=".*embeddings.position_ids.*")
warnings.filterwarnings("ignore", message="You are sending unauthenticated requests")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, index_path: Path, model_name: str = "all-MiniLM-L6-v2"):
        self.index_path = index_path
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = SentenceTransformer(model_name, device="cpu")
        self.data = []
        self.embeddings = None

    def load_from_cache(self, cache_dir: Path):
        """Index all JSON files in the harvest cache."""
        all_items = []
        for f in cache_dir.glob("*.json"):
            try:
                items = json.loads(f.read_text(encoding="utf-8"))
                all_items.extend(items)
            except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError) as e:
                logger.debug(f"Skipping cache file {f}: {e}")
                continue
        
        if not all_items:
            print("No cached data found to index.")
            return

        print(f"Indexing {len(all_items)} items...")
        # Prepare text for embedding: Title + Summary
        texts = [f"{item['title']}: {item['summary']}" for item in all_items]
        self.embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        self.data = all_items
        self.save()

    def save(self):
        with open(self.index_path, "wb") as f:
            pickle.dump({"data": self.data, "embeddings": self.embeddings}, f)
        print(f"Index saved to {self.index_path}")

    def load(self):
        if self.index_path.exists():
            with open(self.index_path, "rb") as f:
                state = pickle.load(f)
                self.data = state["data"]
                self.embeddings = state["embeddings"]
            print(f"Loaded {len(self.data)} items from index.")
            return True
        return False

    def search(self, query: str, top_k: int = 5):
        """Perform semantic search."""
        if self.embeddings is None:
            return []
        
        query_emb = self.model.encode([query], convert_to_numpy=True)
        # Cosine similarity
        similarities = np.dot(self.embeddings, query_emb.T).flatten() / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        
        # Get top K indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "item": self.data[idx],
                "score": float(similarities[idx])
            })
        return results

if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    cache_dir = REPO_ROOT / "rag_cache"
    index_path = REPO_ROOT / "rag_index.pkl"
    
    store = VectorStore(index_path)
    if not store.load():
        store.load_from_cache(cache_dir)
    
    # Test search
    res = store.search("synaptic connections in alzheimers")
    for r in res:
        print(f"[{r['score']:.2f}] {r['item']['title']} ({r['item']['source']})")
