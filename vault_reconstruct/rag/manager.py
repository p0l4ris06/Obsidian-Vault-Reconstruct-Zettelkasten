from pathlib import Path
from .harvester import Harvester, get_vault_tags
from .store import VectorStore

class RAGManager:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.cache_dir = repo_root / "rag_cache"
        self.index_path = repo_root / "rag_index.pkl"
        self.harvester = Harvester(self.cache_dir)
        self.store = VectorStore(self.index_path)

    def search_context(self, queries: list[str], top_k: int = 10) -> str:
        """Search for context across multiple queries and return combined text."""
        if not self.store.load():
            return ""
        
        all_results = []
        for q in queries:
            all_results.extend(self.store.search(q, top_k=top_k//len(queries) + 1))
        
        # Sort by score and remove duplicates
        all_results = sorted(all_results, key=lambda x: x["score"], reverse=True)
        seen = set()
        unique_results = []
        for r in all_results:
            if r["item"]["id"] not in seen:
                unique_results.append(r)
                seen.add(r["item"]["id"])
            if len(unique_results) >= top_k: break

        context_blocks = []
        for r in unique_results:
            item = r["item"]
            src = f"[{item['source'].upper()} - {item['published']}] {item['title']}"
            context_blocks.append(f"SOURCE: {src}\nCONTENT: {item['summary']}\nURL: {item['url']}")
        
        return "\n\n".join(context_blocks)

    def sync(self, tags: list[str]):
        """Harvest new data and rebuild the index."""
        print(f"Syncing knowledge base for {len(tags)} tags...")
        self.harvester.harvest_all(tags)
        self.store.load_from_cache(self.cache_dir)
