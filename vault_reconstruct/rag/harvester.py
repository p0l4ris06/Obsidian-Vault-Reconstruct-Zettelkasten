import arxiv
import wikipedia
from Bio import Entrez
import json
import os
from pathlib import Path
from datetime import datetime
import re
import logging

# NCBI / PubMed setup: Global SSL bypass for strict networks
import ssl
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception: pass

Entrez.email = "vault-reconstructor@local.machine"
try:
    Entrez.context = ssl._create_unverified_context()
except Exception: pass

logger = logging.getLogger(__name__)

class Harvester:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_arxiv(self, query: str, max_results: int = 10):
        """Fetch abstracts from arXiv."""
        print(f"  [arXiv] Searching for '{query}'...", flush=True)
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            results = []
            for res in search.results():
                results.append({
                    "source": "arxiv",
                    "id": res.entry_id,
                    "title": res.title,
                    "summary": res.summary,
                    "authors": [a.name for a in res.authors],
                    "url": res.pdf_url,
                    "published": res.published.strftime("%Y-%m-%d")
                })
            return results
        except Exception as e:
            print(f"    arXiv skip: {str(e)[:100]}")
            return []

    def fetch_wikipedia(self, query: str, max_results: int = 3):
        """Fetch summaries from Wikipedia."""
        print(f"  [Wikipedia] Searching for '{query}'...", flush=True)
        results = []
        titles = wikipedia.search(query, results=max_results)
        for t in titles:
            try:
                page = wikipedia.page(t, auto_suggest=False)
                results.append({
                    "source": "wikipedia",
                    "id": page.pageid,
                    "title": page.title,
                    "summary": page.summary,
                    "url": page.url,
                    "published": datetime.now().strftime("%Y-%m-%d")
                })
            except wikipedia.exceptions.DisambiguationError as e:
                # Try the first option if it's a disambiguation page
                if e.options:
                    try:
                        page = wikipedia.page(e.options[0], auto_suggest=False)
                        results.append({
                            "source": "wikipedia",
                            "id": page.pageid,
                            "title": page.title,
                            "summary": page.summary,
                            "url": page.url,
                            "published": datetime.now().strftime("%Y-%m-%d")
                        })
                    except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError) as inner_e:
                        logger.debug(f"Failed to get disambiguation option for '{t}': {inner_e}")
                        continue
            except wikipedia.exceptions.PageError:
                continue
            except Exception as e:
                # Log only true errors, not just normal disambiguation
                print(f"    Wiki error for {t}: {e}")
        return results

    def fetch_pubmed(self, query: str, max_results: int = 10):
        """Fetch abstracts from PubMed."""
        print(f"  [PubMed] Searching for '{query}'...", flush=True)
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
            record = Entrez.read(handle)
            handle.close()
            ids = record["IdList"]
            if not ids:
                return []
            
            handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="abstract", retmode="xml")
            records = Entrez.read(handle)
            handle.close()
            
            results = []
            for article in records.get("PubmedArticle", []):
                try:
                    medline = article["MedlineCitation"]
                    art = medline["Article"]
                    pmid = medline["PMID"]
                    results.append({
                        "source": "pubmed",
                        "id": str(pmid),
                        "title": art.get("ArticleTitle", "Untitled"),
                        "summary": " ".join(art.get("Abstract", {}).get("AbstractText", ["No abstract available."])),
                        "authors": [auth.get("LastName", "") for auth in art.get("AuthorList", [])],
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        "published": art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {}).get("Year", "Unknown")
                    })
                except (KeyError, TypeError, AttributeError) as e:
                    logger.debug(f"Failed to parse PubMed article: {e}")
                    continue
            return results
        except Exception as e:
            print(f"    PubMed error: {e}")
            return []

    def harvest_all(self, tags: list[str]):
        """Harvest data for all tags and save to cache."""
        from vault_reconstruct.paths import safe_filename
        for i, tag in enumerate(tags):
            tag_clean = tag.replace("#", "").strip()
            # Flatten path characters to keep cache dir flat
            cache_file = self.cache_dir / f"{safe_filename(tag_clean)}.json"
            
            print(f"  [{i+1}/{len(tags)}] Processing: {tag_clean}", flush=True)
            
            # Simple caching: skip if recently fetched
            if cache_file.exists():
                print(f"    Already cached.", flush=True)
                continue

            # Ensure parent directory exists (just in case of complex config)
            cache_file.parent.mkdir(parents=True, exist_ok=True)

            results = []
            try:
                results.extend(self.fetch_arxiv(tag_clean))
                results.extend(self.fetch_wikipedia(tag_clean))
                results.extend(self.fetch_pubmed(tag_clean))
            except Exception as e:
                print(f"    Provider error for '{tag_clean}': {e}")
            
            if results:
                cache_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
                print(f"  Cached {len(results)} items for '{tag_clean}'")
            
            # Simple throttling to respect public API rate limits
            import time
            time.sleep(1.2)

def get_vault_tags(vault_path: Path):
    """Scan vault for all unique tags."""
    tags = set()
    # Simple regex to find tags in markdown files
    tag_re = re.compile(r"(?<!\S)#([a-zA-Z0-9_\-/]+)")
    for md in vault_path.rglob("*.md"):
        try:
            content = md.read_text(encoding="utf-8")
            for t in tag_re.findall(content):
                tags.add(t)
        except (FileNotFoundError, UnicodeDecodeError, OSError) as e:
            logger.debug(f"Failed to read markdown file {md}: {e}")
            continue
    return sorted(list(tags))

if __name__ == "__main__":
    from vault_reconstruct.config import get_vault_paths
    paths = get_vault_paths()
    vault_out = paths.output_vault
    cache_dir = REPO_ROOT / "rag_cache" # REPO_ROOT needs to be defined if run directly
    
    # For now, let's just test with a couple of topics if run directly
    import sys
    from pathlib import Path
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    cache_dir = REPO_ROOT / "rag_cache"
    
    h = Harvester(cache_dir)
    test_tags = ["synaptic plasticity", "alzheimers disease"]
    if len(sys.argv) > 1:
        test_tags = sys.argv[1:]
    
    print(f"Starting harvest for: {test_tags}")
    h.harvest_all(test_tags)
    print("Done.")
