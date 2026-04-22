import sys
from pathlib import Path
# Add repo root to path so we can import the vault_reconstruct package
sys.path.append(str(Path(__file__).resolve().parent.parent))
"""
vault_researcher.py

Synthetic research generator using the custom-trained autoresearch model or grounded RAG.
Synthesizes new connections between tags/topics and places them in 03_Literature.
"""

import argparse
import sys
import os
import warnings
import logging
from pathlib import Path
from datetime import datetime

# Silence noisy dependency warnings for a cleaner TUI
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="No parser was explicitly specified")
# Silence specific BERT and Hub warnings
warnings.filterwarnings("ignore", message=".*embeddings.position_ids.*")
warnings.filterwarnings("ignore", message="You are sending unauthenticated requests")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from vault_reconstruct.env import load_dotenv_no_override
from vault_reconstruct.llm import LlmConfig, make_backend, generate_text_with_retries
from vault_reconstruct.config import get_vault_paths
from vault_reconstruct.paths import safe_filename

load_dotenv_no_override()

def generate_research_note(topic_a: str, topic_b: str, provider: str = "ollama", use_rag: bool = False):
    """Generate a bridge note between two topics."""
    
    # Setup backend
    # We use Qwen 2.5 3B for high-quality grounded research if using Ollama
    model = "qwen2.5-coder:3b" if (provider == "ollama" or provider == "autoresearch") else "champion"
    cfg = LlmConfig(provider=provider if provider != "autoresearch" else "ollama", model=model)
    backend = make_backend(cfg)
    
    context = ""
    if use_rag:
        try:
            from vault_reconstruct.rag.manager import RAGManager
            repo_root = Path(__file__).resolve().parent.parent
            rag = RAGManager(repo_root)
            print(f"Retrieving local context for '{topic_a}' and '{topic_b}'...")
            context = rag.search_context([topic_a, topic_b], top_k=8)
            if not context:
                print("  Warning: No local context found. Run with --sync first.")
        except Exception as e:
            print(f"  RAG Error: {e}")

    rag_instruction = ""
    if context:
        rag_instruction = f"""
Use the following RESEARCH CONTEXT to ground your response. Cite the sources (e.g. PubMed 2023) where applicable.

### RESEARCH CONTEXT
{context}
---
"""

    prompt = f"""# Research Note: {topic_a} and {topic_b}
{rag_instruction}

This document explores the conceptual intersections between [[{topic_a}]] and [[{topic_b}]], 
specifically focusing on their shared roles within the knowledge vault.

## Synthesis
"""
    
    print(f"Generating synthetic research note for: {topic_a} + {topic_b}...")
    content = generate_text_with_retries(backend, prompt=prompt, max_retries=1)
    
    # Build full note
    title = f"Research - {topic_a} and {topic_b}"
    timestamp = datetime.now().strftime("%Y-%m-%d")
    
    # Note ID (Simulating the vault ID generator)
    note_id = datetime.now().strftime("%Y%m%d%H%M")
    
    frontmatter = f"""---
id: {note_id}
title: "{title}"
tags:
  - research
  - {topic_a.lower().replace(' ', '-')}
  - {topic_b.lower().replace(' ', '-')}
type: literature
created: {timestamp}
---

"""
    return title, frontmatter + content

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic research notes.")
    parser.add_argument("topic_a", nargs="?", help="First topic/tag")
    parser.add_argument("topic_b", nargs="?", help="Second topic/tag")
    parser.add_argument("--vault", help="Path to Obsidian vault")
    parser.add_argument("--provider", default="ollama", help="LLM provider (default: ollama)")
    parser.add_argument("--rag", action="store_true", help="Use local RAG context")
    parser.add_argument("--sync", action="store_true", help="Sync knowledge base from vault tags")
    
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    
    if args.sync:
        try:
            from vault_reconstruct.rag.manager import RAGManager
            from vault_reconstruct.rag.harvester import get_vault_tags
            paths = get_vault_paths()
            rag = RAGManager(repo_root)
            tags = get_vault_tags(paths.output_vault)
            print(f"Discovered {len(tags)} tags in vault. Syncing...")
            rag.sync(tags)
            print("Success: Knowledge base synced and indexed.")
        except Exception as e:
            print(f"Sync failed: {e}")
        return

    if not args.topic_a or not args.topic_b:
        parser.error("topic_a and topic_b are required unless using --sync")

    vault_path = Path(args.vault) if args.vault else get_vault_paths().output_vault
    target_dir = vault_path / "03_Literature"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    title, full_content = generate_research_note(args.topic_a, args.topic_b, args.provider, use_rag=args.rag)
    
    filename = safe_filename(title) + ".md"
    file_path = target_dir / filename
    
    file_path.write_text(full_content, encoding="utf-8")
    print(f"Success! Research note created: {file_path}")

if __name__ == "__main__":
    main()

