import sys
from pathlib import Path
# Add repo root to path so we can import the vault_reconstruct package
sys.path.append(str(Path(__file__).resolve().parent.parent))
"""
vault_reconstruct.py

The "Power Reconstructor" — A unified, phase-aware pipeline for Obsidian vaults.
Supports: Splitting, Recovery, Frontmatter, AI-Linking, and MOC Generation.
Backends: Ollama, Gemini, Azure, Autoresearch.
"""

import os
import sys
import time
import json
import re
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict
from collections import defaultdict

try:
    import reconstruct_rust # type: ignore
except ImportError:
    reconstruct_rust = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from vault_reconstruct.env import load_dotenv_no_override
from vault_reconstruct.llm import LlmConfig, LlmBackend, generate_text_with_retries, make_backend
from vault_reconstruct.json_extract import extract_json_array, extract_json_dict
from vault_reconstruct.paths import safe_filename
from vault_reconstruct.text_protect import count_wikilinks, mask_protected, restore_protected
from vault_reconstruct.config import get_vault_paths

# ============================================================================
# REGEX PATTERNS (Pre-compiled for performance)
# ============================================================================

# Extracts metadata JSON from comment block at start of file
# (e.g., <!-- meta: {"tags": ["a"], "type": "zettel"} -->)
META_PATTERN = re.compile(r"^<!-- meta: (.*?) -->\n", re.DOTALL)

# Extracts individual tag values from YAML list (e.g., "  - tag1" → "tag1")
YAML_TAG_VALUE_PATTERN = re.compile(r"  - (.+)")

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    input_vault: str = str(get_vault_paths().input_vault)
    output_vault: str = str(get_vault_paths().output_vault)
    provider:     str = os.environ.get("VAULT_LLM_PROVIDER", "ollama").strip().lower()
    
    # Model defaults
    gemini_model: str = os.environ.get("VAULT_GEMINI_MODEL", "gemini-2.0-flash")
    azure_model:  str = os.environ.get("VAULT_AZURE_MODEL", "gpt-4o-mini")
    ollama_model: str = os.environ.get("VAULT_OLLAMA_MODEL", "llama3")
    ollama_cloud: str = os.environ.get("VAULT_OLLAMA_CLOUD_MODEL", "gemma3:4b-cloud")
    
    min_content_length: int = 50
    request_delay:      float = 0.5
    max_retries:        int = 3
    min_links_before_ai_skip: int = 2
    
    output_folders: list = field(default_factory=lambda: [
        "00_Inbox", "01_MOCs", "02_Zettels", "03_Literature"
    ])
    tracker_filename: str = ".reconstructor_tracker.json"

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ============================================================================
# ID GENERATOR
# ============================================================================

_id_counter = 0

def generate_id() -> str:
    global _id_counter
    _id_counter += 1
    return datetime.now().strftime("%Y%m%d") + f"{_id_counter:04d}"

# ============================================================================
# TRACKER
# ============================================================================

class ProcessingTracker:
    def __init__(self, tracker_path: Path):
        self.path = tracker_path
        self._data: dict[str, set[str]] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, encoding="utf-8") as f:
                    raw = json.load(f)
                self._data = {k: set(v) for k, v in raw.items()}
            except (json.JSONDecodeError, OSError):
                log.warning("Could not read tracker; starting fresh.")

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({k: sorted(v) for k, v in self._data.items()}, f, indent=2)

    def is_done(self, phase: str, key: str) -> bool:
        return key in self._data.get(phase, set())

    def mark_done(self, phase: str, key: str):
        self._data.setdefault(phase, set()).add(key)
        self._save()

# ============================================================================
# BACKEND INITIALIZATION
# ============================================================================

def get_llm_backend(config: Config):
    if config.provider == "gemini":
        cfg = LlmConfig(provider="gemini", model=config.gemini_model)
    elif config.provider == "azure":
        cfg = LlmConfig(provider="azure", model=config.azure_model)
    elif config.provider == "ollama":
        cfg = LlmConfig(provider="ollama", model=config.ollama_model, ollama_cloud_model=config.ollama_cloud)
    elif config.provider == "autoresearch":
        cfg = LlmConfig(provider="autoresearch", model="local-finetune")
    else:
        raise ValueError(f"Unknown provider: {config.provider}")
    return make_backend(cfg)

# ============================================================================
# PHASE 0: RECOVERY
# ============================================================================

RECOVERY_PROMPT = """\
You are a Zettelkasten assistant. Recover the following failed note convert request.
Return a JSON array of atomic notes.
Schema: [{{"title": "...", "content": "# Title\\n...", "tags": ["tag1"], "type": "zettel"}}]

CONTENT:
{content}
"""

def run_phase0(backend: LlmBackend, config: Config) -> None:
    log.info("=== PHASE 0: Recovery ===")
    out_path = Path(config.output_vault)
    inbox = out_path / "00_Inbox"
    tracker = ProcessingTracker(out_path / config.tracker_filename)
    
    q_files = list(inbox.glob("QUARANTINE_*.md"))
    for q_file in tqdm(q_files, desc="Recovering"):
        if tracker.is_done("phase0", q_file.name): continue
        
        content = q_file.read_text(encoding="utf-8")
        prompt = RECOVERY_PROMPT.format(content=content)
        try:
            resp = generate_text_with_retries(backend, prompt=prompt, max_retries=config.max_retries)
            notes = extract_json_array(resp)
            if notes:
                for n in notes:
                    title = safe_filename(n.get("title", "Untitled"))
                    ntype = n.get("type", "zettel").lower()
                    dest_folder = "03_Literature" if ntype == "literature" else "02_Zettels"
                    (out_path / dest_folder / f"{title}.md").write_text(n.get("content", ""), encoding="utf-8")
                q_file.unlink()
                log.info("Recovered %s → %s", q_file.name, dest_folder)
        except Exception as e:
            log.error("Failed recovery for %s: %s", q_file.name, e)
        tracker.mark_done("phase0", q_file.name)

# ============================================================================
# PHASE 1: SPLIT
# ============================================================================

SPLIT_PROMPT = """\
Convert the following note into atomic Zettelkasten notes.
Output ONLY a JSON array.
Schema: [{{"title": "...", "content": "# Title\\n...", "tags": ["tag1"], "type": "zettel"}}]

NOTE:
{content}
"""

def run_phase1(backend: LlmBackend, config: Config) -> None:
    log.info("=== PHASE 1: Splitting ===")
    out_path = Path(config.output_vault)
    tracker = ProcessingTracker(out_path / config.tracker_filename)
    
    for folder in config.output_folders:
        (out_path / folder).mkdir(parents=True, exist_ok=True)
        
    input_files = list(Path(config.input_vault).rglob("*.md"))
    for f in tqdm(input_files, desc="Splitting"):
        if tracker.is_done("phase1", f.name): continue
        content = f.read_text(encoding="utf-8")
        if len(content.strip()) < config.min_content_length:
            tracker.mark_done("phase1", f.name)
            continue
            
        try:
            resp = generate_text_with_retries(backend, prompt=SPLIT_PROMPT.format(content=content), max_retries=config.max_retries)
            notes = extract_json_array(resp)
            if notes:
                for n in notes:
                    title = safe_filename(n.get("title", "Untitled"))
                    tags = n.get("tags", [])
                    ntype = n.get("type", "zettel").lower()
                    dest_folder = "03_Literature" if ntype == "literature" else "02_Zettels"
                    # Store as meta-comment for Phase 3
                    (out_path / dest_folder / f"{title}.md").write_text(
                        f"<!-- meta: {json.dumps({'tags': tags, 'type': ntype})} -->\n{n.get('content','')}",
                        encoding="utf-8"
                    )
            else:
                (out_path / "00_Inbox" / f"QUARANTINE_{f.name}").write_text(content, encoding="utf-8")
        except Exception as e:
            log.error("Error splitting %s: %s", f.name, e)
        tracker.mark_done("phase1", f.name)

# ============================================================================
# PHASE 2: LINKING (RUST)
# ============================================================================

def run_phase2_rust(config: Config) -> None:
    if reconstruct_rust is None:
        log.warning("Rust engine not found. Skipping high-speed link phase.")
        return
    log.info("=== PHASE 2: High-speed Link (Rust) ===")
    try:
        files_modified = reconstruct_rust.run_link_phase(config.output_vault)
        log.info("Rust link complete. %d files modified.", files_modified)
    except Exception as e:
        log.error("Rust link failed: %s", e)

# ============================================================================
# PHASE 3: FRONTMATTER
# ============================================================================

def run_phase3(config: Config) -> None:
    log.info("=== PHASE 3: Frontmatter ===")
    out_path = Path(config.output_vault)
    notes = list(out_path.rglob("*.md"))
    tracker = ProcessingTracker(out_path / config.tracker_filename)
    
    for f in tqdm(notes, desc="Frontmatter"):
        if tracker.is_done("phase3", f.stem): continue
        text = f.read_text(encoding="utf-8")
        if text.startswith("---"): 
            tracker.mark_done("phase3", f.stem)
            continue
            
        meta_match = META_PATTERN.match(text)
        tags, ntype = [], "zettel"
        if meta_match:
            try:
                m = json.loads(meta_match.group(1))
                tags, ntype = m.get("tags", []), m.get("type", "zettel")
                text = text[meta_match.end():]
            except (json.JSONDecodeError, ValueError, IndexError) as e:
                log.debug(f"Failed to parse metadata in {f.stem}: {e}")
                pass

        # Ensure the file is in the correct folder based on type
        correct_folder = "03_Literature" if ntype == "literature" else "02_Zettels"
        correct_path = out_path / correct_folder / f.name
        if f.parent.name != correct_folder and f != correct_path:
            correct_path.parent.mkdir(parents=True, exist_ok=True)
            f.rename(correct_path)
            f = correct_path

        fm = f"---\nid: {generate_id()}\ntitle: \"{f.stem}\"\ntags:\n"
        for t in tags: fm += f"  - {t}\n"
        fm += f"type: {ntype}\ncreated: {datetime.now().strftime('%Y-%m-%d')}\n---\n\n"
        f.write_text(fm + text, encoding="utf-8")
        tracker.mark_done("phase3", f.stem)

# ============================================================================
# PHASE 4: MOCs
# ============================================================================

MOC_PROMPT = """\
Topic: {topic}
Related notes:
{note_list}

Write a Map of Content (MOC) note. 
Group notes into headers. Return the body of the note only.
"""

def run_phase4(backend: LlmBackend, config: Config) -> None:
    log.info("=== PHASE 4: MOC Generation ===")
    out_path = Path(config.output_vault)
    moc_dir = out_path / "01_MOCs"
    moc_dir.mkdir(parents=True, exist_ok=True)
    tracker = ProcessingTracker(out_path / config.tracker_filename)
    
    # Collect tags
    tag_map = defaultdict(list)
    for f in out_path.rglob("*.md"):
        if "MOC" in f.name or "QUARANTINE" in f.name: continue
        try:
            content = f.read_text(encoding="utf-8")
            tags = YAML_TAG_VALUE_PATTERN.findall(content)
            for t in tags: tag_map[t.strip().lower()].append(f.stem)
        except (FileNotFoundError, UnicodeDecodeError, OSError) as e:
            log.debug(f"Failed to read file {f} for tag collection: {e}")
            pass
        
    for tag, note_list in tqdm(tag_map.items()):
        if len(note_list) < 3: continue
        key = f"MOC_{tag}"
        if tracker.is_done("phase4", key): continue
        
        prompt = MOC_PROMPT.format(topic=tag, note_list="\n".join(f"- [[{n}]]" for n in note_list))
        try:
            body = generate_text_with_retries(backend, prompt=prompt, max_retries=config.max_retries)
            if body:
                title = f"MOC - {tag.title()}"
                fm = f"---\nid: {generate_id()}\ntitle: \"{title}\"\ntags:\n  - moc\n  - {tag}\n---\n\n# {title}\n\n{body}\n"
                (moc_dir / f"{safe_filename(title)}.md").write_text(fm, encoding="utf-8")
                log.info("Created MOC for %s", tag)
        except Exception as e:
            log.error(f"Failed to generate MOC for tag '{tag}': {e}")
        tracker.mark_done("phase4", key)

# ============================================================================
# MAIN / ENTRY
# ============================================================================

def main():
    load_dotenv_no_override()
    parser = argparse.ArgumentParser(description="Vault Reconstructor Pipeline")
    parser.add_argument("--phase", type=int, help="Run only specific phase (0-4)")
    args = parser.parse_args()

    cfg = Config()
    backend = get_llm_backend(cfg)
    
    phases = [
        lambda: run_phase0(backend, cfg),
        lambda: run_phase1(backend, cfg),
        lambda: run_phase2_rust(cfg),
        lambda: run_phase3(cfg),
        lambda: run_phase4(backend, cfg),
    ]

    if args.phase is not None:
        if 0 <= args.phase < len(phases):
            phases[args.phase]()
        else:
            log.error("Invalid phase: %s", args.phase)
            sys.exit(1)
    else:
        for p in phases:
            p()
            
    log.info("Reconstruction complete.")

if __name__ == "__main__":
    main()

