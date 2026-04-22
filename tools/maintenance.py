import sys
from pathlib import Path
# Add repo root to path so we can import the vault_reconstruct package
sys.path.append(str(Path(__file__).resolve().parent.parent))
"""
vault_maintenance.py

Unified vault maintenance utility for Vault Reconstructor.
Includes health reports, tag normalization, link repair, and note expansion.
"""

import sys
import os
import re
import json
import logging
import argparse
import difflib
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Set, List, Tuple, Optional

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

from vault_reconstruct.config import get_vault_paths
from vault_reconstruct.env import load_dotenv_no_override

# ============================================================================
# REGEX PATTERNS (Pre-compiled for performance)
# ============================================================================

# Matches hashtags with word boundary (e.g., #tag, #complex/tag, #tag-with-dash)
TAG_PATTERN = re.compile(r"(?<!\w)#([\w/\-]+)")

# Matches YAML frontmatter tags section (e.g., tags:\n  - tag1\n  - tag2)
FRONTMATTER_TAGS_PATTERN = re.compile(r"^---\s*\ntags:\s*\n((?:  - .+\n?)+)", re.M)

# Extracts individual tag values from YAML list (e.g., "  - tag1" → "tag1")
YAML_TAG_VALUE_PATTERN = re.compile(r"  - (.+)")

# Removes frontmatter block from content (e.g., ---\n...\n---\n)
FRONTMATTER_STRIP_PATTERN = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)

# Matches wiki-style links (e.g., [[Note Title]] or [[Note|Display Text]])
WIKILINK_PATTERN = re.compile(r"\[\[([^\]|#]+)")

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    vault_path: Path = Path(get_vault_paths().output_vault)
    dry_run: bool = False
    fix_tags: bool = False
    fix_links: bool = False
    expand_short: bool = False
    repair_quarantine: bool = False
    export_json: bool = False
    short_threshold: int = 100

    # Tag map (UK English standardized)
    tag_mappings: Dict[str, str] = field(default_factory=lambda: {
        "behavior": "behaviour", "color": "colour", "hematology": "haematology",
        "hemorrhage": "haemorrhage", "edema": "oedema", "center": "centre",
        "analyze": "analyse", "paralyze": "paralyse"
    })

# ============================================================================
# CORE ANALYZER
# ============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

class VaultAnalyzer:
    def __init__(self, vault_path: Path):
        self.path = vault_path
        self.notes: Dict[str, Path] = {}
        self.tags: Dict[str, Set[str]] = defaultdict(set)
        self.broken: Dict[str, List[str]] = defaultdict(list)
        self.short: List[Tuple[str, Path, int]] = []
        self.quarantine: List[Path] = []
        self._title_index: Dict[str, str] = {}

    def scan(self, threshold: int, use_native: bool = True):
        scanner_exe = Path(__file__).parent / "scanner.exe"
        if use_native and scanner_exe.exists():
            log.info("Using native C++ scanner engine...")
            import subprocess
            try:
                # First run: Analysis
                res = subprocess.run([str(scanner_exe), str(self.path)], capture_output=True, text=True, check=True)
                data = json.loads(res.stdout)
                log.info(f"Native Scan Result: {data}")
                
                # Update basic metrics (further population happens in manual loop for now to maintain compat)
                # However, for speed, we now have a working health report instantly.
                # We still need notes{} populated for other Python logic.
            except Exception as e:
                log.warning(f"Native scanner failed, falling back to Python: {e}")

        log.info("Scanning vault: %s", self.path)
        for f in self.path.rglob("*.md"):
            if f.name.startswith("."): continue
            if "QUARANTINE_" in f.name:
                self.quarantine.append(f)
                continue
            
            title = f.stem
            self.notes[title] = f
            self._title_index[title.lower()] = title
            try:
                # Identify tags/length only if not already handled or needed for Python logic
                content = f.read_text(encoding="utf-8")
                for tag in TAG_PATTERN.findall(content):
                    self.tags[tag.lower()].add(title)
                fm_match = FRONTMATTER_TAGS_PATTERN.search(content)
                if fm_match:
                    for tag in YAML_TAG_VALUE_PATTERN.findall(fm_match.group(1)):
                        self.tags[tag.strip().lower()].add(title)
                
                body = FRONTMATTER_STRIP_PATTERN.sub("", content).strip()
                if len(body) < threshold: self.short.append((title, f, len(body)))
            except (FileNotFoundError, UnicodeDecodeError, OSError) as e:
                log.debug(f"Failed to scan file {f}: {e}")
                pass

    def analyze_links(self):
        for title, path in self.notes.items():
            try:
                content = path.read_text(encoding="utf-8")
                links = WIKILINK_PATTERN.findall(content)
                for l in links:
                    target = l.strip()
                    if target not in self.notes:
                        self.broken[title].append(target)
            except (FileNotFoundError, UnicodeDecodeError, OSError) as e:
                log.debug(f"Failed to analyze links in {path}: {e}")
                pass

    def find_fuzzy_match(self, broken: str) -> Optional[str]:
        norm = broken.lower().strip()
        if norm in self._title_index: return self._title_index[norm]
        matches = difflib.get_close_matches(norm, self._title_index.keys(), n=1, cutoff=0.8)
        return self._title_index[matches[0]] if matches else None

# ============================================================================
# ACTIONS
# ============================================================================

def perform_tag_fix(analyzer: VaultAnalyzer, cfg: Config):
    log.info("Standardizing tags...")
    modified = 0
    for tag, notes in analyzer.tags.items():
        if tag in cfg.tag_mappings:
            new_tag = cfg.tag_mappings[tag]
            log.info(f"  {tag} -> {new_tag}")
            if not cfg.dry_run:
                for n_title in notes:
                    path = analyzer.notes[n_title]
                    content = path.read_text(encoding="utf-8")
                    updated = content.replace(f"#{tag}", f"#{new_tag}").replace(f"- {tag}", f"- {new_tag}")
                    if updated != content:
                        path.write_text(updated, encoding="utf-8")
                        modified += 1
    log.info(f"Finished. Modified {modified} files.")

def perform_link_fix(analyzer: VaultAnalyzer, cfg: Config):
    log.info("Repairing broken links...")
    fixed = 0
    for note, broken_list in analyzer.broken.items():
        path = analyzer.notes[note]
        content = path.read_text(encoding="utf-8")
        updated = content
        for b in broken_list:
            match = analyzer.find_fuzzy_match(b)
            if match:
                log.info(f"  {note}: [[{b}]] -> [[{match}]]")
                updated = updated.replace(f"[[{b}]]", f"[[{match}]]").replace(f"[[{b}|", f"[[{match}|")
        if updated != content and not cfg.dry_run:
            path.write_text(updated, encoding="utf-8")
            fixed += 1
    log.info(f"Finished. Fixed links in {fixed} files.")

# ============================================================================
# MAIN
# ============================================================================

def main():
    load_dotenv_no_override()
    parser = argparse.ArgumentParser(description="Vault Maintenance")
    parser.add_argument("--fix-tags", action="store_true")
    parser.add_argument("--fix-links", action="store_true")
    parser.add_argument("--expand", action="store_true")
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    cfg.fix_tags = args.fix_tags
    cfg.fix_links = args.fix_links
    cfg.expand_short = args.expand
    cfg.repair_quarantine = args.repair
    cfg.dry_run = args.dry_run

    analyzer = VaultAnalyzer(cfg.vault_path)
    analyzer.scan(cfg.short_threshold)
    analyzer.analyze_links()

    log.info("\n=== VAULT HEALTH SUMMARY ===")
    log.info(f"Total Notes:      {len(analyzer.notes)}")
    log.info(f"Broken Links:     {sum(len(v) for v in analyzer.broken.values())}")
    log.info(f"Short/Empty:      {len(analyzer.short)}")
    log.info(f"Quarantined:      {len(analyzer.quarantine)}")
    
    # If using native, we can run the fix directly
    scanner_exe = Path(__file__).parent / "scanner.exe"
    if scanner_exe.exists() and (cfg.fix_tags or cfg.fix_links) and not cfg.dry_run:
        log.info("Running native auto-fix...")
        import subprocess
        subprocess.run([str(scanner_exe), str(cfg.vault_path), "--fix"], check=True)
    else:
        if cfg.fix_tags: perform_tag_fix(analyzer, cfg)
        if cfg.fix_links: perform_link_fix(analyzer, cfg)
    
    log.info("\nMaintenance cycle complete.")

if __name__ == "__main__":
    main()

