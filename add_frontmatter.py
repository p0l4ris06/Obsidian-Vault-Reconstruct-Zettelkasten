"""
Frontmatter Injector for Obsidian Vaults
========================================
Walks through a vault and ensures all .md files have valid YAML frontmatter.
Special logic: For Anatomy notes, reorders tags by [BODY SYSTEM] -> [CORE] -> [SPECIES].

Usage:
    python add_frontmatter.py --vault "D:\\OBSIDIAN\\Uni Sync"
    python add_frontmatter.py --dry-run
"""

import os
import re
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Set, Tuple

# ============================================================================
# CATEGORIES FOR TAG ORDERING
# ============================================================================

BODY_SYSTEMS = {
    "alimentary", "digestion", "digestive", "gi", "gastrointestinal",
    "renal", "kidney", "urinary", "bladder",
    "respiratory", "respiration",
    "cardiovascular", "cardiology", "cardiac",
    "endocrine", "hormones",
    "neurology", "neuroscience", "nervous",
    "reproduction", "embryology", "reproductive",
    "musculoskeletal", "orthopaedics", "orthopaedic", "skeletal", "muscular",
    "haematology", "blood",
    "microbiology", "microbes",
    "integumentary", "skin",
    "sensory", "eyes", "ears",
}

SPECIES = {
    "canine", "dog", "dogs",
    "feline", "cat", "cats",
    "equine", "horse", "horses",
    "bovine", "cow", "cattle",
    "ovine", "sheep",
    "porcine", "pig",
    "rabbit", "lagomorph",
    "reptile", "snake", "lizard", "tortoise",
    "avian", "bird", "birds",
    "exotics", "smallies",
}

CORE_SUBJECTS = {
    "anatomy", "physiology", "pathology", "pharmacology",
    "clinical", "theory", "surgery", "anaesthesia",
    "nursing", "emergency", "diagnostics",
}

# ============================================================================
# REGEX
# ============================================================================

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_TAG_LINE_RE = re.compile(r"^  - (.+)$", re.MULTILINE)
_INLINE_TAG_RE = re.compile(r"(?<!\w)#([\w/\-]+)")
_ID_RE = re.compile(r"^id:\s*(.+)$", re.MULTILINE)
_TITLE_RE = re.compile(r"^title:\s*\"?(.+?)\"?\s*$", re.MULTILINE)
_TYPE_RE = re.compile(r"^type:\s*(.+)$", re.MULTILINE)

# ============================================================================
# LOGIC
# ============================================================================

def generate_id() -> str:
    """Matches the ID format in the rest of the pipeline."""
    now = datetime.now()
    return now.strftime("%Y%m%d%H%M") + f"{random.randint(0, 9999):04d}"

def transform_tags(tags: Set[str]) -> List[str]:
    """
    Transforms and sorts tags with an explicit hierarchical structure for anatomy:
    anatomy/body-system/cardiac
    Then adds core subjects, species, and other tags.
    """
    systems = sorted([t for t in tags if t in BODY_SYSTEMS])
    species = sorted([t for t in tags if t in SPECIES])
    core = sorted([t for t in tags if t in CORE_SUBJECTS and t != "anatomy"])
    others = sorted([t for t in tags if t not in BODY_SYSTEMS and t not in SPECIES and t not in CORE_SUBJECTS and t != "anatomy"])

    final_tags = []
    is_anatomy = "anatomy" in tags

    if is_anatomy and systems:
        for sys in systems:
            final_tags.append(f"anatomy/body-system/{sys}")
    elif is_anatomy:
        final_tags.append("anatomy")
        for sys in systems: 
            final_tags.append(sys)
    else:
        for sys in systems:
            final_tags.append(sys)

    final_tags.extend(core)
    final_tags.extend(species)
    final_tags.extend(others)
            
    return final_tags

def process_file(fp: Path, dry_run: bool = False) -> bool:
    content = fp.read_text(encoding="utf-8")
    
    # 1. Extract existing frontmatter and ID
    fm_match = _FRONTMATTER_RE.match(content)
    existing_fm = fm_match.group(1) if fm_match else ""
    body = content[fm_match.end():] if fm_match else content
    
    existing_id = None
    if existing_fm:
        id_m = _ID_RE.search(existing_fm)
        if id_m:
            existing_id = id_m.group(1).strip()
            
    # 2. Extract Tags
    tags = set()
    # From YAML
    if existing_fm:
        for t in _TAG_LINE_RE.findall(existing_fm):
            tags.add(t.strip().lower())
    
    # From body (#tag)
    # We only scan the first 2000 chars for performance and to avoid code snippets
    found_inline = _INLINE_TAG_RE.findall(body[:2000])
    for t in found_inline:
        # Avoid things that look like hex colors or headers
        if not re.match(r"^[0-9a-fA-F]{3,6}$", t):
            tags.add(t.strip().lower())
            
    # 3. Determine Type and Title
    note_type = "zettel"
    if existing_fm:
        type_m = _TYPE_RE.search(existing_fm)
        if type_m:
            note_type = type_m.group(1).strip()
    
    if "Year 1" in str(fp) or "Year 2" in str(fp):
        # Could refine this logic if needed
        pass
    
    title = fp.stem
    if existing_fm:
        title_m = _TITLE_RE.search(existing_fm)
        if title_m:
            title = title_m.group(1).strip()

    # 4. Sort Tags
    # Re-ordering rule: for anatomy order tags by BODY SYSTEM not SPECIES
    sorted_tag_list = transform_tags(tags)
    
    # 5. Build New Frontmatter
    note_id = existing_id or generate_id()
    created = datetime.now().strftime("%Y-%m-%d")
    
    fm_lines = [
        "---",
        f"id: {note_id}",
        f"title: \"{title}\"",
        "tags:"
    ]
    if sorted_tag_list:
        for t in sorted_tag_list:
            fm_lines.append(f"  - {t}")
    else:
        fm_lines.append("  - general")
        
    fm_lines.append(f"type: {note_type}")
    fm_lines.append(f"created: {created}")
    fm_lines.append("---\n\n")
    
    new_content = "\n".join(fm_lines) + body.lstrip()
    
    if new_content != content:
        if not dry_run:
            fp.write_text(new_content, encoding="utf-8")
        return True
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=str, default=r"D:\OBSIDIAN\Uni Sync")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    vault = Path(args.vault)
    if not vault.exists():
        print(f"Vault path not found: {vault}")
        return
    
    modified = 0
    total = 0
    
    print(f"Scanning {vault}...")
    for fp in vault.rglob("*.md"):
        if ".obsidian" in fp.parts or ".trash" in fp.parts:
            continue
            
        total += 1
        try:
            if process_file(fp, args.dry_run):
                modified += 1
        except Exception as e:
            print(f"Error processing {fp}: {e}")
            
    if args.dry_run:
        print(f"DRY RUN: Would have modified {modified} / {total} files.")
    else:
        print(f"Successfully updated {modified} / {total} files.")

if __name__ == "__main__":
    main()
