"""
Tag Consolidator for Obsidian Zettelkasten
===========================================
Uses gemma4:31b-cloud to intelligently merge rare/specific tags into broader
parent tags, remove junk tags, and rewrite YAML frontmatter across the vault.

Usage:
    python tag_consolidator.py --dry-run    # preview without writing
    python tag_consolidator.py              # apply changes
    python tag_consolidator.py --min 5      # consolidate tags used <5 times (default 3)
"""

import os
import re
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import ollama
from ollama import Client

VAULT_PATH  = r"C:\Users\Wren C\Documents\Coding stuff\Obsidian Vault"
CLOUD_MODEL = "gemma4:31b-cloud"
LOCAL_MODEL = "gemma3:4b"
MIN_COUNT   = 3
BATCH_SIZE  = 60
REQUEST_DELAY = 0.3

ALWAYS_REMOVE = {
    "dataview", "dataview-publisher", "publisher", "query",
    "Obsidian", "vault", "linking", "note-taking",
    "Flashard", "Flashard/Completed", "Flashcard", "Flashcard/Uncompleted",
    "smartselect", "favorites", "ideas", "session", "zoom",
    "ANAT10008", "ANAT10009", "ANAT10009/Definition", "ANAT20001",
    "PHPH10014", "PHPH10014/Definition", "VETS10021", "VETS20018", "VETS20019",
    "anatomy20001",
    "Unit", "Unit/Organisation", "Unit/Organisation/Course", "UnitHub",
    "LearningOutcomes", "Assignments", "CaseStudies", "Examples",
    "Concepts", "Overview", "Indexing", "Miscellaneous",
    "general", "module", "modules", "course", "curriculum",
    "exam", "exams", "practical", "tutorial", "assessment",
    "study", "note", "notes", "summary", "introduction",
    "definition", "Definitions", "Prefixes", "Suffixes",
    "MedicalTerminology", "AnatomicalTerms", "Theory",
    "reference", "Referrals", "citation", "completed",
    "phase1", "phase2", "bmbtt", "rér", "RER",
    "mg", "mg/kg", "mg/ml", "mcg", "mA", "mAs", "kV", "kVp",
    "pl", "cv", "ep", "ir", "cn", "cn1", "cn2", "cn4", "cn6",
    "cn7", "cn8", "cn11", "cn12", "cnv", "cnvii", "p3", "pu",
    "sid", "thr", "ts", "hr", "TS", "BLS", "ALS",
    "coffee", "geisha", "gesha", "FoodLore", "divas", "varietal",
    "wren", "warrior", "ghost", "RomanEmpire", "Palaeolithic",
    "Anglo-Saxon", "colombia", "europe", "woman", "zoom",
    "facts", "History", "chelonia",
}

FORCE_BROAD_TAGS = {
    "anatomy", "physiology", "pharmacology", "anaesthesia", "anaesthetic",
    "neuroscience", "neurology", "cardiology", "cardiac", "cardiovascular",
    "respiratory", "respiration", "kidney", "renal", "digestion", "digestive",
    "reproduction", "embryology", "endocrine", "hormones", "hormone",
    "behaviour", "ethology", "learning", "nutrition", "diet",
    "surgery", "orthopaedics", "sutures", "clinical", "diagnosis",
    "pharmacokinetics", "opioid", "opioids", "anaesthetics",
    "microbiology", "bacteria", "infection", "parasites",
    "dogs", "cats", "horse", "horses", "rabbit", "rabbits",
    "reptiles", "snake", "lizards", "tortoise", "birds", "avian",
    "ferrets", "rodents", "hedgehog",
    "haematology", "biochemistry", "urinalysis", "urine", "blood",
    "muscle", "muscles", "bone", "joints", "skeleton", "locomotion",
    "skin", "ophthalmology", "dentition", "teeth",
    "immune", "inflammation", "shock", "emergency", "pain", "analgesia",
    "welfare", "stress", "enrichment", "housing",
    "radiography", "imaging", "monitoring", "equipment",
    "fluids", "electrolytes", "acid-base",
    "development", "genetics", "evolution", "taxonomy", "zoology",
    "movement", "biomechanics", "exercise", "thermoregulation",
    "liver", "pancreas", "lungs", "heart", "brain", "thyroid",
    "bladder", "uterus", "ovary", "testis", "placenta",
    "catheter", "medication", "drugs", "dosage", "injection",
    "safety", "hygiene", "disinfection", "sterilisation",
    "regulation", "homeostasis", "transport", "membrane",
    "sensory", "vision", "hearing", "olfaction",
    "communication", "grooming", "feeding", "predation",
    "classification", "morphology", "tissue", "cells",
    "pressure", "temperature", "metabolism", "circulation",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("tag_consolidation.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_TAG_BLOCK_RE   = re.compile(r"(tags:\s*\n)((?:  - .+\n?)*)", re.MULTILINE)
_TAG_LINE_RE    = re.compile(r"^  - (.+)$", re.MULTILINE)


def _load_dotenv():
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    log.info("[.env] loaded")

_load_dotenv()


def _call_model(prompt: str, label: str = "") -> str | None:
    key   = os.environ.get("OLLAMA_API_KEY", "").strip()
    cloud = Client(host="https://ollama.com",
                   headers={"Authorization": f"Bearer {key}"}) if key else None
    pairs = ([(cloud, CLOUD_MODEL, "cloud")] if cloud else []) + [(None, LOCAL_MODEL, "local")]

    for client, model, src in pairs:
        for attempt in range(3):
            try:
                msgs = [{"role": "user", "content": prompt}]
                r = client.chat(model=model, messages=msgs) if client \
                    else ollama.chat(model=model, messages=msgs)
                return r["message"]["content"]
            except ollama.ResponseError:
                break
            except Exception as exc:
                wait = 5 * (2 ** attempt)
                log.warning("[%s] %s error (%s), retry in %ds", label, src, exc, wait)
                time.sleep(wait)
    return None


def scan_vault(vault: Path) -> dict[str, list[Path]]:
    tag_map: dict[str, list[Path]] = defaultdict(list)
    count = 0
    for fp in vault.rglob("*.md"):
        if fp.name.startswith(".") or "QUARANTINE_" in fp.name:
            continue
        count += 1
        text = fp.read_text(encoding="utf-8")
        fm   = _FRONTMATTER_RE.match(text)
        if not fm:
            continue
        for tag in _TAG_LINE_RE.findall(fm.group(1)):
            tag_map[tag.strip()].append(fp)
    log.info("Scanned %d notes, %d unique tags", count, len(tag_map))
    return dict(tag_map)


_MAP_PROMPT = """\
You are a tag consolidation assistant for an Obsidian veterinary nursing Zettelkasten vault.

BROAD TARGET TAGS (the vocabulary we want — map TO these):
{broad_tags}

RARE TAGS TO MAP (each currently used only 1-{min_count} times):
{rare_tags}

For each rare tag, either:
  - Map it to the single most semantically appropriate broad tag
  - Or return "DELETE" if it is: junk, a module code, too vague, non-veterinary, or a unit/dosage

Return ONLY a valid JSON object. No markdown fences. No explanation.
Format: {{"rare_tag": "broad_tag_or_DELETE", ...}}
"""


def ai_map_tags(rare: list[str], broad: set[str], min_count: int) -> dict[str, str]:
    broad_str = ", ".join(sorted(broad))
    rare_str  = "\n".join(f"- {t}" for t in rare)
    raw = _call_model(
        _MAP_PROMPT.format(broad_tags=broad_str, rare_tags=rare_str, min_count=min_count),
        label="tag-map"
    )
    if not raw:
        return {}
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    m = re.search(r"\{.*\}", clean, re.DOTALL)
    if not m:
        return {}
    try:
        result = json.loads(m.group(0))
        return {k: v for k, v in result.items()
                if k in rare and (v == "DELETE" or v in broad)}
    except json.JSONDecodeError:
        return {}


def rewrite_note(fp: Path, tag_map: dict[str, str], dry_run: bool) -> list[str]:
    original = fp.read_text(encoding="utf-8")
    fm_match = _FRONTMATTER_RE.match(original)
    if not fm_match:
        return []
    fm_content = fm_match.group(1)
    tb_match   = _TAG_BLOCK_RE.search(fm_content)
    if not tb_match:
        return []

    old_tags = [t.strip() for t in _TAG_LINE_RE.findall(tb_match.group(2))]
    new_tags = []
    changes  = []

    for tag in old_tags:
        if tag in ALWAYS_REMOVE:
            changes.append(f"DELETE {tag}")
        elif tag in tag_map:
            action = tag_map[tag]
            if action == "DELETE":
                changes.append(f"DELETE {tag}")
            else:
                new_tags.append(action)
                if action != tag:
                    changes.append(f"REMAP {tag} → {action}")
                else:
                    new_tags.append(tag)
        else:
            new_tags.append(tag)

    # Deduplicate preserving order
    seen, deduped = set(), []
    for t in new_tags:
        if t not in seen:
            seen.add(t)
            deduped.append(t)

    if not changes:
        return []

    if not dry_run:
        new_block = "tags:\n" + "".join(f"  - {t}\n" for t in deduped)
        new_fm    = _TAG_BLOCK_RE.sub(new_block, fm_content, count=1)
        new_text  = original.replace(fm_match.group(1), new_fm, 1)
        fp.write_text(new_text, encoding="utf-8")

    return changes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min", type=int, default=MIN_COUNT)
    args = parser.parse_args()

    vault = Path(VAULT_PATH)
    if not vault.exists():
        log.error("Vault not found: %s", vault)
        sys.exit(1)

    if args.dry_run:
        log.info("DRY RUN — no files will be modified")

    tag_counts = scan_vault(vault)

    # Build broad tag set
    broad = set(FORCE_BROAD_TAGS)
    for tag, notes in tag_counts.items():
        if len(notes) >= args.min and tag not in ALWAYS_REMOVE:
            broad.add(tag)
    log.info("%d broad tags identified", len(broad))

    # Find rare tags
    rare = [
        tag for tag, notes in tag_counts.items()
        if len(notes) < args.min
        and tag not in ALWAYS_REMOVE
        and tag not in broad
    ]
    log.info("%d rare tags (used < %d times) to process", len(rare), args.min)

    # Build full mapping
    full_map: dict[str, str] = {t: "DELETE" for t in ALWAYS_REMOVE if t in tag_counts}

    batches = [rare[i:i+BATCH_SIZE] for i in range(0, len(rare), BATCH_SIZE)]
    for i, batch in enumerate(batches):
        log.info("AI batch %d/%d...", i+1, len(batches))
        full_map.update(ai_map_tags(batch, broad, args.min))
        time.sleep(REQUEST_DELAY)

    # Save mapping for review
    map_path = Path(__file__).parent / "tag_mapping.json"
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(full_map.items())), f, indent=2)
    log.info("Mapping saved to tag_mapping.json — review before applying if unsure")

    # Apply to vault
    change_log     = []
    total_changes  = 0
    affected_notes = 0

    for fp in vault.rglob("*.md"):
        if fp.name.startswith(".") or "QUARANTINE_" in fp.name:
            continue
        changes = rewrite_note(fp, full_map, args.dry_run)
        if changes:
            affected_notes += 1
            total_changes  += len(changes)
            change_log.append({"file": fp.name, "changes": changes})

    log_path = Path(__file__).parent / "tag_changes.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(change_log, f, indent=2)

    log.info("Done — %d notes affected, %d tag operations", affected_notes, total_changes)
    log.info("Full change log: tag_changes.json")
    if args.dry_run:
        log.info("Re-run without --dry-run to apply")


if __name__ == "__main__":
    main()
