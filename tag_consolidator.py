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

from vault_reconstruct.json_extract import extract_json_dict
from vault_reconstruct.llm import LlmConfig, generate_text_with_retries, make_backend
from vault_reconstruct.config import get_vault_paths
from vault_reconstruct.env import load_dotenv_no_override

load_dotenv_no_override()

VAULT_PATH = os.environ.get("VAULT_PATH", str(get_vault_paths().output_vault))

# Backend selection:
# - VAULT_LLM_PROVIDER=ollama|gemini|azure
# - For gemini: set GEMINI_API_KEY
# - For azure: set AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY + VAULT_AZURE_MODEL (deployment name)
PROVIDER = os.environ.get("VAULT_LLM_PROVIDER", "ollama").strip().lower()

# Model defaults per provider
OLLAMA_LOCAL_MODEL = os.environ.get("VAULT_OLLAMA_MODEL", "qwen2.5-coder:0.5b-base-q8_0")
OLLAMA_CLOUD_MODEL = os.environ.get("VAULT_OLLAMA_CLOUD_MODEL", "gemma4:31b-cloud")
GEMINI_MODEL = os.environ.get("VAULT_GEMINI_MODEL", "gemini-2.5-flash")
AZURE_MODEL = os.environ.get("VAULT_AZURE_MODEL", "gpt-4.1-mini")
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
    ],
)
log = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_TAG_BLOCK_RE   = re.compile(r"(tags:\s*\n)((?:  - .+\n?)*)", re.MULTILINE)
_TAG_LINE_RE    = re.compile(r"^  - (.+)$", re.MULTILINE)


def _make_backend():
    if PROVIDER == "ollama":
        cfg = LlmConfig(
            provider="ollama",
            model=OLLAMA_LOCAL_MODEL,
            max_retries=3,
            ollama_cloud_model=OLLAMA_CLOUD_MODEL,
        )
    elif PROVIDER == "gemini":
        cfg = LlmConfig(provider="gemini", model=GEMINI_MODEL, max_retries=3)
    elif PROVIDER == "azure":
        cfg = LlmConfig(provider="azure", model=AZURE_MODEL, max_retries=3)
    else:
        raise SystemExit(f"Unknown VAULT_LLM_PROVIDER: {PROVIDER!r} (expected ollama/gemini/azure)")
    return make_backend(cfg)


def _verify_backend_or_raise(backend) -> None:
    """
    Fail-fast connectivity check.
    Note: for Ollama this verifies the local server is reachable; for remote
    providers it verifies credentials/network are working.
    """
    _ = generate_text_with_retries(backend, prompt="ping", max_retries=1)


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


def ai_map_tags(rare: list[str], broad: set[str], min_count: int, *, require_llm: bool) -> dict[str, str]:
    broad_str = ", ".join(sorted(broad))
    rare_str  = "\n".join(f"- {t}" for t in rare)
    backend = _make_backend()
    if require_llm:
        raw = generate_text_with_retries(
            backend,
            prompt=_MAP_PROMPT.format(broad_tags=broad_str, rare_tags=rare_str, min_count=min_count),
            max_retries=3,
        )
    else:
        try:
            raw = generate_text_with_retries(
                backend,
                prompt=_MAP_PROMPT.format(broad_tags=broad_str, rare_tags=rare_str, min_count=min_count),
                max_retries=3,
            )
        except Exception as exc:
            log.warning("[tag-map] LLM error: %s", exc)
            return {}

    result = extract_json_dict(raw or "") or {}
    cleaned: dict[str, str] = {}
    for k, v in result.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        if k in rare and (v == "DELETE" or v in broad):
            cleaned[k] = v
    return cleaned


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
    parser.add_argument(
        "--require-llm",
        action="store_true",
        help="Fail fast if the selected LLM backend is not reachable",
    )
    args = parser.parse_args()

    vault = Path(VAULT_PATH)
    if not vault.exists():
        log.error("Vault not found: %s", vault)
        sys.exit(1)

    if args.dry_run:
        log.info("DRY RUN — no files will be modified")

    try:
        backend = _make_backend()
        if args.require_llm:
            _verify_backend_or_raise(backend)
    except RuntimeError as exc:
        log.critical(str(exc))
        log.critical(
            "Set one of:\n"
            "  - VAULT_LLM_PROVIDER=ollama and run `ollama serve` (optional OLLAMA_API_KEY)\n"
            "  - VAULT_LLM_PROVIDER=gemini and set GEMINI_API_KEY\n"
            "  - VAULT_LLM_PROVIDER=azure and set AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY (+ VAULT_AZURE_MODEL)\n"
        )
        sys.exit(1)
    except Exception as exc:
        if args.require_llm:
            log.critical("LLM backend not reachable: %s", exc)
            sys.exit(1)
        log.warning("LLM backend not reachable (%s). Continuing without AI mappings.", exc)

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
        full_map.update(ai_map_tags(batch, broad, args.min, require_llm=args.require_llm))
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
