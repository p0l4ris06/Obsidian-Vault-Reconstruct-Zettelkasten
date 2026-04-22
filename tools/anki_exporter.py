from __future__ import annotations

import sys
from pathlib import Path
# Add repo root to path so we can import the vault_reconstruct package
sys.path.append(str(Path(__file__).resolve().parent.parent))
"""
Anki Flashcard Generator for Obsidian Zettelkasten
==================================================
Converts vault zettels into Anki-compatible flashcards.

Features:
  - Tracks converted notes (incremental updates)
  - Detects modified notes and regenerates their cards
  - Organises cards into Anki decks matching vault tags
  - Skips MOCs and literature notes (zettels only)
  - Exports one .apkg per subject deck for easy import

Backends (shared with the rest of this repo):
  - Ollama (default): VAULT_LLM_PROVIDER=ollama, optional OLLAMA_API_KEY
  - Gemini:          VAULT_LLM_PROVIDER=gemini, set GEMINI_API_KEY
  - Azure OpenAI:    VAULT_LLM_PROVIDER=azure, set AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY

Usage:
  python anki_exporter.py
  python anki_exporter.py --reset
  python anki_exporter.py --deck anatomy
"""



import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import genanki  # type: ignore
except ImportError:  # pragma: no cover
    print("Install genanki first:  pip install genanki")
    raise

from vault_reconstruct.json_extract import extract_json_array
from vault_reconstruct.llm import LlmConfig, generate_text_with_retries, make_backend
from vault_reconstruct.config import get_vault_paths
from vault_reconstruct.env import load_dotenv_no_override

load_dotenv_no_override()


# ============================================================================
# CONFIG (defaults; override via env or CLI)
# ============================================================================

DEFAULT_VAULT_PATH = os.environ.get("VAULT_PATH", str(get_vault_paths().output_vault))
DEFAULT_OUTPUT_DIR = os.environ.get(
    "ANKI_OUTPUT_DIR",
    r"C:\Users\Wren C\Documents\Coding stuff\Anki Decks",
)
TRACKER_FILE = os.environ.get("ANKI_TRACKER_FILE", "anki_tracker.json")

# Provider selection shared with rest of repo
DEFAULT_PROVIDER = os.environ.get("VAULT_LLM_PROVIDER", "ollama").strip().lower()

# Ollama model defaults (cloud-first optional via OLLAMA_API_KEY)
OLLAMA_LOCAL_MODEL = os.environ.get("VAULT_OLLAMA_MODEL", "qwen2.5-coder:0.5b-base-q8_0")
OLLAMA_CLOUD_MODEL = os.environ.get("VAULT_OLLAMA_CLOUD_MODEL", "gemma4:31b-cloud")

# Gemini / Azure model defaults
GEMINI_MODEL = os.environ.get("VAULT_GEMINI_MODEL", "gemini-2.5-flash")
AZURE_MODEL = os.environ.get("VAULT_AZURE_MODEL", "gpt-4.1-mini")

REQUEST_DELAY = float(os.environ.get("ANKI_REQUEST_DELAY", "0.4"))
MAX_RETRIES = int(os.environ.get("ANKI_MAX_RETRIES", "3"))

# Anki deck naming
DECK_PREFIX = os.environ.get("ANKI_DECK_PREFIX", "Vet Nursing")


DEFAULT_DECK = "General"

# High-level separation:
# - Anatomy is its own top-level deck with body-system subdecks.
# - Clinical Theory is its own top-level deck with subdecks.
_ANATOMY_TRIGGER_TAGS = {"anatomy"}
_CLINICAL_TRIGGER_TAGS = {
    "clinical",
    "emergency",
    "diagnosis",
    "anaesthesia",
    "anaesthetic",
    "surgery",
    "orthopaedics",
}

# Body systems (used as subdecks under Anatomy, and optionally under Clinical Theory).
_SYSTEM_TAG_MAP: dict[str, str] = {
    # Alimentary
    "alimentary": "Alimentary",
    "digestion": "Alimentary",
    "digestive": "Alimentary",
    "gi": "Alimentary",
    "gastrointestinal": "Alimentary",
    # Renal / urinary
    "renal": "Renal",
    "kidney": "Renal",
    "urinary": "Urinary",
    "bladder": "Urinary",
    # Respiratory
    "respiratory": "Respiratory",
    "respiration": "Respiratory",
    # Cardiovascular
    "cardiovascular": "Cardiovascular",
    "cardiology": "Cardiovascular",
    "cardiac": "Cardiovascular",
    # Endocrine
    "endocrine": "Endocrine",
    "hormones": "Endocrine",
    # Nervous system
    "neurology": "Nervous",
    "neuroscience": "Nervous",
    "nervous": "Nervous",
    # Reproductive
    "reproduction": "Reproductive",
    "embryology": "Reproductive",
    # Musculoskeletal
    "musculoskeletal": "Musculoskeletal",
    "orthopaedics": "Musculoskeletal",
    # Blood / microbes (not a body system per se, but useful grouping)
    "haematology": "Haematology",
    "microbiology": "Microbiology",
}

# Clinical theory subdecks (if no body-system tag is present).
_CLINICAL_SUBDECK_TAG_MAP: dict[str, str] = {
    "emergency": "Emergency",
    "diagnosis": "Diagnostics",
    "anaesthesia": "Anaesthesia",
    "anaesthetic": "Anaesthesia",
    "surgery": "Surgery",
    "orthopaedics": "Surgery",
    "clinical": "General",
}

# Fallback mapping for everything else (non-anatomy / non-clinical).
_DECK_TAG_MAP: dict[str, str] = {
    "physiology": "Physiology",
    "pharmacology": "Pharmacology",
    "pharmacokinetics": "Pharmacology",
    "behaviour": "Behaviour",
    "ethology": "Behaviour",
    "nutrition": "Nutrition",
    "dogs": "Species::Dogs",
    "cats": "Species::Cats",
    "horse": "Species::Equine",
    "horses": "Species::Equine",
    "rabbit": "Species::Rabbit",
    "reptiles": "Species::Reptiles",
    "snake": "Species::Reptiles",
    "birds": "Species::Avian",
    "avian": "Species::Avian",
    # If a note is about a system but not tagged anatomy/clinical, still group it.
    "respiratory": "Physiology::Respiratory",
    "renal": "Physiology::Renal",
    "cardiology": "Physiology::Cardiovascular",
    "endocrine": "Physiology::Endocrine",
    "neurology": "Physiology::Nervous",
}


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ============================================================================
# REGEX
# ============================================================================

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_TAG_LINE_RE = re.compile(r"^  - (.+)$", re.MULTILINE)
_TYPE_RE = re.compile(r"^type:\s*(.+)$", re.MULTILINE)
_TITLE_RE = re.compile(r'^title:\s*"?(.+?)"?\s*$', re.MULTILINE)
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")


# ============================================================================
# LLM prompt + backend
# ============================================================================

_CARD_PROMPT = """\
You are creating Anki flashcards for a veterinary nursing student.

Note title: {title}
Note content:
{body}

Generate 2-4 high-quality flashcard Q&A pairs from this note.

Rules:
- Questions should test understanding, not just recall of exact wording
- Answers should be concise (1-3 sentences)
- Cover the most clinically or academically important points
- Do NOT generate trivial or obvious questions
- Each question must be answerable from the note content alone

Return ONLY a valid JSON array. No markdown fences. No explanation.
Format:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]
"""


def _make_llm_backend(provider: str):
    provider = (provider or "ollama").strip().lower()
    if provider == "ollama":
        return make_backend(
            LlmConfig(
                provider="ollama",
                model=OLLAMA_LOCAL_MODEL,
                max_retries=MAX_RETRIES,
                ollama_cloud_model=OLLAMA_CLOUD_MODEL,
            )
        )
    if provider == "gemini":
        return make_backend(
            LlmConfig(
                provider="gemini",
                model=GEMINI_MODEL,
                max_retries=MAX_RETRIES,
            )
        )
    if provider == "azure":
        return make_backend(
            LlmConfig(
                provider="azure",
                model=AZURE_MODEL,
                max_retries=MAX_RETRIES,
            )
        )
    raise SystemExit(f"Unknown VAULT_LLM_PROVIDER: {provider!r} (expected ollama/gemini/azure)")


def generate_cards(backend: Any, note: dict[str, Any]) -> list[dict[str, str]]:
    raw = generate_text_with_retries(
        backend,
        prompt=_CARD_PROMPT.format(title=note["title"], body=note["body"]),
        max_retries=MAX_RETRIES,
    )
    arr = extract_json_array(raw or "")
    if not arr:
        return []

    out: list[dict[str, str]] = []
    for c in arr:
        if not isinstance(c, dict):
            continue
        q = c.get("question")
        a = c.get("answer")
        if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
            out.append({"question": q.strip(), "answer": a.strip()})
    return out


# ============================================================================
# NOTE PARSING
# ============================================================================


def _content_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def parse_note(fp: Path) -> dict[str, Any] | None:
    text = fp.read_text(encoding="utf-8")
    fm = _FRONTMATTER_RE.match(text)
    if not fm:
        return None

    fm_text = fm.group(1)
    type_m = _TYPE_RE.search(fm_text)
    note_type = type_m.group(1).strip().lower() if type_m else "zettel"

    if note_type in ("moc", "literature"):
        return None

    title_m = _TITLE_RE.search(fm_text)
    title = title_m.group(1).strip() if title_m else fp.stem

    tags = [t.strip().lower() for t in _TAG_LINE_RE.findall(fm_text) if t.strip()]

    body = text[fm.end() :].strip()
    body = _WIKILINK_RE.sub(r"\1", body)  # keep display text, drop brackets
    body = re.sub(r"^#+\s+", "", body, flags=re.MULTILINE)  # remove headings

    return {
        "title": title,
        "tags": tags,
        "type": note_type,
        "body": body[:2000],
        "path": str(fp),
        "hash": _content_hash(text),
    }


def _pick_system_subdeck(tags: list[str]) -> str | None:
    for tag in tags:
        mapped = _SYSTEM_TAG_MAP.get(tag)
        if mapped:
            return mapped
    return None


def _pick_clinical_subdeck(tags: list[str]) -> str:
    for tag in tags:
        mapped = _CLINICAL_SUBDECK_TAG_MAP.get(tag)
        if mapped:
            return mapped
    return "General"


def get_deck_name(tags: list[str]) -> str:
    tag_set = set(tags)

    if tag_set & _ANATOMY_TRIGGER_TAGS:
        system = _pick_system_subdeck(tags)
        return f"{DECK_PREFIX}::Anatomy::{system}" if system else f"{DECK_PREFIX}::Anatomy"

    if tag_set & _CLINICAL_TRIGGER_TAGS:
        system = _pick_system_subdeck(tags)
        if system:
            return f"{DECK_PREFIX}::Clinical Theory::{system}"
        return f"{DECK_PREFIX}::Clinical Theory::{_pick_clinical_subdeck(tags)}"

    for tag in tags:
        mapped = _DECK_TAG_MAP.get(tag)
        if mapped:
            return f"{DECK_PREFIX}::{mapped}"

    system = _pick_system_subdeck(tags)
    if system:
        return f"{DECK_PREFIX}::Theory::{system}"

    return f"{DECK_PREFIX}::{DEFAULT_DECK}"


# ============================================================================
# ANKI BUILDING
# ============================================================================


def _stable_id(name: str, salt: int) -> int:
    return int(hashlib.md5(name.encode("utf-8")).hexdigest()[:8], 16) + salt


ANKI_MODEL = genanki.Model(
    _stable_id("VetNursing Basic", 0),
    "Vet Nursing Basic",
    fields=[
        {"name": "Question"},
        {"name": "Answer"},
        {"name": "Source"},
        {"name": "Tags"},
    ],
    templates=[
        {
            "name": "Card 1",
            "qfmt": "<div class='question'>{{Question}}</div><div class='source'>{{Source}}</div>",
            "afmt": "{{FrontSide}}<hr><div class='answer'>{{Answer}}</div>",
        }
    ],
    css="""
        .card { font-family: Arial, sans-serif; font-size: 16px; max-width: 650px; margin: auto; }
        .question { font-weight: bold; margin-bottom: 8px; }
        .answer { color: #2c5f2e; margin-top: 8px; }
        .source { font-size: 11px; color: #888; font-style: italic; margin-top: 8px; }
    """,
)


def _deck_id(name: str) -> int:
    return _stable_id(name, 1)


# ============================================================================
# TRACKER
# ============================================================================


class AnkiTracker:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, Any] = {}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self.data = {}

    def needs_update(self, note_path: str, content_hash: str) -> bool:
        entry = self.data.get(note_path)
        return entry is None or entry.get("hash") != content_hash

    def mark_done(self, note_path: str, content_hash: str, card_count: int) -> None:
        self.data[note_path] = {
            "hash": content_hash,
            "cards": card_count,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")


# ============================================================================
# MAIN
# ============================================================================


def _safe_deck_filename(deck_name: str) -> str:
    # Keep "::" meaningful but filesystem safe.
    base = deck_name.replace("::", "_").strip()
    base = re.sub(r"[^\w\s-]", "", base)
    base = re.sub(r"\s+", " ", base).strip()
    return base[:120] or "Deck"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=str, default=DEFAULT_VAULT_PATH)
    parser.add_argument("--out", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reset", action="store_true", help="Regenerate all cards")
    parser.add_argument("--deck", type=str, default=None, help="Only export notes containing this tag")
    parser.add_argument("--provider", type=str, default=DEFAULT_PROVIDER, help="ollama|gemini|azure")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan + report what would be generated, but do not call the LLM or write .apkg files",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    vault = Path(args.vault)
    if not vault.exists():
        raise SystemExit(f"Vault path does not exist: {vault}")

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker = AnkiTracker(Path(__file__).parent / TRACKER_FILE)
    if args.reset:
        tracker.data = {}
        log.info("Tracker reset — regenerating all cards")

    backend = None if args.dry_run else _make_llm_backend(args.provider)

    all_notes: list[dict[str, Any]] = []
    for fp in vault.rglob("*.md"):
        if fp.name.startswith(".") or fp.name.startswith("QUARANTINE_"):
            continue
        note = parse_note(fp)
        if note:
            all_notes.append(note)

    log.info("%d eligible notes found", len(all_notes))

    pending = [n for n in all_notes if tracker.needs_update(n["path"], n["hash"])]
    log.info("%d notes need card generation", len(pending))

    if args.deck:
        wanted = args.deck.strip().lower()
        pending = [n for n in pending if wanted in n["tags"]]
        log.info("Filtered to %d notes with tag '%s'", len(pending), wanted)

    if args.dry_run:
        decks_count: dict[str, int] = {}
        for n in pending:
            decks_count[get_deck_name(n["tags"])] = decks_count.get(get_deck_name(n["tags"]), 0) + 1
        log.info("[dry-run] Would generate cards for %d notes across %d decks.", len(pending), len(decks_count))
        for deck_name, n_count in sorted(decks_count.items(), key=lambda x: (-x[1], x[0])):
            log.info("[dry-run] Deck: %s  (notes: %d)", deck_name, n_count)
        log.info("[dry-run] No LLM calls made. No .apkg files written.")
        return 0

    all_cards: list[dict[str, Any]] = []
    for i, note in enumerate(pending):
        log.info("[%d/%d] %s", i + 1, len(pending), note["title"][:80])
        try:
            assert backend is not None
            cards = generate_cards(backend, note)
        except Exception as exc:
            log.warning("  → LLM error: %s", exc)
            cards = []

        deck_name = get_deck_name(note["tags"])
        if cards:
            for card in cards:
                all_cards.append(
                    {
                        "question": card["question"],
                        "answer": card["answer"],
                        "deck": deck_name,
                        "source": note["title"],
                        "tags": note["tags"],
                    }
                )
            tracker.mark_done(note["path"], note["hash"], len(cards))
            log.info("  → %d cards for deck: %s", len(cards), deck_name)
        else:
            tracker.mark_done(note["path"], note["hash"], 0)
            log.warning("  → No cards generated")

        if i % 20 == 0:
            tracker.save()
        time.sleep(REQUEST_DELAY)

    tracker.save()
    log.info("Generated %d new cards total", len(all_cards))

    if not all_cards:
        log.info("No new cards to export.")
        return 0

    decks_map: dict[str, list[dict[str, Any]]] = {}
    for card in all_cards:
        decks_map.setdefault(card["deck"], []).append(card)

    for deck_name, cards in decks_map.items():
        deck = genanki.Deck(_deck_id(deck_name), deck_name)
        for card in cards:
            anki_note = genanki.Note(
                model=ANKI_MODEL,
                fields=[
                    card["question"],
                    card["answer"],
                    card["source"],
                    " ".join(card["tags"]),
                ],
                guid=genanki.guid_for(card["question"]),
                tags=card["tags"],
            )
            deck.add_note(anki_note)

        out_path = output_dir / f"{_safe_deck_filename(deck_name)}.apkg"
        genanki.Package(deck).write_to_file(str(out_path))
        log.info("Exported %d cards → %s", len(cards), out_path)

    log.info("All decks exported to %s", output_dir)
    log.info("Import .apkg files into Anki via File > Import")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



