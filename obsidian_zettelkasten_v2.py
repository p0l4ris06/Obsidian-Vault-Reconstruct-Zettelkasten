"""
Obsidian Zettelkasten Converter — v2.1 (Fixed)
Converts lecture notes into a full Zettelkasten vault with atomic notes,
YAML frontmatter, tags, classification, wikilinks, and MOCs.

Fixes over v2:
  - generate_id() now uses timestamp + random suffix (no restart collisions)
  - Phase 3 regex linking now correctly matches multi-word titles
  - Thread-safe wikilink pattern cache (RLock)
  - Tracker saves every write, not every 10 (crash-safe)
  - Exponential backoff in call_ollama
  - AI link skip threshold raised to 8 (was 3, too aggressive)
  - _count_and_maybe_restart passes full env to child process
  - MOC tag_map cached, not rebuilt per-topic
  - Phase 3 uses full-text substring scan before regex (fast pre-filter)

Usage:
    Ensure Ollama is running (ollama serve) and your model is pulled, then:
        python obsidian_zettelkasten_v2.py

    For cloud inference, create a .env file in the same directory:
        OLLAMA_API_KEY=your-key-here
        OLLAMA_HOST=http://localhost:11434

    Or set manually:
        $env:OLLAMA_API_KEY = "your-key-here"   # PowerShell
        export OLLAMA_API_KEY="your-key-here"   # bash/zsh
"""

import json
import os
import re
import random
import subprocess
import sys
import time
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

import ollama
from ollama import Client


# ============================================================================
# .ENV LOADER
# Reads a .env file in the same directory as this script.
# Supports: OLLAMA_API_KEY, OLLAMA_HOST, and any other env vars.
# Does NOT overwrite variables already set in the environment.
# ============================================================================

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
            key   = key.strip()
            value = value.strip().strip('"').strip("'")
            # Don't overwrite vars already set in the shell environment
            os.environ.setdefault(key, value)
    print(f"[.env] Loaded from {env_path}")

_load_dotenv()


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    input_vault:  str = r"C:\Users\Wren C\Documents\Coding stuff\Obsidian Vault"
    output_vault: str = r"C:\Users\Wren C\Documents\Coding stuff\Obsidian Vault"

    # Ollama models — cloud used first if OLLAMA_API_KEY is set
    cloud_model: str = "gemma4:31b-cloud"
    local_model: str = "gemma3:4b"

    # Processing options
    min_content_length:     int   = 50
    request_delay:          float = 0.5
    max_retries:            int   = 3
    # FIX: raised from 3 — notes with few links still benefit from AI linking
    ai_link_skip_threshold: int   = 8
    moc_min_notes:          int   = 15
    moc_max_count:          int   = 50
    moc_min_note_size:      int   = 10

    output_folders: list = field(default_factory=lambda: [
        "00_Inbox", "01_MOCs", "02_Zettels", "03_Literature",
    ])
    tracker_filename: str = ".zettelkasten_tracker.json"
    restart_every: int = 100

    # Concurrency settings
    regex_workers: int = 4

    # AI linking optimisation
    ai_link_use_semantic_filter: bool = True
    ai_link_max_candidates: int = 200


# ============================================================================
# PRE-COMPILED REGEX PATTERNS
# ============================================================================

_UNSAFE_CHARS    = re.compile(r'[\\/:*?"<>|]')
_FRONTMATTER_RE  = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
_CODE_FENCE_RE   = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE  = re.compile(r"`[^`]+`")
_WIKILINK_RE     = re.compile(r"\[\[.*?\]\]")
_META_COMMENT_RE = re.compile(r"^<!-- meta: ({.*?}) -->\n", re.DOTALL)
_WORD_EXTRACT_RE = re.compile(r'\b([A-Za-z][A-Za-z0-9_-]{2,})\b')
_DATAVIEW_RE     = re.compile(r"```dataview|dataview\s+publisher", re.IGNORECASE)
_TAG_LINE_RE     = re.compile(r"^  - (.+)$", re.MULTILINE)

_JUNK_TAGS = frozenset({
    "general", "moc", "publisher", "dataview", "query", "data", "note", "notes",
    "overview", "summary", "introduction", "conclusion", "review", "revision",
    "vets20019", "vets20008", "vets20007", "vets20006", "vets10001", "vets10002",
    "1", "2", "3", "i", "ii", "iii", "iv", "v",
    "abduction", "adduction", "extension", "flexion", "rotation",
    "innervation", "drainage", "supply",
    "muscles", "arteries", "veins", "nerves",
    "lecture", "practical", "tutorial", "assessment", "exam", "semester",
    "structure", "function", "process", "system", "part", "region", "area",
    "types", "methods", "classification", "definition", "mechanism",
})

_MODULE_CODE_RE = re.compile(r'^(vets|vet|bio|biol|anim)\s*\d{4,5}[a-z]?$', re.IGNORECASE)
_NUMERIC_TAG_RE = re.compile(r'^\d+$|^[ivx]+$|^[a-z]\d+$', re.IGNORECASE)

FOLDER_MAP = {
    "moc":        "01_MOCs",
    "literature": "03_Literature",
    "zettel":     "02_Zettels",
}

# FIX: thread-safe wikilink pattern cache
_wikilink_pattern_cache: dict[str, re.Pattern] = {}
_wikilink_cache_lock = threading.RLock()


def _get_wikilink_pattern(title: str) -> re.Pattern:
    """Get or create a cached regex pattern for wikilinking a title. Thread-safe."""
    with _wikilink_cache_lock:
        if title not in _wikilink_pattern_cache:
            _wikilink_pattern_cache[title] = re.compile(
                r"(?<!\[\[)\b(" + re.escape(title) + r")\b(?!\]\])",
                re.IGNORECASE
            )
        return _wikilink_pattern_cache[title]


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("zettelkasten_v2.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ============================================================================
# ZETTEL ID GENERATOR
# FIX: old version used a counter reset to 0 on restart → ID collisions.
# Now uses timestamp + random 4-digit suffix → collision-safe across restarts.
# ============================================================================

def generate_id() -> str:
    """Generate a collision-safe zettel ID: YYYYMMDDHHmm + 4 random digits."""
    ts     = datetime.now().strftime("%Y%m%d%H%M")
    suffix = f"{random.randint(0, 9999):04d}"
    return ts + suffix


# ============================================================================
# TRACKER
# FIX: now saves on every mark_done (was batched every 10 — lost progress on crash)
# ============================================================================

class ProcessingTracker:
    """Phase-aware completion tracker with immediate writes for crash safety."""

    def __init__(self, path: Path):
        self.path  = path
        self._data: dict[str, set[str]] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, encoding="utf-8") as f:
                    raw = json.load(f)
                self._data = {k: set(v) for k, v in raw.items()}
                total = sum(len(v) for v in self._data.values())
                log.info("Tracker: %d total completions.", total)
            except (json.JSONDecodeError, OSError):
                log.warning("Tracker unreadable — starting fresh.")

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({k: sorted(v) for k, v in self._data.items()}, f, indent=2)

    def is_done(self, phase: str, key: str) -> bool:
        with self._lock:
            return key in self._data.get(phase, set())

    def mark_done(self, phase: str, key: str):
        with self._lock:
            self._data.setdefault(phase, set()).add(key)
            self._save()  # FIX: save immediately, not batched

    def flush(self):
        with self._lock:
            self._save()


# ============================================================================
# JSON HELPERS
# ============================================================================

def _strip_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)

def _extract_array_str(text: str) -> Optional[str]:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    return m.group(0) if m else None

def extract_json_array(text: str) -> Optional[list[dict]]:
    for c in (text, _strip_fences(text), _extract_array_str(text)):
        if c is None:
            continue
        try:
            r = json.loads(c)
            if isinstance(r, list):
                return r
        except json.JSONDecodeError:
            pass
    return None

def extract_json_string_array(text: str) -> Optional[list[str]]:
    for c in (text, _strip_fences(text), _extract_array_str(text)):
        if c is None:
            continue
        try:
            r = json.loads(c)
            if isinstance(r, list) and all(isinstance(i, str) for i in r):
                return r
        except json.JSONDecodeError:
            pass
    return None


# ============================================================================
# FILE / VAULT HELPERS
# ============================================================================

def safe_filename(title: str) -> str:
    return _UNSAFE_CHARS.sub("-", title).strip(". ")[:200] or "Untitled"

def classify_folder(note_type: str) -> str:
    return FOLDER_MAP.get(note_type.lower().strip(), "02_Zettels")

def count_wikilinks(text: str) -> int:
    return len(_WIKILINK_RE.findall(text))

def vault_notes(output_path: Path, tracker_filename: str) -> list[Path]:
    return [
        p for p in output_path.rglob("*.md")
        if not p.name.startswith(".")
        and "QUARANTINE_" not in p.name
        and p.name != tracker_filename
    ]


# ============================================================================
# TITLE CACHE
# ============================================================================

class TitleCache:
    """Cached title list with word-index for semantic candidate filtering."""

    def __init__(self):
        self._titles:        list[str]              = []
        self._title_set:     set[str]               = set()
        self._title_lower:   dict[str, str]         = {}  # lower -> original
        self._word_to_titles: dict[str, set[str]]   = {}
        self.dirty = True

    def rebuild(self, notes: list[Path]) -> None:
        titles = sorted(
            [p.stem for p in notes if len(p.stem) > 3],
            key=len, reverse=True,
        )
        self._titles    = titles
        self._title_set = set(titles)
        # FIX: keep lower->original map for case-insensitive lookup
        self._title_lower = {t.lower(): t for t in titles}
        self._build_word_index()
        self.dirty = False
        log.info("Title cache: %d titles indexed.", len(self._titles))

    def _build_word_index(self) -> None:
        self._word_to_titles.clear()
        for title in self._titles:
            for word in _WORD_EXTRACT_RE.findall(title):
                self._word_to_titles.setdefault(word.lower(), set()).add(title)

    def get_titles(self) -> list[str]:
        return self._titles

    def get_title_set(self) -> set[str]:
        return self._title_set

    def get_title_lower(self) -> dict[str, str]:
        return self._title_lower

    def get_candidates_for_note(self, note_title: str, note_content: str,
                                 max_candidates: int = 200) -> list[str]:
        content_words = set(
            w.lower() for w in _WORD_EXTRACT_RE.findall(note_content[:1000])
        )
        content_words.update(w.lower() for w in _WORD_EXTRACT_RE.findall(note_title))

        scores: dict[str, int] = {}
        for word in content_words:
            for title in self._word_to_titles.get(word, set()):
                if title != note_title:
                    scores[title] = scores.get(title, 0) + 1

        return sorted(scores, key=lambda t: (-scores[t], -len(t)))[:max_candidates]

    def mark_dirty(self) -> None:
        self.dirty = True


_title_cache = TitleCache()

def get_title_cache() -> TitleCache:
    return _title_cache


# ============================================================================
# TEXT MASKING
# ============================================================================

def mask_protected(text: str) -> tuple[str, list[tuple[str, str]]]:
    slots: list[tuple[str, str]] = []
    def _sub(m: re.Match) -> str:
        tok = f"\x00PH{len(slots)}\x00"
        slots.append((tok, m.group(0)))
        return tok
    t = _FRONTMATTER_RE.sub(_sub, text, count=1)
    t = _CODE_FENCE_RE.sub(_sub, t)
    t = _INLINE_CODE_RE.sub(_sub, t)
    t = _WIKILINK_RE.sub(_sub, t)
    return t, slots

def restore_protected(text: str, slots: list[tuple[str, str]]) -> str:
    for tok, orig in slots:
        text = text.replace(tok, orig)
    return text


def _write_note(dest: Path, title: str, note_type: str, tags: list[str], content: str):
    dest.write_text(
        f"<!-- meta: {json.dumps({'tags': tags, 'type': note_type})} -->\n" + content,
        encoding="utf-8",
    )


# ============================================================================
# OLLAMA — cloud-first with local fallback
# FIX: exponential backoff (was linear 10s increments)
# ============================================================================

def _make_cloud_client() -> Optional[Client]:
    key = os.environ.get("OLLAMA_API_KEY", "").strip()
    if not key:
        return None
    return Client(host="https://ollama.com", headers={"Authorization": f"Bearer {key}"})

def _raw_chat(client: Optional[Client], model: str, prompt: str) -> str:
    msgs = [{"role": "user", "content": prompt}]
    r    = client.chat(model=model, messages=msgs) if client else ollama.chat(model=model, messages=msgs)
    return r["message"]["content"]

def call_ollama(config: Config, prompt: str, label: str) -> Optional[str]:
    cloud      = _make_cloud_client()
    candidates = []
    if cloud:
        candidates.append((cloud, config.cloud_model, "cloud"))
    candidates.append((None, config.local_model, "local"))

    for client, model, src in candidates:
        for attempt in range(config.max_retries):
            try:
                return _raw_chat(client, model, prompt)
            except ollama.ResponseError as exc:
                log.warning("[%s] %s model error: %s — trying next.", label, src, exc)
                break
            except Exception as exc:
                msg = str(exc).lower()
                if any(t in msg for t in ("connection", "refused", "timeout", "rate")) \
                        and attempt < config.max_retries - 1:
                    # FIX: exponential backoff — 5s, 10s, 20s, ...
                    wait = 5 * (2 ** attempt)
                    log.warning("[%s] %s unreachable, retry in %ds (%d/%d)…",
                                label, src, wait, attempt + 1, config.max_retries)
                    time.sleep(wait)
                else:
                    log.warning("[%s] %s failed (%s) — falling back.", label, src, exc)
                    break

    log.error("[%s] All inference options exhausted.", label)
    return None

def _local_model_names() -> list[str]:
    try:
        models = ollama.list()
        items  = models.get("models", []) if isinstance(models, dict) else list(models)
        return [
            m.get("model", m.get("name", "")) if isinstance(m, dict)
            else getattr(m, "model", str(m))
            for m in items
        ]
    except Exception:
        return []

def verify_ollama(config: Config):
    key      = os.environ.get("OLLAMA_API_KEY", "").strip()
    cloud_ok = False
    local_ok = False

    if key:
        try:
            _make_cloud_client().chat(
                model=config.cloud_model,
                messages=[{"role": "user", "content": "ping"}],
            )
            cloud_ok = True
            log.info("Cloud ready — '%s' reachable.", config.cloud_model)
        except Exception as exc:
            log.warning("Cloud check failed (%s) — local only.", exc)

    try:
        local_ok = any(config.local_model in n for n in _local_model_names())
        if local_ok:
            log.info("Local ready — '%s' found.", config.local_model)
        else:
            log.warning("Local model '%s' not pulled. Run: ollama pull %s",
                        config.local_model, config.local_model)
    except Exception as exc:
        log.warning("Cannot reach local Ollama (%s). Run: ollama serve", exc)

    if not cloud_ok and not local_ok:
        log.critical(
            "Neither cloud nor local Ollama available.\n"
            "  Cloud: set OLLAMA_API_KEY\n"
            "  Local: 'ollama serve' then 'ollama pull %s'",
            config.local_model,
        )
        sys.exit(1)

    if not key:
        log.info("No OLLAMA_API_KEY set — local-only mode.")
    elif cloud_ok:
        log.info("Mode: cloud-first, local fallback.")


# ============================================================================
# PHASE 0 — QUARANTINE RECOVERY
# ============================================================================

_QUARANTINE_PROMPT = """\
You are a Zettelkasten assistant converting a university veterinary nursing \
lecture note into atomic notes.

YOUR ENTIRE RESPONSE MUST BE A VALID JSON ARRAY AND NOTHING ELSE.
No explanation. No preamble. No markdown fences. No trailing text.

Required structure:
[
  {{
    "title": "Short descriptive title",
    "content": "# Short descriptive title\\n\\nNote body here.",
    "tags": ["topic1", "topic2"],
    "type": "zettel"
  }}
]

Rules:
- type: "zettel" or "literature" only — NEVER "moc"
- tags: 2-5 lowercase subject keywords (anatomy, physiology, pharmacology, etc.)
- Split into multiple objects only for genuinely separate concepts
- Escape internal double-quotes as \\"

NOTE TO CONVERT:
{content}"""


def run_phase0(config: Config):
    log.info("=== PHASE 0: Quarantine recovery ===")
    output_path = Path(config.output_vault)
    tracker     = ProcessingTracker(output_path / config.tracker_filename)
    all_q       = list((output_path / "00_Inbox").glob("QUARANTINE_*.md"))
    pending     = [f for f in all_q if not tracker.is_done("phase0", f.name)]

    if not pending:
        log.info("No quarantined notes to recover.")
        return

    log.info("%d quarantined files to attempt.", len(pending))
    recovered = 0

    for fp in tqdm(pending, desc="Recovering"):
        key     = fp.name
        raw     = fp.read_text(encoding="utf-8")
        content = re.sub(r"^<!--.*?-->\s*\n\n", "", raw, flags=re.DOTALL)
        text    = call_ollama(config, _QUARANTINE_PROMPT.format(content=content), key)

        if text is None:
            log.warning("No response for: %s", key)
            tracker.mark_done("phase0", key)
            continue

        notes = extract_json_array(text)
        if notes is None:
            log.warning("Still unparseable: %s — leaving in quarantine.", key)
            tracker.mark_done("phase0", key)
            continue

        for note in notes:
            ntype = note.get("type", "zettel")
            if ntype == "moc":
                ntype = "zettel"
            tags  = [t for t in note.get("tags", []) if t not in _JUNK_TAGS] or ["general"]
            dest  = output_path / classify_folder(ntype) / f"{safe_filename(note.get('title', 'Untitled'))}.md"
            _write_note(dest, note.get("title", "Untitled"), ntype, tags, note.get("content", ""))

        fp.unlink()
        tracker.mark_done("phase0", key)
        recovered += 1
        log.info("Recovered %d note(s) from: %s", len(notes), key)
        time.sleep(config.request_delay)

    log.info("Phase 0 complete — %d file(s) recovered.", recovered)


# ============================================================================
# PHASE 1 — SPLIT NOTES INTO ATOMIC ZETTELS
# ============================================================================

_SPLIT_PROMPT = """\
You are a Zettelkasten assistant. Convert the lecture note below into one or \
more atomic notes for a veterinary nursing student. Each note must cover \
exactly ONE idea.

YOUR ENTIRE RESPONSE MUST BE A VALID JSON ARRAY AND NOTHING ELSE.
No explanation. No preamble. No markdown fences.

Required structure:
[
  {{
    "title": "Concise descriptive title",
    "content": "# Concise descriptive title\\n\\nFull note body here.",
    "tags": ["tag1", "tag2"],
    "type": "zettel"
  }}
]

TYPE RULES:
- "zettel"     : DEFAULT. Any concept, fact, process, definition, mechanism,
                 or clinical topic. Almost always the right choice.
- "literature" : ONLY for notes that are primarily a citation, reference list,
                 or summary of a specific named textbook or paper.
- "moc"        : DO NOT USE — MOCs are generated automatically.

TAG RULES:
- 2-5 lowercase keywords relevant to the content
- Good tags: anatomy, physiology, pharmacology, cardiology, behaviour,
             neurology, microbiology, clinical, pathology, nutrition
- Do NOT use: module codes, exam names, "dataview", "general", "moc"

OTHER:
- Split into multiple objects only for genuinely separate concepts
- Do NOT omit any information from the original note
- Escape internal double-quotes as \\"

NOTE TO PROCESS:
{content}"""


def run_phase1(config: Config):
    log.info("=== PHASE 1: Splitting notes ===")
    output_path = Path(config.output_vault)
    for folder in config.output_folders:
        (output_path / folder).mkdir(parents=True, exist_ok=True)

    tracker   = ProcessingTracker(output_path / config.tracker_filename)
    all_input = [
        p for p in Path(config.input_vault).rglob("*.md")
        if ".obsidian" not in p.parts
    ]
    pending = [f for f in all_input if not tracker.is_done("phase1", f.name)]
    log.info("%d files total, %d remaining.", len(all_input), len(pending))

    if not pending:
        log.info("All input files processed. Skipping Phase 1.")
        return

    for fp in tqdm(pending, desc="Splitting"):
        fname   = fp.name
        content = fp.read_text(encoding="utf-8")

        if len(content.strip()) < config.min_content_length:
            tracker.mark_done("phase1", fname)
            continue

        if _DATAVIEW_RE.search(content):
            log.info("Skipping Dataview artefact: %s", fname)
            tracker.mark_done("phase1", fname)
            continue

        text  = call_ollama(config, _SPLIT_PROMPT.format(content=content), fname)
        notes = extract_json_array(text) if text else None

        if notes is None:
            q = output_path / "00_Inbox" / f"QUARANTINE_{fname}"
            q.write_text(f"<!-- Failed to parse AI response for: {fname} -->\n\n{content}",
                         encoding="utf-8")
            log.warning("Quarantined: %s", fname)
        else:
            for note in notes:
                ntype = note.get("type", "zettel")
                if ntype == "moc":
                    ntype = "zettel"
                tags  = [t for t in note.get("tags", []) if t not in _JUNK_TAGS] or ["general"]
                title = safe_filename(note.get("title", "Untitled"))
                dest  = output_path / classify_folder(ntype) / f"{title}.md"
                _write_note(dest, title, ntype, tags, note.get("content", ""))
            log.info("Split into %d note(s): %s", len(notes), fname)

        tracker.mark_done("phase1", fname)
        _count_and_maybe_restart(config, tracker)
        time.sleep(config.request_delay)

    log.info("Phase 1 complete.")
    get_title_cache().mark_dirty()


# ============================================================================
# PHASE 2 — YAML FRONTMATTER
# ============================================================================

def _build_frontmatter(note_id: str, title: str, tags: list[str], note_type: str) -> str:
    tag_block = "\n".join(f"  - {t}" for t in tags) if tags else "  - general"
    return (
        f"---\n"
        f"id: {note_id}\n"
        f"title: \"{title}\"\n"
        f"tags:\n{tag_block}\n"
        f"type: {note_type}\n"
        f"created: {datetime.now().strftime('%Y-%m-%d')}\n"
        f"---\n\n"
    )


def run_phase2(config: Config):
    log.info("=== PHASE 2: YAML frontmatter ===")
    output_path = Path(config.output_vault)
    tracker     = ProcessingTracker(output_path / config.tracker_filename)
    all_notes   = vault_notes(output_path, config.tracker_filename)
    pending     = [p for p in all_notes if not tracker.is_done("phase2", p.stem)]
    log.info("%d notes total, %d need frontmatter.", len(all_notes), len(pending))

    if not pending:
        log.info("All notes have frontmatter. Skipping Phase 2.")
        return

    for fp in tqdm(pending, desc="Frontmatter"):
        text = fp.read_text(encoding="utf-8")

        if _FRONTMATTER_RE.match(text):
            tracker.mark_done("phase2", fp.stem)
            continue

        tags, ntype = [], "zettel"
        m = _META_COMMENT_RE.match(text)
        if m:
            try:
                meta  = json.loads(m.group(1))
                tags  = meta.get("tags", [])
                ntype = meta.get("type", "zettel")
            except json.JSONDecodeError:
                pass
            text = text[m.end():]

        fp.write_text(_build_frontmatter(generate_id(), fp.stem, tags, ntype) + text,
                      encoding="utf-8")
        tracker.mark_done("phase2", fp.stem)

    log.info("Phase 2 complete.")


# ============================================================================
# PHASE 3 — REGEX LINKING
#
# FIX: The original version extracted single words from the note and checked
# if each word was a title — this completely missed multi-word titles like
# "Cardiac Output" or "Renal Tubular Secretion". The fix:
#
#   1. Build a lowercased content string for fast substring pre-filtering
#   2. Only run the expensive regex for titles whose text actually appears
#      in the note (case-insensitive substring check first)
#   3. Sort titles longest-first so longer matches take priority over
#      sub-phrases (e.g. "Renal Tubular Secretion" before "Secretion")
# ============================================================================

def _regex_link_note(fp: Path, titles_sorted: list[str]) -> int:
    """
    Link all titles that appear in this note.
    titles_sorted must be ordered longest-first to avoid partial overwrites.
    """
    original      = fp.read_text(encoding="utf-8")
    masked, slots = mask_protected(original)
    masked_lower  = masked.lower()
    added         = 0

    for title in titles_sorted:
        if title == fp.stem:
            continue
        # Fast pre-filter: skip regex entirely if the title text isn't present
        if title.lower() not in masked_lower:
            continue
        # Skip if already wikilinked
        if f"[[{title}]]".lower() in masked_lower or f"[[{title.lower()}]]" in masked_lower:
            continue

        pattern       = _get_wikilink_pattern(title)
        masked, n     = pattern.subn(r"[[\1]]", masked, count=1)
        if n:
            masked_lower = masked.lower()
            added       += 1

    result = restore_protected(masked, slots)
    if result != original:
        fp.write_text(result, encoding="utf-8")
    return added


def _regex_link_wrapper(args: tuple[Path, list[str]]) -> tuple[str, int]:
    fp, titles = args
    return fp.stem, _regex_link_note(fp, titles)


def run_phase3(config: Config):
    log.info("=== PHASE 3: Regex linking ===")
    output_path = Path(config.output_vault)
    tracker     = ProcessingTracker(output_path / config.tracker_filename)
    all_notes   = vault_notes(output_path, config.tracker_filename)

    cache = get_title_cache()
    cache.rebuild(all_notes)

    # Titles sorted longest-first — critical for correct multi-word matching
    titles_sorted = sorted(cache.get_titles(), key=len, reverse=True)

    pending = [p for p in all_notes if not tracker.is_done("phase3", p.stem)]
    log.info("%d notes | %d titles | %d remaining.",
             len(all_notes), len(titles_sorted), len(pending))

    if not pending:
        log.info("All notes already regex-linked. Skipping Phase 3.")
        return

    total = 0
    with ThreadPoolExecutor(max_workers=config.regex_workers) as executor:
        args    = [(fp, titles_sorted) for fp in pending]
        futures = {executor.submit(_regex_link_wrapper, arg): arg[0] for arg in args}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Regex linking"):
            stem, added = future.result()
            total      += added
            tracker.mark_done("phase3", stem)
            _count_and_maybe_restart(config, tracker)

    log.info("Phase 3 complete — %d links added.", total)


# ============================================================================
# PHASE 4 — AI LINKING
# ============================================================================

_AI_LINK_PROMPT = """\
You are an Obsidian knowledge linker for a veterinary nursing Zettelkasten.

Note title: {title}
Note content (first 600 chars): {snippet}

Related note titles in the vault ({count} candidates):
{titles}

Return a JSON array of titles from the list above that are directly and
specifically related to this note and would make useful [[wikilinks]].

Rules:
- Only include titles genuinely relevant to this note's content
- Do NOT include this note's own title
- Return [] if nothing is relevant
- Return ONLY the JSON array, no explanation

Example: ["Cardiac Output", "Frank-Starling Law", "Venous Return"]"""


def _ai_link_note(config: Config, fp: Path, cache: TitleCache) -> int:
    text = fp.read_text(encoding="utf-8")

    if config.ai_link_use_semantic_filter:
        candidates = cache.get_candidates_for_note(
            fp.stem, text[:1000], config.ai_link_max_candidates
        )
    else:
        candidates = [t for t in cache.get_titles() if t != fp.stem][:config.ai_link_max_candidates]

    if not candidates:
        return 0

    snippet = text[:600].replace("\n", " ")
    t_str   = "\n".join(f"- {t}" for t in candidates)

    raw       = call_ollama(config,
                            _AI_LINK_PROMPT.format(
                                title=fp.stem,
                                snippet=snippet,
                                titles=t_str,
                                count=len(candidates),
                            ),
                            fp.stem)
    suggested = extract_json_string_array(raw) if raw else None
    if not suggested:
        return 0

    title_set = cache.get_title_set()
    valid     = [t for t in suggested if t in title_set and t != fp.stem]
    if not valid:
        return 0

    masked, slots = mask_protected(text)
    masked_lower  = masked.lower()
    added         = 0
    no_match      = []

    for title in valid:
        if f"[[{title.lower()}]]" in masked_lower:
            continue
        pattern   = _get_wikilink_pattern(title)
        masked, n = pattern.subn(r"[[\1]]", masked, count=1)
        if n:
            masked_lower = masked.lower()
            added       += 1
        else:
            no_match.append(title)

    result = restore_protected(masked, slots)

    if no_match and "## Related" not in result:
        links  = "  ".join(f"[[{t}]]" for t in no_match)
        result = result.rstrip() + f"\n\n## Related\n{links}\n"
        added += len(no_match)

    if result != text:
        fp.write_text(result, encoding="utf-8")

    return added


def run_phase4(config: Config):
    log.info("=== PHASE 4: AI linking ===")
    output_path = Path(config.output_vault)
    tracker     = ProcessingTracker(output_path / config.tracker_filename)
    all_notes   = vault_notes(output_path, config.tracker_filename)

    cache = get_title_cache()
    if cache.dirty:
        cache.rebuild(all_notes)

    pending = [p for p in all_notes if not tracker.is_done("phase4", p.stem)]
    log.info("%d notes total, %d remaining for AI linking.", len(all_notes), len(pending))

    if not pending:
        log.info("All notes AI-linked. Skipping Phase 4.")
        return

    total = 0
    for fp in tqdm(pending, desc="AI linking"):
        if count_wikilinks(fp.read_text(encoding="utf-8")) >= config.ai_link_skip_threshold:
            tracker.mark_done("phase4", fp.stem)
            continue
        added  = _ai_link_note(config, fp, cache)
        total += added
        if added:
            log.info("AI added %d → %s", added, fp.stem)
        tracker.mark_done("phase4", fp.stem)
        _count_and_maybe_restart(config, tracker)
        time.sleep(config.request_delay)

    log.info("Phase 4 complete — %d AI links added.", total)


# ============================================================================
# PHASE 5 — MOC GENERATION
# FIX: tag_map now built once and passed in, not rebuilt per-topic
# ============================================================================

_MOC_PROMPT = """\
You are building a Map of Content (MOC) for an Obsidian Zettelkasten vault \
used by a veterinary nursing student.

Topic: {topic}
Notes to organise ({count} total):
{note_list}

Write a well-structured MOC note:

1. INTRODUCTION (2-3 sentences)
   Introduce the topic and its relevance to veterinary nursing.

2. ORGANISED SECTIONS (2-5 sections, ## headers)
   Choose headers appropriate to the topic, e.g.:
   ## Anatomy, ## Physiology, ## Clinical Relevance, ## Pharmacology,
   ## Assessment, ## Treatment, ## Pathology, ## Behaviour
   Under each header list notes as:
   [[Note Title]] — one-line annotation

3. KEY CONCEPTS (## Key Concepts)
   3-5 bullet-point takeaways.

RULES:
- Every note in the list MUST appear as a [[wikilink]] exactly once
- Do NOT invent notes not in the list
- Do NOT include YAML frontmatter
- Minimum 300 words

Output ONLY the note body. Begin with the introduction paragraph."""


def _is_valid_moc_tag(tag: str) -> bool:
    tag_lower = tag.lower().strip()
    if tag_lower in _JUNK_TAGS:
        return False
    if _MODULE_CODE_RE.match(tag_lower):
        return False
    if _NUMERIC_TAG_RE.match(tag_lower):
        return False
    if len(tag_lower) < 4:
        return False
    return True


def _collect_tag_map(output_path: Path, tracker_filename: str) -> dict[str, list[Path]]:
    """Build tag -> [note paths] map. Called once per phase5 run."""
    tag_map: dict[str, list[Path]] = {}
    for fp in output_path.rglob("*.md"):
        if fp.name.startswith(".") or "QUARANTINE_" in fp.name:
            continue
        text = fp.read_text(encoding="utf-8")
        fm   = _FRONTMATTER_RE.match(text)
        if not fm:
            continue
        fm_text = fm.group(0)
        if re.search(r"^type:\s*moc", fm_text, re.MULTILINE):
            continue
        for tag in _TAG_LINE_RE.findall(fm_text):
            tag = tag.strip()
            if tag and _is_valid_moc_tag(tag):
                tag_map.setdefault(tag, []).append(fp)
    return tag_map


def run_phase5(config: Config):
    log.info("=== PHASE 5: MOC generation ===")
    output_path = Path(config.output_vault)
    tracker     = ProcessingTracker(output_path / config.tracker_filename)
    moc_dir     = output_path / "01_MOCs"

    # FIX: build tag_map once here, not inside the loop
    log.info("Building tag map…")
    tag_map = _collect_tag_map(output_path, config.tracker_filename)

    eligible     = {t: n for t, n in tag_map.items() if len(n) >= config.moc_min_notes}
    sorted_topics = sorted(eligible.items(), key=lambda x: -len(x[1]))

    if config.moc_max_count > 0 and len(sorted_topics) > config.moc_max_count:
        log.info("Limiting to top %d topics (from %d candidates)",
                 config.moc_max_count, len(sorted_topics))
        sorted_topics = sorted_topics[:config.moc_max_count]

    pending = [(t, n) for t, n in sorted_topics
               if not tracker.is_done("phase5", f"MOC_{t}")]

    log.info("%d eligible topics (min %d notes), %d MOCs to generate.",
             len(eligible), config.moc_min_notes, len(pending))

    if not pending:
        log.info("All MOCs generated. Skipping Phase 5.")
        return

    if pending:
        top = [f"{t}({len(n)})" for t, n in pending[:5]]
        log.info("Top topics: %s%s", ", ".join(top), "..." if len(pending) > 5 else "")

    for topic, notes in tqdm(pending, desc="MOCs"):
        moc_key   = f"MOC_{topic}"
        note_list = "\n".join(f"- {p.stem}" for p in notes)
        body      = call_ollama(
            config,
            _MOC_PROMPT.format(topic=topic, note_list=note_list, count=len(notes)),
            moc_key,
        )

        if body is None:
            log.warning("Could not generate MOC for: %s", topic)
            tracker.mark_done("phase5", moc_key)
            continue

        title   = f"MOC - {topic.title()}"
        note_id = generate_id()
        fm      = (f"---\nid: {note_id}\ntitle: \"{title}\"\n"
                   f"tags:\n  - moc\n  - {topic}\ntype: moc\n"
                   f"created: {datetime.now().strftime('%Y-%m-%d')}\n---\n\n")

        dest = moc_dir / f"{safe_filename(title)}.md"
        dest.write_text(fm + f"# {title}\n\n" + body, encoding="utf-8")
        log.info("Created MOC: %s (%d notes)", title, len(notes))
        tracker.mark_done("phase5", moc_key)
        _count_and_maybe_restart(config, tracker)
        time.sleep(config.request_delay)

    log.info("Phase 5 complete.")


# ============================================================================
# PHASE 6 — MOC IMPROVEMENT
# Rewrites existing MOCs that are low quality: missing links, thin content,
# poor structure, or notes that have been added since the MOC was generated.
# ============================================================================

_MOC_IMPROVE_PROMPT = """\
You are improving an existing Map of Content (MOC) note in an Obsidian \
Zettelkasten vault used by a veterinary nursing student.

MOC title: {title}
Current MOC body:
{current_body}

All notes currently tagged with this topic ({count} total):
{note_list}

Problems to fix:
- Missing wikilinks: every note in the list must appear as [[wikilink]] exactly once
- Thin or vague sections: expand with meaningful 1-2 sentence annotations per note
- Poor structure: reorganise into 2-5 logical ## sections appropriate to the topic
- Missing Key Concepts section: add ## Key Concepts with 3-5 bullet takeaways if absent
- Orphaned links: remove [[wikilinks]] to notes not in the provided list

Rewrite the full MOC body with these improvements applied.

RULES:
- Every note in the list MUST appear as [[wikilink]] exactly once
- Do NOT invent notes not in the provided list
- Do NOT include YAML frontmatter — body text only
- Minimum 300 words
- Begin with a 2-3 sentence introduction paragraph

Output ONLY the improved note body."""

_MOC_QUALITY_RE = re.compile(r"\[\[.*?\]\]")


def _moc_needs_improvement(text: str, expected_notes: list[Path]) -> bool:
    """Return True if this MOC is worth rewriting."""
    # Strip frontmatter
    body = _FRONTMATTER_RE.sub("", text, count=1).strip()

    # Too short
    if len(body) < 300:
        return True

    # Fewer than half the expected notes are linked
    linked = set(m.group(0)[2:-2].lower() for m in _MOC_QUALITY_RE.finditer(body))
    expected_stems = {p.stem.lower() for p in expected_notes}
    coverage = len(linked & expected_stems) / max(len(expected_stems), 1)
    if coverage < 0.5:
        return True

    # Missing Key Concepts section
    if "## Key Concepts" not in text and "## key concepts" not in text.lower():
        return True

    return False


def run_phase6(config: Config):
    log.info("=== PHASE 6: MOC improvement ===")
    output_path = Path(config.output_vault)
    tracker     = ProcessingTracker(output_path / config.tracker_filename)
    moc_dir     = output_path / "01_MOCs"

    if not moc_dir.exists():
        log.info("No MOC directory found. Skipping Phase 6.")
        return

    # Build tag map to know which notes belong to each MOC topic
    log.info("Building tag map for MOC improvement…")
    tag_map = _collect_tag_map(output_path, config.tracker_filename)

    moc_files = [p for p in moc_dir.glob("*.md") if not p.name.startswith(".")]
    pending   = [p for p in moc_files if not tracker.is_done("phase6", p.stem)]

    log.info("%d MOCs found, %d to evaluate.", len(moc_files), len(pending))

    if not pending:
        log.info("All MOCs already evaluated. Skipping Phase 6.")
        return

    improved = 0
    skipped  = 0

    for fp in tqdm(pending, desc="Improving MOCs"):
        text = fp.read_text(encoding="utf-8")

        # Extract topic from frontmatter tags (first non-moc tag)
        fm_match = _FRONTMATTER_RE.match(text)
        if not fm_match:
            tracker.mark_done("phase6", fp.stem)
            continue

        fm_text = fm_match.group(0)
        tags    = [t.strip() for t in _TAG_LINE_RE.findall(fm_text)
                   if t.strip() not in ("moc",) and _is_valid_moc_tag(t.strip())]
        topic   = tags[0] if tags else None

        # Get expected notes for this topic
        expected_notes = tag_map.get(topic, []) if topic else []

        if not _moc_needs_improvement(text, expected_notes):
            skipped += 1
            tracker.mark_done("phase6", fp.stem)
            continue

        # Strip frontmatter to get current body
        current_body = _FRONTMATTER_RE.sub("", text, count=1).strip()
        note_list    = "\n".join(f"- {p.stem}" for p in expected_notes) if expected_notes \
                       else "(no tagged notes found — improve structure only)"

        new_body = call_ollama(
            config,
            _MOC_IMPROVE_PROMPT.format(
                title=fp.stem,
                current_body=current_body[:2000],  # cap to avoid huge prompts
                note_list=note_list,
                count=len(expected_notes),
            ),
            f"MOC6:{fp.stem}",
        )

        if new_body is None:
            log.warning("No response for MOC: %s", fp.stem)
            tracker.mark_done("phase6", fp.stem)
            continue

        # Preserve original frontmatter, replace body
        fp.write_text(fm_match.group(0) + "\n" + new_body.strip() + "\n", encoding="utf-8")
        improved += 1
        log.info("Improved MOC: %s", fp.stem)

        tracker.mark_done("phase6", fp.stem)
        _count_and_maybe_restart(config, tracker)
        time.sleep(config.request_delay)

    log.info("Phase 6 complete — %d MOCs improved, %d already good.", improved, skipped)




_files_processed = 0

def _count_and_maybe_restart(config: Config, tracker: "ProcessingTracker", n: int = 1):
    global _files_processed
    if config.restart_every <= 0:
        return
    _files_processed += n
    if _files_processed >= config.restart_every:
        log.info("Processed %d files — restarting to clear memory…", _files_processed)
        tracker.flush()
        for h in logging.getLogger().handlers:
            h.flush()
        # FIX: pass current environment so API keys survive restart
        subprocess.Popen([sys.executable] + sys.argv, env=os.environ.copy())
        sys.exit(0)


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    config = Config()
    verify_ollama(config)

    # Phases 0 and 1 skipped — notes already split
    run_phase2(config)   # YAML frontmatter
    run_phase3(config)   # Regex linking  (parallelized, multi-word fixed)
    run_phase4(config)   # AI linking     (semantic filtering)
    run_phase5(config)   # MOC generation (new notes)
    run_phase6(config)   # MOC improvement (existing weak MOCs)

    log.info("=== All done! ===")


if __name__ == "__main__":
    main()
