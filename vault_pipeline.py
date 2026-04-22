"""
Obsidian Zettelkasten Converter — Threaded & Consolidated Edition
Focuses on concurrent processing of AI links, Tag Consolidation, and MOCs.
"""

import json
import os
import re
import random
import sys
import time
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from vault_reconstruct.config import get_vault_paths
from vault_reconstruct.env import load_dotenv_no_override
from vault_reconstruct.llm import (
    LlmConfig,
    FallbackChain,
    generate_text_with_retries,
    make_backend_thread_local,
    make_fallback_backend,
)
from vault_reconstruct.model_recommend import select_ollama_model_for_mode

try:
    from obsidian_vault_improver import VaultAnalyzer, Config as ImproverConfig, generate_health_report, find_broken_wikilinks_report, generate_empty_notes_report, generate_orphan_suggestions_report, consolidate_mocs, standardize_tags, fix_broken_links
except ImportError:
    VaultAnalyzer = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

# Rust extension — optional, graceful fallback to Python for every function
try:
    import reconstruct_rust as _RUST
    _RUST_AVAILABLE = True
except ImportError:
    _RUST = None
    _RUST_AVAILABLE = False

load_dotenv_no_override()

# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class Config:
    input_vault: str = str(get_vault_paths().input_vault)
    output_vault: str = str(get_vault_paths().output_vault)
    # Model backend selection — supports comma-separated fallback chains:
    #   "ollama"          -> Ollama only (local, unlimited)
    #   "gemini"          -> Gemini only
    #   "azure"           -> Azure OpenAI only
    #   "gemini,ollama"   -> Gemini primary, Ollama fallback when quota hit
    #   "ollama,gemini"   -> Ollama primary, Gemini as rate-limit escape valve
    provider:           str = "azure"
    cloud_model:        str = os.environ.get("VAULT_OLLAMA_CLOUD_MODEL", "gemma3:4b-cloud")
    local_model:        str = os.environ.get("VAULT_OLLAMA_MODEL", "").strip() or ""
    gemini_model:       str = os.environ.get("VAULT_GEMINI_MODEL", "gemini-2.5-flash")
    azure_model:        str = os.environ.get("VAULT_AZURE_MODEL", "gpt-4.1-mini")
    min_content_length:     int   = 50
    request_delay:          float = 0.5
    max_retries:            int   = 3

    # Concurrency
    ai_workers: int = 5
    regex_workers: int = 4

    # Linking & Tags
    ai_link_skip_threshold: int   = 8
    ai_link_use_semantic_filter: bool = True
    ai_link_max_candidates: int = 200
    tag_generalization_threshold: int = 5

    # MOCs
    moc_min_notes:          int   = 15
    moc_max_count:          int   = 50
    output_folders: list = field(default_factory=lambda: ["00_Inbox", "01_MOCs", "02_Zettels", "03_Literature"])
    tracker_filename: str = ".zettelkasten_tracker.json"

# ============================================================================
# REGEX & CONSTANTS
# ============================================================================
_UNSAFE_CHARS    = re.compile(r'[\\/:*?"<>|]')
_FRONTMATTER_RE  = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
_CODE_FENCE_RE   = re.compile(r"`{3}.*?`{3}", re.DOTALL)
_INLINE_CODE_RE  = re.compile(r"`[^`]+`")
_WIKILINK_RE     = re.compile(r"\[\[.*?\]\]")
_META_COMMENT_RE = re.compile(r"^<!-- meta: ({.*?}) -->\n", re.DOTALL)
_WORD_EXTRACT_RE = re.compile(r'\b([A-Za-z][A-Za-z0-9_-]{2,})\b')
_DATAVIEW_RE     = re.compile(r"`{3}dataview|dataview\s+publisher", re.IGNORECASE)
_TAG_LINE_RE     = re.compile(r"^  - (.+)$", re.MULTILINE)

_JUNK_TAGS = frozenset({
    "general", "moc", "publisher", "dataview", "query", "data", "note", "notes",
    "overview", "summary", "introduction", "conclusion", "review", "revision",
    "vets20019", "vets20008", "vets20007", "vets20006", "vets10001", "vets10002",
    "1", "2", "3", "i", "ii", "iii", "iv", "v", "abduction", "adduction",
    "extension", "flexion", "rotation", "innervation", "drainage", "supply",
    "muscles", "arteries", "veins", "nerves", "lecture", "practical", "tutorial",
    "assessment", "exam", "semester", "structure", "function", "process",
    "system", "part", "region", "area", "types", "methods", "classification",
    "definition", "mechanism",
})
_MODULE_CODE_RE = re.compile(r'^(vets|vet|bio|biol|anim)\s*\d{4,5}[a-z]?$', re.IGNORECASE)
_NUMERIC_TAG_RE = re.compile(r'^\d+$|^[ivx]+$|^[a-z]\d+$', re.IGNORECASE)

FOLDER_MAP = {"moc": "01_MOCs", "literature": "03_Literature", "zettel": "02_Zettels"}

def classify_folder(note_type: str) -> str:
    return FOLDER_MAP.get(note_type.lower().strip(), "02_Zettels")

# Pre-compiled wikilink patterns for the Python fallback path only
_wikilink_pattern_cache: dict[str, re.Pattern] = {}
_wikilink_cache_lock = threading.RLock()

def _get_wikilink_pattern(title: str) -> re.Pattern:
    with _wikilink_cache_lock:
        if title not in _wikilink_pattern_cache:
            _wikilink_pattern_cache[title] = re.compile(
                r"(?<!\[\[)\b(" + re.escape(title) + r")\b(?!\]\])", re.IGNORECASE
            )
        return _wikilink_pattern_cache[title]

# ============================================================================
# LOGGING & TRACKER
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

if _RUST_AVAILABLE:
    log.info("Rust engine loaded — phases 4-6 running with native acceleration.")
else:
    log.info(
        "Rust engine not found — running pure Python. "
        "Run `maturin develop` in reconstruct_rust/ to enable acceleration."
    )

_id_lock = threading.Lock()

def generate_id() -> str:
    with _id_lock:
        return datetime.now().strftime("%Y%m%d%H%M") + f"{random.randint(0, 9999):04d}"


class ProcessingTracker:
    _FLUSH_EVERY = 20

    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, set] = {}
        self._lock = threading.Lock()
        self._dirty = False
        self._pending_writes = 0
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, encoding="utf-8") as f:
                    self._data = {k: set(v) for k, v in json.load(f).items()}
            except (json.JSONDecodeError, OSError):
                pass

    def _flush_unsafe(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({k: sorted(v) for k, v in self._data.items()}, f, indent=2)
        self._dirty = False
        self._pending_writes = 0

    def flush(self):
        with self._lock:
            if self._dirty:
                self._flush_unsafe()

    def is_done(self, phase: str, key: str) -> bool:
        with self._lock:
            return key in self._data.get(phase, set())

    def mark_done(self, phase: str, key: str):
        with self._lock:
            self._data.setdefault(phase, set()).add(key)
            self._dirty = True
            self._pending_writes += 1
            if self._pending_writes >= self._FLUSH_EVERY:
                self._flush_unsafe()

    def __del__(self):
        try:
            self.flush()
        except Exception:
            pass


# ============================================================================
# LLM CLIENT — fallback-chain aware, per-thread backend cache
# ============================================================================

def _make_llm_configs(config: Config, *, strict_json: bool) -> list[LlmConfig]:
    raw_providers = [p.strip().lower() for p in config.provider.split(",") if p.strip()]
    if not raw_providers:
        raw_providers = ["ollama"]

    configs: list[LlmConfig] = []
    for provider in raw_providers:
        if provider == "ollama":
            local_model = config.local_model or select_ollama_model_for_mode(strict_json=strict_json)
            configs.append(LlmConfig(
                provider="ollama",
                model=local_model,
                max_retries=max(1, int(config.max_retries)),
                ollama_cloud_model=(config.cloud_model or None),
            ))
        elif provider == "gemini":
            configs.append(LlmConfig(
                provider="gemini",
                model=config.gemini_model,
                max_retries=max(1, int(config.max_retries)),
            ))
        elif provider == "azure":
            configs.append(LlmConfig(
                provider="azure",
                model=config.azure_model,
                max_retries=max(1, int(config.max_retries)),
            ))
        else:
            raise SystemExit(
                f"Unknown provider {provider!r} in VAULT_LLM_PROVIDER={config.provider!r}. "
                f"Valid values: ollama, gemini, azure (comma-separate for fallback chains)."
            )
    return configs


def call_llm(config: Config, prompt: str, label: str, *, strict_json: bool = True) -> Optional[str]:
    try:
        configs = _make_llm_configs(config, strict_json=strict_json)
        backend = make_fallback_backend(configs)
        return generate_text_with_retries(
            backend,
            prompt=prompt,
            max_retries=max(1, int(config.max_retries)),
        )
    except Exception as exc:
        log.warning("[%s] LLM call failed: %s", label, exc)
        return None


# ============================================================================
# CORE PARSING HELPERS
# ============================================================================
def _strip_fences(text: str) -> str:
    return re.sub(r"^`{3}(?:json)?\s*|\s*`{3}$", "", text.strip(), flags=re.MULTILINE)


def extract_json_array(text: str) -> Optional[list]:
    candidates = [text, _strip_fences(text)]
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            r = json.loads(c)
            if isinstance(r, list):
                return r
        except json.JSONDecodeError:
            pass
    return None


def extract_json_dict(text: str) -> Optional[dict]:
    candidates = [text, _strip_fences(text)]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            r = json.loads(c)
            if isinstance(r, dict):
                return r
        except json.JSONDecodeError:
            pass
    return None


def safe_filename(title: str) -> str:
    return _UNSAFE_CHARS.sub("-", title).strip(". ")[:200] or "Untitled"


def count_wikilinks(text: str) -> int:
    """Count [[wikilinks]] — uses Rust regex engine if available."""
    if _RUST_AVAILABLE:
        return _RUST.count_wikilinks(text)
    return len(_WIKILINK_RE.findall(text))


def vault_notes(out_path: Path, t_file: str) -> list[Path]:
    """List all processable notes — uses Rust parallel walk if available."""
    if _RUST_AVAILABLE:
        return [Path(p) for p in _RUST.list_vault_notes(str(out_path), t_file)]
    return [
        p for p in out_path.rglob("*.md")
        if not p.name.startswith(".") and "QUARANTINE_" not in p.name and p.name != t_file
    ]


# ============================================================================
# TITLE CACHE
# ============================================================================
class TitleCache:
    def __init__(self):
        self._titles: list[str] = []
        self._title_set: set[str] = set()
        self._word_to_titles: dict[str, set[str]] = {}
        self.dirty = True
        self._lock = threading.RLock()

    def rebuild(self, notes: list[Path]) -> None:
        titles = sorted([p.stem for p in notes if len(p.stem) > 3], key=len, reverse=True)
        title_set = set(titles)

        if _RUST_AVAILABLE:
            # Rust builds the word->titles index in a single native pass
            raw_index = _RUST.build_title_word_index(titles)
            word_to_titles = {w: set(ts) for w, ts in raw_index.items()}
        else:
            word_to_titles: dict[str, set[str]] = {}
            for t in titles:
                for w in _WORD_EXTRACT_RE.findall(t):
                    word_to_titles.setdefault(w.lower(), set()).add(t)

        with self._lock:
            self._titles = titles
            self._title_set = title_set
            self._word_to_titles = word_to_titles
            self.dirty = False

    def get_titles(self) -> list[str]:
        with self._lock:
            return list(self._titles)

    def get_title_set(self) -> set[str]:
        with self._lock:
            return set(self._title_set)

    def mark_dirty(self) -> None:
        with self._lock:
            self.dirty = True

    def get_candidates_for_note(self, note_title: str, note_content: str, max_candidates: int = 200) -> list[str]:
        with self._lock:
            word_to_titles = self._word_to_titles

        if _RUST_AVAILABLE:
            # Convert set values to lists for the FFI boundary, then score in Rust
            index_for_rust = {w: list(ts) for w, ts in word_to_titles.items()}
            return _RUST.get_note_candidates(
                index_for_rust, note_title, note_content[:1000], max_candidates
            )

        # Python fallback
        content_words = {w.lower() for w in _WORD_EXTRACT_RE.findall(note_content[:1000])}
        content_words.update(w.lower() for w in _WORD_EXTRACT_RE.findall(note_title))

        scores: dict[str, int] = {}
        for w in content_words:
            for t in word_to_titles.get(w, set()):
                if t != note_title:
                    scores[t] = scores.get(t, 0) + 1

        return sorted(scores, key=lambda t: (-scores[t], -len(t)))[:max_candidates]


_title_cache = TitleCache()


def get_title_cache() -> TitleCache:
    return _title_cache


# ============================================================================
# MASK / RESTORE — Python fallback path only
# (Rust functions handle mask/restore internally where they accelerate)
# ============================================================================
def mask_protected(text: str) -> tuple[str, list[tuple[str, str]]]:
    slots: list[tuple[str, str]] = []

    def _sub(m: re.Match) -> str:
        tok = f"\x00PH{len(slots)}\x00"
        slots.append((tok, m.group(0)))
        return tok

    t = _FRONTMATTER_RE.sub(_sub, text, count=1)
    t = _CODE_FENCE_RE.sub(_sub, t)
    return _WIKILINK_RE.sub(_sub, t), slots


def restore_protected(text: str, slots: list[tuple[str, str]]) -> str:
    for tok, orig in slots:
        text = text.replace(tok, orig)
    return text


# ============================================================================
# PHASE 0 — QUARANTINE RECOVERY
# ============================================================================

QUARANTINE_PROMPT = """\
You are a Zettelkasten assistant converting a university lecture note into \
atomic notes.

IMPORTANT: Your ENTIRE response must be one valid JSON array and nothing else.
Do not include any explanation, preamble, or markdown formatting.
Do not wrap in code fences.

Return this exact structure (replace the example values):
[{{"title": "Short descriptive title", "content": "# Short descriptive title\\n\\nNote body here.", "tags": ["topic1", "topic2"], "type": "zettel"}}]

Rules:
- type must be one of: zettel, literature, moc
- tags should be 2-5 lowercase subject keywords
- Split into multiple objects if there are multiple distinct ideas
- Escape internal double quotes as \\"

NOTE TO CONVERT:
{content}"""


def run_phase0(config: Config):
    """Retry all quarantined notes with a stricter, more explicit prompt."""
    log.info("=== PHASE 0: Recovering quarantined notes ===")

    output_path = Path(config.output_vault)
    quarantine_files = list((output_path / "00_Inbox").glob("QUARANTINE_*.md"))

    if not quarantine_files:
        log.info("No quarantined notes found. Skipping Phase 0.")
        return

    tracker = ProcessingTracker(output_path / config.tracker_filename)

    quarantine_files = [f for f in quarantine_files if not tracker.is_done("phase0", f.name)]
    log.info("Found %d quarantined files to recover.", len(quarantine_files))

    if not quarantine_files:
        log.info("All quarantined files already processed. Skipping Phase 0.")
        return

    recovered = 0

    for file_path in tqdm(quarantine_files, desc="Recovering quarantine"):
        key = file_path.name

        raw = file_path.read_text(encoding="utf-8")
        # Strip the quarantine comment header
        content = re.sub(r"^<!--.*?-->\s*\n\n", "", raw, flags=re.DOTALL)

        prompt = QUARANTINE_PROMPT.format(content=content)
        text = call_llm(config, prompt, key)

        if text is None:
            log.warning("Ollama returned nothing for: %s", key)
            tracker.mark_done("phase0", key)
            continue

        notes = extract_json_array(text)

        if notes is None:
            log.warning("Still cannot parse JSON for: %s — leaving in quarantine.", key)
            tracker.mark_done("phase0", key)
            continue

        # Write recovered notes
        for note in notes:
            title        = safe_filename(note.get("title", "Untitled"))
            note_content = note.get("content", "")
            note_type    = note.get("type", "zettel")
            folder       = classify_folder(note_type)
            dest          = output_path / folder / f"{title}.md"
            dest.write_text(note_content, encoding="utf-8")

        # Remove the quarantine file
        file_path.unlink()
        tracker.mark_done("phase0", key)
        recovered += 1
        log.info("Recovered: %s → %d note(s)", key, len(notes))
        time.sleep(config.request_delay)

    log.info("Phase 0 complete. Recovered %d file(s).", recovered)


# ============================================================================
# PHASE 1 — SPLIT NOTES (enhanced: tags + classification)
# ============================================================================

SPLIT_PROMPT = """\
You are a Zettelkasten assistant. Convert the lecture note below into one or \
more atomic notes. Each note must cover exactly ONE idea.

IMPORTANT: Your ENTIRE response must be a valid JSON array and nothing else.
Do not include explanation, preamble, or markdown code fences.

Return this exact structure:
[
  {{
    "title": "Concise descriptive title",
    "content": "# Concise descriptive title\\n\\nFull note body here.",
    "tags": ["tag1", "tag2"],
    "type": "zettel"
  }}
]

Rules:
- type must be exactly one of: zettel, literature, moc
  - zettel: a single atomic concept or fact
  - literature: a reference note summarising a source
  - moc: an overview/index note linking multiple concepts
- tags: 2-5 lowercase subject keywords (e.g. physiology, cardiac, anatomy)
- Split into multiple JSON objects if there are multiple distinct ideas
- Do NOT lose any information from the original note
- Escape internal double quotes as \\"

NOTE TO PROCESS:
{content}"""


def split_note(config: Config, content: str, filename: str) -> list[dict] | None:
    prompt = SPLIT_PROMPT.format(content=content)
    text = call_llm(config, prompt, filename)
    if text is None:
        return None
    notes = extract_json_array(text)
    if notes is None:
        log.warning("[%s] Could not parse JSON from model response.", filename)
    return notes


def run_phase1(config: Config):
    log.info("=== PHASE 1: Splitting notes into Zettelkasten ===")

    output_path = Path(config.output_vault)
    for folder in config.output_folders:
        (output_path / folder).mkdir(parents=True, exist_ok=True)

    tracker = ProcessingTracker(output_path / config.tracker_filename)

    all_input = [
        p for p in Path(config.input_vault).rglob("*.md")
        if ".obsidian" not in p.parts
    ]
    input_files = [f for f in all_input if not tracker.is_done("phase1", f.name)]
    log.info("%d files total, %d remaining.", len(all_input), len(input_files))

    if not input_files:
        log.info("All input files already processed. Skipping Phase 1.")
        return

    for file_path in tqdm(input_files, desc="Splitting notes"):
        filename = file_path.name

        content = file_path.read_text(encoding="utf-8")
        if len(content.strip()) < config.min_content_length:
            tracker.mark_done("phase1", filename)
            continue

        notes = split_note(config, content, filename)

        if notes is None:
            quarantine_path = output_path / "00_Inbox" / f"QUARANTINE_{filename}"
            quarantine_path.write_text(
                f"<!-- Failed to parse AI response for: {filename} -->\n\n{content}",
                encoding="utf-8",
            )
            log.warning("Quarantined: %s", filename)
        else:
            for note in notes:
                title        = safe_filename(note.get("title", "Untitled"))
                note_content = note.get("content", "")
                note_type    = note.get("type", "zettel")
                tags         = note.get("tags", [])
                folder       = classify_folder(note_type)
                dest         = output_path / folder / f"{title}.md"
                # Store tags and type for Phase 2 to use in frontmatter
                dest.write_text(
                    f"<!-- meta: {json.dumps({'tags': tags, 'type': note_type})} -->\n"
                    + note_content,
                    encoding="utf-8",
                )
            log.info("Split into %d note(s): %s", len(notes), filename)

        tracker.mark_done("phase1", filename)
        time.sleep(config.request_delay)

    log.info("Phase 1 complete.")


# ============================================================================
# PHASE 2 — YAML FRONTMATTER
# ============================================================================

_META_COMMENT_RE = re.compile(r"^<!-- meta: ({.*?}) -->\n", re.DOTALL)


def build_frontmatter(note_id: str, title: str, tags: list[str], note_type: str) -> str:
    tag_lines = "\n".join(f"  - {t}" for t in tags) if tags else "  - general"
    return (
        f"---\n"
        f"id: {note_id}\n"
        f"title: \"{title}\"\n"
        f"tags:\n{tag_lines}\n"
        f"type: {note_type}\n"
        f"created: {datetime.now().strftime('%Y-%m-%d')}\n"
        f"---\n\n"
    )


def run_phase2(config: Config):
    log.info("=== PHASE 2: Adding YAML frontmatter ===")

    output_path = Path(config.output_vault)
    tracker     = ProcessingTracker(output_path / config.tracker_filename)

    all_notes_raw = [
        p for p in output_path.rglob("*.md")
        if not p.name.startswith(".")
        and "QUARANTINE_" not in p.name
        and p.name != config.tracker_filename
    ]
    all_notes = [p for p in all_notes_raw if not tracker.is_done("phase2", p.stem)]
    log.info("%d notes total, %d need frontmatter.", len(all_notes_raw), len(all_notes))

    if not all_notes:
        log.info("All notes already have frontmatter. Skipping Phase 2.")
        return

    for file_path in tqdm(all_notes, desc="Adding frontmatter"):
        key = file_path.stem

        text = file_path.read_text(encoding="utf-8")

        # If it already has proper frontmatter, skip
        if _FRONTMATTER_RE.match(text):
            tracker.mark_done("phase2", key)
            continue

        # Extract meta comment left by Phase 1
        meta_match = _META_COMMENT_RE.match(text)
        tags      = []
        note_type = "zettel"

        if meta_match:
            try:
                meta      = json.loads(meta_match.group(1))
                tags      = meta.get("tags", [])
                note_type = meta.get("type", "zettel")
            except json.JSONDecodeError:
                pass
            text = text[meta_match.end():]  # Strip the comment

        note_id     = generate_id()
        frontmatter = build_frontmatter(note_id, file_path.stem, tags, note_type)
        file_path.write_text(frontmatter + text, encoding="utf-8")
        tracker.mark_done("phase2", key)

    log.info("Phase 2 complete.")


# ============================================================================
# PHASE 4 — MULTITHREADED AI LINKING
# ============================================================================
_AI_LINK_PROMPT = """\
You are an Obsidian knowledge linker for a veterinary nursing Zettelkasten.
Note title: {title}
Note content (first 600 chars): {snippet}
Related note titles in the vault ({count} candidates):
{titles}
Return a JSON array of titles from the list above that are directly and specifically related.
Example: ["Cardiac Output", "Frank-Starling Law", "Venous Return"]"""


def _apply_wikilinks_python(fp: Path, text: str, valid: list[str]) -> int:
    """Pure-Python wikilink application — used when Rust is unavailable."""
    masked, slots = mask_protected(text)
    masked_lower = masked.lower()
    added, no_match = 0, []

    for title in valid:
        if f"[[{title.lower()}]]" in masked_lower:
            continue
        pattern = _get_wikilink_pattern(title)
        masked, n = pattern.subn(r"[[\1]]", masked, count=1)
        if n:
            masked_lower = masked.lower()
            added += 1
        else:
            no_match.append(title)

    result = restore_protected(masked, slots)
    if no_match and "## Related" not in result:
        result = result.rstrip() + "\n\n## Related\n" + "\n".join(f"[[{t}]]" for t in no_match) + "\n"
        added += len(no_match)

    if result != text:
        fp.write_text(result, encoding="utf-8")
    return added


def _process_ai_link(fp: Path, config: Config, cache: TitleCache, tracker: ProcessingTracker) -> int:
    text = fp.read_text(encoding="utf-8")
    if count_wikilinks(text) >= config.ai_link_skip_threshold:
        tracker.mark_done("phase4", fp.stem)
        return 0

    if config.ai_link_use_semantic_filter:
        candidates = cache.get_candidates_for_note(fp.stem, text[:1000], config.ai_link_max_candidates)
    else:
        candidates = [t for t in cache.get_titles() if t != fp.stem][:config.ai_link_max_candidates]

    if not candidates:
        tracker.mark_done("phase4", fp.stem)
        return 0

    raw = call_llm(
        config,
        _AI_LINK_PROMPT.format(
            title=fp.stem,
            snippet=text[:600].replace("\n", " "),
            titles="\n".join(f"- {t}" for t in candidates),
            count=len(candidates),
        ),
        fp.stem,
    )
    suggested = extract_json_array(raw) if raw else None

    if not suggested:
        tracker.mark_done("phase4", fp.stem)
        return 0

    title_set = cache.get_title_set()
    # Pre-filter in Python: remove non-strings, unknown titles, self-links
    valid = [t for t in suggested if isinstance(t, str) and t in title_set and t != fp.stem]

    if not valid:
        tracker.mark_done("phase4", fp.stem)
        return 0

    if _RUST_AVAILABLE:
        # Rust: mask -> regex apply -> restore -> write, all in one native call
        added = _RUST.apply_wikilinks_to_file(str(fp), valid)
    else:
        added = _apply_wikilinks_python(fp, text, valid)

    if added:
        log.info("AI added %d -> %s", added, fp.stem)
    tracker.mark_done("phase4", fp.stem)
    return added


def run_phase4(config: Config):
    log.info("\n" + "="*50 + "\n=== PHASE 4: AI Semantic Linking ===\n" + "="*50)
    output_path = Path(config.output_vault)
    tracker = ProcessingTracker(output_path / config.tracker_filename)
    all_notes = vault_notes(output_path, config.tracker_filename)

    cache = get_title_cache()
    if cache.dirty:
        cache.rebuild(all_notes)

    pending = [p for p in all_notes if not tracker.is_done("phase4", p.stem)]
    if not pending:
        return log.info("All notes linked. Skipping.")

    total = 0
    with ThreadPoolExecutor(max_workers=config.ai_workers) as executor:
        futures = {executor.submit(_process_ai_link, fp, config, cache, tracker): fp for fp in pending}
        for future in tqdm(as_completed(futures), total=len(pending), desc="Linking"):
            total += future.result()

    tracker.flush()
    log.info("Phase 4 complete — %d AI links added.", total)


# ============================================================================
# PHASE 4.5 — TAG CONSOLIDATION
# ============================================================================
_TAG_CONSOLIDATION_PROMPT = """\
You are a knowledge architect for a veterinary/medical Zettelkasten.
The following tags are too specific and have very few notes:
{weak_tags}
Map each to a broader, general topic. Prefer using existing strong tags if appropriate:
{strong_tags}
If none fit, use standard broad categories like anatomy, physiology, pharmacology, pathology, clinical, behaviour.
Return ONLY a JSON dictionary mapping specific to broader tag: {{"left-ventricular-hypertrophy": "cardiology"}}"""


def _is_valid_tag(tag: str) -> bool:
    t = tag.lower().strip()
    return t and t not in _JUNK_TAGS and not _MODULE_CODE_RE.match(t) and not _NUMERIC_TAG_RE.match(t)


def _collect_tag_map(output_path: Path, tracker_filename: str) -> dict[str, list[Path]]:
    """
    Collect {tag: [paths]} for every valid tag across all non-MOC notes.

    Rust path: parallel rayon walk + frontmatter parsing (much faster on large vaults).
    Python fallback: serial read of every file.
    Tag validity filtering always runs in Python (it's vault-specific logic).
    """
    if _RUST_AVAILABLE:
        raw = _RUST.collect_tag_map_raw(str(output_path))
        return {
            tag: [Path(p) for p in paths]
            for tag, paths in raw.items()
            if _is_valid_tag(tag)
        }

    # Python fallback (serial)
    tag_map: dict[str, list[Path]] = defaultdict(list)
    for fp in output_path.rglob("*.md"):
        if fp.name.startswith(".") or "QUARANTINE_" in fp.name:
            continue
        text = fp.read_text(encoding="utf-8")
        fm = _FRONTMATTER_RE.match(text)
        if not fm or re.search(r"^type:\s*moc", fm.group(0), re.MULTILINE):
            continue
        for tag in _TAG_LINE_RE.findall(fm.group(0)):
            tag = tag.strip()
            if _is_valid_tag(tag):
                tag_map[tag].append(fp)
    return dict(tag_map)


def _update_note_tags(fp: Path, tag_mapping: dict[str, str]):
    """Apply tag remappings to a single file — uses Rust write path if available."""
    if _RUST_AVAILABLE:
        _RUST.update_note_tags_file(str(fp), tag_mapping)
        return

    # Python fallback
    text = fp.read_text(encoding="utf-8")
    fm_match = _FRONTMATTER_RE.match(text)
    if not fm_match:
        return

    current_tags = [t.strip() for t in _TAG_LINE_RE.findall(fm_match.group(0))]
    new_tags, changed = [], False
    for t in current_tags:
        mapped = tag_mapping.get(t)
        if mapped is not None:
            new_tags.append(mapped)
            changed = True
        else:
            new_tags.append(t)

    if not changed:
        return

    seen: set[str] = set()
    dedup_tags = [x for x in new_tags if not (x in seen or seen.add(x))]
    tag_block = "tags:\n" + "\n".join(f"  - {t}" for t in dedup_tags)
    new_fm = re.sub(r"tags:\n(?:  - .*\n)*", tag_block + "\n", fm_match.group(0))
    fp.write_text(text.replace(fm_match.group(0), new_fm, 1), encoding="utf-8")


def _process_tag_batch(batch: list[str], strong_tags_str: str, config: Config) -> dict:
    raw = call_llm(
        config,
        _TAG_CONSOLIDATION_PROMPT.format(
            weak_tags="\n".join(f"- {t}" for t in batch),
            strong_tags=strong_tags_str,
        ),
        "TagConsolidation",
    )
    return extract_json_dict(raw) or {}


def run_phase_4_5_tag_consolidation(config: Config, tag_map: Optional[dict] = None):
    log.info("\n" + "="*50 + "\n=== PHASE 4.5: Tag Consolidation ===\n" + "="*50)
    if tag_map is None:
        tag_map = _collect_tag_map(Path(config.output_vault), config.tracker_filename)

    strong_tags = {k: len(v) for k, v in tag_map.items() if len(v) >= config.moc_min_notes}
    weak_tags   = {k: len(v) for k, v in tag_map.items() if len(v) < config.tag_generalization_threshold}

    log.info(
        "Strong tags (>= %d notes): %d | Weak tags (< %d notes): %d",
        config.moc_min_notes, len(strong_tags),
        config.tag_generalization_threshold, len(weak_tags),
    )
    if not weak_tags:
        return log.info("All tags have sufficient notes. Skipping.")

    weak_tag_names = list(weak_tags.keys())
    batches = [weak_tag_names[i:i + 30] for i in range(0, len(weak_tag_names), 30)]
    strong_tags_str = ", ".join(strong_tags.keys()) if strong_tags else "anatomy, physiology, clinical, pathology"
    global_mapping: dict[str, str] = {}

    log.info("Mapping %d weak tags using AI (%d batches)...", len(weak_tags), len(batches))
    with ThreadPoolExecutor(max_workers=config.ai_workers) as executor:
        futures = [executor.submit(_process_tag_batch, b, strong_tags_str, config) for b in batches]
        for future in tqdm(as_completed(futures), total=len(batches), desc="Consolidating"):
            for k, v in future.result().items():
                if k in weak_tags:
                    global_mapping[k] = v.lower().strip()

    if not global_mapping:
        return log.warning("AI failed to provide valid tag mappings.")

    # Batch all changes per file so each file is written exactly once
    file_changes: dict[Path, dict[str, str]] = defaultdict(dict)
    for old_tag, new_tag in global_mapping.items():
        if old_tag == new_tag:
            continue
        log.info("  Mapping: '%s' (%d notes) -> '%s'", old_tag, weak_tags[old_tag], new_tag)
        for fp in tag_map.get(old_tag, []):
            file_changes[fp][old_tag] = new_tag

    for fp, mapping in file_changes.items():
        _update_note_tags(fp, mapping)

    log.info(
        "Phase 4.5 complete — %d tags consolidated across %d files.",
        len(global_mapping), len(file_changes),
    )


# ============================================================================
# PHASE 5 — MULTITHREADED MOC GENERATION
# ============================================================================
_MOC_PROMPT = """\
You are building a Map of Content (MOC) for an Obsidian Zettelkasten vault.
Topic: {topic}
Notes to organise ({count} total):
{note_list}
Write a well-structured MOC note. Group notes under relevant ## headings.
List every note provided exactly once as a [[wikilink]]. Do not output YAML."""


def _process_single_moc(topic: str, notes: list[Path], config: Config, moc_dir: Path, tracker: ProcessingTracker):
    moc_key = f"MOC_{topic}"
    body = call_llm(
        config,
        _MOC_PROMPT.format(
            topic=topic,
            note_list="\n".join(f"- {p.stem}" for p in notes),
            count=len(notes),
        ),
        moc_key,
        strict_json=False,
    )
    if not body:
        tracker.mark_done("phase5", moc_key)
        return

    title = f"MOC - {topic.title()}"
    fm = (
        f"---\nid: {generate_id()}\ntitle: \"{title}\"\n"
        f"tags:\n  - moc\n  - {topic}\ntype: moc\n"
        f"created: {datetime.now().strftime('%Y-%m-%d')}\n---\n\n"
    )
    (moc_dir / f"{safe_filename(title)}.md").write_text(fm + f"# {title}\n\n" + body, encoding="utf-8")
    log.info("Created MOC: %s (%d notes)", title, len(notes))
    tracker.mark_done("phase5", moc_key)


def run_phase5(config: Config, tag_map: Optional[dict] = None):
    log.info("\n" + "="*50 + "\n=== PHASE 5: MOC Generation ===\n" + "="*50)
    output_path = Path(config.output_vault)
    tracker = ProcessingTracker(output_path / config.tracker_filename)
    moc_dir = output_path / "01_MOCs"

    if tag_map is None:
        tag_map = _collect_tag_map(output_path, config.tracker_filename)

    run_phase0(config)
    run_phase1(config)
    run_phase2(config)

    eligible = {t: n for t, n in tag_map.items() if len(n) >= config.moc_min_notes}
    sorted_topics = sorted(eligible.items(), key=lambda x: -len(x[1]))
    if config.moc_max_count > 0:
        sorted_topics = sorted_topics[:config.moc_max_count]

    pending = [(t, n) for t, n in sorted_topics if not tracker.is_done("phase5", f"MOC_{t}")]
    log.info("%d eligible topics. %d pending generation.", len(eligible), len(pending))
    if not pending:
        return

    with ThreadPoolExecutor(max_workers=config.ai_workers) as executor:
        futures = {
            executor.submit(_process_single_moc, t, n, config, moc_dir, tracker): t
            for t, n in pending
        }
        for future in tqdm(as_completed(futures), total=len(pending), desc="MOCs"):
            future.result()

    tracker.flush()


# ============================================================================
# PHASE 6 — MULTITHREADED MOC IMPROVEMENT
# ============================================================================
_MOC_IMPROVE_PROMPT = """\
You are improving an existing Map of Content (MOC) note.
MOC Title: {title}
Current Body:
{current_body}
All notes currently tagged with this topic ({count} total):
{note_list}
Fix problems: Ensure EVERY note in the list appears as a [[wikilink]] exactly once. Remove orphaned links. Improve structure. Return ONLY the improved body."""

_MOC_QUALITY_RE = re.compile(r"\[\[.*?\]\]")


def _moc_needs_improvement(body: str, expected_notes: list[Path]) -> bool:
    if len(body) < 300:
        return True
    linked = {m.group(0)[2:-2].lower() for m in _MOC_QUALITY_RE.finditer(body)}
    expected_stems = {p.stem.lower() for p in expected_notes}
    return (len(linked & expected_stems) / max(len(expected_stems), 1)) < 0.5


def _process_improve_moc(
    fp: Path,
    text: str,
    expected_notes: list[Path],
    config: Config,
    tracker: ProcessingTracker,
) -> int:
    fm_match = _FRONTMATTER_RE.match(text)
    if not fm_match:
        tracker.mark_done("phase6", fp.stem)
        return 0

    body = _FRONTMATTER_RE.sub("", text, count=1).strip()
    if not _moc_needs_improvement(body, expected_notes):
        tracker.mark_done("phase6", fp.stem)
        return 0

    new_body = call_llm(
        config,
        _MOC_IMPROVE_PROMPT.format(
            title=fp.stem,
            current_body=body[:2000],
            note_list="\n".join(f"- {p.stem}" for p in expected_notes) if expected_notes else "(none)",
            count=len(expected_notes),
        ),
        f"MOC6:{fp.stem}",
        strict_json=False,
    )
    if not new_body:
        tracker.mark_done("phase6", fp.stem)
        return 0

    fp.write_text(fm_match.group(0) + "\n" + new_body.strip() + "\n", encoding="utf-8")
    log.info("Improved MOC: %s", fp.stem)
    tracker.mark_done("phase6", fp.stem)
    return 1


def run_phase6(config: Config, tag_map: Optional[dict] = None):
    log.info("\n" + "="*50 + "\n=== PHASE 6: MOC Improvement ===\n" + "="*50)
    output_path = Path(config.output_vault)
    tracker = ProcessingTracker(output_path / config.tracker_filename)
    moc_dir = output_path / "01_MOCs"

    if not moc_dir.exists():
        return

    if tag_map is None:
        tag_map = _collect_tag_map(output_path, config.tracker_filename)

    run_phase0(config)
    run_phase1(config)
    run_phase2(config)

    pending: list[tuple[Path, str, list[Path]]] = []
    for fp in moc_dir.glob("*.md"):
        if fp.name.startswith(".") or tracker.is_done("phase6", fp.stem):
            continue
        text = fp.read_text(encoding="utf-8")
        fm_match = _FRONTMATTER_RE.match(text)
        tags = (
            [t.strip() for t in _TAG_LINE_RE.findall(fm_match.group(0))
             if t.strip() != "moc" and _is_valid_tag(t.strip())]
            if fm_match else []
        )
        expected_notes = tag_map.get(tags[0], []) if tags else []
        pending.append((fp, text, expected_notes))

    if not pending:
        return

    improved = 0
    with ThreadPoolExecutor(max_workers=config.ai_workers) as executor:
        futures = [
            executor.submit(_process_improve_moc, fp, text, expected_notes, config, tracker)
            for fp, text, expected_notes in pending
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Improving MOCs"):
            improved += future.result()

    tracker.flush()
    log.info("Phase 6 complete — %d MOCs improved.", improved)


def run_phase8(config: Config):
    \"\"\"
    Phase 8: Autonomous Research Note Generation.
    Uses the local 'autoresearch' engine to generate synthetic literature notes.
    \"\"\"
    log.info("Phase 8: Autonomous Research Initiation...")
    
    # Check if a model exists
    model_path = Path("autoresearch") / "model.pt"
    if not model_path.exists():
        log.warning("Phase 8 skipped: No trained model found in autoresearch/model.pt. Run 'vault_cli.py train' first.")
        return

    print("\n" + "*"*50)
    print("Welcome to the Autonomous Research Laboratory")
    print("*"*50)
    
    while True:
        prompt = input("\nEnter a research query (or 'quit' to finalize): ").strip()
        if not prompt or prompt.lower() in ("quit", "exit", "q"):
            break
            
        log.info("Generating research note for: '%s'...", prompt)
        
        # We run this as a subprocess to keep the torch environment in 'autoresearch' isolated
        import subprocess
        try:
            # We use 'uv run' to ensure dependencies are resolved
            # We pass the prompt as input to avoid shell escaping issues
            process = subprocess.Popen(
                ["uv", "run", "python", "generate.py"],
                cwd="autoresearch",
                stdin=subprocess.PIPE,
                text=True
            )
            process.communicate(input=prompt + "\n")
            log.info("Successfully generated literature note.")
        except Exception as e:
            log.error("Failed to generate research note: %s", e)

# ============================================================================
# ENTRY POINT
# ============================================================================
def main():
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(
            "Obsidian Zettelkasten Converter (threaded)\n\n"
            "Environment:\n"
            "  VAULT_INPUT_PATH / VAULT_OUTPUT_PATH  - vault paths\n"
            "  VAULT_LLM_PROVIDER                    - provider(s), comma-separated for fallback:\n"
            "    gemini,ollama                        - Gemini primary, Ollama when quota hit\n"
            "    ollama,gemini                        - Ollama primary, Gemini as escape valve\n"
            "    ollama / gemini / azure              - single provider\n"
            "  VAULT_OLLAMA_MODEL                    - override local model (default: auto-select)\n"
            "  OLLAMA_HOST                           - Ollama server URL\n"
            "  GEMINI_API_KEY                        - required when provider includes 'gemini'\n"
            "  VAULT_GEMINI_MODEL                    - Gemini model (default: gemini-2.5-flash)\n\n"
            f"Rust engine: {'ACTIVE' if _RUST_AVAILABLE else 'not found — run: cd reconstruct_rust && maturin develop'}\n"
        )
        raise SystemExit(0)

    config = Config()
    output_path = Path(config.output_vault)
    # Collect tag map once — Rust makes this a parallel rayon scan
    tag_map = _collect_tag_map(output_path, config.tracker_filename)

    run_phase0(config)
    run_phase1(config)
    run_phase2(config)

    run_phase4(config)
    run_phase_4_5_tag_consolidation(config, tag_map=tag_map)
    run_phase5(config, tag_map=tag_map)
    run_phase6(config, tag_map=tag_map)
    run_phase8(config)
    log.info("=== Zettelkasten Knowledge Architecture Complete! ===")

    if VaultAnalyzer is not None:
        log.info("\n" + "="*50 + "\n=== VAULT HEALTH CHECKS & IMPROVEMENTS ===\n" + "="*50)
        imp_config = ImproverConfig(vault_path=str(output_path), dry_run=False, fix_tags=True, fix_links=True)
        analyzer = VaultAnalyzer(output_path)
        analyzer.scan_vault(imp_config.short_note_threshold)
        analyzer.analyze_links()
        analyzer.analyze_tags()
        generate_health_report(analyzer, output_path)
        find_broken_wikilinks_report(analyzer, output_path)
        generate_empty_notes_report(analyzer, output_path)
        generate_orphan_suggestions_report(analyzer, output_path)
        consolidate_mocs(analyzer, imp_config)
        standardize_tags(analyzer, imp_config)
        if imp_config.fix_links:
            fix_broken_links(analyzer, imp_config)
        log.info("=== Vault Improver Complete ===")

if __name__ == "__main__":
    main()
