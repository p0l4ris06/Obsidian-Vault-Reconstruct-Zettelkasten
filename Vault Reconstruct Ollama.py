"""
Obsidian Zettelkasten Converter — v2
Splits lecture notes into atomic Zettelkasten notes with full Zettelkasten
structure: unique IDs, YAML frontmatter, tags, classification, AI-assisted
linking, quarantine recovery, and MOC generation.

Usage:
    Make sure Ollama is running (ollama serve) and your model is pulled, then:
    python obsidian_zettelkasten_v2.py

Phases:
    0 — Recover quarantined notes from previous run
    1 — Split notes into atomic zettels (with tags + classification)
    2 — Add YAML frontmatter to every note
    3 — AI-assisted linking (semantically aware, not just regex)
    4 — Generate Maps of Content (MOCs) per topic
"""

import json
import time
import re
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

import os
import ollama
from ollama import Client


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    input_vault:        str   = r"C:\Users\dcrac\Documents\University Vault"
    output_vault:       str   = r"C:\Users\dcrac\Documents\Obsidian Vault"
    # Cloud model (used first if OLLAMA_API_KEY is set); set to None to force local
    cloud_model:        str   = "gemma3:4b-cloud"
    # Local model used as fallback when cloud fails or no API key is found
    local_model:        str   = "gemma3:4b"
    min_content_length: int   = 50
    request_delay:      float = 0.5
    max_retries:        int   = 3
    # Minimum links a note needs before AI linking is skipped (saves time)
    min_links_before_ai_skip: int = 2
    output_folders: list = field(default_factory=lambda: [
        "00_Inbox", "01_MOCs", "02_Zettels", "03_Literature"
    ])
    tracker_filename: str = ".zettelkasten_tracker.json"


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
# ID GENERATOR — unique Zettelkasten IDs per session
# ============================================================================

_id_counter = 0

def generate_id() -> str:
    """Unique ID: YYYYMMDD + zero-padded session counter, e.g. 202501140042."""
    global _id_counter
    _id_counter += 1
    return datetime.now().strftime("%Y%m%d") + f"{_id_counter:04d}"


# ============================================================================
# TRACKER — phase-aware, so each phase can be re-run independently
# ============================================================================

class ProcessingTracker:
    """
    Tracks completion per phase. JSON structure:
    { "phase0": [...], "phase1": [...], "phase2": [...], ... }
    """
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
                total = sum(len(v) for v in self._data.values())
                log.info("Tracker loaded — %d total phase completions recorded.", total)
            except (json.JSONDecodeError, OSError):
                log.warning("Could not read tracker; starting fresh.")

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(
                {k: sorted(v) for k, v in self._data.items()},
                f, indent=2
            )

    def is_done(self, phase: str, key: str) -> bool:
        return key in self._data.get(phase, set())

    def mark_done(self, phase: str, key: str):
        self._data.setdefault(phase, set()).add(key)
        self._save()


# ============================================================================
# JSON HELPERS
# ============================================================================

def extract_json_array(text: str) -> list[dict] | None:
    for candidate in (text, _strip_fences(text), _regex_extract(text)):
        if candidate is None:
            continue
        try:
            result = json.loads(candidate)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    return None


def extract_json_string_array(text: str) -> list[str] | None:
    """Extract a JSON array of strings (used for link suggestions)."""
    for candidate in (text, _strip_fences(text), _regex_extract(text)):
        if candidate is None:
            continue
        try:
            result = json.loads(candidate)
            if isinstance(result, list) and all(isinstance(i, str) for i in result):
                return result
        except json.JSONDecodeError:
            pass
    return None


def _strip_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)


def _regex_extract(text: str) -> str | None:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    return match.group(0) if match else None


# ============================================================================
# HELPERS
# ============================================================================

_UNSAFE_CHARS = re.compile(r'[\\/:*?"<>|]')

def safe_filename(title: str) -> str:
    return _UNSAFE_CHARS.sub("-", title).strip(". ")[:200] or "Untitled"


def _make_cloud_client() -> Client | None:
    """Return an ollama Client pointed at ollama.com if API key is set, else None."""
    api_key = os.environ.get("OLLAMA_API_KEY", "").strip()
    if not api_key:
        return None
    return Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {api_key}"},
    )


def _chat(client: Client | None, model: str, prompt: str) -> str:
    """Call ollama.chat via client or module-level, return content string."""
    messages = [{"role": "user", "content": prompt}]
    if client is not None:
        response = client.chat(model=model, messages=messages)
    else:
        response = ollama.chat(model=model, messages=messages)
    return response["message"]["content"]


def call_ollama(config: Config, prompt: str, label: str) -> str | None:
    """
    Cloud-first with local fallback.
    - If OLLAMA_API_KEY is set, tries cloud_model first.
    - Falls back to local_model automatically on any cloud failure.
    - Retries connection errors with back-off on both paths.
    """
    cloud_client = _make_cloud_client()
    use_cloud    = cloud_client is not None

    # Build ordered list of (client, model, description) to try
    candidates = []
    if use_cloud:
        candidates.append((cloud_client, config.cloud_model, "cloud"))
    candidates.append((None, config.local_model, "local"))

    for client, model, source in candidates:
        for attempt in range(config.max_retries):
            try:
                text = _chat(client, model, prompt)
                if source == "cloud" and attempt == 0 and use_cloud:
                    pass  # happy path, no extra log needed
                else:
                    log.debug("[%s] Response via %s/%s", label, source, model)
                return text

            except ollama.ResponseError as exc:
                log.warning("[%s] %s model error: %s — trying next option.", label, source, exc)
                break  # Model-level error; skip retries, move to next candidate

            except Exception as exc:
                msg = str(exc).lower()
                is_connection = any(t in msg for t in ("connection", "refused", "timeout", "rate"))
                if is_connection and attempt < config.max_retries - 1:
                    wait = 10 * (attempt + 1)
                    log.warning("[%s] %s not responding. Retrying in %ds (%d/%d)…",
                                label, source, wait, attempt + 1, config.max_retries)
                    time.sleep(wait)
                else:
                    log.warning("[%s] %s failed (%s) — falling back.", label, source, exc)
                    break  # Move to next candidate

    log.error("[%s] All inference options exhausted (cloud + local).", label)
    return None

    log.critical(
        "Cannot reach Ollama after %d attempts. Is 'ollama serve' running?",
        config.max_retries,
    )
    sys.exit(1)


_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
_CODE_FENCE_RE  = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]+`")
_WIKILINK_RE    = re.compile(r"\[\[.*?\]\]")


def count_wikilinks(text: str) -> int:
    return len(_WIKILINK_RE.findall(text))


def mask_protected(text: str) -> tuple[str, list[tuple[str, str]]]:
    placeholders: list[tuple[str, str]] = []

    def _replace(m: re.Match) -> str:
        token = f"\x00PH{len(placeholders)}\x00"
        placeholders.append((token, m.group(0)))
        return token

    masked = _FRONTMATTER_RE.sub(_replace, text, count=1)
    masked = _CODE_FENCE_RE.sub(_replace, masked)
    masked = _INLINE_CODE_RE.sub(_replace, masked)
    masked = _WIKILINK_RE.sub(_replace, masked)
    return masked, placeholders


def restore_protected(text: str, placeholders: list[tuple[str, str]]) -> str:
    for token, original in placeholders:
        text = text.replace(token, original)
    return text


# ============================================================================
# FOLDER MAPPING
# ============================================================================

FOLDER_MAP = {
    "moc":        "01_MOCs",
    "literature": "03_Literature",
    "zettel":     "02_Zettels",
}

def classify_folder(note_type: str) -> str:
    return FOLDER_MAP.get(note_type.lower().strip(), "02_Zettels")


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
        text = call_ollama(config, prompt, key)

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
    text = call_ollama(config, prompt, filename)
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
# PHASE 3 — AI-ASSISTED LINKING
# ============================================================================

LINK_PROMPT = """\
You are an Obsidian knowledge linker.

Here is a note:
TITLE: {title}
CONTENT (first 600 chars): {snippet}

Here are all note titles in the vault:
{titles}

Return ONLY a JSON array of titles from the list above that are directly \
conceptually related to this note and should be wikilinked from it.
Be selective — only genuinely relevant titles.
If none are relevant, return an empty array: []

Example output: ["Cardiac Output", "Frank-Starling Law"]"""


def regex_link_pass(file_path: Path, titles: list[str]) -> int:
    """Fast first pass: exact regex matching. Returns number of links added."""
    original = file_path.read_text(encoding="utf-8")
    masked, placeholders = mask_protected(original)
    links_added = 0

    for title in titles:
        if title == file_path.stem:
            continue
        pattern = r"(?<!\[\[)\b(" + re.escape(title) + r")\b(?!\]\])"
        new_masked, count = re.subn(
            pattern, r"[[\1]]", masked, count=1, flags=re.IGNORECASE
        )
        if count:
            masked = new_masked
            links_added += 1

    result = restore_protected(masked, placeholders)
    if result != original:
        file_path.write_text(result, encoding="utf-8")
    return links_added


def ai_link_pass(config: Config, file_path: Path, all_titles: list[str]) -> int:
    """AI-assisted pass for notes that regex couldn't link well."""
    text    = file_path.read_text(encoding="utf-8")
    snippet = text[:600].replace("\n", " ")
    titles_str = "\n".join(f"- {t}" for t in all_titles if t != file_path.stem)

    prompt = LINK_PROMPT.format(
        title=file_path.stem,
        snippet=snippet,
        titles=titles_str,
    )

    raw = call_ollama(config, prompt, file_path.stem)
    if raw is None:
        return 0

    suggested = extract_json_string_array(raw)
    if not suggested:
        return 0

    # Apply the AI-suggested links
    masked, placeholders = mask_protected(text)
    links_added = 0

    for title in suggested:
        # Only link if this title actually exists
        if title not in all_titles:
            continue
        if title == file_path.stem:
            continue
        # Don't re-link something already wikilinked
        if f"[[{title}]]" in text:
            continue

        pattern = r"(?<!\[\[)\b(" + re.escape(title) + r")\b(?!\]\])"
        new_masked, count = re.subn(
            pattern, r"[[\1]]", masked, count=1, flags=re.IGNORECASE
        )
        if count:
            masked = new_masked
            links_added += 1
        else:
            # Title doesn't appear verbatim — append a "Related" section
            # (handled below after the loop)
            pass

    result = restore_protected(masked, placeholders)

    # If AI suggested links but none appeared verbatim, add a Related section
    valid_suggestions = [t for t in suggested if t in all_titles and t != file_path.stem]
    if valid_suggestions and "## Related" not in result:
        related_links = "  ".join(f"[[{t}]]" for t in valid_suggestions)
        result = result.rstrip() + f"\n\n## Related\n{related_links}\n"
        links_added += len(valid_suggestions)

    if result != text:
        file_path.write_text(result, encoding="utf-8")

    return links_added


def run_phase3(config: Config):
    log.info("=== PHASE 3: AI-assisted linking ===")

    output_path = Path(config.output_vault)
    tracker     = ProcessingTracker(output_path / config.tracker_filename)

    all_notes_raw = [
        p for p in output_path.rglob("*.md")
        if not p.name.startswith(".")
        and "QUARANTINE_" not in p.name
        and p.name != config.tracker_filename
    ]

    # Build title list from ALL notes (including done ones) for link targets
    all_titles = sorted(
        [f.stem for f in all_notes_raw if len(f.stem) > 3],
        key=len, reverse=True,
    )

    all_notes = [p for p in all_notes_raw if not tracker.is_done("phase3", p.stem)]
    log.info("%d notes total, %d remaining to link.", len(all_notes_raw), len(all_notes))

    if not all_notes:
        log.info("All notes already linked. Skipping Phase 3.")
        return

    regex_total = 0
    ai_total    = 0

    for file_path in tqdm(all_notes, desc="Linking notes"):
        key = file_path.stem

        # Pass 1: regex
        added = regex_link_pass(file_path, all_titles)
        regex_total += added

        # Pass 2: AI for under-linked notes
        current_text  = file_path.read_text(encoding="utf-8")
        current_links = count_wikilinks(current_text)

        if current_links < config.min_links_before_ai_skip:
            ai_added  = ai_link_pass(config, file_path, all_titles)
            ai_total += ai_added
            if ai_added:
                log.info("AI linked %d → %s", ai_added, key)
            time.sleep(config.request_delay)

        tracker.mark_done("phase3", key)

    log.info("Phase 3 complete. Regex: %d links. AI: %d links.", regex_total, ai_total)


# ============================================================================
# PHASE 4 — MOC GENERATION
# ============================================================================

MOC_PROMPT = """\
You are building a Map of Content (MOC) note for an Obsidian Zettelkasten vault.

Topic: {topic}
Related notes:
{note_list}

Write a short MOC note (200-400 words) that:
1. Briefly introduces the topic
2. Groups the related notes into logical sub-sections using ## headers
3. References each note as a wikilink [[Note Title]]
4. Adds a brief one-line annotation after each link explaining what it covers

Output ONLY the note body (no frontmatter). Start directly with the intro paragraph."""


def collect_tags(output_path: Path) -> dict[str, list[Path]]:
    """Return a mapping of tag → list of note paths that have that tag."""
    tag_map: dict[str, list[Path]] = {}

    for file_path in output_path.rglob("*.md"):
        if file_path.name.startswith(".") or "QUARANTINE_" in file_path.name:
            continue
        text = file_path.read_text(encoding="utf-8")
        fm_match = _FRONTMATTER_RE.match(text)
        if not fm_match:
            continue
        fm_text = fm_match.group(0)
        # Extract tags from frontmatter
        for tag in re.findall(r"^\s+- (.+)$", fm_text, re.MULTILINE):
            tag = tag.strip()
            if tag and tag != "general":
                tag_map.setdefault(tag, []).append(file_path)

    return tag_map


def run_phase4(config: Config):
    log.info("=== PHASE 4: Generating MOCs ===")

    output_path = Path(config.output_vault)
    tracker     = ProcessingTracker(output_path / config.tracker_filename)
    moc_dir     = output_path / "01_MOCs"
    tag_map     = collect_tags(output_path)

    # Only generate MOCs for tags with 3+ notes (avoids noise)
    all_eligible = {tag: notes for tag, notes in tag_map.items() if len(notes) >= 3}
    eligible = {t: n for t, n in all_eligible.items() if not tracker.is_done("phase4", f"MOC_{t}")}
    log.info("%d eligible topics, %d MOCs remaining to generate.", len(all_eligible), len(eligible))

    if not eligible:
        log.info("All MOCs already generated. Skipping Phase 4.")
        return

    for topic, notes in tqdm(eligible.items(), desc="Generating MOCs"):
        moc_key = f"MOC_{topic}"

        note_list = "\n".join(f"- {p.stem}" for p in notes)
        prompt    = MOC_PROMPT.format(topic=topic, note_list=note_list)
        body      = call_ollama(config, prompt, moc_key)

        if body is None:
            log.warning("Could not generate MOC for topic: %s", topic)
            tracker.mark_done("phase4", moc_key)
            continue

        note_id     = generate_id()
        title       = f"MOC - {topic.title()}"
        frontmatter = (
            f"---\n"
            f"id: {note_id}\n"
            f"title: \"{title}\"\n"
            f"tags:\n  - moc\n  - {topic}\n"
            f"type: moc\n"
            f"created: {datetime.now().strftime('%Y-%m-%d')}\n"
            f"---\n\n"
        )

        moc_path = moc_dir / f"{safe_filename(title)}.md"
        moc_path.write_text(frontmatter + f"# {title}\n\n" + body, encoding="utf-8")
        log.info("Created MOC: %s (%d notes)", title, len(notes))
        tracker.mark_done("phase4", moc_key)
        time.sleep(config.request_delay)

    log.info("Phase 4 complete.")


# ============================================================================
# ENTRY POINT
# ============================================================================

def _local_model_names() -> list[str]:
    """Return list of locally available model name strings."""
    try:
        models = ollama.list()
        model_list = models.get("models", []) if isinstance(models, dict) else list(models)
        names = []
        for m in model_list:
            if isinstance(m, dict):
                names.append(m.get("model", m.get("name", "")))
            else:
                names.append(getattr(m, "model", str(m)))
        return names
    except Exception:
        return []


def verify_ollama(config: Config):
    api_key    = os.environ.get("OLLAMA_API_KEY", "").strip()
    cloud_ok   = False
    local_ok   = False

    # Check cloud
    if api_key:
        try:
            client   = _make_cloud_client()
            response = client.chat(
                model=config.cloud_model,
                messages=[{"role": "user", "content": "ping"}],
            )
            cloud_ok = True
            log.info("Cloud ready — model '%s' reachable.", config.cloud_model)
        except Exception as exc:
            log.warning("Cloud check failed (%s). Will fall back to local.", exc)

    # Check local
    try:
        names    = _local_model_names()
        local_ok = any(config.local_model in n for n in names)
        if local_ok:
            log.info("Local ready — model '%s' found.", config.local_model)
        else:
            log.warning(
                "Local model '%s' not pulled. Run:  ollama pull %s",
                config.local_model, config.local_model,
            )
    except Exception as exc:
        log.warning("Cannot reach local Ollama (%s). Run: ollama serve", exc)

    if not cloud_ok and not local_ok:
        log.critical(
            "Neither cloud nor local Ollama is available.\n"
            "  Cloud: set OLLAMA_API_KEY environment variable\n"
            "  Local: run 'ollama serve' and 'ollama pull %s'",
            config.local_model,
        )
        sys.exit(1)

    if not api_key:
        log.info("No OLLAMA_API_KEY found — running in local-only mode.")
    elif cloud_ok:
        log.info("Mode: cloud-first with local fallback.")


def main():
    config = Config()
    verify_ollama(config)

    run_phase0(config)   # Recover quarantined notes
    run_phase1(config)   # Split + classify + tag
    run_phase2(config)   # YAML frontmatter
    run_phase3(config)   # AI-assisted linking
    run_phase4(config)   # MOC generation

    log.info("=== All done! ===")


if __name__ == "__main__":
    main()
