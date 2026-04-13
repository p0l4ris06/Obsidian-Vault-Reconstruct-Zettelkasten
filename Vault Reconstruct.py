"""
Obsidian Zettelkasten Converter
Splits lecture notes into atomic Zettelkasten notes and auto-generates wiki-links.

Usage:
    Set GEMINI_API_KEY as an environment variable (or place in a .env file),
    then run: python obsidian_zettelkasten.py
"""

import os
import json
import time
import re
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, field

# ── Optional dependency: python-dotenv ──────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Fine — just use real environment variables

# ── Optional dependency: tqdm ────────────────────────────────────────────────
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):  # Graceful fallback
        return iterable

from google import genai

# ============================================================================
# CONFIGURATION — edit these two paths; keep the API key in your environment
# ============================================================================

@dataclass
class Config:
    input_vault:  str = r"C:\Users\dcrac\Documents\University Vault"
    output_vault: str = r"C:\Users\dcrac\Documents\Obsidian Vault"
    model:        str = "gemini-2.5-flash"
    min_content_length: int = 50
    request_delay:      float = 4.0   # seconds between successful API calls
    max_retries:        int = 5
    output_folders:     list = field(default_factory=lambda: [
        "00_Inbox", "01_MOCs", "02_Zettels", "03_Literature"
    ])
    # Single tracker file instead of hundreds of hidden files
    tracker_filename: str = ".processed_files.json"


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("zettelkasten_run.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ============================================================================
# TRACKER — persists processed filenames in one clean JSON file
# ============================================================================

class ProcessingTracker:
    def __init__(self, tracker_path: Path):
        self.path = tracker_path
        self._processed: set[str] = set()
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, encoding="utf-8") as f:
                    self._processed = set(json.load(f))
                log.info("Resuming — %d files already processed.", len(self._processed))
            except (json.JSONDecodeError, OSError):
                log.warning("Could not read tracker file; starting fresh.")

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(sorted(self._processed), f, indent=2)

    def is_done(self, key: str) -> bool:
        return key in self._processed

    def mark_done(self, key: str):
        self._processed.add(key)
        self._save()


# ============================================================================
# JSON HELPERS
# ============================================================================

def extract_json_array(text: str) -> list[dict] | None:
    """
    Try three strategies to pull a valid JSON array out of a model response:
      1. Direct parse (model was well-behaved)
      2. Regex extraction of the outermost [...] block
      3. Strip markdown code fences and retry
    Returns None if all strategies fail.
    """
    for attempt in (text, _strip_fences(text), _regex_extract(text)):
        if attempt is None:
            continue
        try:
            result = json.loads(attempt)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    return None


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers."""
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)


def _regex_extract(text: str) -> str | None:
    """Pull the outermost [...] block from text."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    return match.group(0) if match else None


# ============================================================================
# SAFE FILENAME
# ============================================================================

_UNSAFE_CHARS = re.compile(r'[\\/:*?"<>|]')

def safe_filename(title: str) -> str:
    """Strip characters illegal in filenames across platforms."""
    return _UNSAFE_CHARS.sub("-", title).strip(". ")[:200] or "Untitled"


# ============================================================================
# PHASE 1 — GENERATE ATOMIC NOTES
# ============================================================================

SPLIT_PROMPT = """\
You are a Zettelkasten assistant. Restructure the note below into one or more
atomic notes. Each note should cover exactly one idea. Do NOT omit any
information — add context where needed for continuity.

RULES:
- Output ONLY a valid JSON array; no preamble, no markdown fences.
- Schema: [{{"title": "...", "content": "# Title\\n..."}}]
- Escape any double-quotes inside strings as \\".

NOTE TO PROCESS:
{content}
"""

def split_note(client, config: Config, content: str, filename: str) -> list[dict] | None:
    """
    Call the Gemini API with exponential back-off.
    Returns a list of note dicts on success, or None if all retries failed.
    """
    prompt = SPLIT_PROMPT.format(content=content)

    for attempt in range(config.max_retries):
        try:
            response = client.models.generate_content(
                model=config.model,
                contents=prompt,
            )
            notes = extract_json_array(response.text)
            if notes is not None:
                return notes
            log.warning("[%s] Response was not valid JSON (attempt %d/%d).",
                        filename, attempt + 1, config.max_retries)
            return None  # Bad JSON isn't fixed by retrying — quarantine it

        except Exception as exc:
            msg = str(exc).lower()
            is_rate_limit = any(tok in msg for tok in ("429", "quota", "exhausted", "rate"))

            if is_rate_limit:
                wait = 60 * (2 ** attempt)  # Exponential back-off: 60 s, 120 s, 240 s…
                log.warning("Rate-limited. Waiting %d s before retry %d/%d…",
                            wait, attempt + 1, config.max_retries)
                time.sleep(wait)
            else:
                log.error("[%s] Unexpected error: %s — skipping.", filename, exc)
                return None  # Non-retriable

    log.critical(
        "Hit max retries. You've likely exhausted the daily free-tier quota.\n"
        "Run the script again tomorrow — it will resume where it left off."
    )
    sys.exit(1)


def run_phase1(client, config: Config):
    log.info("=== PHASE 1: Splitting notes into Zettelkasten ===")

    output_path = Path(config.output_vault)
    for folder in config.output_folders:
        (output_path / folder).mkdir(parents=True, exist_ok=True)

    tracker = ProcessingTracker(output_path / config.tracker_filename)

    input_files = [
        p for p in Path(config.input_vault).rglob("*.md")
        if ".obsidian" not in p.parts
    ]
    log.info("Found %d markdown files.", len(input_files))

    for file_path in tqdm(input_files, desc="Splitting notes"):
        filename = file_path.name

        if tracker.is_done(filename):
            continue

        content = file_path.read_text(encoding="utf-8")
        if len(content.strip()) < config.min_content_length:
            tracker.mark_done(filename)
            continue

        notes = split_note(client, config, content, filename)

        if notes is None:
            # Quarantine: save raw content for manual review
            quarantine_path = output_path / "00_Inbox" / f"QUARANTINE_{filename}"
            quarantine_path.write_text(
                f"<!-- Failed to parse AI response for: {filename} -->\n\n{content}",
                encoding="utf-8",
            )
            log.warning("Quarantined: %s", filename)
        else:
            zettel_dir = output_path / "02_Zettels"
            for note in notes:
                title   = safe_filename(note.get("title", "Untitled"))
                note_content = note.get("content", "")
                (zettel_dir / f"{title}.md").write_text(note_content, encoding="utf-8")
            log.info("Split into %d note(s): %s", len(notes), filename)

        tracker.mark_done(filename)
        time.sleep(config.request_delay)

    log.info("Phase 1 complete.")


# ============================================================================
# PHASE 2 — AUTO-LINK THE VAULT
# ============================================================================

# Matches YAML front-matter at the top of a file
_FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
# Matches fenced code blocks  ```...```
_CODE_FENCE_RE  = re.compile(r"```.*?```", re.DOTALL)
# Matches inline code  `...`
_INLINE_CODE_RE = re.compile(r"`[^`]+`")
# Matches existing wiki-links  [[...]]
_WIKILINK_RE    = re.compile(r"\[\[.*?\]\]")


def _mask_protected_regions(text: str) -> tuple[str, list[tuple[str, str]]]:
    """
    Replace front-matter, code blocks, and existing wiki-links with unique
    placeholders so Phase 2 never corrupts them. Returns the masked text and
    the placeholder→original mapping.
    """
    placeholders: list[tuple[str, str]] = []

    def _replace(m: re.Match) -> str:
        token = f"\x00PLACEHOLDER_{len(placeholders)}\x00"
        placeholders.append((token, m.group(0)))
        return token

    masked = _FRONTMATTER_RE.sub(_replace, text, count=1)
    masked = _CODE_FENCE_RE.sub(_replace, masked)
    masked = _INLINE_CODE_RE.sub(_replace, masked)
    masked = _WIKILINK_RE.sub(_replace, masked)
    return masked, placeholders


def _restore_protected_regions(text: str, placeholders: list[tuple[str, str]]) -> str:
    for token, original in placeholders:
        text = text.replace(token, original)
    return text


def run_phase2(config: Config):
    log.info("=== PHASE 2: Auto-generating wiki-links ===")

    output_md_files = [
        p for p in Path(config.output_vault).rglob("*.md")
        if not p.name.startswith(".") and p.name != config.tracker_filename
    ]

    # Build title list; ignore very short stems (too many false positives)
    titles = sorted(
        (f.stem for f in output_md_files if len(f.stem) > 3),
        key=len,
        reverse=True,  # Longest-first avoids partial replacement
    )
    log.info("Linking %d titles across %d files…", len(titles), len(output_md_files))

    for file_path in tqdm(output_md_files, desc="Linking notes"):
        try:
            original = file_path.read_text(encoding="utf-8")
            masked, placeholders = _mask_protected_regions(original)

            links_added = 0
            for title in titles:
                if title == file_path.stem:
                    continue  # Don't link a note to itself

                # Word-boundary match, case-insensitive, first occurrence only
                pattern = r"\b(" + re.escape(title) + r")\b"
                new_masked, count = re.subn(
                    pattern, r"[[\1]]", masked, count=1, flags=re.IGNORECASE
                )
                if count:
                    masked = new_masked
                    links_added += 1

            result = _restore_protected_regions(masked, placeholders)

            if result != original:
                file_path.write_text(result, encoding="utf-8")
                log.info("Added %d link(s) → %s", links_added, file_path.stem)

        except OSError as exc:
            log.error("Could not process %s: %s", file_path.stem, exc)

    log.info("Phase 2 complete.")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        log.critical(
            "GEMINI_API_KEY environment variable is not set.\n"
            "Set it with:  export GEMINI_API_KEY='your-key-here'\n"
            "Or place it in a .env file in the same directory as this script."
        )
        sys.exit(1)

    config = Config()
    client = genai.Client(api_key=api_key)

    run_phase1(client, config)
    run_phase2(config)

    log.info("All done!")


if __name__ == "__main__":
    main()
