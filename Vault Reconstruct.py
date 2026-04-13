"""
Obsidian Zettelkasten Converter
Splits lecture notes into atomic Zettelkasten notes and auto-generates wiki-links.

Usage:
    Set GEMINI_API_KEY as an environment variable (or place in a .env file),
    then run: python obsidian_zettelkasten.py
"""

import os
import time
import re
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, field

from vault_reconstruct.env import load_dotenv_no_override
load_dotenv_no_override()

# ── Optional dependency: tqdm ────────────────────────────────────────────────
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):  # Graceful fallback
        return iterable

from vault_reconstruct.json_extract import extract_json_array
from vault_reconstruct.llm import LlmConfig, generate_text_with_retries, make_backend
from vault_reconstruct.paths import safe_filename
from vault_reconstruct.text_protect import mask_protected, restore_protected
from vault_reconstruct.config import get_vault_paths

# ============================================================================
# CONFIGURATION — edit these two paths; keep the API key in your environment
# ============================================================================

@dataclass
class Config:
    input_vault: str = str(get_vault_paths().input_vault)
    output_vault: str = str(get_vault_paths().output_vault)
    provider:     str = os.environ.get("VAULT_LLM_PROVIDER", "gemini").strip().lower()
    model:        str = os.environ.get("VAULT_GEMINI_MODEL", "gemini-2.5-flash")
    azure_model:  str = os.environ.get("VAULT_AZURE_MODEL", "gpt-4.1-mini")
    ollama_model: str = os.environ.get("VAULT_OLLAMA_MODEL", "qwen2.5-coder:0.5b-base-q8_0")
    min_content_length: int = 50
    request_delay:      float = 1.0   # seconds between successful calls (extra quota safety)
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


def _make_backend(config: Config):
    if config.provider == "gemini":
        cfg = LlmConfig(provider="gemini", model=config.model, max_retries=config.max_retries)
    elif config.provider == "azure":
        cfg = LlmConfig(provider="azure", model=config.azure_model, max_retries=config.max_retries)
    elif config.provider == "ollama":
        cfg = LlmConfig(provider="ollama", model=config.ollama_model, max_retries=config.max_retries)
    else:
        raise SystemExit(f"Unknown VAULT_LLM_PROVIDER: {config.provider!r} (expected gemini/azure/ollama)")
    return make_backend(cfg)


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
    Call the configured LLM backend with retry/backoff.
    Returns a list of note dicts on success, or None if all retries failed.
    """
    prompt = SPLIT_PROMPT.format(content=content)
    try:
        text = generate_text_with_retries(client, prompt=prompt, max_retries=max(1, int(config.max_retries)))
    except Exception as exc:
        log.error("[%s] LLM error: %s — skipping.", filename, exc)
        return None

    notes = extract_json_array(text or "")
    if notes is not None:
        return notes
    log.warning("[%s] Response was not valid JSON — quarantining.", filename)
    return None


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
    return mask_protected(text)


def _restore_protected_regions(text: str, placeholders: list[tuple[str, str]]) -> str:
    return restore_protected(text, placeholders)


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
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(
            "Vault Reconstruct (Gemini/Azure/Ollama splitter + regex linker)\n\n"
            "Environment:\n"
            "  VAULT_INPUT_PATH   (input vault path)\n"
            "  VAULT_OUTPUT_PATH  (output vault path)\n"
            "  VAULT_LLM_PROVIDER gemini|azure|ollama (default: gemini)\n"
            "  GEMINI_API_KEY     (if provider=gemini)\n"
            "  AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY (if provider=azure)\n"
            "  VAULT_GEMINI_MODEL / VAULT_AZURE_MODEL / VAULT_OLLAMA_MODEL (optional)\n"
        )
        raise SystemExit(0)
    config = Config()
    try:
        client = _make_backend(config)
    except RuntimeError as exc:
        log.critical(str(exc))
        log.critical(
            "Set one of:\n"
            "  - VAULT_LLM_PROVIDER=gemini and GEMINI_API_KEY\n"
            "  - VAULT_LLM_PROVIDER=azure and AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY\n"
            "  - VAULT_LLM_PROVIDER=ollama and run `ollama serve`\n"
        )
        sys.exit(1)

    run_phase1(client, config)
    run_phase2(config)

    log.info("All done!")


if __name__ == "__main__":
    main()
