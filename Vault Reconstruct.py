"""
Obsidian Zettelkasten Converter
Splits lecture notes into atomic Zettelkasten notes and auto-generates wiki-links.

Usage:
    Set GEMINI_API_KEY as an environment variable (or place in a .env file),
    then run: python obsidian_zettelkasten.py
"""

import os
import time
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, field

import reconstruct_rust # New Rust module

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
from vault_reconstruct.paths import safe_filename # Keep for Phase 1
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

def run_phase2(config: Config):
    # This phase is now handled by the Rust module
    try:
        files_modified = reconstruct_rust.run_link_phase(config.output_vault)
        log.info("Phase 2 (Rust) complete. %d files modified.", files_modified)
    except Exception as exc:
        log.error("Phase 2 (Rust) failed: %s", exc)
        raise


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
