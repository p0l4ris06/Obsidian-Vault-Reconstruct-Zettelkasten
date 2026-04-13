"""
Regex-Only Vault Linker
Adds [[wikilinks]] to your Obsidian vault using exact title matching.
No Ollama required — runs entirely offline and quickly.

Uses the same tracker as obsidian_zettelkasten_v2.py so results are
compatible. Notes marked done here won't be re-processed by the main script.

Usage:
    python regex_link_only.py --vault "D:\Obsidian Vault"
"""

import argparse
import json
import re
import logging
import sys
from pathlib import Path
from dataclasses import dataclass
from vault_reconstruct.config import get_vault_paths
from vault_reconstruct.env import load_dotenv_no_override

load_dotenv_no_override()

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ============================================================================
# CONFIGURATION — must match your main script paths
# ============================================================================

@dataclass
class Config:
    output_vault: str = str(get_vault_paths().output_vault)
    tracker_filename: str = ".zettelkasten_tracker.json"
    # Only process notes with fewer than this many existing wikilinks
    # Set to 999 to process every note regardless
    max_existing_links: int = 999


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
# TRACKER
# ============================================================================

class ProcessingTracker:
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
                log.info("Tracker loaded — %d total phase completions.", total)
            except (json.JSONDecodeError, OSError):
                log.warning("Could not read tracker; starting fresh.")

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(
                {k: sorted(v) for k, v in self._data.items()},
                f, indent=2,
            )

    def is_done(self, phase: str, key: str) -> bool:
        return key in self._data.get(phase, set())

    def mark_done(self, phase: str, key: str):
        self._data.setdefault(phase, set()).add(key)
        self._save()


from vault_reconstruct.text_protect import count_wikilinks, mask_protected, restore_protected


# ============================================================================
# REGEX LINK PASS
# ============================================================================

def regex_link_pass(file_path: Path, titles: list[str]) -> int:
    """
    Scan the note for any title that appears verbatim as a plain word
    and wrap the first occurrence in [[wikilink]] brackets.
    Returns the number of new links added.
    """
    original = file_path.read_text(encoding="utf-8")
    masked, placeholders = mask_protected(original)
    links_added = 0

    for title in titles:
        if title == file_path.stem:
            continue
        # Negative lookbehind/ahead to avoid double-bracketing already linked text
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


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Regex-only wikilinker (offline).")
    parser.add_argument(
        "--vault",
        type=str,
        default=None,
        help="Path to Obsidian vault (defaults to VAULT_OUTPUT_PATH/VAULT_PATH or repo default).",
    )
    parser.add_argument(
        "--max-existing-links",
        type=int,
        default=Config().max_existing_links,
        help="Skip notes that already have >= this many wikilinks (default: 999).",
    )
    ns = parser.parse_args()

    config = Config()
    output_path = Path(ns.vault).expanduser() if ns.vault else Path(config.output_vault)
    config.max_existing_links = int(ns.max_existing_links)
    tracker     = ProcessingTracker(output_path / config.tracker_filename)

    # Collect all vault notes
    all_notes_raw = [
        p for p in output_path.rglob("*.md")
        if not p.name.startswith(".")
        and "QUARANTINE_" not in p.name
        and p.name != config.tracker_filename
    ]

    # Build full title list (including already-done notes, needed as link targets)
    all_titles = sorted(
        [f.stem for f in all_notes_raw if len(f.stem) > 3],
        key=len, reverse=True,  # Longest first avoids partial matches
    )

    # Filter to only unprocessed notes
    pending = [
        p for p in all_notes_raw
        if not tracker.is_done("phase3", p.stem)
    ]

    log.info(
        "%d notes total | %d titles available | %d remaining to link",
        len(all_notes_raw), len(all_titles), len(pending),
    )

    if not pending:
        log.info("All notes already processed. Nothing to do.")
        return

    total_links = 0
    skipped     = 0

    for file_path in tqdm(pending, desc="Linking"):
        text = file_path.read_text(encoding="utf-8")

        # Skip notes that already have enough links
        if count_wikilinks(text) >= config.max_existing_links:
            tracker.mark_done("phase3", file_path.stem)
            skipped += 1
            continue

        added = regex_link_pass(file_path, all_titles)
        total_links += added
        tracker.mark_done("phase3", file_path.stem)

    log.info(
        "Done. %d links added across %d notes. %d notes skipped (already linked).",
        total_links, len(pending) - skipped, skipped,
    )


if __name__ == "__main__":
    main()
