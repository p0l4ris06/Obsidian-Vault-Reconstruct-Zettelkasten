"""
Fix QUARANTINE and ERROR notes in the Obsidian vault.

For QUARANTINE files: Remove the header comment and rename
For ERROR files: Parse the JSON and extract individual zettel notes
"""

import argparse
import os
import json
import re
import shutil
from pathlib import Path
from datetime import datetime
from vault_reconstruct.config import DEFAULT_VAULT_PATH, get_vault_paths
from vault_reconstruct.env import load_dotenv_no_override

load_dotenv_no_override()


def _default_vault_path() -> Path:
    return get_vault_paths().output_vault if "VAULT_PATH" in os.environ or "VAULT_OUTPUT_PATH" in os.environ else DEFAULT_VAULT_PATH


def fix_quarantine_files(inbox_path: Path, zettels_path: Path):
    """Fix QUARANTINE_*.md files - remove header and rename."""
    quarantine_files = list(inbox_path.glob("QUARANTINE_*.md"))
    print(f"Found {len(quarantine_files)} QUARANTINE files")

    fixed_count = 0
    for qfile in quarantine_files:
        try:
            content = qfile.read_text(encoding="utf-8")

            # Remove the HTML comment at the top
            content = re.sub(r'^<!-- Failed to parse AI response for: .+? -->\s*\n*', '', content)
            content = content.strip()

            # Determine new filename (remove QUARANTINE_ prefix)
            new_name = qfile.stem.replace("QUARANTINE_", "")
            new_path = zettels_path / f"{new_name}.md"

            # Check if target exists
            if new_path.exists():
                existing = new_path.read_text(encoding="utf-8")
                if len(content) > len(existing):
                    new_path.write_text(content, encoding="utf-8")
                    print(f"  Updated: {new_name}")
                else:
                    print(f"  Skipped (exists): {new_name}")
            else:
                new_path.write_text(content, encoding="utf-8")
                print(f"  Created: {new_name}")

            # Remove original file
            qfile.unlink()
            fixed_count += 1

        except Exception as e:
            print(f"  Error processing {qfile.name}: {e}")

    print(f"Fixed {fixed_count} QUARANTINE files")
    return fixed_count


def extract_notes_from_json_like(json_str: str) -> list:
    """Extract notes from potentially malformed JSON using regex."""
    notes = []

    # Try to find individual note objects
    # Pattern matches {"title": "...", "content": "..."} blocks
    note_pattern = re.compile(
        r'\{\s*"title"\s*:\s*"([^"]+)"\s*,\s*"content"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}',
        re.DOTALL
    )

    for match in note_pattern.finditer(json_str):
        title = match.group(1)
        content = match.group(2)

        # Unescape JSON strings
        content = content.replace('\\n', '\n')
        content = content.replace('\\"', '"')
        content = content.replace('\\\\', '\\')

        notes.append({"title": title, "content": content})

    return notes


def fix_error_files(inbox_path: Path, zettels_path: Path):
    """Fix ERROR_*.md files - parse JSON and extract individual notes."""
    error_files = list(inbox_path.glob("ERROR_*.md"))
    print(f"Found {len(error_files)} ERROR files")

    total_notes = 0
    for efile in error_files:
        try:
            content = efile.read_text(encoding="utf-8")

            # Find JSON content (between ```json and ```)
            json_match = re.search(r'```json\s*\n(.*?)\s*\n```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find array directly
                json_match = re.search(r'(\[.*\])\s*$', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = content

            # First try standard JSON parse
            notes = []
            try:
                parsed = json.loads(json_str)
                notes = parsed if isinstance(parsed, list) else [parsed]
            except json.JSONDecodeError:
                # Fall back to regex extraction
                notes = extract_notes_from_json_like(json_str)

            print(f"\n  {efile.name}: {len(notes)} notes extracted")

            for note in notes:
                title = note.get("title", "Untitled")
                note_content = note.get("content", "")

                if not note_content:
                    continue

                # Clean title for filename
                safe_title = re.sub(r'[<>:"/\\|?*]', '', title)
                safe_title = safe_title.strip()[:200]  # Limit length

                # Add frontmatter
                today = datetime.now().strftime("%Y-%m-%d")
                full_content = f"""---
tags:
  - zettel
created: {today}
---

{note_content}
"""
                new_path = zettels_path / f"{safe_title}.md"

                # Check if file exists
                if new_path.exists():
                    existing = new_path.read_text(encoding="utf-8")
                    if len(full_content) > len(existing):
                        new_path.write_text(full_content, encoding="utf-8")
                        print(f"    Updated: {safe_title[:60]}")
                else:
                    new_path.write_text(full_content, encoding="utf-8")
                    print(f"    Created: {safe_title[:60]}")
                total_notes += 1

            # Remove original error file
            efile.unlink()
            print(f"  Removed: {efile.name}")

        except Exception as e:
            print(f"  Error processing {efile.name}: {e}")

    print(f"\nExtracted {total_notes} notes from ERROR files")
    return total_notes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", default=str(_default_vault_path()), help="Path to Obsidian vault")
    args = parser.parse_args()

    vault_path = Path(args.vault)
    inbox_path = vault_path / "00_Inbox"
    zettels_path = vault_path / "02_Zettels"
    zettels_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Fixing QUARANTINE and ERROR notes")
    print("=" * 60)

    q_fixed = fix_quarantine_files(inbox_path, zettels_path)
    e_fixed = fix_error_files(inbox_path, zettels_path)

    print("\n" + "=" * 60)
    print(f"Summary: Fixed {q_fixed} QUARANTINE files, extracted {e_fixed} notes from ERROR files")
    print("=" * 60)


if __name__ == "__main__":
    main()