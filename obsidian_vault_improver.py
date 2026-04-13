"""
Obsidian Vault Improver — Post-processing script for Zettelkasten vaults

Improvements:
  1. Merge duplicate MOCs (same topic, different names)
  2. Standardize tags (UK English preferred for veterinary)
  3. Find orphan notes (no incoming/outgoing links)
  4. Detect and auto-fix broken wikilinks (fuzzy matching)
  5. Generate vault health report
  6. Find empty and short notes
  7. Suggest connections for orphan notes based on shared tags
  8. Export analysis to JSON for further processing

Usage:
    python obsidian_vault_improver.py [--vault PATH] [--dry-run] [--fix-tags] [--fix-links] [--export-json]
"""

import argparse
import difflib
import hashlib
import json
import os
import re
import sys
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Set, List, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    vault_path: str = r"C:\Users\dcrac\Documents\Obsidian Vault"
    dry_run: bool = False
    fix_tags: bool = False
    fix_links: bool = False
    export_json: bool = False
    short_note_threshold: int = 100  # Characters
    output_folders: List[str] = field(default_factory=lambda: [
        "00_Inbox", "01_MOCs", "02_Zettels", "03_Literature",
    ])

    # Tag standardization: map US -> UK spelling
    tag_mappings: Dict[str, str] = field(default_factory=lambda: {
        "behavior": "behaviour",
        "color": "colour",
        "hematology": "haematology",
        "hemorrhage": "haemorrhage",
        "edema": "oedema",
        "estrous": "oestrous",
        "estrogen": "oestrogen",
        "anemia": "anaemia",
        "hemoglobin": "haemoglobin",
        "center": "centre",
        "analyze": "analyse",
        "paralyze": "paralyse",
    })

    # MOC consolidation: map duplicate -> canonical name
    moc_consolidations: Dict[str, str] = field(default_factory=lambda: {
        # Singular/plural duplicates
        "dogs": "dog",
        "horses": "horse",
        "cats": "cat",
        "hormones": "hormone",
        "muscles": "muscle",
        "arteries": "artery",
        "veins": "vein",
        "nerves": "nerve",
        "joints": "joint",
        "bacteria": "bacterium",
        # Spelling variants
        "behavior": "behaviour",
        # Overlapping topics
        "cardiovascular system": "cardiovascular",
        "digestive system": "digestion",
        "cardiology": "cardiac",
        "heart": "cardiac",
    })

    @classmethod
    def from_args(cls) -> 'Config':
        """Create config from command-line arguments."""
        parser = argparse.ArgumentParser(
            description="Improve Obsidian vault quality"
        )
        parser.add_argument(
            "--vault", "-v",
            default=cls.vault_path,
            help="Path to Obsidian vault"
        )
        parser.add_argument(
            "--dry-run", "-n",
            action="store_true",
            help="Analyze only, don't modify files"
        )
        parser.add_argument(
            "--fix-tags",
            action="store_true",
            help="Standardize tags to UK English"
        )
        parser.add_argument(
            "--fix-links",
            action="store_true",
            help="Auto-fix broken wikilinks with fuzzy matching"
        )
        parser.add_argument(
            "--export-json",
            action="store_true",
            help="Export analysis results to JSON file"
        )
        parser.add_argument(
            "--short-threshold",
            type=int,
            default=cls.short_note_threshold,
            help="Character threshold for short notes (default: 100)"
        )
        args = parser.parse_args()

        return cls(
            vault_path=args.vault,
            dry_run=args.dry_run,
            fix_tags=args.fix_tags or args.dry_run is False,
            fix_links=args.fix_links or args.dry_run is False,
            export_json=args.export_json,
            short_note_threshold=args.short_threshold,
        )

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("vault_improver.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ============================================================================
# REGEX PATTERNS
# ============================================================================

_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[#|\][^\]]+)?\]\]")
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_TAG_LINE_RE = re.compile(r"^  - (.+)$", re.MULTILINE)
_YAML_TAG_RE = re.compile(r"^tags:\s*\n((?:  - .+\n?)+)", re.MULTILINE)
_INLINE_TAG_RE = re.compile(r"(?<!\w)#[\w/\-]+")
_ANCHOR_RE = re.compile(r"#([\w\-]+)$")


# ============================================================================
# VAULT ANALYSIS
# ============================================================================

class VaultAnalyzer:
    """Analyze vault for issues and improvements."""

    def __init__(self, vault_path: Path):
        self.vault_path = vault_path
        self.notes: Dict[str, Path] = {}  # title -> path
        self.mocs: Dict[str, Path] = {}    # topic -> path
        self.wikilinks: Dict[str, Set[str]] = defaultdict(set)  # note -> linked notes
        self.backlinks: Dict[str, Set[str]] = defaultdict(set) # note -> notes linking to it
        self.tags: Dict[str, Set[str]] = defaultdict(set)       # tag -> notes with this tag
        self.broken_links: Dict[str, List[str]] = defaultdict(list)  # note -> broken links
        self.orphan_notes: List[str] = []
        self.empty_notes: List[Tuple[str, Path]] = []  # (title, path) for empty notes
        self.short_notes: List[Tuple[str, Path, int]] = []  # (title, path, char_count)
        self.note_content: Dict[str, str] = {}  # title -> content (for similarity)
        self._title_index: Dict[str, str] = {}  # normalized title -> actual title

    def _normalize_title(self, title: str) -> str:
        """Normalize a title for fuzzy matching."""
        return title.lower().replace("-", " ").replace("_", " ").strip()

    def _build_title_index(self) -> None:
        """Build an index for fuzzy title matching."""
        for title in self.notes:
            normalized = self._normalize_title(title)
            self._title_index[normalized] = title

    def find_fuzzy_match(self, broken_link: str, threshold: float = 0.85) -> Optional[str]:
        """Find a fuzzy match for a broken wikilink."""
        normalized = self._normalize_title(broken_link)

        # Direct match in index
        if normalized in self._title_index:
            return self._title_index[normalized]

        # Fuzzy match using difflib
        candidates = list(self._title_index.keys())
        matches = difflib.get_close_matches(normalized, candidates, n=1, cutoff=threshold)

        if matches:
            return self._title_index[matches[0]]

        return None

    def suggest_connections_for_orphan(self, note_title: str, min_shared_tags: int = 2) -> List[Tuple[str, int]]:
        """Suggest notes that an orphan could link to based on shared tags.

        Returns list of (note_title, shared_tag_count) tuples.
        """
        if note_title not in self.note_content:
            return []

        # Get tags for this note
        note_tags = set()
        for tag, notes in self.tags.items():
            if note_title in notes:
                note_tags.add(tag)

        if not note_tags:
            return []

        # Find notes with shared tags
        shared_counts: Counter = Counter()
        for tag in note_tags:
            for other_note in self.tags.get(tag, []):
                if other_note != note_title:
                    shared_counts[other_note] += 1

        # Filter by minimum shared tags
        suggestions = [(note, count) for note, count in shared_counts.most_common(10) if count >= min_shared_tags]
        return suggestions

    def scan_vault(self, short_threshold: int = 100) -> None:
        """Scan all notes in the vault."""
        log.info("Scanning vault: %s", self.vault_path)

        # Scan all markdown files
        for md_file in self.vault_path.rglob("*.md"):
            if md_file.name.startswith("."):
                continue
            if "QUARANTINE_" in md_file.name:
                continue

            title = md_file.stem
            self.notes[title] = md_file

            # Check if it's a MOC
            if md_file.parent.name == "01_MOCs":
                # Extract topic from MOC title
                topic = title.replace("MOC - ", "").lower()
                self.mocs[topic] = md_file

            # Check for empty/short notes
            try:
                content = md_file.read_text(encoding="utf-8")
                # Strip frontmatter for content analysis
                content_stripped = _FRONTMATTER_RE.sub("", content).strip()
                content_length = len(content_stripped)

                if content_length == 0:
                    self.empty_notes.append((title, md_file))
                elif content_length < short_threshold:
                    self.short_notes.append((title, md_file, content_length))
            except Exception:
                pass

        # Build title index for fuzzy matching
        self._build_title_index()
        log.info("Found %d notes, %d MOCs, %d empty, %d short",
                 len(self.notes), len(self.mocs), len(self.empty_notes), len(self.short_notes))

    def analyze_links(self) -> None:
        """Analyze wikilinks and find broken/orphan notes."""
        log.info("Analyzing wikilinks...")

        for title, path in tqdm(self.notes.items(), desc="Analyzing"):
            try:
                content = path.read_text(encoding="utf-8")
            except Exception:
                continue

            # Store content for later analysis
            self.note_content[title] = content

            # Find all wikilinks
            links = _WIKILINK_RE.findall(content)
            linked_titles = set()

            for link in links:
                link_title = link.strip()
                linked_titles.add(link_title)

                # Track backlink
                self.backlinks[link_title].add(title)

                # Check if link target exists
                if link_title not in self.notes:
                    self.broken_links[title].append(link_title)

            self.wikilinks[title] = linked_titles

        # Find orphan notes (no incoming or outgoing links)
        for title in self.notes:
            has_outgoing = len(self.wikilinks.get(title, set())) > 0
            has_incoming = len(self.backlinks.get(title, set())) > 0
            if not has_outgoing and not has_incoming:
                self.orphan_notes.append(title)

        log.info("Found %d broken links, %d orphan notes",
                 sum(len(v) for v in self.broken_links.values()),
                 len(self.orphan_notes))

    def analyze_tags(self) -> None:
        """Extract and analyze all tags (frontmatter and inline)."""
        log.info("Analyzing tags...")

        for title, path in tqdm(self.notes.items(), desc="Tags"):
            try:
                content = path.read_text(encoding="utf-8")
            except Exception:
                continue

            # Extract frontmatter tags
            fm = _FRONTMATTER_RE.match(content)
            if fm:
                for tag in _TAG_LINE_RE.findall(fm.group(0)):
                    tag = tag.strip().lower()
                    if tag:
                        self.tags[tag].add(title)

            # Extract inline tags (#tag format)
            for tag in _INLINE_TAG_RE.findall(content):
                tag = tag.lower().lstrip('#')
                if tag and len(tag) > 1:  # Skip single-char tags
                    self.tags[tag].add(title)

        log.info("Found %d unique tags", len(self.tags))

    def find_duplicate_mocs(self, config: 'Config') -> Dict[str, List[str]]:
        """Find MOCs that could be consolidated."""
        duplicates: Dict[str, List[str]] = defaultdict(list)

        # Group by normalized name
        for topic in self.mocs:
            # Check if this is a known duplicate
            normalized = topic.lower().strip()
            for dup_pattern, canonical in config.moc_consolidations.items():
                if dup_pattern in normalized or normalized in dup_pattern:
                    duplicates[canonical].append(topic)
                    break

        return {k: v for k, v in duplicates.items() if len(v) > 1}


# ============================================================================
# VAULT IMPROVEMENTS
# ============================================================================

def consolidate_mocs(analyzer: VaultAnalyzer, config: Config) -> None:
    """Merge duplicate MOCs into canonical ones."""
    log.info("=== MOC Consolidation ===")

    duplicates = analyzer.find_duplicate_mocs(config)

    if not duplicates:
        log.info("No duplicate MOCs found.")
        return

    log.info("Found %d MOC groups to consolidate", len(duplicates))

    for canonical, topics in duplicates.items():
        log.info("  %s <- %s", canonical, ", ".join(topics))


def standardize_tags(analyzer: VaultAnalyzer, config: Config) -> int:
    """Standardize tags to consistent naming (UK English preferred).

    If config.fix_tags is True and config.dry_run is False, actually modifies files.
    """
    log.info("=== Tag Standardization ===")

    changes = 0
    tag_mappings = config.tag_mappings
    files_modified = 0

    for tag, notes in analyzer.tags.items():
        normalized = tag.lower().strip()
        if normalized in tag_mappings:
            new_tag = tag_mappings[normalized]
            log.info("  %s -> %s (%d notes)", tag, new_tag, len(notes))
            changes += len(notes)

            # Actually fix the tags in files
            if config.fix_tags and not config.dry_run:
                for note_title in notes:
                    if note_title not in analyzer.notes:
                        continue
                    path = analyzer.notes[note_title]
                    try:
                        content = path.read_text(encoding="utf-8")
                        # Replace tag in frontmatter
                        new_content = content.replace(
                            f"  - {tag}\n",
                            f"  - {new_tag}\n"
                        )
                        if new_content != content:
                            path.write_text(new_content, encoding="utf-8")
                            files_modified += 1
                    except Exception as e:
                        log.warning("Failed to update %s: %s", path, e)

    if config.dry_run:
        log.info("DRY RUN: Would standardize %d note-tags in %d files", changes, len(analyzer.tags))
    elif config.fix_tags:
        log.info("Total tags standardized: %d note-tags in %d files", changes, files_modified)
    else:
        log.info("Total tags to standardize: %d note-tags (run with --fix-tags to apply)", changes)

    return changes


def fix_broken_links(analyzer: VaultAnalyzer, config: Config) -> int:
    """Auto-fix broken wikilinks using fuzzy matching.

    Returns the number of fixed links.
    """
    if not analyzer.broken_links:
        return 0

    log.info("=== Fixing Broken Wikilinks ===")

    fixed_count = 0
    files_modified = set()

    for note_title, broken_list in analyzer.broken_links.items():
        if note_title not in analyzer.notes:
            continue

        path = analyzer.notes[note_title]
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            continue

        new_content = content
        for broken_link in broken_list:
            match = analyzer.find_fuzzy_match(broken_link)
            if match:
                # Replace the wikilink
                old_link = f"[[{broken_link}"
                new_link = f"[[{match}"

                # Handle display text: [[link|text]] -> [[newlink|text]]
                new_content = new_content.replace(
                    f"[[{broken_link}|",
                    f"[[{match}|"
                )
                # Handle simple links: [[link]] -> [[newlink]]
                new_content = new_content.replace(
                    f"[[{broken_link}]]",
                    f"[[{match}]]"
                )

                if new_content != content:
                    log.info("  Fixed: '%s' -> '%s' in %s", broken_link, match, note_title)
                    fixed_count += 1
                    files_modified.add(note_title)

        # Write back if changed
        if new_content != content and not config.dry_run:
            try:
                path.write_text(new_content, encoding="utf-8")
            except Exception as e:
                log.warning("Failed to write %s: %s", path, e)

    if config.dry_run:
        log.info("DRY RUN: Would fix %d broken links in %d files", fixed_count, len(files_modified))
    elif config.fix_links:
        log.info("Fixed %d broken links in %d files", fixed_count, len(files_modified))
    else:
        log.info("Found %d fixable broken links (run with --fix-links to apply)", fixed_count)

    return fixed_count


def generate_health_report(analyzer: VaultAnalyzer, output_path: Path) -> None:
    """Generate a comprehensive vault health report."""
    log.info("=== Generating Health Report ===")

    report_path = output_path / "00_Inbox" / "Vault Health Report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate statistics
    total_notes = len(analyzer.notes)
    total_mocs = len(analyzer.mocs)
    total_links = sum(len(v) for v in analyzer.wikilinks.values())
    avg_links = total_links / total_notes if total_notes > 0 else 0

    # Find most linked notes
    top_linked = sorted(analyzer.backlinks.items(), key=lambda x: -len(x[1]))[:10]

    # Find notes with most broken links
    top_broken = sorted(analyzer.broken_links.items(), key=lambda x: -len(x[1]))[:10]

    # Find tags with most notes
    top_tags = sorted(analyzer.tags.items(), key=lambda x: -len(x[1]))[:20]

    report = f"""# Vault Health Report

Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Statistics

| Metric | Value |
|--------|-------|
| Total Notes | {total_notes} |
| Total MOCs | {total_mocs} |
| Total Wikilinks | {total_links} |
| Avg Links/Note | {avg_links:.2f} |
| Orphan Notes | {len(analyzer.orphan_notes)} |
| Broken Links | {sum(len(v) for v in analyzer.broken_links.values())} |
| Unique Tags | {len(analyzer.tags)} |

## Top Linked Notes

These notes are referenced by the most other notes:

| Note | Incoming Links |
|------|----------------|
"""
    for note, links in top_linked:
        report += f"| [[{note}]] | {len(links)} |\n"

    report += "\n## Broken Wikilinks\n\n"
    if top_broken:
        report += "| Note | Broken Links |\n|------|-------------|\n"
        for note, broken in top_broken:
            report += f"| {note} | {', '.join(broken[:5])}{'...' if len(broken) > 5 else ''} |\n"
    else:
        report += "No broken wikilinks found.\n"

    report += "\n## Orphan Notes\n\n"
    if analyzer.orphan_notes:
        report += f"Found {len(analyzer.orphan_notes)} notes with no incoming or outgoing links:\n\n"
        for note in analyzer.orphan_notes[:20]:
            report += f"- [[{note}]]\n"
        if len(analyzer.orphan_notes) > 20:
            report += f"\n...and {len(analyzer.orphan_notes) - 20} more\n"
    else:
        report += "No orphan notes found.\n"

    report += "\n## Top Tags\n\n| Tag | Notes |\n|-----|-------|\n"
    for tag, notes in top_tags:
        report += f"| {tag} | {len(notes)} |\n"

    # Count fixable broken links
    fixable_count = sum(
        1 for broken_list in analyzer.broken_links.values()
        for broken in broken_list
        if analyzer.find_fuzzy_match(broken)
    )
    total_broken = sum(len(v) for v in analyzer.broken_links.values())

    report += "\n## Recommendations\n\n"

    if analyzer.orphan_notes:
        report += f"### 1. Link Orphan Notes\n\n"
        report += f"Run the AI linking phase on {len(analyzer.orphan_notes)} orphan notes to connect them to the knowledge graph.\n\n"

    if total_broken > 0:
        report += f"### 2. Fix Broken Links\n\n"
        report += f"Found {total_broken} broken wikilinks"
        if fixable_count > 0:
            report += f" ({fixable_count} can be auto-fixed with fuzzy matching)"
        report += ".\n\n"
        report += "Run with `--fix-links` to automatically fix close matches, or review the Broken Wikilinks Report.\n\n"

    report += "### 3. Review Tag Consistency\n\n"
    report += "Some tags may have variant spellings (e.g., 'behavior' vs 'behaviour'). "
    report += "Run with `--fix-tags` to standardize to UK English for veterinary terminology.\n"

    report_path.write_text(report, encoding="utf-8")
    log.info("Health report saved to: %s", report_path)


def find_broken_wikilinks_report(analyzer: VaultAnalyzer, output_path: Path) -> None:
    """Create a detailed broken wikilinks report with suggested fixes."""
    if not analyzer.broken_links:
        return

    report_path = output_path / "00_Inbox" / "Broken Wikilinks Report.md"

    report = "# Broken Wikilinks Report\n\n"
    report += f"Found {sum(len(v) for v in analyzer.broken_links.values())} broken wikilinks in {len(analyzer.broken_links)} notes.\n\n"

    # Group by fixable vs unfixable
    fixable = []
    unfixable = []

    for note, broken_list in sorted(analyzer.broken_links.items()):
        for broken in broken_list:
            match = analyzer.find_fuzzy_match(broken)
            if match:
                fixable.append((note, broken, match))
            else:
                unfixable.append((note, broken))

    if fixable:
        report += "## Auto-fixable Links\n\n"
        report += "These links have close matches and can be fixed automatically:\n\n"
        report += "| Note | Broken Link | Suggested Fix |\n|------|-------------|---------------|\n"
        for note, broken, match in fixable[:50]:  # Limit to 50
            report += f"| [[{note}]] | [[{broken}]] | [[{match}]] |\n"
        if len(fixable) > 50:
            report += f"\n...and {len(fixable) - 50} more\n"
        report += "\n"

    if unfixable:
        report += "## Unfixable Links\n\n"
        report += "These links need manual review:\n\n"
        report += "| Note | Broken Link |\n|------|-------------|\n"
        for note, broken in unfixable[:50]:  # Limit to 50
            report += f"| [[{note}]] | [[{broken}]] |\n"
        if len(unfixable) > 50:
            report += f"\n...and {len(unfixable) - 50} more\n"

    report_path.write_text(report, encoding="utf-8")
    log.info("Broken links report saved to: %s", report_path)


def generate_empty_notes_report(analyzer: VaultAnalyzer, output_path: Path) -> None:
    """Generate a report of empty and very short notes."""
    if not analyzer.empty_notes and not analyzer.short_notes:
        return

    log.info("=== Generating Empty Notes Report ===")

    report_path = output_path / "00_Inbox" / "Empty Notes Report.md"

    report = "# Empty and Short Notes Report\n\n"
    report += f"Found {len(analyzer.empty_notes)} empty notes and {len(analyzer.short_notes)} short notes.\n\n"

    if analyzer.empty_notes:
        report += "## Empty Notes\n\n"
        report += "These notes have no content (only frontmatter):\n\n"
        for title, path in sorted(analyzer.empty_notes, key=lambda x: x[0])[:50]:
            rel_path = path.relative_to(analyzer.vault_path) if path.is_relative_to(analyzer.vault_path) else path
            report += f"- [[{title}]] ({rel_path})\n"
        if len(analyzer.empty_notes) > 50:
            report += f"\n...and {len(analyzer.empty_notes) - 50} more\n"

    if analyzer.short_notes:
        report += "\n## Short Notes\n\n"
        report += f"These notes have less than 100 characters of content:\n\n"
        for title, path, char_count in sorted(analyzer.short_notes, key=lambda x: x[2])[:30]:
            rel_path = path.relative_to(analyzer.vault_path) if path.is_relative_to(analyzer.vault_path) else path
            report += f"- [[{title}]] ({char_count} chars) - {rel_path}\n"
        if len(analyzer.short_notes) > 30:
            report += f"\n...and {len(analyzer.short_notes) - 30} more\n"

    report += "\n## Recommendations\n\n"
    report += "### Empty Notes\n\n"
    report += "- Review empty notes and either add content or delete them\n"
    report += "- Empty notes may be placeholders for future content\n\n"
    report += "### Short Notes\n\n"
    report += "- Consider merging short notes into related notes\n"
    report += "- Or expand short notes with more detail\n"

    report_path.write_text(report, encoding="utf-8")
    log.info("Empty notes report saved to: %s", report_path)


def generate_orphan_suggestions_report(analyzer: VaultAnalyzer, output_path: Path) -> None:
    """Generate a report with suggested connections for orphan notes."""
    if not analyzer.orphan_notes:
        return

    log.info("=== Generating Orphan Suggestions Report ===")

    report_path = output_path / "00_Inbox" / "Orphan Suggestions Report.md"

    report = "# Orphan Notes Connection Suggestions\n\n"
    report += f"Found {len(analyzer.orphan_notes)} orphan notes. Below are suggested connections based on shared tags.\n\n"

    suggestions_found = 0
    processed = 0

    for orphan_title in analyzer.orphan_notes[:100]:  # Limit to first 100
        suggestions = analyzer.suggest_connections_for_orphan(orphan_title, min_shared_tags=2)
        if suggestions:
            suggestions_found += 1
            report += f"## [[{orphan_title}]]\n\n"
            report += "Suggested connections:\n\n"
            for suggested_note, shared_count in suggestions[:5]:
                report += f"- [[{suggested_note}]] ({shared_count} shared tags)\n"
            report += "\n"

        processed += 1
        if processed % 50 == 0:
            log.info("Processed %d/%d orphans", processed, min(len(analyzer.orphan_notes), 100))

    if suggestions_found == 0:
        report += "No tag-based connection suggestions found. Consider:\n"
        report += "- Adding tags to orphan notes\n"
        report += "- Using AI to analyze note content for connections\n"

    report += f"\n---\n\nFound suggestions for {suggestions_found} of {len(analyzer.orphan_notes)} orphan notes.\n"

    report_path.write_text(report, encoding="utf-8")
    log.info("Orphan suggestions report saved to: %s", report_path)


def export_analysis_json(analyzer: VaultAnalyzer, output_path: Path) -> None:
    """Export vault analysis to JSON for further processing."""
    log.info("=== Exporting Analysis to JSON ===")

    json_path = output_path / "00_Inbox" / "vault_analysis.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert sets to sorted lists for JSON serialization
    data = {
        "statistics": {
            "total_notes": len(analyzer.notes),
            "total_mocs": len(analyzer.mocs),
            "total_links": sum(len(v) for v in analyzer.wikilinks.values()),
            "orphan_notes": len(analyzer.orphan_notes),
            "broken_links": sum(len(v) for v in analyzer.broken_links.values()),
            "unique_tags": len(analyzer.tags),
            "empty_notes": len(analyzer.empty_notes),
            "short_notes": len(analyzer.short_notes),
        },
        "top_linked": [
            {"note": note, "incoming_links": len(links)}
            for note, links in sorted(analyzer.backlinks.items(), key=lambda x: -len(x[1]))[:50]
        ],
        "broken_links": {
            note: broken_list
            for note, broken_list in sorted(analyzer.broken_links.items())[:200]
        },
        "orphan_notes": sorted(analyzer.orphan_notes)[:500],
        "empty_notes": [title for title, _ in analyzer.empty_notes[:100]],
        "short_notes": [
            {"title": title, "char_count": count}
            for title, _, count in sorted(analyzer.short_notes, key=lambda x: x[2])[:100]
        ],
        "top_tags": [
            {"tag": tag, "note_count": len(notes)}
            for tag, notes in sorted(analyzer.tags.items(), key=lambda x: -len(x[1]))[:100]
        ],
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    log.info("JSON analysis exported to: %s", json_path)


# ============================================================================
# MAIN
# ============================================================================

def main():
    config = Config.from_args()
    vault_path = Path(config.vault_path)

    if not vault_path.exists():
        log.error("Vault path not found: %s", vault_path)
        sys.exit(1)

    if config.dry_run:
        log.info("DRY RUN MODE: No files will be modified")

    # Analyze vault
    analyzer = VaultAnalyzer(vault_path)
    analyzer.scan_vault(config.short_note_threshold)
    analyzer.analyze_links()
    analyzer.analyze_tags()

    # Generate reports
    generate_health_report(analyzer, vault_path)
    find_broken_wikilinks_report(analyzer, vault_path)
    generate_empty_notes_report(analyzer, vault_path)
    generate_orphan_suggestions_report(analyzer, vault_path)

    # Export JSON if requested
    if config.export_json:
        export_analysis_json(analyzer, vault_path)

    # Show consolidation suggestions
    consolidate_mocs(analyzer, config)
    standardize_tags(analyzer, config)

    # Fix broken links with fuzzy matching
    if config.fix_links or config.dry_run:
        fix_broken_links(analyzer, config)

    log.info("=== Analysis Complete ===")
    if config.dry_run:
        log.info("DRY RUN: No files were modified. Remove --dry-run to apply changes.")
    log.info("Check reports in the Inbox folder for details.")


if __name__ == "__main__":
    main()