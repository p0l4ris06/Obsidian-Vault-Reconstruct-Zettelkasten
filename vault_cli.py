"""
vault_cli.py

Single entrypoint for the (previously separate) vault scripts in this repo.
This keeps the repo compact without having to constantly remember filenames.

Examples:
  python vault_cli.py convert-ollama
  python vault_cli.py convert-gemini
  python vault_cli.py architecture
  python vault_cli.py improve -- --vault "D:\\Obsidian Vault" --dry-run
  python vault_cli.py tag-consolidate -- --dry-run --min 5

Notes:
  - Everything after `--` is passed through verbatim to the underlying script.
  - This wrapper intentionally uses subprocess so the existing scripts can stay
    largely unchanged.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from vault_reconstruct.env import load_dotenv_no_override
from vault_reconstruct.runner import build_python_command, run_script_inprocess


def _detect_repo_root() -> Path:
    """
    When packaged with PyInstaller --onefile, __file__ points into a temp dir.
    We want the folder containing the scripts (usually the repo root).
    """
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        # Common layout (onefile): repo_root/dist/VaultCLI.exe (or VaultHUD.exe)
        if exe_dir.name.lower() == "dist" and (exe_dir.parent / "vault_cli.py").exists():
            return exe_dir.parent

        # Common layout (onedir): repo_root/dist/<appname>/VaultHUD_onedir.exe
        # Here exe_dir is repo_root/dist/<appname>
        if (exe_dir.parent.parent / "vault_cli.py").exists():
            return exe_dir.parent.parent

        return exe_dir
    return Path(__file__).resolve().parent


REPO_ROOT = _detect_repo_root()

load_dotenv_no_override(repo_root=REPO_ROOT)


def _script(path: str) -> Path:
    p = (REPO_ROOT / path).resolve()
    if not p.exists():
        raise SystemExit(f"Missing script: {p}")
    return p


def _run(script_path: Path, passthrough: list[str]) -> int:
    if getattr(sys, "frozen", False):
        return run_script_inprocess(
            repo_root=REPO_ROOT,
            script=script_path,
            passthrough=passthrough,
            on_line=print,
        )
    cmd = build_python_command(script_path, passthrough)
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        prog="vault_cli.py",
        description="Run vault reconstruction / improvement scripts via one CLI.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_cmd(name: str, help_text: str):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument(
            "passthrough",
            nargs=argparse.REMAINDER,
            help="Args after `--` are passed to underlying script",
        )

    add_cmd("convert-gemini", "Split + link using Gemini (Vault Reconstruct.py)")
    add_cmd("convert-ollama", "Split + tag + frontmatter + link + MOCs (Vault Reconstruct Ollama.py)")
    add_cmd("architecture", "Threaded linking + tag consolidation + MOCs (obsidian_zettelkasten_v3_threaded.py)")
    add_cmd("regex-link", "Regex-only wikilinking (regex_link_only.py)")
    add_cmd("improve", "Vault health + tag/link fixes + reports (obsidian_vault_improver.py)")
    add_cmd("tag-consolidate", "Standalone tag consolidation (tag_consolidator.py)")
    add_cmd("fix-quarantine", "Fix QUARANTINE_/ERROR_ notes (fix_quarantined_notes.py)")
    add_cmd("expand-short", "Expand very short notes using templates (expand_short_notes.py)")
    add_cmd("anki-export", "Generate Anki decks from zettels (anki_exporter.py)")
    add_cmd("doctor", "Dry-run checks: LLM + env + paths (vault_doctor.py)")

    args = parser.parse_args(argv)

    # Allow `--` separator but don't require it.
    passthrough = list(args.passthrough)
    if passthrough[:1] == ["--"]:
        passthrough = passthrough[1:]

    return args, passthrough


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv
    if getattr(sys, "frozen", False) and len(raw_argv) >= 2 and raw_argv[0] == "--_run-script":
        # Internal execution mode (used by frozen parent process).
        script = Path(raw_argv[1]).resolve()
        passthrough = raw_argv[2:]
        if passthrough[:1] == ["--"]:
            passthrough = passthrough[1:]
        return run_script_inprocess(repo_root=REPO_ROOT, script=script, passthrough=passthrough, on_line=print)

    args, passthrough = _parse_args(raw_argv)

    mapping: dict[str, str] = {
        "convert-gemini": "Vault Reconstruct.py",
        "convert-ollama": "Vault Reconstruct Ollama.py",
        "architecture": "obsidian_zettelkasten_v3_threaded.py",
        "regex-link": "regex_link_only.py",
        "improve": "obsidian_vault_improver.py",
        "tag-consolidate": "tag_consolidator.py",
        "fix-quarantine": "fix_quarantined_notes.py",
        "expand-short": "expand_short_notes.py",
        "anki-export": "anki_exporter.py",
        "doctor": "vault_doctor.py",
    }

    script_path = _script(mapping[args.cmd])
    return _run(script_path, passthrough)


if __name__ == "__main__":
    raise SystemExit(main())

