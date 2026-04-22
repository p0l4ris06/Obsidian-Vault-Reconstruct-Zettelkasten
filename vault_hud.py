"""
vault_hud.py

Inline, sleek CLI HUD for running vault scripts.
Replicates the Gemini CLI aesthetic with scrolling terminal output.
Categorized nested navigation for Vault Reconstructor.
"""

import os
import sys
import time
import json
import shlex
import asyncio
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Union

from rich.console import Console
# Branding: the logo is already 'VAULT RECONSTRUCTOR' from previous turn
from rich.text import Text
from rich.live import Live
from rich.prompt import Prompt
from rich.table import Table
from rich.style import Style

from vault_reconstruct.runner import (
    detect_python_launcher,
    build_python_command,
    popen_script,
    run_script_inprocess,
)
from vault_reconstruct.config import get_vault_paths
from vault_reconstruct.env import load_dotenv_no_override

# --- CONFIG & PATHS ---

def _detect_repo_root() -> Path:
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        if exe_dir.name.lower() == "dist" and (exe_dir.parent / "vault_cli.py").exists():
            return exe_dir.parent
        if (exe_dir.parent.parent / "vault_cli.py").exists():
            return exe_dir.parent.parent
        return exe_dir
    return Path(__file__).resolve().parent

REPO_ROOT = _detect_repo_root()
_HUD_SETTINGS_PATH = REPO_ROOT / ".vault_hud_settings.json"

def _load_hud_settings() -> dict:
    try:
        if not _HUD_SETTINGS_PATH.exists():
            return {}
        data = json.loads(_HUD_SETTINGS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _save_hud_settings(data: dict) -> None:
    try:
        _HUD_SETTINGS_PATH.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception:
        pass

@dataclass(frozen=True)
class Operation:
    id: str
    name: str
    description: str
    script: str
    icon: str = "▸"
    tags: list[str] = field(default_factory=list)
    args_hint: str = ""
    default_args: str = ""

    def script_path(self) -> Path:
        p = (REPO_ROOT / self.script).resolve()
        if not p.exists():
            raise FileNotFoundError(str(p))
        return p

@dataclass(frozen=True)
class Category:
    id: str
    name: str
    description: str
    items: list[Union['Operation', 'Category']]

# --- CATEGORIZED OPERATIONS ---

OPERATIONS_TREE: list[Union[Operation, Category]] = [
    Category(
        id="reconstruct",
        name="AI RECONSTRUCTION",
        description="Run the full vault pipeline — splitting, linking, and organising your notes with AI.",
        items=[
            Operation(
                id="full-recon",
                name="Full Pipeline",
                description="Runs everything start to finish: recovers broken notes, splits large ones into atomic zettels, adds wikilinks, builds MOC index pages. Use this for a full vault rebuild.",
                script="tools/reconstruct.py",
                icon="[P]",
                args_hint="Leave blank to run all phases, or add --phase 0-4 to run one step",
            ),
            Operation(
                id="rust-link",
                name="Quick Wikilinks Only",
                description="Just adds [[wikilinks]] between your notes — no AI, no splitting, very fast. Good for a quick refresh after you've added new notes manually.",
                script="tools/reconstruct.py",
                default_args="--phase 2",
                icon="[R]",
                args_hint="--phase 2  (runs the fast linker only)",
            ),
        ]
    ),
    Category(
        id="maintenance",
        name="VAULT MAINTENANCE",
        description="Fix broken links, clean up tags, and find notes that need attention.",
        items=[
            Operation(
                id="health-check",
                name="Health Report",
                description="Scans your vault and produces a report — shows broken links, orphaned notes (nothing links to them), and tag inconsistencies. Doesn't change anything, safe to run anytime.",
                script="tools/maintenance.py",
                icon="[H]",
                args_hint="No args needed — just press Enter to run",
            ),
            Operation(
                id="fix-auto",
                name="Auto-fix Tags & Links",
                description="Automatically corrects common tag formatting issues and repairs fuzzy wikilinks (e.g. where the note was renamed but links weren't updated). Makes changes in-place.",
                script="tools/maintenance.py",
                default_args="--fix-tags --fix-links",
                icon="[F]",
                args_hint="--fix-tags --fix-links  (both fixes on by default)",
            ),
            Operation(
                id="repair-q",
                name="Rescue Stuck Notes",
                description="Tries to recover notes that ended up in the QUARANTINE folder (notes the AI couldn't process cleanly). Attempts to parse and re-file them.",
                script="tools/maintenance.py",
                default_args="--repair",
                icon="[!]",
                args_hint="--repair  (attempts automatic rescue)",
            ),
        ]
    ),
    Category(
        id="utils",
        name="UTILITIES",
        description="Export to Anki, or check that your AI providers are set up correctly.",
        items=[
            Operation(
                id="anki-export",
                name="Export to Anki",
                description="Turns your zettels into Anki flashcard decks. Picks up Q&A-style notes automatically. Run this whenever you want to refresh your Anki collection from the vault.",
                script="tools/anki_exporter.py",
                icon="[A]",
                args_hint="Leave blank for defaults, or: --deck anatomy --out C:\\path\\to\\decks",
            ),
            Operation(
                id="doctor",
                name="Check Configuration",
                description="Tests that your AI keys and providers are working — checks Ollama, Gemini, and Azure connections without touching any vault files. Run this if something isn't working.",
                script="tools/doctor.py",
                default_args="--all",
                icon="[D]",
                args_hint="--all  (checks everything)",
            ),
        ]
    ),
    Category(
        id="autoresearch",
        name="AUTORESEARCH & RAG",
        description="Generate fact-grounded research notes using your local vault and academic sources.",
        items=[
            Operation(
                id="rag-sync",
                name="Sync Knowledge (RAG)",
                description="Crawls arXiv, PubMed, and Wikipedia for all tags current in your vault and builds a local factual index. Run this to 'seed' your research bank.",
                script="tools/research.py",
                default_args="--sync",
                icon="[S]",
                args_hint="--sync  (crawls external sources for all vault tags)",
            ),
            Operation(
                id="rag-research",
                name="Grounded Research Note",
                description="Generates a note between two topics using the local RAG index. Provides citations and grounded facts. Uses Qwen 2.5 3B.",
                script="tools/research.py",
                default_args='--rag --provider ollama',
                icon="[G]",
                args_hint='Type two topics, e.g: "synaptic plasticity" "cancer"',
            ),
            Operation(
                id="research-note",
                name="Fast Synthetic Note",
                description="Generates a connection note using only the model's internal weights (no external lookup). Fast but higher hallucination risk.",
                script="tools/research.py",
                icon="[~]",
                args_hint='Type two topic names, e.g: "Zettelkasten" "spaced repetition"',
            ),
        ]
    ),
]

# --- THEME & LOGO ---

VAULT_LOGO = """
[bold #7895f5]  ██╗   ██╗ █████╗ ██╗   ██╗██╗     ████████╗ [/]
[bold #8a8df0]  ██║   ██║██╔══██╗██║   ██║██║     ╚══██╔══╝ [/]
[bold #9d85eb]  ██║   ██║███████║██║   ██║██║        ██║    [/]
[bold #af7de6]  ╚██╗ ██╔╝██╔══██║██║   ██║██║        ██║    [/]
[bold #c275e1]  ╚████╔╝ ██║  ██║╚██████╔╝███████╗   ██║   [/]
[bold #af7de6]                                            [/]
[bold #af7de6] ██████╗ ███████╗ ██████╗  ██████╗ ███╗   ██╗███████╗████████╗██████╗ ██╗   ██╗ ██████╗ ████████╗ ██████╗ ██████╗  [/]
[bold #9d85eb] ██╔══██╗██╔════╝██╔════╝ ██╔═══██╗████╗  ██║██╔════╝╚══██╔══╝██╔══██╗██║   ██║██╔════╝ ╚══██╔══╝██╔═══██╗██╔══██╗ [/]
[bold #8a8df0] ██████╔╝█████╗  ██║      ██║   ██║██╔██╗ ██║███████╗   ██║   ██████╔╝██║   ██║██║         ██║   ██║   ██║██████╔╝ [/]
[bold #9d85eb] ██╔══██╗██╔══╝  ██║      ██║   ██║██║╚██╗██║╚════██║   ██║   ██╔══██╗██║   ██║██║         ██║   ██║   ██║██╔══██╗ [/]
[bold #af7de6] ██║  ██║███████╗╚██████╗ ╚██████╔╝██║ ╚████║███████║   ██║   ██║  ██║╚██████╔╝╚██████╗    ██║   ╚██████╔╝██║  ██║ [/]
"""

class VaultReconstructorCLI:
    def __init__(self):
        self.console = Console()
        load_dotenv_no_override(repo_root=REPO_ROOT)
        self.settings = _load_hud_settings()
        self.title = os.environ.get("VAULT_HUD_TITLE", "Vault Reconstructor").upper()
        self.history: list[list[Union[Operation, Category]]] = [OPERATIONS_TREE]

    @property
    def current_menu(self) -> list[Union[Operation, Category]]:
        return self.history[-1]

    def print_header(self):
        self.console.print(VAULT_LOGO)
        
        # Sleek gradient title bar
        title_text = Text()
        title_text.append(f" {self.title} ", style="bold #0d0d0d on #7895f5")
        title_text.append(f" {datetime.now().year} EDITION ", style="dim #7895f5")
        self.console.print(title_text)
        
        self.console.print("\n[bold #c8c8c8]Getting Started[/]")
        self.console.print(" [dim]•[/] [#a8b8ff]/help[/] for command reference")
        self.console.print(" [dim]•[/] [#a8b8ff]/env[/]  diagnose vault configuration")
        self.console.print(" [dim]•[/] Select a category or operation below.\n")

    def get_footer_info(self):
        paths = get_vault_paths()
        provider = (os.environ.get("VAULT_LLM_PROVIDER", "ollama") or "ollama").strip().lower()
        model = (os.environ.get("VAULT_OLLAMA_MODEL", "llama3") or "").strip()
        if provider == "gemini":
            model = (os.environ.get("VAULT_GEMINI_MODEL", "gemini-2.5-flash") or "").strip()
        
        info = Text()
        info.append("── ", style="dim")
        info.append(f"vault:recon", style="#888888")
        info.append(" | ", style="dim")
        info.append(f"llm:{provider}:{model}", style="#7895f5")
        info.append(" | ", style="dim")
        info.append(f"{datetime.now().strftime('%H:%M:%S')}", style="dim")
        return info

    def show_menu(self):
        # If we are deep in nested menu, show breadcrumbs or just indent
        if len(self.history) > 1:
            self.console.print(f"[bold #7895f5]SUBMENU > [/][dim #a8b8ff]Navigate back with 'b'[/]\n")

        for i, item in enumerate(self.current_menu):
            line = Text()
            line.append(f" {i+1:02d} ", style="bold #a8b8ff")
            if isinstance(item, Category):
                line.append(f" {item.name:<25}", style="bold #c275e1")
                line.append(f" folder view (contains {len(item.items)} items)", style="dim")
            else:
                line.append(f" {item.name:<25}", style="#c8c8c8")
                line.append(f" {item.description.splitlines()[0]}", style="dim")
            self.console.print(line)
        
        if len(self.history) > 1:
            self.console.print(f" [bold #cc5577]b[/]  Go Back")

        self.console.print("")

    async def run_operation(self, op: Operation, args_str: str):
        try:
            args = shlex.split(args_str, posix=False) if args_str else []
        except ValueError as exc:
            self.console.print(f"[bold #cc5577]Error parsing arguments:[/] {exc}")
            return

        script_path = op.script_path()
        self.console.print(f"\n[bold #7895f5]━━━ TRIGGERING PIPELINE: {op.name} ━━━[/]")

        # For the offline autoresearch pipeline, force the local model provider
        extra_env = {}
        if op.id == "recon-autoresearch":
            extra_env["VAULT_LLM_PROVIDER"] = "autoresearch"
            self.console.print("[dim]  Using local autoresearch model (offline mode)[/]")
        
        run_env = {**os.environ, **extra_env}

        with Live(Text("⠋ Just a moment, I'm tuning the algorithms...", style="#af7de6"), refresh_per_second=12, transient=True) as live:
            if getattr(sys, "frozen", False):
                cmd = [sys.executable, "--_run-script", str(script_path), "--", *args]
                proc = subprocess.Popen(
                    cmd, cwd=str(REPO_ROOT), env=run_env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
                )
            else:
                # Temporarily apply extra env vars since popen_script inherits os.environ
                _old = {k: os.environ.get(k) for k in extra_env}
                os.environ.update(extra_env)
                try:
                    proc = popen_script(repo_root=REPO_ROOT, script=script_path, passthrough=args)
                finally:
                    for k, v in _old.items():
                        if v is None:
                            os.environ.pop(k, None)
                        else:
                            os.environ[k] = v

            while True:
                line = await asyncio.to_thread(proc.stdout.readline)
                if not line and proc.poll() is not None:
                    break
                if line:
                    self.console.print(f"  [dim]│[/] {line.strip()}", style="#888888")
            
            code = proc.wait()
            self.console.print(f"  [bold #a8b8ff]✓[/] Sequence ended (code {code})")

    async def main_loop(self):
        self.print_header()
        while True:
            self.show_menu()
            
            choices = [str(i+1) for i in range(len(self.current_menu))] + ["q"]
            if len(self.history) > 1:
                choices.append("b")

            choice = Prompt.ask("[bold #7895f5]>[/] Select action", choices=choices)
            
            if choice == "q":
                break
            if choice == "b":
                self.history.pop()
                self.console.print("\n" + "─" * 40 + "\n")
                continue
            
            item = self.current_menu[int(choice)-1]
            
            if isinstance(item, Category):
                self.history.append(item.items)
                self.console.print(f"\n[bold #c275e1]📂 {item.name}[/]")
                self.console.print(f"[dim]{item.description}[/]\n")
                continue
            
            # It's an operation
            op = item
            self.console.print(f"\n[dim]{op.description}[/]\n")
            if op.args_hint:
                self.console.print(f"[dim]  hint: {op.args_hint}[/]")
            args = Prompt.ask(f"[bold #7895f5]>[/] Args", default=op.default_args)
            
            # Save settings
            self.settings["selected_op_id"] = op.id
            if "args_by_op" not in self.settings: self.settings["args_by_op"] = {}
            self.settings["args_by_op"][op.id] = args
            _save_hud_settings(self.settings)

            await self.run_operation(op, args)
            self.console.print(self.get_footer_info())
            self.console.print("\n" + "─" * 40 + "\n")

def _ensure_utf8() -> None:
    if sys.platform != "win32": return
    try:
        import ctypes
        import ctypes.wintypes
        kernel32 = ctypes.windll.kernel32
        if kernel32.GetConsoleCP() not in (65001, 0):
            kernel32.SetConsoleCP(65001)
            kernel32.SetConsoleOutputCP(65001)
        # Enable VT100
        h = kernel32.GetStdHandle(-11)
        mode = ctypes.wintypes.DWORD(0)
        if kernel32.GetConsoleMode(h, ctypes.byref(mode)):
            kernel32.SetConsoleMode(h, mode.value | 0x0004 | 0x0001)
    except Exception: pass
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception: pass

def main_entry():
    """Entry point for vault-recon."""
    _ensure_utf8()
    cli = VaultReconstructorCLI()
    try:
        asyncio.run(cli.main_loop())
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Print basic diagnostics and exit.",
    )
    parser.add_argument(
        "--_run-script",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    ns, rest = parser.parse_known_args()

    if ns._run_script:
        script = Path(ns._run_script).resolve()
        passthrough = rest
        if passthrough[:1] == ["--"]:
            passthrough = passthrough[1:]
        code = run_script_inprocess(
            repo_root=REPO_ROOT,
            script=script,
            passthrough=passthrough,
            on_line=print,
        )
        raise SystemExit(code)

    if ns.self_test:
        print("VaultReconstructor HUD self-test")
        print(f"- repo_root: {REPO_ROOT}")
        print(f"- python: {sys.executable}")
        raise SystemExit(0)

    main_entry()

