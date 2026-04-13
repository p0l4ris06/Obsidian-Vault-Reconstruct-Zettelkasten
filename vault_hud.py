"""
vault_hud.py

Interactive HUD/TUI for running this repo's vault scripts on Windows
from PowerShell/CMD.

Run:
  python vault_hud.py
  python vault_hud.py --self-test

Keys:
  - Up/Down: select action
  - Enter: run action
  - Ctrl+C: quit (safe)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from vault_reconstruct.runner import (
    detect_python_launcher,
    build_python_command,
    popen_script,
    run_script_inprocess,
)
from vault_reconstruct.config import get_vault_paths
from vault_reconstruct.env import load_dotenv_no_override

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    ProgressBar,
    RichLog,
    Static,
)


def _detect_repo_root() -> Path:
    """
    When packaged with PyInstaller --onefile, __file__ points into a temp dir.
    We want the folder containing the scripts (usually the repo root).
    """
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        # Common layout (onefile): repo_root/dist/VaultHUD.exe
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
        return


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


OPERATIONS: list[Operation] = [
    Operation(
        id="convert-ollama",
        name="Convert (Ollama)",
        description="Split + tags + frontmatter + linking + MOCs.\nRequires Ollama running. Uses OLLAMA_API_KEY for cloud-first if set.",
        script="Vault Reconstruct Ollama.py",
        icon="[O]",
        tags=["AI", "OLLAMA"],
        args_hint="(optional) extra args passed to script",
    ),
    Operation(
        id="convert-gemini",
        name="Convert (Gemini)",
        description="Split + regex linking via Gemini API.\nRequires GEMINI_API_KEY env var or .env.",
        script="Vault Reconstruct.py",
        icon="[G]",
        tags=["AI", "CLOUD"],
        args_hint="(optional) extra args passed to script",
    ),
    Operation(
        id="architecture",
        name="Architecture (threaded)",
        description="AI linking + tag consolidation + MOCs.\nThreaded architecture pass with parallel processing.",
        script="obsidian_zettelkasten_v3_threaded.py",
        icon="[T]",
        tags=["AI", "THREADED"],
        args_hint="(optional) extra args passed to script",
    ),
    Operation(
        id="improve",
        name="Improve vault",
        description="Health reports + broken link fixes + tag cleanup.\nSafe start: run with --dry-run and point --vault if needed.",
        script="obsidian_vault_improver.py",
        icon="[+]",
        tags=["HEALTH"],
        default_args="--dry-run",
        args_hint='--vault "C:\\Users\\Wren C\\Documents\\Coding stuff\\Obsidian Vault" --dry-run',
    ),
    Operation(
        id="regex-link",
        name="Regex-only wikilinks",
        description="Add [[wikilinks]] fast using regex. No API required.\nFully offline, fast.",
        script="regex_link_only.py",
        icon="[R]",
        tags=["OFFLINE", "FAST"],
        args_hint="(optional) extra args passed to script",
    ),
    Operation(
        id="tag-consolidate",
        name="Tag consolidation",
        description="Standalone tag merge + dedup + normalize.\nStarts in preview mode by default.",
        script="tag_consolidator.py",
        icon="[T]",
        tags=["TAGS"],
        default_args="--dry-run",
        args_hint="--dry-run --min 5",
    ),
    Operation(
        id="fix-quarantine",
        name="Fix QUARANTINE/ERROR",
        description="Process notes flagged QUARANTINE_ or ERROR_.\nAttempts automated repair and cleanup.",
        script="fix_quarantined_notes.py",
        icon="[!]",
        tags=["REPAIR"],
        args_hint='--vault "C:\\Users\\Wren C\\Documents\\Coding stuff\\Obsidian Vault"',
    ),
    Operation(
        id="expand-short",
        name="Expand short notes",
        description="Find notes below content threshold and expand with templates.\nOffline templates; use carefully.",
        script="expand_short_notes.py",
        icon="[^]",
        tags=["EXPAND"],
        args_hint='--vault "C:\\Users\\Wren C\\Documents\\Coding stuff\\Obsidian Vault"',
    ),
    Operation(
        id="anki-export",
        name="Anki export",
        description="Generate Anki decks from zettels.\nRequires genanki installed (optional).",
        script="anki_exporter.py",
        icon="[A]",
        tags=["EXPORT"],
        args_hint='--deck anatomy --out "C:\\path\\to\\decks"',
    ),
    Operation(
        id="doctor",
        name="Doctor (dry-run checks)",
        description="Dry-run checks for env vars, provider config, and basic backend init.\nUse this to validate Azure/Gemini/Ollama config without touching your vault.",
        script="vault_doctor.py",
        icon="[D]",
        tags=["DIAG"],
        default_args="--all",
        args_hint="--all  (or: --providers ollama,gemini,azure)",
    ),
]


class RunRequested(Message):
    def __init__(self, op: Operation, args: list[str]) -> None:
        super().__init__()
        self.op = op
        self.args = args


HUD_CSS = """
Screen {
    background: #06080c;
    color: #7fefaa;
}

#main-grid {
    layout: horizontal;
    height: 1fr;
}

#op-panel {
    width: 38;
    border: tall #163826;
    background: #0a0e14;
    padding: 0;
}

#op-panel-title {
    dock: top;
    height: 3;
    background: #0d1420;
    color: #00ff88;
    text-style: bold;
    content-align: center middle;
    border-bottom: hkey #1a3a2a;
}

#op-list {
    background: transparent;
    scrollbar-size: 1 1;
    padding: 0;
}

ListItem:hover {
    background: #0d2018;
}

ListItem.-highlight {
    background: #003322;
    color: #00ffaa;
}

#detail-area {
    width: 1fr;
    padding: 1 2;
    background: #080c10;
}

#detail-header {
    height: auto;
    max-height: 10;
    background: #0a1018;
    border: round #1a3a2a;
    padding: 1 2;
    margin-bottom: 1;
}

#detail-title {
    color: #00ff88;
    text-style: bold;
}

#detail-desc {
    color: #4a8a6a;
    margin-top: 1;
}

#detail-tags {
    color: #2a6a4a;
    margin-top: 1;
}

#config-status {
    color: #9fe7c0;
    margin-top: 1;
}

#args-row {
    height: 3;
    margin-bottom: 1;
    layout: horizontal;
}

#args-label {
    width: 7;
    height: 3;
    content-align: left middle;
    color: #3a7a5a;
    text-style: bold;
}

#args-input {
    width: 1fr;
    background: #0a0e14;
    border: tall #1a3a2a;
    color: #7fefaa;
}

#args-input:focus {
    border: tall #00ff88;
}

#btn-row {
    height: 3;
    layout: horizontal;
    margin-bottom: 1;
}

#btn-run {
    width: 1fr;
    background: #004422;
    color: #00ff88;
    border: tall #00aa55;
    text-style: bold;
    margin-right: 1;
}

#btn-clear {
    width: 1fr;
    background: #443300;
    color: #ffaa00;
    border: tall #aa7700;
    text-style: bold;
    margin-right: 1;
}

#btn-stop {
    width: 1fr;
    background: #440000;
    color: #ff4444;
    border: tall #aa0000;
    text-style: bold;
}

#output-panel {
    height: 1fr;
    background: #060a0e;
    border: round #1a3a2a;
    padding: 0;
}

#output-header {
    dock: top;
    height: 1;
    background: #0a1018;
    color: #2a5a4a;
    padding: 0 1;
    border-bottom: hkey #1a3a2a;
}

#output-log {
    background: transparent;
    scrollbar-size: 1 1;
    padding: 0 1;
    color: #5faf8a;
}

#status-bar {
    dock: bottom;
    height: 1;
    background: #0a1018;
    color: #2a5a4a;
    layout: horizontal;
    padding: 0 1;
    border-top: hkey #1a3a2a;
}

#status-left {
    width: 1fr;
    color: #2a6a4a;
}

#status-clock {
    width: auto;
    color: #ffaa00;
    text-style: bold;
}
"""


class OpItem(ListItem):
    def __init__(self, op: Operation) -> None:
        super().__init__()
        self.op = op

    def compose(self) -> ComposeResult:
        tag_str = " ".join(f"[{t}]" for t in self.op.tags)
        yield Label(f" {self.op.icon}  {self.op.name}  [dim]{tag_str}[/]", markup=True)


class VaultHud(App):
    """Futuristic HUD that runs the real scripts and streams output."""

    CSS = HUD_CSS
    TITLE = "VaultHUD"

    BINDINGS = [
        Binding("ctrl+r", "run_op", "Run", show=True, priority=True),
        Binding("ctrl+l", "clear_output", "Clear", show=True, priority=True),
        Binding("ctrl+c", "stop_op", "Stop", show=True, priority=True),
        Binding("ctrl+e", "ensure_env", ".env", show=True, priority=True),
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding("escape", "quit", "Quit", show=False),
    ]

    selected_op: reactive[Operation | None] = reactive(None)
    is_running: reactive[bool] = reactive(False)

    def __init__(self) -> None:
        super().__init__()
        self._proc: subprocess.Popen[str] | None = None
        self._stream_task: asyncio.Task[None] | None = None
        self._inproc_task: asyncio.Task[int] | None = None
        self._hud_settings: dict = _load_hud_settings()
        self._hud_op_index_by_id: dict[str, int] = {op.id: i for i, op in enumerate(OPERATIONS)}
        self._hud_settings_dirty: bool = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal(id="main-grid"):
            with Vertical(id="op-panel"):
                yield Static("OPERATIONS", id="op-panel-title")
                yield ListView(*[OpItem(op) for op in OPERATIONS], id="op-list")

            with Vertical(id="detail-area"):
                yield Container(
                    Static("Select an operation ←", id="detail-title"),
                    Static("", id="detail-desc"),
                    Static("", id="detail-tags"),
                    Static("", id="config-status"),
                    id="detail-header",
                )
                with Horizontal(id="args-row"):
                    yield Static("Args: ", id="args-label")
                    yield Input(placeholder="--flag value", id="args-input")

                with Horizontal(id="btn-row"):
                    yield Button("▶ RUN", id="btn-run", variant="success")
                    yield Button("✕ CLEAR", id="btn-clear", variant="warning")
                    yield Button("■ STOP", id="btn-stop", variant="error")

                yield ProgressBar(total=100, show_eta=False, id="progress-bar")

                with Container(id="output-panel"):
                    yield Static(" ◧  OUTPUT", id="output-header")
                    yield RichLog(highlight=True, markup=True, wrap=True, id="output-log", min_width=40)

        yield Container(
            Static("^r Run  ^l Clear  ^c Stop  ^q Quit", id="status-left"),
            Static("", id="status-clock"),
            id="status-bar",
        )

        yield Footer()

    def on_mount(self) -> None:
        op_list = self.query_one("#op-list", ListView)

        selected_id = str(self._hud_settings.get("selected_op_id", "") or "").strip()
        idx = self._hud_op_index_by_id.get(selected_id, 0)
        idx = max(0, min(idx, len(OPERATIONS) - 1))
        op_list.index = idx

        # Make the right pane show the selected op immediately.
        self._set_selected_op(OPERATIONS[idx], prefer_saved_args=True)
        self._update_config_status()
        self._update_clock()
        self.set_interval(1, self._update_clock)
        self.set_interval(2, self._update_config_status)

        log = self.query_one("#output-log", RichLog)
        log.write("[dim]╔══════════════════════════════════════════╗[/]")
        log.write("[dim]║[/]  [bold #00ff88]VaultHUD[/] — [dim]ready[/]")
        log.write("[dim]║[/]  Select an operation and press [bold #00ff88]^r[/] to run")
        log.write("[dim]╚══════════════════════════════════════════╝[/]")

    def _update_clock(self) -> None:
        self.query_one("#status-clock", Static).update(datetime.now().strftime("%H:%M:%S"))

    def _update_config_status(self) -> None:
        s = self.query_one("#config-status", Static)

        load_dotenv_no_override(repo_root=REPO_ROOT)

        if self._hud_settings_dirty:
            self._persist_hud_settings(flush=True)

        provider = (os.environ.get("VAULT_LLM_PROVIDER", "ollama") or "ollama").strip().lower()
        if provider not in ("ollama", "gemini", "azure"):
            provider = "ollama"

        paths = get_vault_paths()

        gemini_ok = bool(os.environ.get("GEMINI_API_KEY", "").strip())
        azure_ok = (
            bool(os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip())
            and bool(os.environ.get("AZURE_OPENAI_API_KEY", "").strip())
            and bool(os.environ.get("VAULT_AZURE_MODEL", "").strip())
        )
        ollama_cloud_ok = bool(os.environ.get("OLLAMA_API_KEY", "").strip())

        ollama_model = (os.environ.get("VAULT_OLLAMA_MODEL", "") or "").strip()
        gemini_model = (os.environ.get("VAULT_GEMINI_MODEL", "gemini-2.5-flash") or "").strip()
        azure_model = (os.environ.get("VAULT_AZURE_MODEL", "") or "").strip()

        def yn(v: bool) -> str:
            return "[bold #00ff88]yes[/]" if v else "[bold #ff4444]no[/]"

        lines = [
            f"[dim]provider:[/] [bold #00ff88]{provider}[/]",
            f"[dim]vault in:[/] {paths.input_vault}",
            f"[dim]vault out:[/] {paths.output_vault}",
        ]
        if ollama_model:
            lines.append(f"[dim]ollama model:[/] {ollama_model}")
        lines.extend(
            [
                f"[dim]ollama cloud key:[/] {yn(ollama_cloud_ok)}",
                f"[dim]gemini key:[/] {yn(gemini_ok)}  [dim]model:[/] {gemini_model}",
                f"[dim]azure ready:[/] {yn(azure_ok)}  [dim]deployment:[/] {azure_model or '(unset)'}",
            ]
        )
        s.update("\n".join(lines))

    def _set_selected_op(self, op: Operation, *, prefer_saved_args: bool = False) -> None:
        self.selected_op = op
        title = self.query_one("#detail-title", Static)
        desc = self.query_one("#detail-desc", Static)
        tags = self.query_one("#detail-tags", Static)
        args_input = self.query_one("#args-input", Input)

        title.update(f"{op.icon}  {op.name}")
        desc.update(op.description)
        tag_str = "  ".join(f"[bold #2a6a4a]<{t}>[/]" for t in op.tags)
        tags.update(tag_str)
        args_input.placeholder = op.args_hint or "--flag value"

        saved = ""
        if prefer_saved_args:
            by_op = self._hud_settings.get("args_by_op")
            if isinstance(by_op, dict):
                v = by_op.get(op.id, "")
                if isinstance(v, str):
                    saved = v

        if saved.strip():
            args_input.value = saved
        elif not args_input.value.strip():
            args_input.value = op.default_args

    def _stash_args_for_selected_op(self) -> None:
        op = self.selected_op
        if not op:
            return

        args_input = self.query_one("#args-input", Input)
        args_by_op = self._hud_settings.get("args_by_op")
        if not isinstance(args_by_op, dict):
            args_by_op = {}

        args_by_op[op.id] = args_input.value
        self._hud_settings["args_by_op"] = args_by_op
        self._hud_settings_dirty = True

    def _persist_hud_settings(self, *, flush: bool) -> None:
        op = self.selected_op
        if not op:
            return

        self._hud_settings["selected_op_id"] = op.id
        if flush:
            _save_hud_settings(self._hud_settings)
            self._hud_settings_dirty = False

    @on(ListView.Highlighted, "#op-list")
    def _op_highlighted(self, event: ListView.Highlighted) -> None:
        item = event.item
        if isinstance(item, OpItem):
            # Stash args for the previously highlighted op before switching UI state.
            self._stash_args_for_selected_op()
            self._set_selected_op(item.op, prefer_saved_args=True)
            self._persist_hud_settings(flush=True)

    @on(Input.Changed, "#args-input")
    def _args_changed(self, _event: Input.Changed) -> None:
        self._stash_args_for_selected_op()

    async def action_quit(self) -> None:
        self._stash_args_for_selected_op()
        _save_hud_settings(self._hud_settings)
        self._hud_settings_dirty = False
        await super().action_quit()

    def action_clear_output(self) -> None:
        self.query_one("#output-log", RichLog).clear()
        self.query_one("#progress-bar", ProgressBar).update(progress=0)

    def action_ensure_env(self) -> None:
        """
        Create `.env` from `.env.example` if missing. Never overwrites.
        """
        log = self.query_one("#output-log", RichLog)
        env_path = REPO_ROOT / ".env"
        example_path = REPO_ROOT / ".env.example"

        if env_path.exists():
            log.write(f"[bold #00ff88]✓[/] .env exists: {env_path}")
            return
        if not example_path.exists():
            log.write(f"[bold #ff4444]Missing:[/] {example_path}")
            return

        try:
            env_path.write_text(example_path.read_text(encoding="utf-8"), encoding="utf-8")
            log.write(f"[bold #00ff88]✓[/] created .env from .env.example")
        except Exception as exc:
            log.write(f"[bold #ff4444]Failed:[/] {exc}")

    def action_stop_op(self) -> None:
        self.is_running = False
        self._terminate_proc()
        if self._inproc_task and not self._inproc_task.done():
            self._inproc_task.cancel()

    def action_run_op(self) -> None:
        if self.is_running:
            return
        if not self.selected_op:
            return
        args_str = self.query_one("#args-input", Input).value.strip()
        try:
            args = shlex.split(args_str, posix=False) if args_str else []
        except ValueError as exc:
            self.query_one("#output-log", RichLog).write(f"[bold #ff4444]args parse error:[/] {exc}")
            return
        self.post_message(RunRequested(self.selected_op, args))

    async def on_run_requested(self, msg: RunRequested) -> None:
        if (self._proc and self._proc.poll() is None) or (self._inproc_task and not self._inproc_task.done()):
            self.query_one("#output-log", RichLog).write("[bold #ffaa00]Busy:[/] stop the running process first.")
            return

        log = self.query_one("#output-log", RichLog)
        progress = self.query_one("#progress-bar", ProgressBar)

        try:
            script_path = msg.op.script_path()
        except FileNotFoundError as exc:
            log.write(f"[bold #ff4444]Missing script:[/] {exc}")
            return

        log.write(f"\n[bold #ffaa00]━━━ {msg.op.icon}  {msg.op.name} ━━━[/]")
        try:
            preview_cmd = build_python_command(script_path, msg.args)
        except Exception:
            preview_cmd = [str(script_path), *msg.args]
        log.write(f"[dim]$ {' '.join(_quote(c) for c in preview_cmd)}[/]")

        self.is_running = True
        progress.update(progress=0)

        if getattr(sys, "frozen", False):
            cmd = [sys.executable, "--_run-script", str(script_path), "--", *msg.args]
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self._stream_task = asyncio.create_task(self._stream_output())
            await self._stream_task
            code = self._proc.wait() if self._proc else 1
        else:
            try:
                self._proc = popen_script(repo_root=REPO_ROOT, script=script_path, passthrough=msg.args)
            except Exception as exc:
                log.write(f"[bold #ff4444]Failed to start:[/] {exc}")
                self.is_running = False
                return

            self._stream_task = asyncio.create_task(self._stream_output())
            await self._stream_task

            code = self._proc.wait() if self._proc else 1
        log.write(f"[bold #00ff88]✓[/] exit {code}")
        self.is_running = False
        self._proc = None
        self._inproc_task = None

    async def _stream_output(self) -> None:
        assert self._proc is not None
        assert self._proc.stdout is not None

        log = self.query_one("#output-log", RichLog)
        progress = self.query_one("#progress-bar", ProgressBar)

        tick = 0
        while True:
            if not self.is_running:
                break

            line = await asyncio.to_thread(self._proc.stdout.readline)
            if not line and self._proc.poll() is not None:
                break
            if line:
                log.write(line.rstrip("\n"))
                tick = min(100, tick + 1)
                progress.update(progress=tick)

        progress.update(progress=100 if (self._proc and self._proc.poll() == 0) else progress.progress)

    def _terminate_proc(self) -> None:
        if not self._proc or self._proc.poll() is not None:
            return
        try:
            self._proc.terminate()
            self.query_one("#output-log", RichLog).write("[bold #ff4444]■ terminate sent[/]")
        except Exception as exc:
            self.query_one("#output-log", RichLog).write(f"[bold #ff4444]stop error:[/] {exc}")

    @on(Button.Pressed, "#btn-run")
    def _btn_run(self) -> None:
        self.action_run_op()

    @on(Button.Pressed, "#btn-clear")
    def _btn_clear(self) -> None:
        self.action_clear_output()

    @on(Button.Pressed, "#btn-stop")
    def _btn_stop(self) -> None:
        self.action_stop_op()


def _quote(s: str) -> str:
    if not s:
        return '""'
    if any(ch.isspace() for ch in s) or any(ch in s for ch in ('"', "'")):
        return '"' + s.replace('"', '\\"') + '"'
    return s


def _looks_like_cmd() -> bool:
    """
    Heuristic: when launched from cmd.exe, PROMPT is usually set.
    (PowerShell typically doesn't define PROMPT env var; it defines a prompt function.)
    """
    return bool(__import__("os").environ.get("PROMPT"))


def _relaunch_in_powershell(argv: list[str]) -> None:
    """
    Launch a new PowerShell window running this program, then exit.
    Uses UTF-8 codepage to improve box-drawing + symbols rendering.
    """
    exe = Path(sys.executable).resolve()

    if getattr(sys, "frozen", False):
        target = f"& '{str(exe)}' --no-relaunch"
    else:
        script = Path(__file__).resolve()
        target = f"& '{sys.executable}' '{str(script)}' --no-relaunch"

    # Preserve user args except our relaunch flags.
    passthrough = [a for a in argv if a not in ("--relaunch-powershell", "--no-relaunch")]
    if passthrough:
        # naive quoting; PowerShell will receive as a single string
        extra = " " + " ".join("'" + a.replace("'", "''") + "'" for a in passthrough)
    else:
        extra = ""

    command = f"chcp 65001 > $null; {target}{extra}"
    subprocess.Popen(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-NoExit",
            "-Command",
            command,
        ],
        cwd=str(REPO_ROOT),
        creationflags=subprocess.CREATE_NEW_CONSOLE,  # type: ignore[attr-defined]
    )


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
    parser.add_argument(
        "--no-relaunch",
        action="store_true",
        help="Disable auto-relaunch into PowerShell when started from cmd.exe.",
    )
    ns, rest = parser.parse_known_args()

    if ns._run_script:
        # Internal execution mode used by packaged exe to run helper scripts in a child process.
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
        missing = []
        for a in OPERATIONS:
            try:
                a.script_path()
            except FileNotFoundError:
                missing.append(a.script)

        print("VaultHUD self-test")
        print(f"- repo_root: {REPO_ROOT}")
        print(f"- python: {sys.executable}")
        if getattr(sys, "frozen", False):
            print("- run_mode: child-exe")
            launcher = detect_python_launcher()
            print(f"- python_launcher(optional): {launcher.argv0 if launcher else 'none'}")
        print(f"- actions: {len(OPERATIONS)}")
        print(f"- missing_scripts: {missing if missing else 'none'}")
        raise SystemExit(0)

    # Default behavior: if launched from cmd.exe, relaunch in PowerShell to improve rendering.
    if (not ns.no_relaunch) and _looks_like_cmd():
        _relaunch_in_powershell(sys.argv[1:])
        raise SystemExit(0)

    VaultHud().run()

