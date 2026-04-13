from __future__ import annotations

import contextlib
import os
import runpy
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


@dataclass(frozen=True)
class PythonLauncher:
    argv0: str
    args_prefix: list[str]


def detect_python_launcher() -> PythonLauncher | None:
    """
    When running from a packaged exe, sys.executable is the exe, not Python.
    Prefer:
      - py -3 (Windows Python launcher)
      - python (PATH)
    """
    if shutil.which("py"):
        return PythonLauncher(argv0="py", args_prefix=["-3"])
    if shutil.which("python"):
        return PythonLauncher(argv0="python", args_prefix=[])
    return None


def build_python_command(script: Path, passthrough: list[str]) -> list[str]:
    if getattr(sys, "frozen", False):
        launcher = detect_python_launcher()
        if launcher is None:
            # In frozen builds we can still run in-process; this is mainly used for previews.
            return [str(script), *passthrough]
        return [launcher.argv0, *launcher.args_prefix, str(script), *passthrough]
    return [sys.executable, str(script), *passthrough]


def popen_script(*, repo_root: Path, script: Path, passthrough: list[str]) -> subprocess.Popen[str]:
    cmd = build_python_command(script, passthrough)
    return subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


class _LineWriter:
    def __init__(self, on_line: Callable[[str], None]) -> None:
        self._on_line = on_line
        self._buf = ""

    def write(self, s: str) -> int:  # file-like
        if not s:
            return 0
        self._buf += s
        while True:
            idx = self._buf.find("\n")
            if idx < 0:
                break
            line = self._buf[:idx]
            self._buf = self._buf[idx + 1 :]
            self._on_line(line.rstrip("\r"))
        return len(s)

    def flush(self) -> None:  # file-like
        if self._buf:
            self._on_line(self._buf.rstrip("\r"))
            self._buf = ""


def run_script_inprocess(
    *,
    repo_root: Path,
    script: Path,
    passthrough: list[str],
    on_line: Callable[[str], None],
) -> int:
    """
    Execute a .py script inside the current interpreter (used for frozen exe),
    capturing stdout/stderr and forwarding lines via on_line.
    """
    prev_cwd = Path.cwd()
    prev_argv = sys.argv[:]

    writer = _LineWriter(on_line)
    try:
        os.chdir(str(repo_root))
        sys.argv = [str(script), *passthrough]
        with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
            try:
                runpy.run_path(str(script), run_name="__main__")
                writer.flush()
                return 0
            except SystemExit as exc:
                writer.flush()
                code = exc.code
                if code is None:
                    return 0
                if isinstance(code, int):
                    return code
                return 1
            except Exception as exc:
                writer.write(f"[error] {type(exc).__name__}: {exc}\n")
                writer.flush()
                return 1
    finally:
        sys.argv = prev_argv
        os.chdir(str(prev_cwd))

