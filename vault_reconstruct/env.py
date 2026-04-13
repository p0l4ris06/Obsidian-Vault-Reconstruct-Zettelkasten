from __future__ import annotations

import os
from pathlib import Path


def load_dotenv_no_override(*, repo_root: Path | None = None) -> None:
    """
    Minimal `.env` loader used across scripts.

    - Quiet by default
    - Does not override already-set environment variables
    - Supports basic `KEY=VALUE` lines, strips surrounding quotes
    """
    root = repo_root or Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if not env_path.exists():
        return

    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        return

