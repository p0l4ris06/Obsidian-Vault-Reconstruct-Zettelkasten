from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VaultPaths:
    input_vault: Path
    output_vault: Path


def _get_default_vault_path() -> Path:
    """
    Get default vault path using relative path fallback.
    
    Priority:
    1. VAULT_PATH environment variable
    2. Relative path from this script location: ../Obsidian Vault
    """
    if vault_path := os.environ.get("VAULT_PATH"):
        return Path(vault_path)
    
    # Use relative path fallback: ../Obsidian Vault (from vault_reconstruct/config.py)
    config_dir = Path(__file__).parent
    repo_root = config_dir.parent
    default_path = repo_root / ".." / "Obsidian Vault"
    return default_path.resolve()


def _validate_vault_path(path: Path) -> None:
    """
    Validate that vault path exists.
    
    Raises:
        FileNotFoundError: If vault path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Vault path does not exist: {path}\n"
            f"Set VAULT_PATH environment variable or ensure the Obsidian Vault exists at: {path}"
        )


def get_vault_paths() -> VaultPaths:
    """
    Get vault input and output paths with validation.

    Priority order:
    1. VAULT_INPUT_PATH / VAULT_OUTPUT_PATH (specific paths)
    2. VAULT_PATH (acts as both input and output)
    3. Relative fallback: ../Obsidian Vault from repo root
    
    All paths are validated to exist.
    """
    base = _get_default_vault_path()
    inp = Path(os.environ.get("VAULT_INPUT_PATH", str(base)))
    out = Path(os.environ.get("VAULT_OUTPUT_PATH", str(base)))
    
    # Validate paths exist
    try:
        _validate_vault_path(inp)
        _validate_vault_path(out)
    except FileNotFoundError as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        raise
    
    return VaultPaths(input_vault=inp, output_vault=out)

