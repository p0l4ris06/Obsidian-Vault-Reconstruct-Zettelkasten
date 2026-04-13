from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# Hard-locked default vault path for this repo.
DEFAULT_VAULT_PATH = Path(r"C:\Users\Wren C\Documents\Coding stuff\Obsidian Vault")


@dataclass(frozen=True)
class VaultPaths:
    input_vault: Path
    output_vault: Path


def get_vault_paths() -> VaultPaths:
    """
    Centralised path selection.

    Defaults are hard-locked to DEFAULT_VAULT_PATH per your instruction.
    Env vars are still supported for automation, but should be rarely needed:
      - VAULT_INPUT_PATH
      - VAULT_OUTPUT_PATH
      - VAULT_PATH (acts as both)
    """
    base = Path(os.environ.get("VAULT_PATH", str(DEFAULT_VAULT_PATH)))
    inp = Path(os.environ.get("VAULT_INPUT_PATH", str(base)))
    out = Path(os.environ.get("VAULT_OUTPUT_PATH", str(base)))
    return VaultPaths(input_vault=inp, output_vault=out)

