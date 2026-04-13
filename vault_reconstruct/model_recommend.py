from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Literal


Category = Literal["coding", "reasoning", "creative", "balanced"]


@dataclass(frozen=True)
class LlmCheckerRecommendation:
    model: str
    recommended_model: str | None = None
    confidence: float | None = None
    raw: dict[str, Any] | None = None


def _ollama_installed_models() -> list[str]:
    """
    Best-effort list of locally installed Ollama model tags.

    Uses the `ollama` CLI if available. Returns an empty list on any failure.
    """
    exe = shutil.which("ollama")
    if not exe:
        return []
    try:
        p = subprocess.run([exe, "list"], check=False, capture_output=True, text=True, timeout=15)
    except Exception:
        return []
    lines = (p.stdout or "").splitlines()
    if not lines:
        return []

    models: list[str] = []
    for line in lines[1:]:  # skip header
        parts = line.split()
        if parts:
            models.append(parts[0].strip())
    return [m for m in models if m]


def _run_llm_checker_json(args: list[str], *, timeout_s: int = 60) -> dict[str, Any] | None:
    exe = shutil.which("llm-checker")
    if not exe:
        return None
    try:
        p = subprocess.run(
            [exe, *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except Exception:
        return None

    out = (p.stdout or "").strip()
    if not out:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def recommend_ollama_model(*, category: Category = "reasoning") -> LlmCheckerRecommendation | None:
    """
    Best-effort local model recommendation via `llm-checker`.

    This is intentionally optional:
    - If `llm-checker` isn't installed or fails, returns None.
    - Callers should fall back to their own default model.
    """
    # Prefer a task hint from env if set (lets you tune without code changes).
    env_cat = os.environ.get("VAULT_LLM_CHECKER_CATEGORY", "").strip().lower()
    if env_cat in ("coding", "reasoning", "creative", "balanced"):
        category = env_cat  # type: ignore[assignment]

    # Use `smart-recommend` (supports stable JSON output via `-j`).
    use_case = {"coding": "coding", "reasoning": "reasoning", "creative": "creative", "balanced": "general"}.get(
        category, "general"
    )
    data = _run_llm_checker_json(["smart-recommend", "-u", use_case, "-l", "1", "-j"])
    if not isinstance(data, dict):
        return None

    # Current `smart-recommend -j` schema:
    #   { "topPicks": { "best": { "variant": { "tag": "qwen2.5:..." }, "score": ... } } }
    recommended = (
        (((data.get("topPicks") or {}).get("best") or {}).get("variant") or {}).get("tag")
        if isinstance(data.get("topPicks"), dict)
        else None
    )
    if not isinstance(recommended, str) or not recommended.strip():
        return None

    model = recommended.strip()

    # If it's not installed locally, prefer a reasonable installed fallback so
    # "dry runs" and local-first flows work without an extra pull step.
    installed = set(_ollama_installed_models())
    if installed and model.strip() not in installed:
        for preferred in (
            "qwen2.5-coder:3b",
            "qwen-slim:latest",
            "gemma3:4b",
        ):
            if preferred in installed:
                model = preferred
                break
        else:
            model = next(iter(installed))

    conf = (((data.get("topPicks") or {}).get("best") or {}).get("confidence")) if isinstance(data.get("topPicks"), dict) else None
    confidence = float(conf) if isinstance(conf, (int, float)) else None
    return LlmCheckerRecommendation(model=model.strip(), recommended_model=recommended.strip(), confidence=confidence, raw=data)


def _is_instruction_tuned(tag: str) -> bool:
    """
    Heuristic: models tuned for chat/instruction tend to follow JSON-only prompts better.

    Base / pretrain weights (often tagged `base`) are intentionally treated as not
    instruction-tuned for routing purposes.
    """
    t = tag.lower()
    if "instruct" in t or "-it-" in t or ":it-" in t:
        return True
    if "chat" in t and "base" not in t:
        return True
    # Common chat-tuned families without the word "instruct" in the tag.
    if any(p in t for p in ("gemma3:", "gemma2:", "llama3.", "llama3:", "mistral:", "phi3:", "phi4:")):
        if "base" in t:
            return False
        return True
    if "coder" in t and "instruct" not in t and "chat" not in t:
        return False
    if "base" in t or ":base" in t or "-base-" in t:
        return False
    return False


def select_ollama_model_for_mode(*, strict_json: bool, category: Category = "reasoning") -> str:
    """
    Pick an Ollama model tag for this process.

    - If `VAULT_OLLAMA_MODEL` is set, it always wins (single model for all tasks).
    - If `strict_json` is False, use `llm-checker` recommendation (+ installed fallback),
      optimized for general/prose speed.
    - If `strict_json` is True, prefer an instruction-tuned *installed* model:
        `VAULT_OLLAMA_INSTRUCT_MODEL` if set, else any installed tag passing `_is_instruction_tuned`,
        else the llm-checker pick if it already looks instruction-tuned,
        else a conservative default string (may require `ollama pull`).
    """
    override = os.environ.get("VAULT_OLLAMA_MODEL", "").strip()
    if override:
        return override

    installed = _ollama_installed_models()
    installed_set = set(installed)

    if not strict_json:
        rec = recommend_ollama_model(category=category)
        return rec.model if rec else "qwen2.5:1.5b-instruct-q8_0"

    instruct_env = os.environ.get("VAULT_OLLAMA_INSTRUCT_MODEL", "").strip()
    if instruct_env and instruct_env in installed_set:
        return instruct_env

    for tag in installed:
        if _is_instruction_tuned(tag):
            return tag

    rec = recommend_ollama_model(category=category)
    if rec and _is_instruction_tuned(rec.model):
        return rec.model

    # Prefer pulling a small instruct model if nothing suitable is installed.
    for candidate in (
        "qwen2.5:1.5b-instruct-q8_0",
        "qwen2.5:3b-instruct-q4_K_M",
        "gemma3:4b",
    ):
        if candidate in installed_set:
            return candidate

    return rec.model if rec else "qwen2.5:1.5b-instruct-q8_0"

