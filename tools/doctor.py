from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

# Add repo root to path so we can import the vault_reconstruct package
sys.path.append(str(Path(__file__).resolve().parent.parent))

from vault_reconstruct.config import get_vault_paths
from vault_reconstruct.env import load_dotenv_no_override
from vault_reconstruct.llm import LlmConfig, generate_text_with_retries, make_backend
from vault_reconstruct.model_recommend import recommend_ollama_model, select_ollama_model_for_mode

load_dotenv_no_override()

ProviderName = Literal["ollama", "gemini", "azure"]


@dataclass(frozen=True)
class DoctorResult:
    ok: bool
    skipped: bool
    provider: str
    model: str
    total_latency_s: float | None
    per_ping_s: list[float]
    notes: list[str]


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _detect_provider() -> ProviderName:
    return (_env("VAULT_LLM_PROVIDER") or "ollama").lower()


def _effective_model(provider: ProviderName) -> str:
    if provider == "ollama":
        if _env("VAULT_OLLAMA_MODEL"):
            return _env("VAULT_OLLAMA_MODEL")
        # Match JSON-heavy phases in the vault pipelines (split / link / tags).
        return select_ollama_model_for_mode(strict_json=True)
    if provider == "gemini":
        return _env("VAULT_GEMINI_MODEL") or "gemini-2.5-flash"
    if provider == "azure":
        return _env("VAULT_AZURE_MODEL") or "gpt-4.1-mini"
    return ""


def _build_cfg(provider: ProviderName, model: str) -> LlmConfig:
    if provider == "ollama":
        cloud_model = _env("VAULT_OLLAMA_CLOUD_MODEL") or "gemma3:4b-cloud"
        return LlmConfig(provider="ollama", model=model, ollama_cloud_model=cloud_model, max_retries=3)
    if provider == "gemini":
        return LlmConfig(provider="gemini", model=model, max_retries=3)
    if provider == "azure":
        return LlmConfig(provider="azure", model=model, max_retries=3)
    raise SystemExit(f"Unknown provider: {provider!r}")


def _provider_ready(provider: ProviderName) -> tuple[bool, str | None]:
    if provider == "ollama":
        return True, None
    if provider == "gemini":
        if not _env("GEMINI_API_KEY"):
            return False, "Missing GEMINI_API_KEY"
        return True, None
    if provider == "azure":
        if not _env("AZURE_OPENAI_ENDPOINT") or not _env("AZURE_OPENAI_API_KEY"):
            return False, "Missing AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_API_KEY"
        if not _env("VAULT_AZURE_MODEL"):
            return False, "Missing VAULT_AZURE_MODEL (Azure deployment name)"
        return True, None
    return False, f"Unknown provider: {provider!r}"


def _ollama_pull(model: str, *, timeout_s: int = 60 * 30) -> tuple[bool, str]:
    exe = shutil.which("ollama")
    if not exe:
        return False, "ollama CLI not found on PATH"
    try:
        p = subprocess.run(
            [exe, "pull", model],
            check=False,
            text=True,
            # Don't capture output; allow progress display in terminal.
            timeout=timeout_s,
        )
    except Exception as exc:
        return False, f"ollama pull failed: {exc}"

    if p.returncode != 0:
        return False, f"ollama pull exited {p.returncode}"
    return True, "pulled"


def run_doctor_single(*, provider: ProviderName, ping: bool, ping_repeats: int, pull_recommended: bool) -> DoctorResult:
    notes: list[str] = []

    model = _effective_model(provider)
    if provider == "ollama" and not _env("VAULT_OLLAMA_MODEL"):
        rec = recommend_ollama_model(category="reasoning")
        if rec and rec.recommended_model:
            notes.append(f"llm-checker smart-recommend: {rec.recommended_model}")
        notes.append(f"JSON-route model (this ping): {model}")
        if pull_recommended and rec and rec.recommended_model:
            ok, msg = _ollama_pull(rec.recommended_model)
            if ok:
                notes.append(f"pulled llm-checker tag: {rec.recommended_model}")
                model = rec.recommended_model
            else:
                notes.append(f"WARNING: could not pull recommended model: {msg[:200]}")
    if not model:
        return DoctorResult(
            ok=False,
            skipped=False,
            provider=provider,
            model=model,
            total_latency_s=None,
            per_ping_s=[],
            notes=["No model resolved."],
        )

    # Dependency hints only (don’t mutate anything).
    if shutil.which("llm-checker"):
        notes.append("llm-checker: found")
    else:
        notes.append("llm-checker: not found (optional)")

    # Vault paths sanity (no writes).
    paths = get_vault_paths()
    notes.append(f"vault input: {paths.input_vault}")
    notes.append(f"vault output: {paths.output_vault}")
    if not paths.input_vault.exists():
        notes.append("WARNING: input vault path does not exist")
    if not paths.output_vault.exists():
        notes.append("WARNING: output vault path does not exist")

    # Provider-specific env checks.
    ready, why = _provider_ready(provider)
    if not ready:
        return DoctorResult(
            ok=True,
            skipped=True,
            provider=provider,
            model=model,
            total_latency_s=None,
            per_ping_s=[],
            notes=notes + [f"SKIP: {why or 'not ready'}"],
        )

    # Backend construction validation.
    try:
        backend = make_backend(_build_cfg(provider, model))
        notes.append("backend: init ok")
    except Exception as exc:
        return DoctorResult(
            ok=False,
            skipped=False,
            provider=provider,
            model=model,
            total_latency_s=None,
            per_ping_s=[],
            notes=notes + [f"backend init failed: {exc}"],
        )

    if not ping:
        return DoctorResult(
            ok=True,
            skipped=False,
            provider=provider,
            model=model,
            total_latency_s=None,
            per_ping_s=[],
            notes=notes + ["ping: skipped"],
        )

    # Minimal ping (no vault changes).
    prompt = "Reply with exactly: pong"
    last = ""
    per_ping: list[float] = []
    try:
        for _ in range(max(1, int(ping_repeats))):
            t0 = time.perf_counter()
            last = generate_text_with_retries(backend, prompt=prompt, max_retries=3)
            per_ping.append(time.perf_counter() - t0)
    except Exception as exc:
        return DoctorResult(
            ok=False,
            skipped=False,
            provider=provider,
            model=model,
            total_latency_s=None,
            per_ping_s=per_ping,
            notes=notes + [f"ping failed: {exc}"],
        )

    trimmed = last.strip()
    notes.append(f"ping response (trimmed): {trimmed[:80]!r}")
    if trimmed.lower() != "pong":
        notes.append("WARNING: ping response was not exactly 'pong' (model instruction-following may be weak)")
    total_latency = sum(per_ping) if per_ping else None
    if per_ping:
        notes.append(f"ping warm (s): {per_ping[0]:.3f}")
        if len(per_ping) > 1:
            notes.append(f"ping hot avg (s): {sum(per_ping[1:]) / len(per_ping[1:]):.3f}")
    return DoctorResult(
        ok=True,
        skipped=False,
        provider=provider,
        model=model,
        total_latency_s=total_latency,
        per_ping_s=per_ping,
        notes=notes,
    )


def run_doctor(*, providers: Iterable[ProviderName], ping: bool, ping_repeats: int, pull_recommended: bool) -> list[DoctorResult]:
    results: list[DoctorResult] = []
    for p in providers:
        results.append(
            run_doctor_single(
                provider=p,
                ping=ping,
                ping_repeats=ping_repeats,
                pull_recommended=pull_recommended,
            )
        )
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Dry-run integration + performance checks (no vault writes).")
    p.add_argument("--ping", action="store_true", help="Run a tiny timed LLM ping.")
    p.add_argument("--ping-repeats", type=int, default=1, help="Number of ping calls to average startup+latency effects.")
    p.add_argument(
        "--providers",
        default="",
        help="Comma list: ollama,gemini,azure. Default: current VAULT_LLM_PROVIDER only.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Test ollama, gemini, and azure (skips those missing keys).",
    )
    p.add_argument(
        "--pull-recommended",
        action="store_true",
        help="If llm-checker recommends an Ollama model that isn't installed, run `ollama pull` for it (opt-in).",
    )
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    if args.all:
        providers: list[ProviderName] = ["ollama", "gemini", "azure"]
    elif str(args.providers).strip():
        raw = [x.strip().lower() for x in str(args.providers).split(",") if x.strip()]
        providers = [x for x in raw if x in ("ollama", "gemini", "azure")]  # type: ignore[assignment]
        if not providers:
            raise SystemExit("No valid providers in --providers. Use: ollama,gemini,azure")
    else:
        providers = [_detect_provider()]

    results = run_doctor(
        providers=providers,
        ping=bool(args.ping),
        ping_repeats=int(args.ping_repeats),
        pull_recommended=bool(args.pull_recommended),
    )
    all_ok = True
    for r in results:
        # Skips should not fail the overall doctor run.
        all_ok = all_ok and (r.ok or r.skipped)
        print(f"ok={r.ok}")
        print(f"skipped={r.skipped}")
        print(f"provider={r.provider}")
        print(f"model={r.model}")
        if r.total_latency_s is not None:
            print(f"ping_total_s={r.total_latency_s:.3f}")
            if r.per_ping_s:
                print("ping_each_s=" + ",".join(f"{x:.3f}" for x in r.per_ping_s))
        for n in r.notes:
            print(f"- {n}")
        print()
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())



