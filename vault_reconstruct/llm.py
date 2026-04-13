from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal, Protocol


class LlmBackend(Protocol):
    def generate_text(self, *, prompt: str) -> str:
        ...


Provider = Literal["ollama", "gemini", "azure"]


@dataclass(frozen=True)
class LlmConfig:
    provider: Provider
    model: str
    max_retries: int = 5

    ollama_cloud_model: str | None = None

    gemini_api_key: str | None = None

    azure_endpoint: str | None = None
    azure_api_key: str | None = None
    azure_api_version: str = "2024-10-21"


def _is_transient_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(t in msg for t in ("timeout", "timed out", "rate", "429", "quota", "connection", "refused", "temporar"))


def _backoff_seconds(attempt: int) -> float:
    return min(60.0, (2**attempt)) + random.uniform(0.0, 0.8)


def _normalize_ollama_host(raw: str) -> str:
    """
    Normalize an Ollama host string for client connections.

    Common Windows pitfall: OLLAMA_HOST can be set to `0.0.0.0:11434` (bind
    address). That is not connectable; use localhost instead.
    """
    h = (raw or "").strip()
    if not h:
        return ""
    if h.startswith("0.0.0.0"):
        h = h.replace("0.0.0.0", "127.0.0.1", 1)
    if "://" not in h:
        h = "http://" + h
    return h


class OllamaBackend:
    def __init__(self, *, model: str, cloud_model: str | None = None) -> None:
        import ollama  # type: ignore
        from ollama import Client  # type: ignore

        self._ollama = ollama
        self._local_model = model
        self._local_client = None

        # Prefer an explicit client when OLLAMA_HOST is set (lets us normalize
        # problematic values like `0.0.0.0:11434`).
        raw_host = os.environ.get("OLLAMA_HOST", "").strip()
        norm_host = _normalize_ollama_host(raw_host)
        if norm_host:
            self._local_client = Client(host=norm_host)

        api_key = os.environ.get("OLLAMA_API_KEY", "").strip()
        self._cloud_model = cloud_model if (api_key and cloud_model) else None
        self._cloud_client = (
            Client(host="https://ollama.com", headers={"Authorization": f"Bearer {api_key}"})
            if self._cloud_model
            else None
        )

    def generate_text(self, *, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        if self._cloud_client and self._cloud_model:
            try:
                r = self._cloud_client.chat(model=self._cloud_model, messages=messages)
                return r["message"]["content"]
            except Exception:
                pass
        if self._local_client is not None:
            r = self._local_client.chat(model=self._local_model, messages=messages)
        else:
            r = self._ollama.chat(model=self._local_model, messages=messages)
        return r["message"]["content"]


class GeminiBackend:
    def __init__(self, *, model: str, api_key: str) -> None:
        from google import genai  # type: ignore

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate_text(self, *, prompt: str) -> str:
        r = self._client.models.generate_content(model=self._model, contents=prompt)
        return r.text or ""


class AzureOpenAIBackend:
    def __init__(self, *, model: str, endpoint: str, api_key: str, api_version: str) -> None:
        from openai import AzureOpenAI  # type: ignore

        self._client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        self._model = model

    def generate_text(self, *, prompt: str) -> str:
        r = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return (r.choices[0].message.content or "").strip()


def make_backend(cfg: LlmConfig) -> LlmBackend:
    if cfg.provider == "ollama":
        return OllamaBackend(model=cfg.model, cloud_model=cfg.ollama_cloud_model)
    if cfg.provider == "gemini":
        key = cfg.gemini_api_key or os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("Missing GEMINI_API_KEY for Gemini provider.")
        return GeminiBackend(model=cfg.model, api_key=key)
    if cfg.provider == "azure":
        endpoint = cfg.azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
        key = cfg.azure_api_key or os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "").strip() or cfg.azure_api_version
        if not endpoint or not key:
            raise RuntimeError("Missing AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_API_KEY for Azure provider.")
        return AzureOpenAIBackend(model=cfg.model, endpoint=endpoint, api_key=key, api_version=api_version)
    raise ValueError(f"Unknown provider: {cfg.provider}")


_thread_local = threading.local()


def make_backend_thread_local(cfg: LlmConfig) -> LlmBackend:
    """
    Return a per-thread cached backend instance.

    Some SDK clients are not guaranteed to be thread-safe; this keeps concurrency
    in higher-level scripts predictable without forcing those scripts to manage
    their own client pooling.
    """
    cache: dict[tuple[Any, ...], LlmBackend] = getattr(_thread_local, "cache", None)
    if cache is None:
        cache = {}
        setattr(_thread_local, "cache", cache)

    key = (
        cfg.provider,
        cfg.model,
        cfg.ollama_cloud_model,
        cfg.gemini_api_key,
        cfg.azure_endpoint,
        cfg.azure_api_key,
        cfg.azure_api_version,
    )
    backend = cache.get(key)
    if backend is None:
        backend = make_backend(cfg)
        cache[key] = backend
    return backend


def generate_text_with_retries(backend: LlmBackend, *, prompt: str, max_retries: int) -> str:
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return backend.generate_text(prompt=prompt)
        except Exception as exc:
            last_exc = exc
            if _is_transient_error(exc) and attempt < max_retries - 1:
                time.sleep(_backoff_seconds(attempt))
                continue
            raise
    raise RuntimeError("LLM call failed") from last_exc

