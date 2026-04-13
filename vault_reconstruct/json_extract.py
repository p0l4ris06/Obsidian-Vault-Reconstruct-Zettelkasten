from __future__ import annotations

import json
import re
from typing import Any


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip())


def _regex_extract(text: str, open_ch: str, close_ch: str) -> str | None:
    # Simple greedy extract; good enough for typical LLM outputs.
    pattern = re.compile(re.escape(open_ch) + r".*" + re.escape(close_ch), re.DOTALL)
    m = pattern.search(text)
    return m.group(0) if m else None


def extract_json_array(text: str) -> list[Any] | None:
    for candidate in (text, _strip_fences(text), _regex_extract(text, "[", "]")):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, list) else None
        except json.JSONDecodeError:
            continue
    return None


def extract_json_dict(text: str) -> dict[str, Any] | None:
    for candidate in (text, _strip_fences(text), _regex_extract(text, "{", "}")):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            continue
    return None

