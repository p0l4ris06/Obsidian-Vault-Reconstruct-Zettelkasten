from __future__ import annotations

import re


_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]+`")
_WIKILINK_RE = re.compile(r"\[\[.*?\]\]")


def count_wikilinks(text: str) -> int:
    return len(_WIKILINK_RE.findall(text))


def mask_protected(text: str) -> tuple[str, list[tuple[str, str]]]:
    placeholders: list[tuple[str, str]] = []

    def _replace(m: re.Match) -> str:
        token = f"\x00PH{len(placeholders)}\x00"
        placeholders.append((token, m.group(0)))
        return token

    masked = _FRONTMATTER_RE.sub(_replace, text, count=1)
    masked = _CODE_FENCE_RE.sub(_replace, masked)
    masked = _INLINE_CODE_RE.sub(_replace, masked)
    masked = _WIKILINK_RE.sub(_replace, masked)
    return masked, placeholders


def restore_protected(text: str, placeholders: list[tuple[str, str]]) -> str:
    for token, original in placeholders:
        text = text.replace(token, original)
    return text

