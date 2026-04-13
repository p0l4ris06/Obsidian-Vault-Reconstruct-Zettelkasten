from __future__ import annotations

import re


_UNSAFE_CHARS = re.compile(r'[\\/:*?"<>|]')


def safe_filename(title: str) -> str:
    return _UNSAFE_CHARS.sub("-", title).strip(". ")[:200] or "Untitled"

