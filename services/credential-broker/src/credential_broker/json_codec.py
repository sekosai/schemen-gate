"""Unambiguous finite JSON objects on authorization boundaries."""

from __future__ import annotations

import json
import re
from typing import Any, NoReturn

from .models import canonical

MAX_DEPTH = 64
# Strings are skipped whole so brackets inside them do not count as nesting.
_NESTING = re.compile(rb'"(?:[^"\\]|\\.)*"|[\[\]{}]', re.DOTALL)


def check_depth(raw: str | bytes) -> None:
    """Bound nesting explicitly: Python 3.14's json no longer hits RecursionError."""
    if isinstance(raw, str):
        raw = raw.encode("utf-8", "surrogatepass")
    depth = 0
    for match in _NESTING.finditer(raw):
        token = match.group()
        if token in (b"[", b"{"):
            depth += 1
            if depth > MAX_DEPTH:
                raise ValueError("JSON nesting too deep")
        elif token in (b"]", b"}"):
            depth -= 1


def strict_object(raw: bytes) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(value: str) -> NoReturn:
        raise ValueError("nonfinite JSON")

    check_depth(raw)
    data = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid_constant)
    if not isinstance(data, dict):
        raise ValueError("object required")
    canonical(data)  # Also reject numeric overflow to infinity (e.g. JSON 1e999).
    return data
