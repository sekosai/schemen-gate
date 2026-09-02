#!/usr/bin/env python3
"""Validate local links in every tracked Markdown document."""

from __future__ import annotations

import re
import subprocess  # nosec B404 - fixed Git argument vector, no shell
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?\[[^\]]*\]\((?P<target><[^>]+>|[^\s)]+)")
REMOTE_SCHEMES = ("http://", "https://", "mailto:")


def tracked_markdown() -> list[Path]:
    result = subprocess.run(  # nosec B603
        ("git", "ls-files", "-z", "--", "*.md"),
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / raw.decode("utf-8") for raw in result.stdout.split(b"\0") if raw]


def prose_lines(document: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    fence: str | None = None
    for number, line in enumerate(document.splitlines(), start=1):
        stripped = line.lstrip()
        marker = (
            "```" if stripped.startswith("```") else "~~~" if stripped.startswith("~~~") else None
        )
        if marker is not None:
            fence = None if fence == marker else marker if fence is None else fence
            continue
        if fence is None:
            lines.append((number, line))
    return lines


def main() -> int:
    failures: list[str] = []
    checked = 0
    root = ROOT.resolve()
    documents = tracked_markdown()
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for line_number, line in prose_lines(text):
            for match in LINK.finditer(line):
                raw_target = match.group("target")
                if raw_target.startswith("<"):
                    raw_target = raw_target[1:-1]
                target = unquote(raw_target).split("#", 1)[0]
                if not target or target.startswith(REMOTE_SCHEMES):
                    continue
                checked += 1
                if target.startswith("/") or "\\" in target or "\x00" in target:
                    failures.append(
                        f"{document.relative_to(ROOT)}:{line_number}: unsafe local link {raw_target!r}"
                    )
                    continue
                candidate = (document.parent / target).resolve()
                try:
                    candidate.relative_to(root)
                except ValueError:
                    failures.append(
                        f"{document.relative_to(ROOT)}:{line_number}: link escapes repository {raw_target!r}"
                    )
                    continue
                if not candidate.exists():
                    failures.append(
                        f"{document.relative_to(ROOT)}:{line_number}: missing local link {raw_target!r}"
                    )

    if failures:
        raise SystemExit("Markdown link check failed:\n" + "\n".join(failures))
    print(
        f"Documentation links: {len(documents)} tracked Markdown files, "
        f"{checked} local targets passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
