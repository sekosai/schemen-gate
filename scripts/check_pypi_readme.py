#!/usr/bin/env python3
"""Fail closed on links that would break in the PyPI long description."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPI_README = ROOT / "PYPI.md"
CANONICAL_REPOSITORY = "https://github.com/sekosai/schemen-gate"
LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def main() -> int:
    text = PYPI_README.read_text(encoding="utf-8")
    contract = json.loads((ROOT / "release-contract.json").read_text(encoding="utf-8"))
    tag = contract.get("tag")
    if contract.get("repository") != CANONICAL_REPOSITORY or not isinstance(tag, str):
        raise SystemExit("PyPI README check failed: release contract is invalid")
    immutable_prefixes = (
        f"{CANONICAL_REPOSITORY}/blob/{tag}/",
        f"{CANONICAL_REPOSITORY}/tree/{tag}/",
    )
    operational_targets = {f"{CANONICAL_REPOSITORY}/security/advisories/new"}
    failures: list[str] = []
    for match in (*LINK.finditer(text), *IMAGE.finditer(text)):
        target = match.group(1).strip().split(maxsplit=1)[0]
        if (
            target.startswith(immutable_prefixes)
            or target in operational_targets
            or target == "mailto:ryan@sekos.ai"
        ):
            continue
        line = text[: match.start()].count("\n") + 1
        failures.append(f"PYPI.md:{line}: non-canonical or relative target {target!r}")
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    if 'readme = {file = "PYPI.md", content-type = "text/markdown"}' not in project:
        failures.append("pyproject.toml does not select PYPI.md as Markdown metadata")
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    if "include PYPI.md\n" not in manifest:
        failures.append("MANIFEST.in does not include PYPI.md")
    if failures:
        raise SystemExit("PyPI README check failed:\n" + "\n".join(failures))
    print(f"PyPI README: canonical {tag}-bound links passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
