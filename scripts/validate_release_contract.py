#!/usr/bin/env python3
"""Validate duplicated release mechanics against release-contract.json."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "release-contract.json"
SEMVER = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)")


def python_constants(path: str) -> dict[str, Any]:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path)
    values: dict[str, Any] = {}
    for statement in tree.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.Constant)
        ):
            values[statement.targets[0].id] = statement.value.value
    return values


def require_contains(failures: list[str], path: str, *needles: str) -> None:
    text = (ROOT / path).read_text(encoding="utf-8")
    for needle in needles:
        if needle not in text:
            failures.append(f"{path} is missing release marker {needle!r}")


def main() -> int:
    try:
        contract: Any = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"release contract is unreadable: {exc}") from exc
    expected_fields = {"schema", "package", "version", "tag", "repository"}
    if not isinstance(contract, dict) or set(contract) != expected_fields:
        raise SystemExit("release contract fields do not match the v1 schema")
    version = contract["version"]
    repository = contract["repository"]
    tag = contract["tag"]
    if (
        contract["schema"] != "schemen/gate-release-contract-v1"
        or contract["package"] != "schemen-gate"
        or not isinstance(version, str)
        or SEMVER.fullmatch(version) is None
        or tag != f"v{version}"
        or repository != "https://github.com/sekosai/schemen-gate"
    ):
        raise SystemExit("release contract values are invalid")

    failures: list[str] = []
    constant_bindings = {
        "src/schemen_gate/_release.py": {
            "GATE_PACKAGE": contract["package"],
            "GATE_VERSION": version,
            "GATE_SOURCE_REPOSITORY": repository,
        },
        "scripts/build_release.py": {
            "EXPECTED_VERSION": version,
            "EXPECTED_REPOSITORY": repository,
        },
        "scripts/stamp_release.py": {"EXPECTED_VERSION": version},
        "scripts/verify_dist.py": {
            "EXPECTED_VERSION": version,
            "EXPECTED_REPOSITORY": repository,
        },
        "scripts/release_check.py": {
            "EXPECTED_VERSION": version,
            "SOURCE_REPOSITORY": repository,
        },
        "tests/test_release_version.py": {"EXPECTED_VERSION": version},
    }
    for path, expected in constant_bindings.items():
        actual = python_constants(path)
        for name, value in expected.items():
            if actual.get(name) != value:
                failures.append(f"{path}:{name} expected {value!r}, found {actual.get(name)!r}")

    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    require_contains(
        failures,
        "pyproject.toml",
        f'name = "{contract["package"]}"',
        f'version = "{version}"',
        f'Repository = "{repository}"',
        f'Issues = "{repository}/issues"',
    )
    if project.count(f'version = "{version}"') != 1:
        failures.append("pyproject.toml has an ambiguous project version")

    research_lock = json.loads(
        (ROOT / "research/cdp/experiments/schemen-library-lock.json").read_text(encoding="utf-8")
    )["libraries"]["schemen-gate"]
    if research_lock.get("version") != version:
        failures.append("research Gate lock version differs from the release contract")
    if research_lock.get("repository") != repository:
        failures.append("research Gate lock repository differs from the release contract")

    require_contains(
        failures,
        "CITATION.cff",
        f"version: {version}",
        f'repository-code: "{repository}"',
        f'url: "{repository}/tree/{tag}"',
    )
    require_contains(
        failures,
        "research/cdp/CITATION.cff",
        f"version: {version}",
        f'repository-code: "{repository}"',
        f'url: "{repository}/blob/{tag}/research/cdp/paper/cdp.pdf"',
    )
    require_contains(
        failures,
        ".github/workflows/ci.yml",
        f'tags: ["{tag}"]',
        f"--version {version}",
        f"SCHEMEN_GATE_SOURCE_REPOSITORY: {repository}",
    )
    require_contains(
        failures,
        ".github/workflows/release-attestation.yml",
        f"head_branch == '{tag}'",
        f"dist/schemen_gate-{version}-py3-none-any.whl",
        f"dist/schemen_gate-{version}.tar.gz",
        f'"version": "{version}"',
        f'"source_repository": "{repository}"',
        f'"source_ref": "refs/tags/{tag}"',
    )
    require_contains(
        failures,
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        f"placeholder: {tag} or full commit hash",
    )
    require_contains(
        failures,
        "CHANGELOG.md",
        f"## {version} - Production-ready release candidate",
        "## 1.0.1 - Historical retained release",
        "## 1.0.0 - Initial release line",
    )
    require_contains(
        failures,
        "research/cdp/CHANGELOG.md",
        f"## {version} - Release candidate",
    )
    require_contains(
        failures,
        "PYPI.md",
        f"{repository}/blob/{tag}/examples/ai_pki_quickstart.py",
        f"{repository}/blob/{tag}/docs/ARCHITECTURE.md",
        f"{repository}/blob/{tag}/docs/X509_PROFILE.md",
        f"{repository}/blob/{tag}/docs/PRODUCTION_DEPLOYMENT.md",
        f"{repository}/blob/{tag}/research/cdp/LICENSES.md",
        f"{repository}/tree/{tag}/research/cdp",
    )
    require_contains(failures, "MANIFEST.in", "include release-contract.json")

    if failures:
        raise SystemExit("release contract validation failed:\n" + "\n".join(failures))
    print(f"Release contract: {contract['package']} {version} at {repository} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
