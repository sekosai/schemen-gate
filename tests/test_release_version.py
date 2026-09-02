"""Keep the Schemen Gate package and source metadata on one release version."""

from __future__ import annotations

from pathlib import Path

import schemen_gate

EXPECTED_VERSION = "1.0.2"
ROOT = Path(__file__).resolve().parents[1]


def test_release_version_is_consistent() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f'version = "{EXPECTED_VERSION}"\n' in project
    assert schemen_gate.__version__ == EXPECTED_VERSION
    assert schemen_gate.GATE_VERSION == EXPECTED_VERSION
    assert schemen_gate.current_release_identity().version == EXPECTED_VERSION
