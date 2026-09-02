"""Executable contract for the primary AI-PKI quickstart."""

from __future__ import annotations

import builtins
import runpy
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_ai_pki_quickstart_runs_success_and_denial_paths() -> None:
    result = subprocess.run(
        [sys.executable, ROOT / "examples" / "ai_pki_quickstart.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "certificate -> signed grant -> resolved Regime -> Gate" in result.stdout
    assert "wrong trust root denied before Gate" in result.stdout
    assert "wrong recipient certificate denied before Gate" in result.stdout


def test_modal_quickstart_remote_import_needs_no_repository_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RemoteImage:
        @classmethod
        def debian_slim(cls, **_kwargs: object) -> RemoteImage:
            return cls()

        def uv_pip_install(self, *_packages: str) -> RemoteImage:
            return self

        def env(self, _values: dict[str, str]) -> RemoteImage:
            raise AssertionError("remote import attempted client-side image configuration")

        def add_local_file(self, *_args: object, **_kwargs: object) -> RemoteImage:
            raise AssertionError("remote import attempted a client-side source mount")

    class RemoteApp:
        def __init__(self, _name: str) -> None:
            pass

        def function(self, **_kwargs: object) -> Any:
            return _identity_decorator

        def local_entrypoint(self) -> Any:
            return _identity_decorator

    def _identity_decorator(function: Any) -> Any:
        return function

    fake_modal = types.ModuleType("modal")
    fake_modal.App = RemoteApp  # type: ignore[attr-defined]
    fake_modal.Image = RemoteImage  # type: ignore[attr-defined]
    fake_modal.is_local = lambda: False  # type: ignore[attr-defined]
    fake_modal.fastapi_endpoint = (  # type: ignore[attr-defined]
        lambda **_kwargs: _identity_decorator
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setenv("SCHEMEN_GATE_EXPECTED_VERSION", "1.0.2")
    monkeypatch.setenv(
        "SCHEMEN_GATE_EXPECTED_REPOSITORY",
        "https://github.com/sekosai/schemen-gate",
    )
    monkeypatch.setenv("SCHEMEN_GATE_EXPECTED_COMMIT", "a" * 40)

    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals_: dict[str, Any] | None = None,
        locals_: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "scripts" or name.startswith("scripts."):
            raise ModuleNotFoundError("repository helper is absent remotely")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    namespace = runpy.run_path(
        str(ROOT / "examples" / "modal_quickstart.py"),
        run_name="modal_quickstart_remote_test",
    )

    assert namespace["SOURCE_EXPORT"] is None
    assert namespace["EXPECTED_GATE_VERSION"] == "1.0.2"
    assert namespace["EXPECTED_SOURCE_COMMIT"] == "a" * 40
