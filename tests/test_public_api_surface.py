"""The documented public surface is exactly the eager names plus one lazy table."""

from __future__ import annotations

import importlib

import pytest

import schemen_gate
from schemen_gate import _LAZY_EXPORTS


def _eager_public_names() -> set[str]:
    return {name for name in schemen_gate.__all__ if name in vars(schemen_gate)}


def test_every_public_name_is_eager_or_lazily_mapped_exactly_once() -> None:
    eager = _eager_public_names()
    lazy = set(_LAZY_EXPORTS)

    assert eager | lazy == set(schemen_gate.__all__)
    assert not (eager & lazy)
    assert len(schemen_gate.__all__) == len(set(schemen_gate.__all__))


def test_every_lazy_export_resolves_to_its_declared_module_attribute() -> None:
    for name, (module_name, attribute) in _LAZY_EXPORTS.items():
        module = importlib.import_module(f"schemen_gate.{module_name}")
        assert getattr(schemen_gate, name) is getattr(module, attribute), name


def test_lazy_export_aliases_are_limited_to_the_von_prefix() -> None:
    aliases = {name: target for name, target in _LAZY_EXPORTS.items() if name != target[1]}
    assert aliases
    assert all(
        name.startswith("von_") and module == "_von" for name, (module, _) in aliases.items()
    )


def test_unknown_attribute_is_an_attribute_error() -> None:
    with pytest.raises(AttributeError, match="no attribute"):
        schemen_gate.definitely_not_exported  # noqa: B018
