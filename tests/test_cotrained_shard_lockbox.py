"""Verification test for the co-trained embedded-weights + lockbox example.

Wraps examples/cotrained_shard_lockbox/demo.py and asserts every V5 claim
it exercises: observed gate-aware task learning (correct-mask ≫ wrong-mask), R4 released
exactness, R5 non-keyed zeros, R1 AAD rejection, custody ciphertext opacity.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch extra required")
crypto = pytest.importorskip("cryptography")

DEMO = Path(__file__).resolve().parent.parent / "examples" / "cotrained_shard_lockbox" / "demo.py"
spec = importlib.util.spec_from_file_location("cotrained_shard_lockbox_demo", DEMO)
demo = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = demo
spec.loader.exec_module(demo)


def test_cotrained_lockbox_all_claims():
    assert demo.main() == 0
