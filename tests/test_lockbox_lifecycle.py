"""Executable evidence for lockbox reissue, diagnostics, and document parsing."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from schemen_gate import GateKey
from schemen_gate._key_wrapping import generate_x25519_keypair
from schemen_gate._lockbox import (
    GrantSpec,
    HierarchyDef,
    Lockbox,
    RegimeCapability,
    create_lockbox,
    diagnose_grants,
    diagnose_operational,
    load_lockbox,
    reissue_lockbox,
    save_lockbox,
    seal_grant,
)

MASTER = GateKey(b"m" * 32)


def _lockbox() -> Lockbox:
    read_value = RegimeCapability(0, "read", ["value"])
    write_value = RegimeCapability(1, "write", ["value"])
    return create_lockbox(
        MASTER,
        chain_hash=hashlib.sha256(b"chain").hexdigest(),
        chain_name="audit-chain",
        n_dims=8,
        n_regimes=2,
        hierarchy_def=[
            HierarchyDef("admin", "both regimes", [0, 1], [read_value, write_value]),
            HierarchyDef("reader", "regime 0 only", [0], [read_value]),
        ],
    )


def _grant(lockbox: Lockbox, recipient: str, level: int) -> bytes:
    _, public_key = generate_x25519_keypair()
    seal_grant(lockbox, MASTER, recipient, lockbox.hierarchy[level], public_key)
    return public_key


def test_diagnose_grants_reports_uncovered_levels_duplicates_and_redundancy() -> None:
    lockbox = _lockbox()
    assert [w for w in diagnose_grants(lockbox) if "has no grants issued" in w] == [
        "Access level 'admin' (level 0) has no grants issued",
        "Access level 'reader' (level 1) has no grants issued",
    ]

    _grant(lockbox, "alice", 0)
    _grant(lockbox, "alice", 1)
    _grant(lockbox, "bob", 0)
    _grant(lockbox, "bob", 0)
    warnings = diagnose_grants(lockbox)

    assert any("Recipient 'alice'" in w and "redundant" in w for w in warnings)
    assert "Recipient 'bob' has duplicate grants at the same access level" in warnings
    assert not any("has no grants issued" in w for w in warnings)


def test_reissue_lockbox_applies_revocations_and_additions_without_signing() -> None:
    source = _lockbox()
    _grant(source, "alice", 0)
    _grant(source, "bob", 0)
    _, carol_public_key = generate_x25519_keypair()

    reissued = reissue_lockbox(
        source,
        MASTER,
        revoke={"bob"},
        add=[GrantSpec("carol", "admin", carol_public_key)],
    )

    assert [g.recipient_id for g in reissued.grants] == ["alice", "carol"]
    assert reissued.authority is None
    assert reissued.hierarchy_hash == source.hierarchy_hash
    assert reissued.gate_release == source.gate_release
    assert [g.recipient_id for g in source.grants] == ["alice", "bob"]
    carol = reissued.grants[1]
    assert carol.access_fingerprint == source.hierarchy[0].fingerprint
    assert {sk.regime_id for sk in carol.sealed_keys} == {0, 1}

    with pytest.raises(ValueError, match="not found in hierarchy"):
        reissue_lockbox(source, MASTER, add=[GrantSpec("dave", "owner", carol_public_key)])


def test_diagnose_operational_flags_revoked_recipients_lineage_and_missing_binding() -> None:
    lockbox = _lockbox()
    _grant(lockbox, "alice", 0)

    warnings = diagnose_operational(
        lockbox,
        revoked_recipients={"alice"},
        expected_chain_hash="0" * 64,
    )

    assert "Grant for revoked recipient 'alice' is still present" in warnings
    assert any(w.startswith("Chain hash mismatch") for w in warnings)
    assert any("unsigned" in w for w in warnings)
    assert any("no model artifact hash" in w for w in warnings)
    assert diagnose_operational(lockbox, expected_chain_hash=lockbox.chain_hash) == [
        "Lockbox is unsigned — no authority section present",
        "Lockbox has no model artifact hash — model binding is missing",
    ]


def test_saved_lockbox_round_trips_through_yaml(tmp_path: Path) -> None:
    # The fixture shares one capability object between two levels; the saved
    # document must still contain no YAML anchors, which the loader rejects.
    lockbox = _lockbox()
    _grant(lockbox, "alice", 1)
    path = tmp_path / "lockbox.yaml"
    save_lockbox(lockbox, path)
    assert "&id" not in path.read_text(encoding="utf-8")

    loaded = load_lockbox(path, require_authority=False)

    assert loaded.hierarchy == lockbox.hierarchy
    assert loaded.hierarchy_hash == lockbox.hierarchy_hash
    assert loaded.grants == lockbox.grants
    assert loaded.gate_release == lockbox.gate_release
    assert loaded.authority is None


def _mutations() -> list[tuple[str, Callable[[dict[str, Any]], None]]]:
    def drop_hierarchy(lb: dict[str, Any]) -> None:
        lb.pop("hierarchy")

    def hierarchy_not_a_list(lb: dict[str, Any]) -> None:
        lb["hierarchy"] = 5

    def capability_missing_regime(lb: dict[str, Any]) -> None:
        lb["hierarchy"][0]["capabilities"][0].pop("regime")

    def grant_not_a_mapping(lb: dict[str, Any]) -> None:
        lb["grants"] = [7]

    def sealed_key_not_base64(lb: dict[str, Any]) -> None:
        lb["grants"][0]["sealed_keys"][0]["nonce"] = "!!not base64!!"

    def authority_missing_fields(lb: dict[str, Any]) -> None:
        lb["authority"] = {"signature": 1}

    def token_release_not_a_mapping(lb: dict[str, Any]) -> None:
        lb["grants"][0]["mask_tokens"][0]["gate_release"] = "1.0.2"

    return [
        ("drop_hierarchy", drop_hierarchy),
        ("hierarchy_not_a_list", hierarchy_not_a_list),
        ("capability_missing_regime", capability_missing_regime),
        ("grant_not_a_mapping", grant_not_a_mapping),
        ("sealed_key_not_base64", sealed_key_not_base64),
        ("authority_missing_fields", authority_missing_fields),
        ("token_release_not_a_mapping", token_release_not_a_mapping),
    ]


@pytest.mark.parametrize(("label", "mutate"), _mutations(), ids=[m[0] for m in _mutations()])
def test_load_lockbox_reports_malformed_structure_as_value_error(
    tmp_path: Path, label: str, mutate: Callable[[dict[str, Any]], None]
) -> None:
    lockbox = _lockbox()
    _grant(lockbox, "alice", 0)
    path = tmp_path / "lockbox.yaml"
    save_lockbox(lockbox, path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))

    mutate(document["schemen_lockbox"])
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid"):
        load_lockbox(path, require_authority=False)


def test_load_lockbox_requires_the_root_section_to_be_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "lockbox.yaml"
    path.write_text("schemen_lockbox: [1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_lockbox(path, require_authority=False)
