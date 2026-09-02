"""GateRights refuses coerced or ambiguous authority fields before signing."""

from __future__ import annotations

import pytest

from schemen_gate import (
    GATE_PACKAGE,
    GATE_SOURCE_REPOSITORY,
    GATE_VERSION,
    GateKey,
    GateReleaseIdentity,
    GateRights,
)

PINNED = GateReleaseIdentity(
    package=GATE_PACKAGE,
    version=GATE_VERSION,
    source_repository=GATE_SOURCE_REPOSITORY,
    source_commit="1" * 40,
)


def _rights(**overrides: object) -> GateRights:
    fields: dict[str, object] = {"regime_id": 1, "can_use": True, "gate_release": PINNED}
    fields.update(overrides)
    return GateRights(**fields)  # type: ignore[arg-type]


def test_well_formed_rights_sign_and_authorize() -> None:
    key = GateKey(b"k" * 32)
    rights = _rights()
    signature = rights.sign(key)
    assert GateRights.verify(rights, signature, key, expected_release=PINNED)
    assert GateRights.authorizes(
        rights,
        signature,
        key,
        "use",
        expected_regime_id=1,
        expected_issuer="root",
        expected_release=PINNED,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"regime_id": True},
        {"regime_id": "1"},
        {"regime_id": -1},
        {"version": 0},
        {"version": True},
        {"can_use": 1},
        {"can_destroy": "yes"},
        {"issuer": ""},
        {"issuer": "root\x00"},
        {"parent_regime": True},
        {"parent_regime": -2},
        {"expires_epoch": True},
        {"expires_epoch": -5},
        {"expires_epoch": "soon"},
        {"metadata": [("k", "v")]},
        {"metadata": {1: "one"}},
        {"gate_release": "1.0.2"},
    ],
)
def test_malformed_authority_fields_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _rights(**overrides)


def test_boolean_regime_cannot_be_signed_to_authorize_regime_one() -> None:
    # Before validation, a signer could sign regime_id=True and a verifier
    # comparing `True == 1` would authorize regime 1.
    with pytest.raises(ValueError, match="exact integer"):
        GateRights(regime_id=True, can_use=True, gate_release=PINNED)  # type: ignore[arg-type]
