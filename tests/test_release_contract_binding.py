"""Release identity is authenticated across every Gate contract family."""

from __future__ import annotations

from dataclasses import replace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from schemen_gate import (
    AdapterToken,
    CapabilityToken,
    GateKey,
    GateReleaseIdentity,
    GateRights,
    HierarchyDef,
    RegimeCapability,
    TokenAuthenticationError,
    compute_lockbox_hash,
    create_lockbox,
    derive_tenant_key,
    issue_adapter_token,
    issue_mask_token,
    redeem_adapter_token,
    redeem_mask_token,
    sign_capability,
    verify_capability,
)

RELEASE_A = GateReleaseIdentity(
    package="schemen-gate",
    version="1.0.0",
    source_repository="https://github.com/sekosai/schemen-gate",
    source_commit="a" * 40,
)
RELEASE_B = replace(RELEASE_A, source_commit="b" * 40)
MASTER = GateKey(b"r" * 32)


def test_mask_token_aad_and_runtime_expectation_bind_release() -> None:
    token = issue_mask_token(
        MASTER,
        "tenant",
        0,
        8,
        release_identity=RELEASE_A,
    )
    tenant_key = derive_tenant_key(MASTER, 0, "tenant")
    assert (
        redeem_mask_token(
            token,
            tenant_key,
            expected_release=RELEASE_A,
        ).sum()
        == 4
    )
    with pytest.raises(TokenAuthenticationError, match="Gate release differs"):
        redeem_mask_token(token, tenant_key, expected_release=RELEASE_B)

    tampered = replace(token, gate_release=RELEASE_B)
    with pytest.raises(TokenAuthenticationError, match="authentication failed"):
        redeem_mask_token(tampered, tenant_key, expected_release=RELEASE_B)


def test_adapter_token_aad_and_runtime_expectation_bind_release() -> None:
    weights = b"release-bound-weights"
    token = issue_adapter_token(
        MASTER,
        "tenant",
        0,
        8,
        2,
        weights,
        "8:2:8",
        release_identity=RELEASE_A,
    )
    tenant_key = derive_tenant_key(MASTER, 0, "tenant")
    assert (
        redeem_adapter_token(
            token,
            tenant_key,
            expected_release=RELEASE_A,
        )
        == weights
    )
    with pytest.raises(TokenAuthenticationError, match="Gate release differs"):
        redeem_adapter_token(token, tenant_key, expected_release=RELEASE_B)

    tampered = AdapterToken(**{**token.__dict__, "gate_release": RELEASE_B})
    with pytest.raises(TokenAuthenticationError, match="authentication failed"):
        redeem_adapter_token(tampered, tenant_key, expected_release=RELEASE_B)


def test_rights_hmac_and_capability_signature_bind_release() -> None:
    rights = GateRights(regime_id=0, gate_release=RELEASE_A)
    signature = rights.sign(MASTER)
    assert GateRights.verify(
        rights,
        signature,
        MASTER,
        expected_release=RELEASE_A,
    )
    assert not GateRights.verify(
        rights,
        signature,
        MASTER,
        expected_release=RELEASE_B,
    )
    assert not GateRights.verify(
        replace(rights, gate_release=RELEASE_B),
        signature,
        MASTER,
        expected_release=RELEASE_B,
    )

    signer = Ed25519PrivateKey.from_private_bytes(b"s" * 32)
    capability = sign_capability(
        CapabilityToken(
            action="read",
            target="resource",
            phase="execute",
            iteration=1,
            nonce="nonce",
            delegation_ref="delegation",
            gate_release=RELEASE_A,
        ),
        signer,
    )
    assert verify_capability(
        capability,
        signer.public_key(),
        expected_release=RELEASE_A,
    )
    assert not verify_capability(
        capability,
        signer.public_key(),
        expected_release=RELEASE_B,
    )
    assert not verify_capability(
        replace(capability, gate_release=RELEASE_B),
        signer.public_key(),
        expected_release=RELEASE_B,
    )


def test_unstamped_release_cannot_authorize_operational_contracts() -> None:
    signer = Ed25519PrivateKey.from_private_bytes(b"u" * 32)
    unstamped = replace(RELEASE_A, source_commit=None)
    capability = sign_capability(
        CapabilityToken(
            action="read",
            target="resource",
            phase="execute",
            iteration=1,
            nonce="nonce",
            delegation_ref="delegation",
            gate_release=unstamped,
        ),
        signer,
    )
    assert not verify_capability(
        capability,
        signer.public_key(),
        expected_release=unstamped,
    )


def test_authority_signed_lockbox_digest_binds_release() -> None:
    lockbox = create_lockbox(
        MASTER,
        chain_hash="c" * 64,
        chain_name="test",
        n_dims=8,
        n_regimes=2,
        hierarchy_def=[
            HierarchyDef(
                name="all",
                description="all regimes",
                regimes=[0, 1],
                capabilities=[RegimeCapability(0, "read", ["value"])],
            ),
            HierarchyDef(
                name="one",
                description="one regime",
                regimes=[0],
                capabilities=[RegimeCapability(0, "read", ["value"])],
            ),
        ],
        release_identity=RELEASE_A,
    )
    assert lockbox.version == "2"
    assert lockbox.gate_release == RELEASE_A
    assert compute_lockbox_hash(lockbox) != compute_lockbox_hash(
        replace(lockbox, gate_release=RELEASE_B)
    )
