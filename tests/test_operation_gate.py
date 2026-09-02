"""Exact finite-operation AAD, one-use redemption, and sub-gate attenuation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from schemen_gate import (
    GateKey,
    GateReleaseIdentity,
    TokenAuthenticationError,
    TokenExpiredError,
)
from schemen_gate._operation_gate import (
    LEARNED_ROUTE,
    NATIVE_ROUTE,
    OperationGateAttenuationError,
    OperationGateError,
    OperationGatePublicAttestation,
    OperationGateRedemptionReceipt,
    OperationGateReplayError,
    OperationGateToken,
    OperationGateVerifier,
    OperationProposalOrigin,
    OperationTransitionAAD,
    authenticate_operation_gate,
    derive_operation_gate_key,
    issue_operation_gate,
    issue_operation_subgate,
    sign_operation_redemption,
    verify_operation_public_attestation,
    verify_operation_public_attestation_self_consistency,
    verify_operation_redemption,
)

ROOT = GateKey(b"r" * 32)
PINNED_RELEASE = GateReleaseIdentity(
    package="schemen-gate",
    version="1.0.0",
    source_repository="https://github.com/sekosai/schemen-gate",
    source_commit="a" * 40,
)


def _contract(**changes: object) -> OperationTransitionAAD:
    values: dict[str, object] = {
        "language_id": "workflow-language",
        "language_seal": "language-seal",
        "machine_id": "machine-id",
        "machine_seal": "machine-seal",
        "decoder_id": "decoder-id",
        "decoder_seal": "decoder-seal",
        "c_rasp_decision_seal": "decision-seal",
        "c_rasp_status": "MEMBER",
        "route_seal": "route-seal",
        "route_disposition": LEARNED_ROUTE,
        "native_executor_id": "native-transition-factory",
        "learned_proposer_id": "tiny-agent",
        "proposal_origin": OperationProposalOrigin.LEARNED_PROPOSAL,
        "source_state": "ready",
        "source_snapshot_seal": "source-seal",
        "operation_symbol": "approve",
        "target_state": "approved",
        "target_snapshot_seal": "target-seal",
        "transition_id": "transition-id",
        "transition_seal": "transition-seal",
        "sequence": 1,
        "operation_target": "ontology:invoice-7",
        "arguments_sha256": "a" * 64,
        "required_conditions": ("tenant=alpha", "risk<=low"),
        "child_symbols": ("notify", "archive"),
        "delegation_depth_remaining": 2,
        "parent_gate_ref": "",
    }
    values.update(changes)
    return OperationTransitionAAD(**values)  # type: ignore[arg-type]


def _key() -> GateKey:
    return derive_operation_gate_key(ROOT, "workflow-language")


def test_exact_operation_round_trip_and_receipt_signature() -> None:
    contract = _contract()
    token = issue_operation_gate(_key(), contract, issuer_context="agent-ontology")
    verifier = OperationGateVerifier(_key())

    receipt = verifier.redeem(token, expected_contract=contract)

    assert receipt.transition_seal == contract.transition_seal
    assert receipt.aad_id == contract.aad_id
    assert receipt.consumption_index == 0
    assert verify_operation_redemption(receipt, _key())
    assert verifier.consumed_token_refs == (token.token_ref,)
    assert token.expires_epoch is not None


def test_non_expiring_operation_gate_requires_explicit_policy() -> None:
    token = issue_operation_gate(
        _key(),
        _contract(),
        issuer_context="explicit-non-expiring-test",
        expires_epoch=None,
        allow_non_expiring=True,
    )
    assert token.expires_epoch is None


def test_release_version_and_source_commit_are_authenticated_contract_fields() -> None:
    contract = _contract(gate_release=PINNED_RELEASE)
    token = issue_operation_gate(
        _key(),
        contract,
        issuer_context="agent-ontology",
        release_identity=PINNED_RELEASE,
    )
    verifier = OperationGateVerifier(_key(), release_identity=PINNED_RELEASE)
    receipt = verifier.redeem(token, expected_contract=contract)
    signer = Ed25519PrivateKey.from_private_bytes(b"s" * 32)
    attestation = sign_operation_redemption(
        receipt,
        signer,
        signed_at="2026-08-27T21:00:00+00:00",
    )

    assert token.to_dict()["aad"]["gate_release"] == PINNED_RELEASE.to_dict()
    assert receipt.body()["gate_release"] == PINNED_RELEASE.to_dict()
    assert verify_operation_redemption(
        receipt,
        _key(),
        expected_release=PINNED_RELEASE,
    )
    assert verify_operation_public_attestation(
        attestation,
        expected_signer_public_key=attestation.signer_public_key,
        expected_release=PINNED_RELEASE,
    )
    # Standalone verification never degrades to signature-only acceptance:
    # absent an override, it pins to the running Gate release.
    assert not verify_operation_redemption(receipt, _key())
    assert verify_operation_public_attestation_self_consistency(
        attestation,
        expected_release=PINNED_RELEASE,
    )

    other_commit = GateReleaseIdentity(
        package="schemen-gate",
        version="1.0.0",
        source_repository="https://github.com/sekosai/schemen-gate",
        source_commit="b" * 40,
    )
    assert not verify_operation_redemption(
        receipt,
        _key(),
        expected_release=other_commit,
    )
    assert not verify_operation_public_attestation(
        attestation,
        expected_signer_public_key=attestation.signer_public_key,
        expected_release=other_commit,
    )
    with pytest.raises(TokenAuthenticationError, match="contract differs"):
        authenticate_operation_gate(
            token,
            _key(),
            expected_contract=replace(contract, gate_release=other_commit),
            expected_release=other_commit,
        )


def test_release_identity_tampering_breaks_receipt_signature() -> None:
    contract = _contract(gate_release=PINNED_RELEASE)
    token = issue_operation_gate(
        _key(),
        contract,
        issuer_context="agent-ontology",
        release_identity=PINNED_RELEASE,
    )
    receipt = OperationGateVerifier(_key(), release_identity=PINNED_RELEASE).redeem(
        token, expected_contract=contract
    )
    tampered_release = GateReleaseIdentity(
        package="schemen-gate",
        version="1.0.1",
        source_repository="https://github.com/sekosai/schemen-gate",
        source_commit="a" * 40,
    )

    assert not verify_operation_redemption(
        replace(receipt, gate_release=tampered_release),
        _key(),
    )


def test_concurrent_redemption_consumes_token_exactly_once() -> None:
    contract = _contract()
    token = issue_operation_gate(_key(), contract, issuer_context="agent-ontology")
    verifier = OperationGateVerifier(_key())

    def redeem_once() -> str:
        try:
            verifier.redeem(token, expected_contract=contract)
            return "accepted"
        except OperationGateReplayError:
            return "replay"

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(lambda _: redeem_once(), range(64)))

    assert results.count("accepted") == 1
    assert results.count("replay") == 63


def test_operation_integer_fields_reject_floats() -> None:
    with pytest.raises(OperationGateError, match="sequence"):
        _contract(sequence=1.5)


def test_wire_documents_reject_type_coercion_and_unknown_fields() -> None:
    contract = _contract()
    token = issue_operation_gate(_key(), contract, issuer_context="agent-ontology")
    token_document = token.to_dict()
    token_document["expires_epoch"] = "100"
    with pytest.raises(OperationGateError, match="expires_epoch"):
        OperationGateToken.from_dict(token_document)

    aad_document = contract.to_dict()
    aad_document["native_verification_required"] = 1
    with pytest.raises(OperationGateError, match="exact boolean"):
        OperationTransitionAAD.from_dict(aad_document)

    token_document = token.to_dict()
    token_document["untrusted"] = True
    with pytest.raises(OperationGateError, match="fields are not canonical"):
        OperationGateToken.from_dict(token_document)


def test_redemption_can_be_publicly_attested_and_signer_pinned() -> None:
    contract = _contract()
    token = issue_operation_gate(_key(), contract, issuer_context="agent-ontology")
    redemption = OperationGateVerifier(_key()).redeem(token, expected_contract=contract)
    signer = Ed25519PrivateKey.from_private_bytes(b"s" * 32)

    attestation = sign_operation_redemption(
        redemption,
        signer,
        signed_at="2026-08-15T12:00:00+00:00",
    )

    assert verify_operation_public_attestation(
        attestation,
        expected_signer_public_key=attestation.signer_public_key,
    )
    assert OperationGatePublicAttestation.from_dict(attestation.to_dict()) == attestation
    with pytest.raises(TypeError):
        verify_operation_public_attestation(attestation)  # type: ignore[call-arg]
    assert not verify_operation_public_attestation(
        attestation,
        expected_signer_public_key="00" * 32,
    )
    assert not verify_operation_public_attestation(
        replace(attestation, signed_at="2026-08-15T12:00:01+00:00"),
        expected_signer_public_key=attestation.signer_public_key,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_state", "other"),
        ("source_snapshot_seal", "stale"),
        ("operation_symbol", "delete"),
        ("target_state", "rejected"),
        ("target_snapshot_seal", "forged"),
        ("transition_seal", "other-transition"),
        ("c_rasp_decision_seal", "other-decision"),
        ("route_seal", "other-route"),
        ("required_conditions", ("tenant=alpha",)),
        ("child_symbols", ("notify", "destroy")),
    ],
)
def test_expected_contract_drift_is_cryptographically_rejected(field: str, value: object) -> None:
    contract = _contract()
    token = issue_operation_gate(_key(), contract, issuer_context="agent-ontology")
    expected = replace(contract, **{field: value})

    with pytest.raises(TokenAuthenticationError, match="contract differs"):
        authenticate_operation_gate(token, _key(), expected_contract=expected)


def test_token_metadata_or_wrong_key_cannot_authenticate() -> None:
    contract = _contract()
    token = issue_operation_gate(_key(), contract, issuer_context="agent-ontology")
    tampered = replace(token, issuer_context="other-context")

    with pytest.raises(TokenAuthenticationError):
        authenticate_operation_gate(tampered, _key(), expected_contract=contract)
    with pytest.raises(TokenAuthenticationError):
        authenticate_operation_gate(
            token,
            GateKey(b"x" * 32),
            expected_contract=contract,
        )


def test_expired_and_replayed_tokens_fail_closed() -> None:
    contract = _contract()
    token = issue_operation_gate(
        _key(),
        contract,
        issuer_context="agent-ontology",
        expires_epoch=2_000_000_100,
    )
    verifier = OperationGateVerifier(_key())
    receipt = verifier.redeem(token, expected_contract=contract, now_epoch=2_000_000_099)
    assert receipt.consumption_index == 0
    with pytest.raises(OperationGateReplayError):
        verifier.redeem(token, expected_contract=contract, now_epoch=2_000_000_099)

    fresh = issue_operation_gate(
        _key(),
        contract,
        issuer_context="agent-ontology",
        expires_epoch=2_000_000_100,
    )
    with pytest.raises(TokenExpiredError):
        verifier.redeem(fresh, expected_contract=contract, now_epoch=2_000_000_100)


@pytest.mark.parametrize(
    "now_epoch",
    [True, "99", float("nan"), float("inf"), float("-inf"), -1],
)
def test_operation_gate_rejects_invalid_verifier_clock_overrides(now_epoch) -> None:
    contract = _contract()
    token = issue_operation_gate(
        _key(),
        contract,
        issuer_context="agent-ontology",
        expires_epoch=2_000_000_100,
    )
    verifier = OperationGateVerifier(_key())

    with pytest.raises(OperationGateError, match="now_epoch"):
        verifier.redeem(token, expected_contract=contract, now_epoch=now_epoch)

    assert verifier.consumed_token_refs == ()


def test_retained_token_and_redemption_replay_in_exact_order() -> None:
    contract = _contract()
    token = issue_operation_gate(_key(), contract, issuer_context="agent-ontology")
    original = OperationGateVerifier(_key())
    receipt = original.redeem(token, expected_contract=contract)

    restored = OperationGateVerifier(_key())
    restored.restore(
        OperationGateToken.from_dict(token.to_dict()),
        OperationGateRedemptionReceipt.from_dict(receipt.to_dict()),
        expected_contract=OperationTransitionAAD.from_dict(contract.to_dict()),
    )

    assert restored.consumed_token_refs == (token.token_ref,)
    with pytest.raises(OperationGateReplayError):
        restored.restore(token, receipt, expected_contract=contract)


def test_learned_proposal_requires_positive_c_rasp_route() -> None:
    with pytest.raises(OperationGateError, match="does not admit"):
        _contract(route_disposition=NATIVE_ROUTE)

    native = _contract(
        route_disposition=NATIVE_ROUTE,
        proposal_origin=OperationProposalOrigin.NATIVE_EXACT,
    )
    assert native.route_disposition == NATIVE_ROUTE


def _child(parent: OperationGateToken, **changes: object) -> OperationTransitionAAD:
    values: dict[str, object] = {
        "proposal_origin": OperationProposalOrigin.NATIVE_EXACT,
        "source_state": "approved",
        "source_snapshot_seal": "child-source-seal",
        "operation_symbol": "notify",
        "target_state": "notified",
        "target_snapshot_seal": "child-target-seal",
        "transition_id": "child-transition-id",
        "transition_seal": "child-transition-seal",
        "sequence": 2,
        "operation_target": "subagent:mailer",
        "arguments_sha256": "b" * 64,
        "required_conditions": ("tenant=alpha", "risk<=low", "channel=email"),
        "child_symbols": ("archive",),
        "delegation_depth_remaining": 1,
        "parent_gate_ref": parent.token_ref,
    }
    values.update(changes)
    return replace(parent.aad, **values)


def test_subgate_propagates_with_strictly_attenuated_aad() -> None:
    parent_contract = _contract()
    parent = issue_operation_gate(
        _key(),
        parent_contract,
        issuer_context="agent-ontology",
        expires_epoch=2_000_000_200,
    )
    child_contract = _child(parent)

    child_key, child = issue_operation_subgate(
        parent,
        _key(),
        child_contract,
        child_scope="mailer",
        expires_epoch=2_000_000_150,
    )
    receipt = OperationGateVerifier(child_key).redeem(
        child,
        expected_contract=child_contract,
        now_epoch=2_000_000_100,
    )

    assert child.aad.parent_gate_ref == parent.token_ref
    assert set(child.aad.required_conditions) > set(parent.aad.required_conditions)
    assert set(child.aad.child_symbols) < set(parent.aad.child_symbols)
    assert child.aad.delegation_depth_remaining == 1
    assert verify_operation_redemption(receipt, child_key)
    with pytest.raises(TokenAuthenticationError):
        authenticate_operation_gate(child, _key(), expected_contract=child_contract)


@pytest.mark.parametrize(
    "change",
    [
        {"parent_gate_ref": "wrong-parent"},
        {"operation_symbol": "destroy"},
        {"child_symbols": ("archive", "destroy")},
        {"required_conditions": ("tenant=alpha",)},
        {"delegation_depth_remaining": 2},
        {"language_seal": "other-language"},
    ],
)
def test_subgate_authority_expansion_is_refused(change: dict[str, object]) -> None:
    parent = issue_operation_gate(
        _key(),
        _contract(),
        issuer_context="agent-ontology",
        expires_epoch=2_000_000_200,
    )
    with pytest.raises(OperationGateAttenuationError):
        issue_operation_subgate(
            parent,
            _key(),
            _child(parent, **change),
            child_scope="mailer",
            expires_epoch=2_000_000_150,
        )


def test_subgate_cannot_outlive_parent_or_descend_from_depth_zero() -> None:
    parent = issue_operation_gate(
        _key(),
        _contract(),
        issuer_context="agent-ontology",
        expires_epoch=2_000_000_200,
    )
    with pytest.raises(OperationGateAttenuationError, match="expiry"):
        issue_operation_subgate(
            parent,
            _key(),
            _child(parent),
            child_scope="mailer",
            expires_epoch=2_000_000_201,
        )

    terminal_parent = issue_operation_gate(
        _key(),
        _contract(delegation_depth_remaining=0),
        issuer_context="agent-ontology",
    )
    with pytest.raises(OperationGateAttenuationError, match="cannot create"):
        issue_operation_subgate(
            terminal_parent,
            _key(),
            _child(terminal_parent, delegation_depth_remaining=0),
            child_scope="mailer",
        )
