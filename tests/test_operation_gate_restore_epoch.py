"""Restoring a retained operation admission verifies it at its issue epoch.

``OperationGateVerifier.restore`` replays a token that was already redeemed.
Its expiry must be judged at the epoch the token was live, not at the wall
clock of the restart: a run restored more than one token lifetime later must
still restore.  The signature and the AAD are verified on every path.

Every test drives a fake clock.  Nothing sleeps.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_operation_gate import _child, _contract, _key

from schemen_gate import GateKey, TokenAuthenticationError, TokenExpiredError
from schemen_gate import _operation_gate as operation_gate
from schemen_gate._operation_gate import (
    NATIVE_ROUTE,
    OperationGateError,
    OperationGateRedemptionReceipt,
    OperationGateReplayError,
    OperationGateToken,
    OperationGateVerifier,
    OperationProposalOrigin,
    OperationTransitionAAD,
    _canonical,
    issue_operation_gate,
    issue_operation_subgate,
)

T0 = 2_000_000_000
TTL = 3600
HOUR = 3600

Scenario = tuple[GateKey, OperationGateToken, OperationTransitionAAD]


class FakeClock:
    """Replaces the module's ``time`` so tests never sleep."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, start: float) -> None:
        self.now = float(start)
        monkeypatch.setattr(operation_gate, "time", SimpleNamespace(time=lambda: self.now))


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    return FakeClock(monkeypatch, T0)


def _root_learned() -> Scenario:
    contract = _contract()
    return _key(), issue_operation_gate(_key(), contract, issuer_context="agent"), contract


def _root_native() -> Scenario:
    contract = _contract(
        route_disposition=NATIVE_ROUTE,
        proposal_origin=OperationProposalOrigin.NATIVE_EXACT,
    )
    return _key(), issue_operation_gate(_key(), contract, issuer_context="agent"), contract


def _subgate() -> Scenario:
    parent = issue_operation_gate(_key(), _contract(), issuer_context="agent")
    contract = _child(parent)
    key, token = issue_operation_subgate(
        parent, _key(), contract, child_scope="mailer", expires_epoch=T0 + TTL - 600
    )
    return key, token, contract


SCENARIOS: dict[str, Callable[[], Scenario]] = {
    "learned-root": _root_learned,
    "native-root": _root_native,
    "subgate": _subgate,
}

every_shape = pytest.mark.parametrize("shape", sorted(SCENARIOS))


def _consumed(shape: str) -> tuple[Scenario, OperationGateRedemptionReceipt]:
    key, token, contract = SCENARIOS[shape]()
    receipt = OperationGateVerifier(key).redeem(token, expected_contract=contract)
    return (key, token, contract), receipt


def _forged_receipt(
    key: GateKey, token: OperationGateToken, contract: OperationTransitionAAD
) -> OperationGateRedemptionReceipt:
    """A receipt that is correctly signed for ``token`` by the key holder."""

    unsigned = OperationGateRedemptionReceipt(
        token_ref=token.token_ref,
        aad_id=contract.aad_id,
        transition_seal=contract.transition_seal,
        parent_gate_ref=contract.parent_gate_ref,
        issuer_context=token.issuer_context,
        consumption_index=0,
        gate_signature="0" * 64,
        gate_release=contract.gate_release,
    )
    signature = hmac.new(key.secret, _canonical(unsigned.body()), hashlib.sha256).hexdigest()
    return replace(unsigned, gate_signature=signature)


# --- the bug: an already-consumed admission older than the token lifetime ---


@every_shape
def test_consumed_admission_older_than_one_hour_restores_at_its_issue_epoch(
    clock: FakeClock, shape: str
) -> None:
    (key, token, contract), receipt = _consumed(shape)

    clock.now = T0 + 3 * HOUR  # the process restarts long after the token's expiry

    restored = OperationGateVerifier(key)
    restored.restore(token, receipt, expected_contract=contract, issued_epoch=T0)

    assert restored.consumed_token_refs == (token.token_ref,)
    with pytest.raises(OperationGateReplayError):
        restored.restore(token, receipt, expected_contract=contract, issued_epoch=T0)


@every_shape
def test_restore_after_expiry_without_an_issue_epoch_is_still_refused(
    clock: FakeClock, shape: str
) -> None:
    """Backward compatibility: the legacy call keeps the strict wall-clock check."""

    (key, token, contract), receipt = _consumed(shape)
    clock.now = T0 + 3 * HOUR

    restored = OperationGateVerifier(key)
    with pytest.raises(TokenExpiredError):
        restored.restore(token, receipt, expected_contract=contract)
    assert restored.consumed_token_refs == ()


@every_shape
def test_restore_before_expiry_without_an_issue_epoch_still_works(
    clock: FakeClock, shape: str
) -> None:
    (key, token, contract), receipt = _consumed(shape)
    clock.now = T0 + 60

    restored = OperationGateVerifier(key)
    restored.restore(token, receipt, expected_contract=contract)

    assert restored.consumed_token_refs == (token.token_ref,)


# --- a fresh expired admission is still refused ---


@every_shape
def test_fresh_expired_token_cannot_be_redeemed(clock: FakeClock, shape: str) -> None:
    key, token, contract = SCENARIOS[shape]()
    clock.now = T0 + 2 * HOUR

    verifier = OperationGateVerifier(key)
    with pytest.raises(TokenExpiredError):
        verifier.redeem(token, expected_contract=contract)
    assert verifier.consumed_token_refs == ()


@every_shape
def test_expired_token_cannot_be_restored_by_claiming_a_later_issue_epoch(
    clock: FakeClock, shape: str
) -> None:
    """An unredeemed, expired token with a receipt the key holder signed stays refused."""

    key, token, contract = SCENARIOS[shape]()
    receipt = _forged_receipt(key, token, contract)
    clock.now = T0 + 2 * HOUR
    assert token.expires_epoch is not None

    for claimed in (token.expires_epoch, token.expires_epoch + 1, T0 + 2 * HOUR):
        verifier = OperationGateVerifier(key)
        with pytest.raises(TokenExpiredError):
            verifier.restore(token, receipt, expected_contract=contract, issued_epoch=claimed)
        assert verifier.consumed_token_refs == ()


# --- the signature and the AAD are always verified ---


@every_shape
def test_tampered_receipt_signature_is_refused_on_a_late_restore(
    clock: FakeClock, shape: str
) -> None:
    (key, token, contract), receipt = _consumed(shape)
    flipped = ("0" if receipt.gate_signature[0] != "0" else "1") + receipt.gate_signature[1:]
    tampered = replace(receipt, gate_signature=flipped)
    clock.now = T0 + 3 * HOUR

    verifier = OperationGateVerifier(key)
    with pytest.raises(OperationGateError, match="signature is invalid"):
        verifier.restore(token, tampered, expected_contract=contract, issued_epoch=T0)
    assert verifier.consumed_token_refs == ()


@every_shape
def test_tampered_ciphertext_is_refused_on_a_late_restore(clock: FakeClock, shape: str) -> None:
    (key, token, contract), _ = _consumed(shape)
    flipped = bytes([token.ciphertext[0] ^ 1]) + token.ciphertext[1:]
    tampered = replace(token, ciphertext=flipped)
    receipt = _forged_receipt(key, tampered, contract)
    clock.now = T0 + 3 * HOUR

    verifier = OperationGateVerifier(key)
    with pytest.raises(TokenAuthenticationError):
        verifier.restore(tampered, receipt, expected_contract=contract, issued_epoch=T0)
    assert verifier.consumed_token_refs == ()


@every_shape
def test_tampered_issuer_context_aad_is_refused_on_a_late_restore(
    clock: FakeClock, shape: str
) -> None:
    (key, token, contract), _ = _consumed(shape)
    tampered = replace(token, issuer_context=token.issuer_context + "-forged")
    receipt = _forged_receipt(key, tampered, contract)  # a valid HMAC receipt for the forgery
    clock.now = T0 + 3 * HOUR

    verifier = OperationGateVerifier(key)
    with pytest.raises(TokenAuthenticationError):
        verifier.restore(tampered, receipt, expected_contract=contract, issued_epoch=T0)
    assert verifier.consumed_token_refs == ()


@every_shape
def test_extended_expiry_is_refused_because_it_is_bound_into_the_aad(
    clock: FakeClock, shape: str
) -> None:
    (key, token, contract), _ = _consumed(shape)
    assert token.expires_epoch is not None
    tampered = replace(token, expires_epoch=token.expires_epoch + 10 * HOUR)
    receipt = _forged_receipt(key, tampered, contract)
    clock.now = T0 + 3 * HOUR

    verifier = OperationGateVerifier(key)
    with pytest.raises(TokenAuthenticationError):
        verifier.restore(tampered, receipt, expected_contract=contract, issued_epoch=T0)
    assert verifier.consumed_token_refs == ()


@every_shape
def test_contract_that_differs_from_the_token_aad_is_refused_on_a_late_restore(
    clock: FakeClock, shape: str
) -> None:
    (key, token, contract), receipt = _consumed(shape)
    other = replace(contract, arguments_sha256="c" * 64)
    clock.now = T0 + 3 * HOUR

    verifier = OperationGateVerifier(key)
    with pytest.raises(OperationGateError):
        verifier.restore(token, receipt, expected_contract=other, issued_epoch=T0)
    assert verifier.consumed_token_refs == ()


def test_wrong_gate_key_is_refused_on_a_late_restore(clock: FakeClock) -> None:
    (key, token, contract), receipt = _consumed("learned-root")
    clock.now = T0 + 3 * HOUR

    verifier = OperationGateVerifier(GateKey(b"w" * 32))
    with pytest.raises(TokenAuthenticationError):
        verifier.restore(token, receipt, expected_contract=contract, issued_epoch=T0)
    assert verifier.consumed_token_refs == ()


# --- a missing or invalid issue time is refused ---


@pytest.mark.parametrize(
    "issued",
    [True, False, "2000000000", b"x", float("nan"), float("inf"), float("-inf"), -1, object()],
)
def test_invalid_issue_epoch_is_refused(clock: FakeClock, issued: object) -> None:
    (key, token, contract), receipt = _consumed("learned-root")
    clock.now = T0 + 3 * HOUR

    verifier = OperationGateVerifier(key)
    with pytest.raises(OperationGateError, match="issued_epoch"):
        verifier.restore(
            token,
            receipt,
            expected_contract=contract,
            issued_epoch=issued,  # type: ignore[arg-type]
        )
    assert verifier.consumed_token_refs == ()


@every_shape
def test_issue_epoch_at_or_after_expiry_is_refused(clock: FakeClock, shape: str) -> None:
    (key, token, contract), receipt = _consumed(shape)
    assert token.expires_epoch is not None
    clock.now = T0 + 3 * HOUR

    for issued in (token.expires_epoch, token.expires_epoch + 1, token.expires_epoch + 10 * HOUR):
        verifier = OperationGateVerifier(key)
        with pytest.raises(TokenExpiredError):
            verifier.restore(token, receipt, expected_contract=contract, issued_epoch=issued)
        assert verifier.consumed_token_refs == ()


@every_shape
def test_issue_epoch_just_before_expiry_is_the_last_one_accepted(
    clock: FakeClock, shape: str
) -> None:
    (key, token, contract), receipt = _consumed(shape)
    assert token.expires_epoch is not None
    clock.now = T0 + 3 * HOUR

    verifier = OperationGateVerifier(key)
    verifier.restore(
        token, receipt, expected_contract=contract, issued_epoch=token.expires_epoch - 1
    )
    assert verifier.consumed_token_refs == (token.token_ref,)


def test_non_expiring_token_restores_with_a_valid_issue_epoch(clock: FakeClock) -> None:
    contract = _contract()
    token = issue_operation_gate(_key(), contract, issuer_context="agent", allow_non_expiring=True)
    receipt = OperationGateVerifier(_key()).redeem(token, expected_contract=contract)
    clock.now = T0 + 100 * HOUR

    restored = OperationGateVerifier(_key())
    restored.restore(token, receipt, expected_contract=contract, issued_epoch=T0)
    assert restored.consumed_token_refs == (token.token_ref,)


def test_replayed_history_keeps_order_and_reuse_checks_with_an_issue_epoch(
    clock: FakeClock,
) -> None:
    key, token, contract = _root_learned()
    receipt = OperationGateVerifier(key).redeem(token, expected_contract=contract)
    clock.now = T0 + 3 * HOUR

    out_of_order = OperationGateVerifier(key)
    out_of_order._consumed.append("x" * 64)  # a ledger that already holds one entry
    with pytest.raises(OperationGateReplayError, match="out of order"):
        out_of_order.restore(token, receipt, expected_contract=contract, issued_epoch=T0)
