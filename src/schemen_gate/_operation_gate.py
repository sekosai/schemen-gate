"""AAD-bound gates for exact finite-language operations.

An operation language is any declared finite alphabet with an exact transition
function.  The symbols can be natural-language tokens, task-state edges,
ontology mutations, workflow actions, or tool calls; Schemen Gate does not
assign semantics to them.  It authenticates the *declared operation contract*.

The complete transition contract is AES-256-GCM Additional Authenticated Data
(AAD).  Changing the current state, operation symbol, target state, native
transition receipt, C-RASP disposition, conditions, or delegation lineage
makes redemption cryptographically impossible.

Sub-gates use HKDF-derived child keys and a strict attenuation law:

* required conditions can only accumulate;
* the child alphabet can only shrink;
* delegation depth can only decrease;
* expiry cannot outlive the parent; and
* the exact parent token reference is part of the child AAD.

The learned system remains a proposer.  ``LEARNED_PROPOSAL`` contracts are
valid only when the declared route is
``LEARNED_PROPOSAL_WITH_NATIVE_VERIFICATION``; every admitted operation still
names an exact native transition receipt.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Any

from schemen_gate._release import (
    GateReleaseIdentity,
    current_release_identity,
    release_identity_matches,
)
from schemen_gate._tokens import GateKey, TokenAuthenticationError, TokenExpiredError

OPERATION_TRANSITION_AAD_SCHEMA = "schemen/operation-transition-aad-v2"
# This public domain-separation label is not a credential.
OPERATION_GATE_TOKEN_SCHEMA = "schemen/operation-gate-token-v2"  # nosec B105
OPERATION_GATE_REDEMPTION_SCHEMA = "schemen/operation-gate-redemption-v2"
OPERATION_GATE_PUBLIC_ATTESTATION_SCHEMA = "schemen/operation-gate-public-attestation-v2"

LEARNED_ROUTE = "LEARNED_PROPOSAL_WITH_NATIVE_VERIFICATION"
NATIVE_ROUTE = "NATIVE_EXACT_ONLY"


class OperationGateError(ValueError):
    """Base class for exact-operation gate refusals."""


class OperationGateAttenuationError(OperationGateError):
    """A requested sub-gate would expand its parent's authority."""


class OperationGateReplayError(OperationGateError):
    """A one-use operation token was already consumed."""


class OperationProposalOrigin(str, Enum):
    """Who selected the symbol presented to the native verifier."""

    NATIVE_EXACT = "NATIVE_EXACT"
    LEARNED_PROPOSAL = "LEARNED_PROPOSAL"


def _verification_epoch(now_epoch: object | None) -> float:
    """Resolve a verifier clock override without permitting fail-open values."""

    if now_epoch is None:
        return time.time()
    if isinstance(now_epoch, bool) or not isinstance(now_epoch, Real):
        raise OperationGateError("now_epoch must be a finite non-negative number")
    try:
        resolved = float(now_epoch)
    except (OverflowError, TypeError, ValueError) as exc:
        raise OperationGateError("now_epoch must be a finite non-negative number") from exc
    if not math.isfinite(resolved) or resolved < 0:
        raise OperationGateError("now_epoch must be a finite non-negative number")
    return resolved


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise OperationGateError("operation gate value must be canonical JSON") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _text(value: object, *, name: str, allow_empty: bool = False) -> str:
    if type(value) is not str or "\x00" in value or (not allow_empty and not value):
        qualifier = "an exact string" if allow_empty else "a non-empty exact string"
        raise OperationGateError(f"{name} must be {qualifier} without NUL")
    return value


def _hex_bytes(value: object, *, name: str, length: int | None = None) -> bytes:
    text = _text(value, name=name)
    if len(text) % 2 or any(character not in "0123456789abcdef" for character in text):
        raise OperationGateError(f"{name} must be lowercase hexadecimal")
    decoded = bytes.fromhex(text)
    if length is not None and len(decoded) != length:
        raise OperationGateError(f"{name} must encode exactly {length} bytes")
    return decoded


def _require_document_fields(
    value: dict[str, Any],
    expected: set[str],
    *,
    name: str,
) -> None:
    received = set(value)
    if received != expected:
        missing = sorted(expected - received)
        unknown = sorted(str(field) for field in received - expected)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise OperationGateError(f"{name} fields are not canonical: {', '.join(details)}")


def _canonical_set(values: Iterable[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise OperationGateError(f"{name} must be a collection of exact strings")
    result = tuple(_text(value, name=name) for value in values)
    if len(set(result)) != len(result):
        raise OperationGateError(f"{name} must not contain duplicates")
    return tuple(sorted(result))


@dataclass(frozen=True)
class OperationTransitionAAD:
    """Complete authenticated contract for one exact finite-state operation."""

    language_id: str
    language_seal: str
    machine_id: str
    machine_seal: str
    decoder_id: str
    decoder_seal: str
    c_rasp_decision_seal: str
    c_rasp_status: str
    route_seal: str
    route_disposition: str
    native_executor_id: str
    learned_proposer_id: str
    proposal_origin: OperationProposalOrigin
    source_state: str
    source_snapshot_seal: str
    operation_symbol: str
    target_state: str
    target_snapshot_seal: str
    transition_id: str
    transition_seal: str
    sequence: int
    operation_target: str
    arguments_sha256: str
    required_conditions: tuple[str, ...] = ()
    child_symbols: tuple[str, ...] = ()
    delegation_depth_remaining: int = 0
    parent_gate_ref: str = ""
    native_verification_required: bool = True
    learned_execution_authority: bool = False
    semantic_authority: bool = False
    prediction_authority: bool = False
    gate_release: GateReleaseIdentity = field(default_factory=current_release_identity)

    def __post_init__(self) -> None:
        for field_name in (
            "language_id",
            "language_seal",
            "machine_id",
            "machine_seal",
            "decoder_id",
            "decoder_seal",
            "c_rasp_decision_seal",
            "c_rasp_status",
            "route_seal",
            "route_disposition",
            "native_executor_id",
            "learned_proposer_id",
            "source_state",
            "source_snapshot_seal",
            "operation_symbol",
            "target_state",
            "target_snapshot_seal",
            "transition_id",
            "transition_seal",
            "operation_target",
            "arguments_sha256",
        ):
            _text(getattr(self, field_name), name=field_name)
        if len(self.arguments_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.arguments_sha256
        ):
            raise OperationGateError("arguments_sha256 must be a lowercase SHA-256 digest")
        _text(self.parent_gate_ref, name="parent_gate_ref", allow_empty=True)
        origin = OperationProposalOrigin(self.proposal_origin)
        if self.route_disposition not in {LEARNED_ROUTE, NATIVE_ROUTE}:
            raise OperationGateError("route_disposition is outside the closed route vocabulary")
        if origin is OperationProposalOrigin.LEARNED_PROPOSAL and (
            self.route_disposition != LEARNED_ROUTE
        ):
            raise OperationGateError("C-RASP route does not admit a learned proposal")
        if not self.native_verification_required:
            raise OperationGateError("operation gates require native exact verification")
        if self.learned_execution_authority or self.semantic_authority or self.prediction_authority:
            raise OperationGateError("operation AAD cannot promote learned or semantic authority")
        for field_name in (
            "native_verification_required",
            "learned_execution_authority",
            "semantic_authority",
            "prediction_authority",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise OperationGateError(f"{field_name} must be an exact boolean")
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 1
        ):
            raise OperationGateError("sequence must be an integer >= 1")
        if (
            isinstance(self.delegation_depth_remaining, bool)
            or not isinstance(self.delegation_depth_remaining, int)
            or self.delegation_depth_remaining < 0
        ):
            raise OperationGateError("delegation_depth_remaining must be an integer >= 0")
        conditions = _canonical_set(self.required_conditions, name="required_conditions")
        children = _canonical_set(self.child_symbols, name="child_symbols")
        if not isinstance(self.gate_release, GateReleaseIdentity):
            raise OperationGateError("gate_release must be a validated GateReleaseIdentity")
        object.__setattr__(self, "proposal_origin", origin)
        object.__setattr__(self, "required_conditions", conditions)
        object.__setattr__(self, "child_symbols", children)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": OPERATION_TRANSITION_AAD_SCHEMA,
            "language_id": self.language_id,
            "language_seal": self.language_seal,
            "machine_id": self.machine_id,
            "machine_seal": self.machine_seal,
            "decoder_id": self.decoder_id,
            "decoder_seal": self.decoder_seal,
            "c_rasp_decision_seal": self.c_rasp_decision_seal,
            "c_rasp_status": self.c_rasp_status,
            "route_seal": self.route_seal,
            "route_disposition": self.route_disposition,
            "native_executor_id": self.native_executor_id,
            "learned_proposer_id": self.learned_proposer_id,
            "proposal_origin": self.proposal_origin.value,
            "source_state": self.source_state,
            "source_snapshot_seal": self.source_snapshot_seal,
            "operation_symbol": self.operation_symbol,
            "target_state": self.target_state,
            "target_snapshot_seal": self.target_snapshot_seal,
            "transition_id": self.transition_id,
            "transition_seal": self.transition_seal,
            "sequence": self.sequence,
            "operation_target": self.operation_target,
            "arguments_sha256": self.arguments_sha256,
            "required_conditions": list(self.required_conditions),
            "child_symbols": list(self.child_symbols),
            "delegation_depth_remaining": self.delegation_depth_remaining,
            "parent_gate_ref": self.parent_gate_ref,
            "native_verification_required": self.native_verification_required,
            "learned_execution_authority": self.learned_execution_authority,
            "semantic_authority": self.semantic_authority,
            "prediction_authority": self.prediction_authority,
            "gate_release": self.gate_release.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> OperationTransitionAAD:
        if type(value) is not dict:
            raise OperationGateError("operation-transition AAD must be a dictionary")
        _require_document_fields(
            value,
            set(cls.__dataclass_fields__) | {"schema"},
            name="operation-transition AAD",
        )
        if value.get("schema") != OPERATION_TRANSITION_AAD_SCHEMA:
            raise OperationGateError("unsupported operation-transition AAD schema")
        try:
            fields = dict(value)
            fields.pop("schema")
            if type(fields["required_conditions"]) is not list:
                raise OperationGateError("required_conditions must be a JSON array")
            if type(fields["child_symbols"]) is not list:
                raise OperationGateError("child_symbols must be a JSON array")
            if type(fields["proposal_origin"]) is not str:
                raise OperationGateError("proposal_origin must be an exact string")
            if type(fields["gate_release"]) is not dict:
                raise OperationGateError("gate_release must be a dictionary")
            fields["proposal_origin"] = OperationProposalOrigin(fields["proposal_origin"])
            fields["gate_release"] = GateReleaseIdentity.from_dict(fields["gate_release"])
            return cls(**fields)
        except OperationGateError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise OperationGateError("malformed operation-transition AAD") from exc

    def to_aad(self) -> bytes:
        return _canonical(self.to_dict())

    @property
    def aad_id(self) -> str:
        return _sha256(self.to_aad())


@dataclass(frozen=True)
class OperationGateToken:
    """AES-GCM token whose authentication tag binds an operation contract."""

    aad: OperationTransitionAAD
    issuer_context: str
    nonce: bytes
    ciphertext: bytes
    expires_epoch: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.aad, OperationTransitionAAD):
            raise OperationGateError("aad must be an OperationTransitionAAD")
        _text(self.issuer_context, name="issuer_context")
        if not isinstance(self.nonce, bytes) or len(self.nonce) != 12:
            raise OperationGateError("operation token nonce must be 12 bytes")
        if not isinstance(self.ciphertext, bytes) or not self.ciphertext:
            raise OperationGateError("operation token ciphertext must be non-empty")
        if self.expires_epoch is not None and (
            isinstance(self.expires_epoch, bool)
            or not isinstance(self.expires_epoch, int)
            or self.expires_epoch < 0
        ):
            raise OperationGateError("expires_epoch must be a non-negative integer or None")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": OPERATION_GATE_TOKEN_SCHEMA,
            "aad": self.aad.to_dict(),
            "issuer_context": self.issuer_context,
            "nonce": self.nonce.hex(),
            "ciphertext": self.ciphertext.hex(),
            "expires_epoch": self.expires_epoch,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> OperationGateToken:
        if type(value) is not dict:
            raise OperationGateError("operation-gate token must be a dictionary")
        _require_document_fields(
            value,
            set(cls.__dataclass_fields__) | {"schema"},
            name="operation-gate token",
        )
        if value.get("schema") != OPERATION_GATE_TOKEN_SCHEMA:
            raise OperationGateError("unsupported operation-gate token schema")
        try:
            if type(value["aad"]) is not dict:
                raise OperationGateError("aad must be a dictionary")
            return cls(
                aad=OperationTransitionAAD.from_dict(value["aad"]),
                issuer_context=value["issuer_context"],
                nonce=_hex_bytes(value["nonce"], name="nonce", length=12),
                ciphertext=_hex_bytes(value["ciphertext"], name="ciphertext"),
                expires_epoch=value["expires_epoch"],
            )
        except OperationGateError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise OperationGateError("malformed operation-gate token") from exc

    @property
    def token_ref(self) -> str:
        return _sha256(_canonical(self.to_dict()))


@dataclass(frozen=True)
class OperationGateRedemptionReceipt:
    """HMAC-authenticated proof that one exact operation token was consumed."""

    token_ref: str
    aad_id: str
    transition_seal: str
    parent_gate_ref: str
    issuer_context: str
    consumption_index: int
    gate_signature: str
    gate_release: GateReleaseIdentity

    def __post_init__(self) -> None:
        for field_name in (
            "token_ref",
            "aad_id",
            "transition_seal",
            "issuer_context",
        ):
            _text(getattr(self, field_name), name=field_name)
        _text(self.parent_gate_ref, name="parent_gate_ref", allow_empty=True)
        for field_name in ("token_ref", "aad_id", "gate_signature"):
            value = _text(getattr(self, field_name), name=field_name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise OperationGateError(f"{field_name} must be a lowercase SHA-256 digest")
        if (
            isinstance(self.consumption_index, bool)
            or not isinstance(self.consumption_index, int)
            or self.consumption_index < 0
        ):
            raise OperationGateError("consumption_index must be an integer >= 0")
        if not isinstance(self.gate_release, GateReleaseIdentity):
            raise OperationGateError("gate_release must be a validated GateReleaseIdentity")

    def body(self) -> dict[str, object]:
        return {
            "schema": OPERATION_GATE_REDEMPTION_SCHEMA,
            "token_ref": self.token_ref,
            "aad_id": self.aad_id,
            "transition_seal": self.transition_seal,
            "parent_gate_ref": self.parent_gate_ref,
            "issuer_context": self.issuer_context,
            "consumption_index": self.consumption_index,
            "gate_release": self.gate_release.to_dict(),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.body(), "gate_signature": self.gate_signature}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> OperationGateRedemptionReceipt:
        if type(value) is not dict:
            raise OperationGateError("operation-gate redemption must be a dictionary")
        _require_document_fields(
            value,
            set(cls.__dataclass_fields__) | {"schema"},
            name="operation-gate redemption",
        )
        if value.get("schema") != OPERATION_GATE_REDEMPTION_SCHEMA:
            raise OperationGateError("unsupported operation-gate redemption schema")
        try:
            if type(value["gate_release"]) is not dict:
                raise OperationGateError("gate_release must be a dictionary")
            return cls(
                token_ref=value["token_ref"],
                aad_id=value["aad_id"],
                transition_seal=value["transition_seal"],
                parent_gate_ref=value["parent_gate_ref"],
                issuer_context=value["issuer_context"],
                consumption_index=value["consumption_index"],
                gate_signature=value["gate_signature"],
                gate_release=GateReleaseIdentity.from_dict(value["gate_release"]),
            )
        except OperationGateError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise OperationGateError("malformed operation-gate redemption") from exc

    @property
    def receipt_id(self) -> str:
        return _sha256(_canonical(self.to_dict()))


@dataclass(frozen=True)
class OperationGatePublicAttestation:
    """Publicly verifiable Ed25519 wrapper for a symmetric redemption.

    The HMAC receipt proves possession to peers sharing the gate key.  This
    wrapper separately lets auditors that must not hold that key verify which
    declared signer attested the exact redemption bytes.
    """

    redemption: OperationGateRedemptionReceipt
    signed_at: str
    signer_public_key: str
    signature: str

    def __post_init__(self) -> None:
        if not isinstance(self.redemption, OperationGateRedemptionReceipt):
            raise OperationGateError("redemption must be an OperationGateRedemptionReceipt")
        _text(self.signed_at, name="signed_at")
        for field_name, size in (("signer_public_key", 32), ("signature", 64)):
            value = _text(getattr(self, field_name), name=field_name)
            if len(value) != size * 2 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise OperationGateError(
                    f"{field_name} must encode exactly {size} bytes as lowercase hexadecimal"
                )

    def body(self) -> dict[str, object]:
        return {
            "schema": OPERATION_GATE_PUBLIC_ATTESTATION_SCHEMA,
            "redemption": self.redemption.to_dict(),
            "signed_at": self.signed_at,
            "signer_public_key": self.signer_public_key,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.body(), "signature": self.signature}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> OperationGatePublicAttestation:
        if type(value) is not dict:
            raise OperationGateError("operation-gate attestation must be a dictionary")
        _require_document_fields(
            value,
            set(cls.__dataclass_fields__) | {"schema"},
            name="operation-gate attestation",
        )
        if value.get("schema") != OPERATION_GATE_PUBLIC_ATTESTATION_SCHEMA:
            raise OperationGateError("unsupported operation-gate public attestation schema")
        try:
            if type(value["redemption"]) is not dict:
                raise OperationGateError("redemption must be a dictionary")
            return cls(
                redemption=OperationGateRedemptionReceipt.from_dict(value["redemption"]),
                signed_at=value["signed_at"],
                signer_public_key=value["signer_public_key"],
                signature=value["signature"],
            )
        except OperationGateError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise OperationGateError("malformed operation-gate attestation") from exc

    @property
    def attestation_id(self) -> str:
        return _sha256(_canonical(self.to_dict()))


def sign_operation_redemption(
    receipt: OperationGateRedemptionReceipt,
    private_key: Any,
    *,
    signed_at: str,
) -> OperationGatePublicAttestation:
    """Bind a symmetric redemption into a public Ed25519 custody chain."""
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    timestamp = _text(signed_at, name="signed_at")
    public_key = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    body = {
        "schema": OPERATION_GATE_PUBLIC_ATTESTATION_SCHEMA,
        "redemption": receipt.to_dict(),
        "signed_at": timestamp,
        "signer_public_key": public_key,
    }
    signature = private_key.sign(_canonical(body)).hex()
    return OperationGatePublicAttestation(
        redemption=receipt,
        signed_at=timestamp,
        signer_public_key=public_key,
        signature=signature,
    )


def verify_operation_public_attestation(
    attestation: OperationGatePublicAttestation,
    *,
    expected_signer_public_key: str,
    expected_release: GateReleaseIdentity | None = None,
) -> bool:
    """Verify public custody against the running Gate release.

    ``expected_release`` supports an explicit verifier-owned release identity;
    when omitted, verification remains release-pinned to this installed build.
    """
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    if not expected_signer_public_key or not hmac.compare_digest(
        attestation.signer_public_key, expected_signer_public_key
    ):
        return False
    release = expected_release or current_release_identity()
    if not release_identity_matches(
        attestation.redemption.gate_release,
        release,
        require_source_commit=True,
    ):
        return False
    try:
        public_key = Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(attestation.signer_public_key)
        )
        public_key.verify(
            bytes.fromhex(attestation.signature),
            _canonical(attestation.body()),
        )
        return True
    except (AttributeError, InvalidSignature, TypeError, ValueError):
        return False


def verify_operation_public_attestation_self_consistency(
    attestation: OperationGatePublicAttestation,
    *,
    expected_release: GateReleaseIdentity | None = None,
) -> bool:
    """Verify receipt integrity without treating its embedded key as trusted."""
    return verify_operation_public_attestation(
        attestation,
        expected_signer_public_key=attestation.signer_public_key,
        expected_release=expected_release,
    )


def derive_operation_gate_key(master: GateKey, language_id: str) -> GateKey:
    """Derive the root key for one declared operation language."""

    language = _text(language_id, name="language_id")
    info = b"schemen:v1:operation-language:" + language.encode("utf-8")
    return GateKey(hmac.new(master.secret, info + b"\x01", hashlib.sha256).digest())


def derive_subordinate_operation_key(
    parent_key: GateKey, parent_gate_ref: str, child_scope: str
) -> GateKey:
    """Derive a child-only key from an authenticated parent gate."""

    parent_ref = _text(parent_gate_ref, name="parent_gate_ref")
    scope = _text(child_scope, name="child_scope")
    info = (
        b"schemen:v1:operation-subgate:" + parent_ref.encode("ascii") + b":" + scope.encode("utf-8")
    )
    return GateKey(hmac.new(parent_key.secret, info + b"\x01", hashlib.sha256).digest())


def _token_aad(
    contract: OperationTransitionAAD,
    issuer_context: str,
    expires_epoch: int | None,
) -> bytes:
    return _canonical(
        {
            "schema": OPERATION_GATE_TOKEN_SCHEMA,
            "contract": contract.to_dict(),
            "issuer_context": issuer_context,
            "expires_epoch": expires_epoch,
        }
    )


def issue_operation_gate(
    gate_key: GateKey,
    contract: OperationTransitionAAD,
    *,
    issuer_context: str,
    expires_epoch: int | None = None,
    allow_non_expiring: bool = False,
    release_identity: GateReleaseIdentity | None = None,
) -> OperationGateToken:
    """Issue one AAD-bound exact-operation token."""

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    context = _text(issuer_context, name="issuer_context")
    expected_release = release_identity or current_release_identity()
    if not release_identity_matches(
        contract.gate_release,
        expected_release,
        require_source_commit=True,
    ):
        raise OperationGateError("operation contract Gate release differs from the issuing runtime")
    if expires_epoch is None:
        if allow_non_expiring is not True:
            expires_epoch = int(time.time()) + 3600
    elif isinstance(expires_epoch, bool) or expires_epoch <= int(time.time()):
        raise OperationGateError("expires_epoch must be a future integer")
    plaintext = _canonical(
        {
            "disposition": "ADMIT",
            "aad_id": contract.aad_id,
            "transition_seal": contract.transition_seal,
        }
    )
    import os

    nonce = os.urandom(12)
    ciphertext = AESGCM(gate_key.secret).encrypt(
        nonce,
        plaintext,
        _token_aad(contract, context, expires_epoch),
    )
    return OperationGateToken(contract, context, nonce, ciphertext, expires_epoch)


def authenticate_operation_gate(
    token: OperationGateToken,
    gate_key: GateKey,
    *,
    expected_contract: OperationTransitionAAD,
    now_epoch: float | None = None,
    expected_release: GateReleaseIdentity | None = None,
) -> None:
    """Cryptographically authenticate a token against the caller's contract."""

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    verification_epoch = _verification_epoch(now_epoch)
    release = expected_release or current_release_identity()
    if not release_identity_matches(
        expected_contract.gate_release,
        release,
        require_source_commit=True,
    ):
        raise TokenAuthenticationError(
            "Operation contract Gate release differs from the verifying runtime."
        )
    if token.expires_epoch is not None and verification_epoch >= token.expires_epoch:
        raise TokenExpiredError(f"Operation token expired at epoch {token.expires_epoch}")
    if not hmac.compare_digest(token.aad.to_aad(), expected_contract.to_aad()):
        raise TokenAuthenticationError(
            "Operation contract differs from the expected AAD; the channel does not exist."
        )
    try:
        plaintext = AESGCM(gate_key.secret).decrypt(
            token.nonce,
            token.ciphertext,
            _token_aad(token.aad, token.issuer_context, token.expires_epoch),
        )
    except Exception as exc:
        raise TokenAuthenticationError(
            "Operation token authentication failed; the channel does not exist."
        ) from exc
    expected = _canonical(
        {
            "disposition": "ADMIT",
            "aad_id": expected_contract.aad_id,
            "transition_seal": expected_contract.transition_seal,
        }
    )
    if not hmac.compare_digest(plaintext, expected):
        raise TokenAuthenticationError("Operation token payload does not match its AAD")


def verify_operation_redemption(
    receipt: OperationGateRedemptionReceipt,
    gate_key: GateKey,
    *,
    expected_release: GateReleaseIdentity | None = None,
) -> bool:
    """Verify a receipt's HMAC against the running Gate release by default."""

    release = expected_release or current_release_identity()
    if not release_identity_matches(
        receipt.gate_release,
        release,
        require_source_commit=True,
    ):
        return False
    try:
        expected = hmac.new(gate_key.secret, _canonical(receipt.body()), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, receipt.gate_signature)
    except (AttributeError, TypeError, ValueError):
        return False


class OperationGateVerifier:
    """Stateful one-use verifier for an operation-gate key."""

    def __init__(
        self,
        gate_key: GateKey,
        *,
        release_identity: GateReleaseIdentity | None = None,
    ) -> None:
        self._gate_key = gate_key
        self._release_identity = release_identity or current_release_identity()
        self._consumed: list[str] = []
        self._consumed_set: set[str] = set()
        self._lock = threading.Lock()

    @property
    def consumed_token_refs(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._consumed)

    def redeem(
        self,
        token: OperationGateToken,
        *,
        expected_contract: OperationTransitionAAD,
        now_epoch: float | None = None,
    ) -> OperationGateRedemptionReceipt:
        token_ref = token.token_ref
        with self._lock:
            if token_ref in self._consumed_set:
                raise OperationGateReplayError("operation token has already been consumed")
            authenticate_operation_gate(
                token,
                self._gate_key,
                expected_contract=expected_contract,
                now_epoch=now_epoch,
                expected_release=self._release_identity,
            )
            consumption_index = len(self._consumed)
            body = {
                "schema": OPERATION_GATE_REDEMPTION_SCHEMA,
                "token_ref": token_ref,
                "aad_id": expected_contract.aad_id,
                "transition_seal": expected_contract.transition_seal,
                "parent_gate_ref": expected_contract.parent_gate_ref,
                "issuer_context": token.issuer_context,
                "consumption_index": consumption_index,
                "gate_release": expected_contract.gate_release.to_dict(),
            }
            signature = hmac.new(
                self._gate_key.secret, _canonical(body), hashlib.sha256
            ).hexdigest()
            receipt = OperationGateRedemptionReceipt(
                token_ref=token_ref,
                aad_id=expected_contract.aad_id,
                transition_seal=expected_contract.transition_seal,
                parent_gate_ref=expected_contract.parent_gate_ref,
                issuer_context=token.issuer_context,
                consumption_index=consumption_index,
                gate_signature=signature,
                gate_release=expected_contract.gate_release,
            )
            self._consumed.append(token_ref)
            self._consumed_set.add(token_ref)
            return receipt

    def restore(
        self,
        token: OperationGateToken,
        receipt: OperationGateRedemptionReceipt,
        *,
        expected_contract: OperationTransitionAAD,
    ) -> None:
        """Replay one retained token/receipt into the one-use ledger."""

        with self._lock:
            if receipt.consumption_index != len(self._consumed):
                raise OperationGateReplayError("redemption history is out of order")
            if (
                receipt.token_ref != token.token_ref
                or receipt.aad_id != expected_contract.aad_id
                or receipt.gate_release != expected_contract.gate_release
            ):
                raise OperationGateError("redemption receipt differs from its retained token")
            if receipt.transition_seal != expected_contract.transition_seal:
                raise OperationGateError("redemption receipt differs from its transition")
            authenticate_operation_gate(
                token,
                self._gate_key,
                expected_contract=expected_contract,
                expected_release=self._release_identity,
            )
            if not verify_operation_redemption(
                receipt,
                self._gate_key,
                expected_release=self._release_identity,
            ):
                raise OperationGateError("redemption receipt signature is invalid")
            if receipt.token_ref in self._consumed_set:
                raise OperationGateReplayError("redemption history reuses a token")
            self._consumed.append(receipt.token_ref)
            self._consumed_set.add(receipt.token_ref)


def _same_parent_authority(
    parent: OperationTransitionAAD,
    child: OperationTransitionAAD,
) -> bool:
    fields = (
        "language_id",
        "language_seal",
        "machine_id",
        "machine_seal",
        "decoder_id",
        "decoder_seal",
        "c_rasp_decision_seal",
        "c_rasp_status",
        "route_seal",
        "route_disposition",
        "native_executor_id",
        "learned_proposer_id",
        "native_verification_required",
        "learned_execution_authority",
        "semantic_authority",
        "prediction_authority",
        "gate_release",
    )
    return all(getattr(parent, field) == getattr(child, field) for field in fields)


def issue_operation_subgate(
    parent_token: OperationGateToken,
    parent_key: GateKey,
    child_contract: OperationTransitionAAD,
    *,
    child_scope: str,
    expires_epoch: int | None = None,
) -> tuple[GateKey, OperationGateToken]:
    """Authenticate a parent and issue one strictly attenuated sub-gate."""

    authenticate_operation_gate(
        parent_token,
        parent_key,
        expected_contract=parent_token.aad,
        expected_release=parent_token.aad.gate_release,
    )
    parent = parent_token.aad
    if parent.delegation_depth_remaining < 1:
        raise OperationGateAttenuationError("parent gate cannot create a subordinate")
    if child_contract.parent_gate_ref != parent_token.token_ref:
        raise OperationGateAttenuationError("child AAD does not bind the exact parent gate")
    if not _same_parent_authority(parent, child_contract):
        raise OperationGateAttenuationError("child changes the parent's language authority")
    if child_contract.operation_symbol not in set(parent.child_symbols):
        raise OperationGateAttenuationError("child operation is outside the parent alphabet")
    if not set(child_contract.child_symbols).issubset(parent.child_symbols):
        raise OperationGateAttenuationError("child alphabet expands parent authority")
    if not set(child_contract.required_conditions).issuperset(parent.required_conditions):
        raise OperationGateAttenuationError("child drops a required parent condition")
    if child_contract.delegation_depth_remaining >= parent.delegation_depth_remaining:
        raise OperationGateAttenuationError("child delegation depth is not attenuated")
    if parent_token.expires_epoch is not None and (
        expires_epoch is None or expires_epoch > parent_token.expires_epoch
    ):
        raise OperationGateAttenuationError("child expiry exceeds parent expiry")
    child_key = derive_subordinate_operation_key(
        parent_key,
        parent_token.token_ref,
        child_scope,
    )
    child_token = issue_operation_gate(
        child_key,
        child_contract,
        issuer_context=f"subgate:{parent_token.token_ref}",
        expires_epoch=expires_epoch,
        release_identity=parent.gate_release,
    )
    return child_key, child_token


__all__ = [
    "LEARNED_ROUTE",
    "NATIVE_ROUTE",
    "OPERATION_GATE_REDEMPTION_SCHEMA",
    "OPERATION_GATE_PUBLIC_ATTESTATION_SCHEMA",
    "OPERATION_GATE_TOKEN_SCHEMA",
    "OPERATION_TRANSITION_AAD_SCHEMA",
    "OperationGateAttenuationError",
    "OperationGateError",
    "OperationGateRedemptionReceipt",
    "OperationGatePublicAttestation",
    "OperationGateReplayError",
    "OperationGateToken",
    "OperationGateVerifier",
    "OperationProposalOrigin",
    "OperationTransitionAAD",
    "authenticate_operation_gate",
    "derive_operation_gate_key",
    "derive_subordinate_operation_key",
    "issue_operation_gate",
    "issue_operation_subgate",
    "sign_operation_redemption",
    "verify_operation_public_attestation",
    "verify_operation_public_attestation_self_consistency",
    "verify_operation_redemption",
]
