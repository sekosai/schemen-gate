"""Standalone capability and attestation tokens for Schemen Gate.

The token chain covers authority delegation, action capability, execution
attestation, phase transition, and revocation. All signatures use Ed25519 over
canonical JSON. Version 2 delegations bind distinct authority, policy, and
execution-attestation public keys; the policy private key is derived from
non-public authority custody and cannot be reconstructed from a certificate or
signature.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
)

from schemen_gate._release import (
    GateReleaseIdentity,
    current_release_identity,
    release_identity_matches,
)
from schemen_gate._tokens import hkdf_expand_sha256

_LEGACY_POLICY_KEY_INFO = b"policy-engine:v1"
_BOUND_POLICY_KEY_INFO = b"schemen-gate:policy-engine:v2\0"


# ---------------------------------------------------------------------------
# Canonical serialization
# ---------------------------------------------------------------------------


def canonicalize(payload: dict[str, Any]) -> bytes:
    """Stable JSON bytes for signing: sorted keys, compact separators."""
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("capability payload must be strict canonical JSON") from exc


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _signable(token_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the token dict with signature removed."""
    d = dict(token_dict)
    d.pop("signature", None)
    return d


def _sign_dict(d: dict[str, Any], private_key: Ed25519PrivateKey) -> str:
    """Sign a dict (excluding 'signature' field) and return hex signature."""
    canonical = canonicalize(_signable(d))
    return private_key.sign(canonical).hex()


def _verify_dict(d: dict[str, Any], signature_hex: str, public_key: Ed25519PublicKey) -> bool:
    """Verify a hex Ed25519 signature over the canonical dict."""
    canonical = canonicalize(_signable(d))
    try:
        public_key.verify(bytes.fromhex(signature_hex), canonical)
        return True
    except (InvalidSignature, ValueError):
        return False


def _public_key_hex(key: Ed25519PublicKey) -> str:
    return key.public_bytes(Encoding.Raw, PublicFormat.Raw).hex()


def _public_key_from_hex(hex_str: str) -> Ed25519PublicKey:
    return Ed25519PublicKey.from_public_bytes(bytes.fromhex(hex_str))


# ---------------------------------------------------------------------------
# Key derivation
# ---------------------------------------------------------------------------


def derive_policy_key(delegation_signature: str) -> Ed25519PrivateKey:
    """Derive the legacy v1 policy key from a delegation signature.

    .. warning::
       This is a compatibility decoder, not a secure authority primitive.
       A delegation signature is public, so every holder can derive this key.
       Enforced execution MUST require a v2 delegation and
       :func:`derive_bound_policy_key` instead.
    """
    sig_bytes = bytes.fromhex(delegation_signature)
    key_material = hkdf_expand_sha256(sig_bytes[:32], _LEGACY_POLICY_KEY_INFO, length=32)
    return Ed25519PrivateKey.from_private_bytes(key_material)


def derive_policy_verify_key(delegation_signature: str) -> Ed25519PublicKey:
    """Return the legacy v1 policy key (historical verification only)."""
    return derive_policy_key(delegation_signature).public_key()


def derive_bound_policy_key(
    authority_secret: bytes,
    mission_id: str,
) -> Ed25519PrivateKey:
    """Derive a mission-bound v2 policy key from non-public authority custody.

    ``authority_secret`` must be secret, high-entropy material held by the
    delegated-authority process.  The mission identifier is domain-separated
    into the derivation so one leaked mission key does not authorize another.
    Neither the signed delegation nor its signature contains enough material
    to reproduce the private key.
    """
    if not isinstance(authority_secret, bytes) or len(authority_secret) < 32:
        raise ValueError("authority_secret must contain at least 32 bytes")
    if type(mission_id) is not str or not mission_id or "\x00" in mission_id:
        raise ValueError("mission_id must be a non-empty exact string without NUL")
    seed = hkdf_expand_sha256(
        authority_secret,
        _BOUND_POLICY_KEY_INFO + mission_id.encode("utf-8"),
        length=32,
    )
    return Ed25519PrivateKey.from_private_bytes(seed)


# ---------------------------------------------------------------------------
# Token: DelegationCertificate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DelegationCertificate:
    """Signed by the DA at mission start. Root of the trust chain."""

    mission_id: str
    principal: str
    granted_phases: list[str]
    granted_tools: list[str]
    granted_paths: list[str]
    gate_rights: dict[str, Any]
    max_iterations: int
    issued_at: str
    expires_at: str
    da_public_key: str
    authority_version: int = 1
    policy_public_key: str = ""
    runtime_public_key: str = ""
    signature: str = ""
    gate_release: GateReleaseIdentity = field(default_factory=current_release_identity)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "principal": self.principal,
            "granted_phases": list(self.granted_phases),
            "granted_tools": list(self.granted_tools),
            "granted_paths": list(self.granted_paths),
            "gate_rights": dict(self.gate_rights),
            "max_iterations": self.max_iterations,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "da_public_key": self.da_public_key,
            "authority_version": self.authority_version,
            "policy_public_key": self.policy_public_key,
            "runtime_public_key": self.runtime_public_key,
            "gate_release": self.gate_release.to_dict(),
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DelegationCertificate:
        return cls(
            mission_id=d["mission_id"],
            principal=d["principal"],
            granted_phases=d["granted_phases"],
            granted_tools=d["granted_tools"],
            granted_paths=d["granted_paths"],
            gate_rights=d["gate_rights"],
            max_iterations=d["max_iterations"],
            issued_at=d["issued_at"],
            expires_at=d["expires_at"],
            da_public_key=d["da_public_key"],
            authority_version=d.get("authority_version", 1),
            policy_public_key=d.get("policy_public_key", ""),
            runtime_public_key=d.get("runtime_public_key", ""),
            gate_release=GateReleaseIdentity.from_dict(dict(d["gate_release"])),
            signature=d.get("signature", ""),
        )

    def ref(self) -> str:
        """SHA-256 of the canonical signed form, used as a chain reference."""
        return _sha256_hex(canonicalize(self.to_dict()))


def sign_delegation(
    cert: DelegationCertificate,
    private_key: Ed25519PrivateKey,
) -> DelegationCertificate:
    """Sign a DelegationCertificate, returning a new instance with signature."""
    d = cert.to_dict()
    sig = _sign_dict(d, private_key)
    return DelegationCertificate(**{**d, "gate_release": cert.gate_release, "signature": sig})


def verify_delegation(
    cert: DelegationCertificate,
    *,
    expected_da_public_key: str,
    expected_release: GateReleaseIdentity | None = None,
) -> bool:
    """Verify a delegation against a verifier-owned DA trust anchor."""
    if not expected_da_public_key or not cert.signature or not cert.da_public_key:
        return False
    release = expected_release or current_release_identity()
    if not release_identity_matches(
        cert.gate_release,
        release,
        require_source_commit=True,
    ):
        return False
    if not hmac.compare_digest(cert.da_public_key, expected_da_public_key):
        return False
    try:
        pub = _public_key_from_hex(cert.da_public_key)
        return _verify_dict(cert.to_dict(), cert.signature, pub)
    except ValueError:
        return False


def verify_delegation_self_consistency(
    cert: DelegationCertificate,
    *,
    expected_release: GateReleaseIdentity | None = None,
) -> bool:
    """Verify historical signature integrity without establishing authority."""
    if not cert.signature or not cert.da_public_key:
        return False
    release = expected_release or current_release_identity()
    if not release_identity_matches(cert.gate_release, release):
        return False
    try:
        return _verify_dict(
            cert.to_dict(),
            cert.signature,
            _public_key_from_hex(cert.da_public_key),
        )
    except ValueError:
        return False


def verify_enforced_delegation(
    cert: DelegationCertificate,
    *,
    expected_da_public_key: str,
    now: datetime | None = None,
    expected_release: GateReleaseIdentity | None = None,
) -> bool:
    """Verify a delegation suitable for downstream enforced execution.

    Historical v1 certificates remain structurally readable and verifiable,
    but are rejected here because their policy private key is public-derived
    and they do not bind a runtime attestation identity.
    """
    if not expected_da_public_key or not cert.mission_id or not cert.principal:
        return False
    if not hmac.compare_digest(cert.da_public_key, expected_da_public_key):
        return False
    if (
        not verify_delegation(
            cert,
            expected_da_public_key=expected_da_public_key,
            expected_release=expected_release,
        )
        or cert.authority_version != 2
    ):
        return False
    if (
        isinstance(cert.max_iterations, bool)
        or not isinstance(cert.max_iterations, int)
        or cert.max_iterations <= 0
        or not cert.granted_phases
        or not cert.granted_tools
        or not cert.granted_paths
    ):
        return False
    try:
        issued = datetime.fromisoformat(cert.issued_at)
        expires = datetime.fromisoformat(cert.expires_at)
        if issued.tzinfo is None or expires.tzinfo is None:
            return False
        check_time = now or datetime.now(timezone.utc)
        if check_time.tzinfo is None:
            return False
        if issued > check_time or expires <= check_time or expires <= issued:
            return False
        policy_key = _public_key_from_hex(cert.policy_public_key)
        runtime_key = _public_key_from_hex(cert.runtime_public_key)
        policy_hex = _public_key_hex(policy_key)
        runtime_hex = _public_key_hex(runtime_key)
        if len({cert.da_public_key, policy_hex, runtime_hex}) != 3:
            return False
    except (TypeError, ValueError):
        return False
    return True


def delegation_policy_verify_key(
    cert: DelegationCertificate,
    *,
    expected_da_public_key: str,
) -> Ed25519PublicKey:
    """Return the signed v2 policy verification key, failing closed for v1."""
    if not verify_enforced_delegation(cert, expected_da_public_key=expected_da_public_key):
        raise ValueError("enforced execution requires a valid v2 delegation")
    return _public_key_from_hex(cert.policy_public_key)


def delegation_runtime_verify_key(
    cert: DelegationCertificate,
    *,
    expected_da_public_key: str,
) -> Ed25519PublicKey:
    """Return the signed v2 runtime verification key, failing closed for v1."""
    if not verify_enforced_delegation(cert, expected_da_public_key=expected_da_public_key):
        raise ValueError("enforced execution requires a valid v2 delegation")
    return _public_key_from_hex(cert.runtime_public_key)


# ---------------------------------------------------------------------------
# Token: CapabilityToken
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapabilityToken:
    """Signed by the policy engine per tool action."""

    action: str
    target: str
    phase: str
    iteration: int
    nonce: str
    delegation_ref: str
    constraints: dict[str, Any] = field(default_factory=dict)
    issued_at: str = ""
    signature: str = ""
    gate_release: GateReleaseIdentity = field(default_factory=current_release_identity)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "target": self.target,
            "phase": self.phase,
            "iteration": self.iteration,
            "nonce": self.nonce,
            "delegation_ref": self.delegation_ref,
            "constraints": dict(self.constraints),
            "issued_at": self.issued_at,
            "gate_release": self.gate_release.to_dict(),
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CapabilityToken:
        return cls(
            action=d["action"],
            target=d["target"],
            phase=d["phase"],
            iteration=d["iteration"],
            nonce=d["nonce"],
            delegation_ref=d["delegation_ref"],
            constraints=d.get("constraints", {}),
            issued_at=d.get("issued_at", ""),
            gate_release=GateReleaseIdentity.from_dict(dict(d["gate_release"])),
            signature=d.get("signature", ""),
        )

    def ref(self) -> str:
        """SHA-256 of the canonical signed form."""
        return _sha256_hex(canonicalize(self.to_dict()))


def make_nonce() -> str:
    """Generate a cryptographically random nonce for capability tokens."""
    return os.urandom(16).hex()


def sign_capability(
    token: CapabilityToken,
    private_key: Ed25519PrivateKey,
) -> CapabilityToken:
    """Sign a CapabilityToken, returning a new instance with signature."""
    d = token.to_dict()
    sig = _sign_dict(d, private_key)
    return CapabilityToken(**{**d, "gate_release": token.gate_release, "signature": sig})


def verify_capability(
    token: CapabilityToken,
    public_key: Ed25519PublicKey,
    *,
    expected_release: GateReleaseIdentity | None = None,
) -> bool:
    """Verify a CapabilityToken against the policy engine's public key."""
    if not token.signature:
        return False
    release = expected_release or current_release_identity()
    if not release_identity_matches(
        token.gate_release,
        release,
        require_source_commit=True,
    ):
        return False
    return _verify_dict(token.to_dict(), token.signature, public_key)


# ---------------------------------------------------------------------------
# Token: AttestationReceipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttestationReceipt:
    """Signed by the runtime after tool execution."""

    action: str
    target: str
    capability_ref: str
    result_hash: str
    success: bool
    executed_at: str
    content_hash: str = ""
    exit_code: int | None = None
    runtime_public_key: str = ""
    signature: str = ""
    gate_release: GateReleaseIdentity = field(default_factory=current_release_identity)

    def __post_init__(self) -> None:
        for name in ("action", "target", "capability_ref", "result_hash"):
            value = getattr(self, name)
            if type(value) is not str or not value or "\x00" in value:
                raise ValueError(f"{name} must be a non-empty exact string without NUL")
        if type(self.success) is not bool:
            raise ValueError("success must be an exact boolean")
        if type(self.executed_at) is not str or not self.executed_at:
            raise ValueError("executed_at must be a timezone-aware ISO 8601 timestamp")
        try:
            executed = datetime.fromisoformat(self.executed_at)
        except ValueError as exc:
            raise ValueError("executed_at must be a timezone-aware ISO 8601 timestamp") from exc
        if executed.tzinfo is None:
            raise ValueError("executed_at must be a timezone-aware ISO 8601 timestamp")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "target": self.target,
            "capability_ref": self.capability_ref,
            "result_hash": self.result_hash,
            "content_hash": self.content_hash,
            "exit_code": self.exit_code,
            "success": self.success,
            "executed_at": self.executed_at,
            "runtime_public_key": self.runtime_public_key,
            "gate_release": self.gate_release.to_dict(),
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AttestationReceipt:
        if type(d) is not dict:
            raise ValueError("attestation receipt must be a dictionary")
        required = {
            "action",
            "target",
            "capability_ref",
            "result_hash",
            "success",
            "executed_at",
            "gate_release",
        }
        allowed = required | {
            "content_hash",
            "exit_code",
            "runtime_public_key",
            "signature",
        }
        if not required.issubset(d) or set(d) - allowed:
            raise ValueError("attestation receipt has missing or unknown fields")
        return cls(
            action=d["action"],
            target=d["target"],
            capability_ref=d["capability_ref"],
            result_hash=d["result_hash"],
            content_hash=d.get("content_hash", ""),
            exit_code=d.get("exit_code"),
            success=d["success"],
            executed_at=d["executed_at"],
            runtime_public_key=d.get("runtime_public_key", ""),
            gate_release=GateReleaseIdentity.from_dict(dict(d["gate_release"])),
            signature=d.get("signature", ""),
        )

    def ref(self) -> str:
        return _sha256_hex(canonicalize(self.to_dict()))


def sign_attestation(
    receipt: AttestationReceipt,
    private_key: Ed25519PrivateKey,
) -> AttestationReceipt:
    """Sign an AttestationReceipt, returning a new instance with signature."""
    pub_hex = _public_key_hex(private_key.public_key())
    d = receipt.to_dict()
    d["runtime_public_key"] = pub_hex
    sig = _sign_dict(d, private_key)
    return AttestationReceipt(**{**d, "gate_release": receipt.gate_release, "signature": sig})


def verify_attestation(
    receipt: AttestationReceipt,
    expected_runtime_public_key: str,
    *,
    expected_release: GateReleaseIdentity | None = None,
) -> bool:
    """Verify an attestation against a verifier-owned runtime key."""
    if not expected_runtime_public_key or not receipt.signature or not receipt.runtime_public_key:
        return False
    release = expected_release or current_release_identity()
    if not release_identity_matches(
        receipt.gate_release,
        release,
        require_source_commit=True,
    ):
        return False
    if not hmac.compare_digest(receipt.runtime_public_key, expected_runtime_public_key):
        return False
    try:
        pub = _public_key_from_hex(receipt.runtime_public_key)
        return _verify_dict(receipt.to_dict(), receipt.signature, pub)
    except ValueError:
        return False


def verify_attestation_self_consistency(
    receipt: AttestationReceipt,
    *,
    expected_release: GateReleaseIdentity | None = None,
) -> bool:
    """Verify historical receipt integrity without establishing runtime authority."""
    if not receipt.signature or not receipt.runtime_public_key:
        return False
    release = expected_release or current_release_identity()
    if not release_identity_matches(receipt.gate_release, release):
        return False
    try:
        return _verify_dict(
            receipt.to_dict(),
            receipt.signature,
            _public_key_from_hex(receipt.runtime_public_key),
        )
    except ValueError:
        return False


def verify_enforced_attestation(
    receipt: AttestationReceipt,
    *,
    expected_runtime_public_key: str,
    expected_action: str,
    expected_target: str,
    expected_capability_ref: str,
    expected_success: bool,
    now: datetime | None = None,
    max_age_seconds: int = 300,
    expected_release: GateReleaseIdentity | None = None,
) -> bool:
    """Verify an externally pinned receipt and its exact execution contract."""
    if (
        not expected_runtime_public_key
        or type(expected_success) is not bool
        or isinstance(max_age_seconds, bool)
        or not isinstance(max_age_seconds, int)
        or max_age_seconds <= 0
        or receipt.action != expected_action
        or receipt.target != expected_target
        or receipt.capability_ref != expected_capability_ref
        or receipt.success is not expected_success
    ):
        return False
    try:
        executed = datetime.fromisoformat(receipt.executed_at)
        check_time = now or datetime.now(timezone.utc)
        if executed.tzinfo is None or check_time.tzinfo is None:
            return False
        age = (check_time - executed).total_seconds()
        if age < 0 or age > max_age_seconds:
            return False
    except (TypeError, ValueError):
        return False
    return verify_attestation(
        receipt,
        expected_runtime_public_key=expected_runtime_public_key,
        expected_release=expected_release,
    )


# ---------------------------------------------------------------------------
# Token: PhaseGateReceipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseGateReceipt:
    """Signed by the policy engine at phase transitions."""

    from_phase: str
    to_phase: str
    required_attestations: list[str]
    satisfied_at: str = ""
    signature: str = ""
    gate_release: GateReleaseIdentity = field(default_factory=current_release_identity)

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_phase": self.from_phase,
            "to_phase": self.to_phase,
            "required_attestations": list(self.required_attestations),
            "satisfied_at": self.satisfied_at,
            "gate_release": self.gate_release.to_dict(),
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PhaseGateReceipt:
        return cls(
            from_phase=d["from_phase"],
            to_phase=d["to_phase"],
            required_attestations=d["required_attestations"],
            satisfied_at=d.get("satisfied_at", ""),
            gate_release=GateReleaseIdentity.from_dict(dict(d["gate_release"])),
            signature=d.get("signature", ""),
        )


def sign_phase_gate(
    receipt: PhaseGateReceipt,
    private_key: Ed25519PrivateKey,
) -> PhaseGateReceipt:
    d = receipt.to_dict()
    sig = _sign_dict(d, private_key)
    return PhaseGateReceipt(**{**d, "gate_release": receipt.gate_release, "signature": sig})


def verify_phase_gate(
    receipt: PhaseGateReceipt,
    public_key: Ed25519PublicKey,
    *,
    expected_release: GateReleaseIdentity | None = None,
) -> bool:
    if not receipt.signature:
        return False
    release = expected_release or current_release_identity()
    if not release_identity_matches(
        receipt.gate_release,
        release,
        require_source_commit=True,
    ):
        return False
    return _verify_dict(receipt.to_dict(), receipt.signature, public_key)


# ---------------------------------------------------------------------------
# Token: RevocationNotice
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RevocationNotice:
    """Signed by the DA to revoke capabilities mid-mission."""

    scope: str
    reason: str
    issued_at: str = ""
    signature: str = ""
    gate_release: GateReleaseIdentity = field(default_factory=current_release_identity)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "reason": self.reason,
            "issued_at": self.issued_at,
            "gate_release": self.gate_release.to_dict(),
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RevocationNotice:
        return cls(
            scope=d["scope"],
            reason=d["reason"],
            issued_at=d.get("issued_at", ""),
            gate_release=GateReleaseIdentity.from_dict(dict(d["gate_release"])),
            signature=d.get("signature", ""),
        )


def sign_revocation(
    notice: RevocationNotice,
    private_key: Ed25519PrivateKey,
) -> RevocationNotice:
    d = notice.to_dict()
    sig = _sign_dict(d, private_key)
    return RevocationNotice(**{**d, "gate_release": notice.gate_release, "signature": sig})


def verify_revocation(
    notice: RevocationNotice,
    public_key: Ed25519PublicKey,
    *,
    expected_release: GateReleaseIdentity | None = None,
) -> bool:
    if not notice.signature:
        return False
    release = expected_release or current_release_identity()
    if not release_identity_matches(
        notice.gate_release,
        release,
        require_source_commit=True,
    ):
        return False
    return _verify_dict(notice.to_dict(), notice.signature, public_key)
