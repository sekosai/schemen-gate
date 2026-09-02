"""Authenticated token issuance and redemption for masks and adapters.

Requires ``pip install schemen-gate[crypto]`` (for AES-256-GCM via
the ``cryptography`` package).

Tokens are AEAD bundles: the payload is encrypted, and the full contract
between issuer and consumer is bound as Additional Authenticated Data.
The contract IS the channel — without mutual agreement on every field,
decryption is a cryptographic impossibility.

Canonical token implementation for the Schemen gate.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import json
import math
import os
import re
import time as _time
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Optional

import numpy as np

from schemen_gate._crypto import derive_partition
from schemen_gate._release import (
    GateReleaseIdentity,
    current_release_identity,
    release_identity_matches,
)

# ---------------------------------------------------------------------------
# Error hierarchy — distinct types for distinct failure modes
# ---------------------------------------------------------------------------


class SchemenTokenError(Exception):
    """Base class for all token-related errors."""


class TokenAuthenticationError(SchemenTokenError):
    """Raised when AES-GCM decryption fails (wrong key, tampered AAD, etc).

    This is not a software check — it means the cryptographic channel
    between issuer and consumer does not exist.
    """


class TokenExpiredError(SchemenTokenError):
    """Raised when a token's expiry epoch has passed."""


class WeightIntegrityError(SchemenTokenError):
    """Raised when decrypted weight bytes don't match the committed hash."""


class RegimePermissionError(SchemenTokenError):
    """Raised when an operation requires a permission the context lacks."""


class StoreCapacityError(SchemenTokenError):
    """Raised when a RegimeStore has reached its max_contexts limit."""


def _verification_epoch(now_epoch: object | None) -> float:
    """Resolve a verifier clock override without permitting fail-open values."""

    if now_epoch is None:
        return _time.time()
    if isinstance(now_epoch, bool) or not isinstance(now_epoch, Real):
        raise ValueError("now_epoch must be a finite non-negative number")
    try:
        resolved = float(now_epoch)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("now_epoch must be a finite non-negative number") from exc
    if not math.isfinite(resolved) or resolved < 0:
        raise ValueError("now_epoch must be a finite non-negative number")
    return resolved


# ---------------------------------------------------------------------------
# Key types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateKey:
    """A 32-byte master secret used to derive gate partitions."""

    secret: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.secret, bytes) or len(self.secret) != 32:
            raise ValueError("invalid key material")

    def __repr__(self) -> str:
        return "GateKey(secret=<REDACTED 32 bytes>)"

    @classmethod
    def generate(cls) -> GateKey:
        return cls(secret=os.urandom(32))


@dataclass(frozen=True)
class DerivedKey:
    """A 32-byte key derived from a GateKey via HKDF, scoped to a context."""

    secret: bytes
    context: str

    def __post_init__(self) -> None:
        if not isinstance(self.secret, bytes) or len(self.secret) != 32:
            raise ValueError("invalid key material")
        if not isinstance(self.context, str) or not self.context or "\x00" in self.context:
            raise ValueError("invalid key context")

    def __repr__(self) -> str:
        return f"DerivedKey(context={self.context!r}, secret=<REDACTED 32 bytes>)"


# ---------------------------------------------------------------------------
# HKDF + key derivation
# ---------------------------------------------------------------------------


def hkdf_expand_sha256(prk: bytes, info: bytes, length: int = 32) -> bytes:
    """Single-block HKDF-Expand using HMAC-SHA256 (RFC 5869)."""
    if not isinstance(prk, bytes) or not isinstance(info, bytes):
        raise ValueError("HKDF prk and info must be bytes")
    if isinstance(length, bool) or not isinstance(length, int) or not 1 <= length <= 32:
        raise ValueError("Single-block HKDF-Expand length must be in [1, 32]")
    return _hmac.new(prk, info + b"\x01", hashlib.sha256).digest()[:length]


def derive_regime_key(master: GateKey, regime_id: int) -> DerivedKey:
    if isinstance(regime_id, bool) or not isinstance(regime_id, int) or regime_id < 0:
        raise ValueError("regime_id must be a non-negative integer")
    info = f"schemen:v1:regime:{regime_id}".encode()
    secret = hkdf_expand_sha256(master.secret, info)
    return DerivedKey(secret=secret, context=f"regime:{regime_id}")


def derive_tenant_key(master: GateKey, regime_id: int, tenant_id: str) -> DerivedKey:
    if isinstance(regime_id, bool) or not isinstance(regime_id, int) or regime_id < 0:
        raise ValueError("regime_id must be a non-negative integer")
    if not isinstance(tenant_id, str) or not tenant_id or ":" in tenant_id or "\x00" in tenant_id:
        raise ValueError("tenant_id must be non-empty and must not contain ':' or NUL")
    info = f"schemen:v1:regime:{regime_id}:tenant:{tenant_id}".encode()
    secret = hkdf_expand_sha256(master.secret, info)
    return DerivedKey(secret=secret, context=f"regime:{regime_id}:tenant:{tenant_id}")


# ---------------------------------------------------------------------------
# MaskToken (AAD-bound gate mask)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaskToken:
    """AES-256-GCM encrypted gate mask with metadata bound as AAD."""

    tenant_id: str
    regime_id: int
    nonce: bytes
    ciphertext: bytes
    n_dims: int
    n_regimes: int
    gate_release: GateReleaseIdentity = field(default_factory=current_release_identity)


def _build_mask_aad(
    tenant_id: str,
    regime_id: int,
    n_dims: int,
    n_regimes: int,
    gate_release: GateReleaseIdentity,
) -> bytes:
    return json.dumps(
        {
            "gate_release": gate_release.to_dict(),
            "n_dims": n_dims,
            "n_regimes": n_regimes,
            "regime_id": regime_id,
            "schema": "schemen/mask-token-aad-v2",
            "tenant_id": tenant_id,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _encode_indices(indices: list[int]) -> bytes:
    for idx in indices:
        if idx < 0 or idx > 0xFFFF:
            raise ValueError(f"Index {idx} out of uint16 range")
    return b"".join(idx.to_bytes(2, "big") for idx in indices)


def _decode_indices(data: bytes, n_dims: int) -> list[int]:
    if len(data) % 2 != 0:
        raise ValueError("Encoded indices must have even byte length")
    indices = [int.from_bytes(data[i : i + 2], "big") for i in range(0, len(data), 2)]
    if any(idx >= n_dims for idx in indices):
        raise ValueError(f"Decrypted index out of range [0, {n_dims})")
    if len(indices) != len(set(indices)):
        raise ValueError("Decoded indices contain duplicates")
    return indices


def issue_mask_token(
    master: GateKey,
    tenant_id: str,
    regime_id: int,
    n_dims: int,
    n_regimes: int = 2,
    *,
    release_identity: GateReleaseIdentity | None = None,
) -> MaskToken:
    """Issue an AES-256-GCM encrypted mask token for a tenant."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if regime_id < 0 or regime_id >= n_regimes:
        raise ValueError(f"regime_id {regime_id} out of range [0, {n_regimes})")
    groups = derive_partition(master.secret, n_dims, n_regimes)
    indices = sorted(groups[regime_id])
    tenant_key = derive_tenant_key(master, regime_id, tenant_id)
    plaintext = _encode_indices(indices)
    release = release_identity or current_release_identity()
    aad = _build_mask_aad(tenant_id, regime_id, n_dims, n_regimes, release)
    nonce = os.urandom(12)
    ct = AESGCM(tenant_key.secret).encrypt(nonce, plaintext, aad)
    return MaskToken(
        tenant_id=tenant_id,
        regime_id=regime_id,
        nonce=nonce,
        ciphertext=ct,
        n_dims=n_dims,
        n_regimes=n_regimes,
        gate_release=release,
    )


def redeem_mask_token(
    token: MaskToken,
    tenant_key: DerivedKey,
    *,
    expected_release: GateReleaseIdentity | None = None,
) -> np.ndarray:
    """Redeem a mask token -> binary mask array. AAD mismatch = failure."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    release = expected_release or current_release_identity()
    if not release_identity_matches(
        token.gate_release,
        release,
        require_source_commit=True,
    ):
        raise TokenAuthenticationError(
            "Mask token Gate release differs from the verifying runtime."
        )
    aad = _build_mask_aad(
        token.tenant_id,
        token.regime_id,
        token.n_dims,
        token.n_regimes,
        token.gate_release,
    )
    try:
        plaintext = AESGCM(tenant_key.secret).decrypt(token.nonce, token.ciphertext, aad)
    except Exception:
        raise TokenAuthenticationError(
            "Mask token authentication failed — wrong key or tampered metadata. "
            "The channel does not exist."
        ) from None
    indices = _decode_indices(plaintext, token.n_dims)
    mask = np.zeros(token.n_dims, dtype=np.float64)
    mask[indices] = 1.0
    return mask


# ---------------------------------------------------------------------------
# AdapterToken (AAD-bound adapter weights — full contract as channel)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdapterToken:
    """AES-256-GCM encrypted adapter weights with full contract as AAD.

    The contract between issuer and consumer — geometry, topology, version,
    permissions, expiry, and lockbox binding — IS the channel. Without mutual
    agreement on every field, decryption is a cryptographic impossibility.
    """

    tenant_id: str
    regime_id: int
    nonce: bytes
    ciphertext: bytes
    n_dims: int
    n_regimes: int
    topology: str
    weight_hash: str
    version: int
    permissions: str
    parent_lockbox_hash: str
    expires_epoch: Optional[int] = None
    gate_release: GateReleaseIdentity = field(default_factory=current_release_identity)


def _build_adapter_aad(
    tenant_id: str,
    regime_id: int,
    n_dims: int,
    n_regimes: int,
    topology: str,
    weight_hash: str,
    version: int,
    permissions: str,
    parent_lockbox_hash: str,
    expires_epoch: Optional[int] = None,
    gate_release: GateReleaseIdentity | None = None,
) -> bytes:
    """Canonical, unambiguous AAD for the complete adapter contract."""
    release = gate_release or current_release_identity()
    contract = {
        "expires_epoch": expires_epoch,
        "gate_release": release.to_dict(),
        "n_dims": n_dims,
        "n_regimes": n_regimes,
        "parent_lockbox_hash": parent_lockbox_hash,
        "permissions": permissions,
        "regime_id": regime_id,
        "schema": "schemen/adapter-aad-v4",
        "tenant_id": tenant_id,
        "topology": topology,
        "version": version,
        "weight_hash": weight_hash,
    }
    try:
        return json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("Adapter contract must be canonical JSON") from exc


def canonical_permissions(perms: list[str] | str) -> str:
    """Produce a canonical, sorted, comma-separated permission string."""
    if isinstance(perms, str):
        parts = [p.strip() for p in perms.split(",") if p.strip()]
    else:
        parts = list(perms)
    if not parts or any(not isinstance(part, str) or not part.strip() for part in parts):
        raise ValueError("permissions must contain at least one non-empty string")
    parts = [part.strip() for part in parts]
    if any(re.fullmatch(r"[A-Za-z][A-Za-z0-9._:-]{0,63}", part) is None for part in parts):
        raise ValueError("permission names must use the canonical token grammar")
    return ",".join(sorted(set(parts)))


def _validate_adapter_contract(
    *,
    tenant_id: str,
    regime_id: int,
    n_dims: int,
    n_regimes: int,
    topology: str,
    weight_hash: str,
    version: int,
    permissions: str,
    parent_lockbox_hash: str,
    expires_epoch: Optional[int],
    gate_release: GateReleaseIdentity,
    issuing: bool,
) -> None:
    if not tenant_id or ":" in tenant_id:
        raise ValueError("tenant_id must be non-empty and must not contain ':'")
    if isinstance(n_dims, bool) or not isinstance(n_dims, int) or not 1 <= n_dims <= 65536:
        raise ValueError("n_dims must be an integer in [1, 65536]")
    if (
        isinstance(n_regimes, bool)
        or not isinstance(n_regimes, int)
        or not 1 <= n_regimes <= n_dims
    ):
        raise ValueError("n_regimes must be an integer in [1, n_dims]")
    if (
        isinstance(regime_id, bool)
        or not isinstance(regime_id, int)
        or not 0 <= regime_id < n_regimes
    ):
        raise ValueError(f"regime_id {regime_id!r} out of range [0, {n_regimes})")
    if (
        not isinstance(topology, str)
        or re.fullmatch(
            r"[1-9][0-9]*(?::[1-9][0-9]*)*",
            topology,
        )
        is None
    ):
        raise ValueError("topology must be colon-separated positive dimensions")
    if re.fullmatch(r"[0-9a-f]{64}", weight_hash) is None:
        raise ValueError("weight_hash must be a lowercase SHA-256 digest")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("version must be a positive integer")
    if canonical_permissions(permissions) != permissions:
        raise ValueError("permissions must use canonical sorted form")
    if (
        parent_lockbox_hash
        and re.fullmatch(
            r"[0-9a-f]{64}",
            parent_lockbox_hash,
        )
        is None
    ):
        raise ValueError("parent_lockbox_hash must be empty or a SHA-256 digest")
    if expires_epoch is not None and (
        isinstance(expires_epoch, bool) or not isinstance(expires_epoch, int) or expires_epoch < 1
    ):
        raise ValueError("expires_epoch must be a positive integer or None")
    if issuing and expires_epoch is not None and expires_epoch <= int(_time.time()):
        raise ValueError("cannot issue an already-expired adapter token")
    if not isinstance(gate_release, GateReleaseIdentity):
        raise ValueError("gate_release must be a validated GateReleaseIdentity")


def hash_adapter_weights(weights: bytes) -> str:
    """SHA-256 hex digest of raw serialized adapter weights."""
    return hashlib.sha256(weights).hexdigest()


def adapter_topology_string(layer_dims: list[int]) -> str:
    """Canonical topology string, e.g. [2048, 256, 2048] -> '2048:256:2048'."""
    return ":".join(str(d) for d in layer_dims)


def issue_adapter_token(
    master: GateKey,
    tenant_id: str,
    regime_id: int,
    n_dims: int,
    n_regimes: int,
    raw_weights: bytes,
    topology: str,
    version: int = 1,
    permissions: str | list[str] = "use",
    parent_lockbox_hash: str = "",
    expires_epoch: Optional[int] = None,
    release_identity: GateReleaseIdentity | None = None,
    *,
    allow_non_expiring: bool = False,
) -> AdapterToken:
    """Issue an AES-256-GCM encrypted adapter token with full contract as AAD."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if expires_epoch is None and allow_non_expiring is not True:
        expires_epoch = int(_time.time()) + 3600
    perm_str = canonical_permissions(permissions)
    w_hash = hash_adapter_weights(raw_weights)
    release = release_identity or current_release_identity()
    _validate_adapter_contract(
        tenant_id=tenant_id,
        regime_id=regime_id,
        n_dims=n_dims,
        n_regimes=n_regimes,
        topology=topology,
        weight_hash=w_hash,
        version=version,
        permissions=perm_str,
        parent_lockbox_hash=parent_lockbox_hash,
        expires_epoch=expires_epoch,
        gate_release=release,
        issuing=True,
    )
    tenant_key = derive_tenant_key(master, regime_id, tenant_id)
    aad = _build_adapter_aad(
        tenant_id,
        regime_id,
        n_dims,
        n_regimes,
        topology,
        w_hash,
        version,
        perm_str,
        parent_lockbox_hash,
        expires_epoch,
        release,
    )
    nonce = os.urandom(12)
    ct = AESGCM(tenant_key.secret).encrypt(nonce, raw_weights, aad)
    return AdapterToken(
        tenant_id=tenant_id,
        regime_id=regime_id,
        nonce=nonce,
        ciphertext=ct,
        n_dims=n_dims,
        n_regimes=n_regimes,
        topology=topology,
        weight_hash=w_hash,
        version=version,
        permissions=perm_str,
        parent_lockbox_hash=parent_lockbox_hash,
        expires_epoch=expires_epoch,
        gate_release=release,
    )


def issue_use_only_token(
    master: GateKey,
    tenant_id: str,
    regime_id: int,
    n_dims: int,
    n_regimes: int,
    raw_weights: bytes,
    topology: str,
    parent_lockbox_hash: str = "",
    expires_epoch: Optional[int] = None,
    release_identity: GateReleaseIdentity | None = None,
    *,
    allow_non_expiring: bool = False,
) -> AdapterToken:
    """Convenience: issue a use-only token with minimal parameters.

    This is the default secure posture — inference only, no training,
    no delegation, no destruction.
    """
    return issue_adapter_token(
        master,
        tenant_id,
        regime_id,
        n_dims,
        n_regimes,
        raw_weights,
        topology,
        version=1,
        permissions="use",
        parent_lockbox_hash=parent_lockbox_hash,
        expires_epoch=expires_epoch,
        release_identity=release_identity,
        allow_non_expiring=allow_non_expiring,
    )


def redeem_adapter_token(
    token: AdapterToken,
    tenant_key: DerivedKey,
    *,
    expected_release: GateReleaseIdentity | None = None,
) -> bytes:
    """Redeem an adapter token -> raw weight bytes. Contract mismatch = failure.

    Raises:
        TokenExpiredError: If the token has an expiry and it has passed.
        TokenAuthenticationError: If the key is wrong or any AAD field
            was tampered with. The channel does not exist.
        WeightIntegrityError: If the decrypted weight hash doesn't match.
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    _validate_adapter_contract(
        tenant_id=token.tenant_id,
        regime_id=token.regime_id,
        n_dims=token.n_dims,
        n_regimes=token.n_regimes,
        topology=token.topology,
        weight_hash=token.weight_hash,
        version=token.version,
        permissions=token.permissions,
        parent_lockbox_hash=token.parent_lockbox_hash,
        expires_epoch=token.expires_epoch,
        gate_release=token.gate_release,
        issuing=False,
    )
    release = expected_release or current_release_identity()
    if not release_identity_matches(
        token.gate_release,
        release,
        require_source_commit=True,
    ):
        raise TokenAuthenticationError(
            "Adapter token Gate release differs from the verifying runtime."
        )
    if len(token.nonce) != 12 or len(token.ciphertext) < 16:
        raise ValueError("adapter token has invalid AEAD field lengths")
    if token.expires_epoch is not None and _time.time() >= token.expires_epoch:
        raise TokenExpiredError(
            f"Token expired at epoch {token.expires_epoch} "
            f"(current: {int(_time.time())}). "
            "A new token must be issued."
        )

    aad = _build_adapter_aad(
        token.tenant_id,
        token.regime_id,
        token.n_dims,
        token.n_regimes,
        token.topology,
        token.weight_hash,
        token.version,
        token.permissions,
        token.parent_lockbox_hash,
        token.expires_epoch,
        token.gate_release,
    )
    try:
        plaintext = AESGCM(tenant_key.secret).decrypt(token.nonce, token.ciphertext, aad)
    except Exception:
        raise TokenAuthenticationError(
            "Adapter token authentication failed — context mismatch or tampering. "
            "The channel does not exist without mutual agreement on all contract fields."
        ) from None
    actual_hash = hash_adapter_weights(plaintext)
    if not _hmac.compare_digest(actual_hash, token.weight_hash):
        raise WeightIntegrityError("Adapter weight hash mismatch after decryption")
    return plaintext


# ---------------------------------------------------------------------------
# GateRights (authenticated permission metadata)
# ---------------------------------------------------------------------------


@dataclass
class GateRights:
    """Authenticated permission metadata for a gate — the 'Bill of Rights'.

    Signed with HMAC-SHA256 against a GateKey. Tampering with any field
    invalidates the signature.
    """

    regime_id: int
    version: int = 1
    can_use: bool = False
    can_update: bool = False
    can_delegate: bool = False
    can_create_subordinate: bool = False
    can_destroy: bool = False
    can_inspect: bool = False
    issuer: str = "root"
    parent_regime: Optional[int] = None
    expires_epoch: Optional[int] = field(default_factory=lambda: int(_time.time()) + 3600)
    metadata: dict[str, Any] = field(default_factory=dict)
    gate_release: GateReleaseIdentity = field(default_factory=current_release_identity)

    def __post_init__(self) -> None:
        """Reject coerced or ambiguous authority fields before they can be signed."""
        for name in ("regime_id", "version"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an exact integer")
        if self.regime_id < 0 or self.version < 1:
            raise ValueError("regime_id must be >= 0 and version must be >= 1")
        for name in (
            "can_use",
            "can_update",
            "can_delegate",
            "can_create_subordinate",
            "can_destroy",
            "can_inspect",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be an exact boolean")
        if type(self.issuer) is not str or not self.issuer or "\x00" in self.issuer:
            raise ValueError("issuer must be a non-empty exact string without NUL")
        if self.parent_regime is not None and (
            isinstance(self.parent_regime, bool)
            or not isinstance(self.parent_regime, int)
            or self.parent_regime < 0
        ):
            raise ValueError("parent_regime must be a non-negative integer or None")
        if self.expires_epoch is not None and (
            isinstance(self.expires_epoch, bool)
            or not isinstance(self.expires_epoch, int)
            or self.expires_epoch < 0
        ):
            raise ValueError("expires_epoch must be a non-negative integer or None")
        if type(self.metadata) is not dict or any(type(key) is not str for key in self.metadata):
            raise ValueError("metadata must be a dictionary with string keys")
        if not isinstance(self.gate_release, GateReleaseIdentity):
            raise ValueError("gate_release must be a validated GateReleaseIdentity")

    def to_bytes(self) -> bytes:
        # The v1 rights byte format (default json separators, no schema field)
        # is frozen for signature compatibility; a future schema must adopt the
        # strict canonical encoding used by the other authenticated contracts.
        return json.dumps(
            {
                "regime_id": self.regime_id,
                "version": self.version,
                "can_use": self.can_use,
                "can_update": self.can_update,
                "can_delegate": self.can_delegate,
                "can_create_subordinate": self.can_create_subordinate,
                "can_destroy": self.can_destroy,
                "can_inspect": self.can_inspect,
                "issuer": self.issuer,
                "parent_regime": self.parent_regime,
                "expires_epoch": self.expires_epoch,
                "metadata": self.metadata,
                "gate_release": self.gate_release.to_dict(),
            },
            sort_keys=True,
        ).encode()

    def sign(self, key: GateKey) -> bytes:
        return _hmac.new(key.secret, self.to_bytes(), hashlib.sha256).digest()

    def is_expired(self, *, now_epoch: float | None = None) -> bool:
        """Return whether these rights have expired at ``now_epoch``."""
        now = _verification_epoch(now_epoch)
        return self.expires_epoch is not None and now >= self.expires_epoch

    @staticmethod
    def verify(
        rights: "GateRights",
        signature: bytes,
        key: GateKey,
        *,
        now_epoch: float | None = None,
        expected_release: GateReleaseIdentity | None = None,
    ) -> bool:
        """Authenticate a rights object and require that it is current."""
        try:
            expected = _hmac.new(key.secret, rights.to_bytes(), hashlib.sha256).digest()
            release = expected_release or current_release_identity()
            return (
                _hmac.compare_digest(signature, expected)
                and not rights.is_expired(now_epoch=now_epoch)
                and release_identity_matches(
                    rights.gate_release,
                    release,
                    require_source_commit=True,
                )
            )
        except (AttributeError, TypeError, ValueError):
            return False

    def permissions_set(self, *, now_epoch: float | None = None) -> set[str]:
        """Return the set of active permission names."""
        if self.is_expired(now_epoch=now_epoch):
            return set()
        perms = set()
        if self.can_use:
            perms.add("use")
        if self.can_update:
            perms.add("update")
        if self.can_delegate:
            perms.add("delegate")
        if self.can_create_subordinate:
            perms.add("create")
        if self.can_destroy:
            perms.add("destroy")
        if self.can_inspect:
            perms.add("inspect")
        return perms

    @staticmethod
    def authorizes(
        rights: "GateRights",
        signature: bytes,
        key: GateKey,
        permission: str,
        *,
        expected_regime_id: int,
        expected_issuer: str,
        expected_version: int = 1,
        now_epoch: float | None = None,
        expected_release: GateReleaseIdentity | None = None,
    ) -> bool:
        """Authenticate one current, scoped permission decision."""
        return (
            GateRights.verify(
                rights,
                signature,
                key,
                now_epoch=now_epoch,
                expected_release=expected_release,
            )
            and rights.version == expected_version
            and rights.regime_id == expected_regime_id
            and _hmac.compare_digest(rights.issuer, expected_issuer)
            and permission in rights.permissions_set(now_epoch=now_epoch)
        )

    # ------------------------------------------------------------------
    # Boolean algebra over the 6-bit permission vector.
    #
    # GateRights forms a Boolean algebra under AND (compose), OR
    # (union), and NOT (complement) on the per-bit permissions.
    # The auxiliary fields (regime_id, version, expires_epoch, ...)
    # are reconciled with these rules:
    #
    #   * regime_id MUST match between two operands; mixing rights
    #     for different regimes is a category error and raises.
    #   * version takes the max (the result reflects the newer of
    #     the two policies).
    #   * issuer is rewritten to record the composition: e.g.
    #     "compose(a,b)", "union(a,b)", "complement(a)".  The result
    #     is a *derived* GateRights and is unsigned -- the caller
    #     must sign it with whichever authority has the right to
    #     issue the composed permission, OR treat it as ephemeral
    #     inside a frame's effective-rights computation.
    #   * expires_epoch: AND takes min (most restrictive); OR takes
    #     max (least restrictive); NOT preserves the value.
    #   * parent_regime, metadata: passed through; metadata keys
    #     'compose_of' / 'union_of' / 'complement_of' are populated
    #     with the operand fingerprints for audit.
    #
    # The key property: AND-of-three-sources collapses to the
    # strictest bit on every axis, so superimposing
    # (delegation_rights, partition_rights, situational_rights)
    # produces an effective right where any False bit anywhere makes
    # the bit False everywhere -- the ratchet semantics fall out of
    # Boolean algebra without any policy-enum machinery.
    # ------------------------------------------------------------------

    def _fingerprint_short(self) -> str:
        """SHA-256 prefix of the rights bytes; identifies this
        permission state for audit metadata.  Excludes signature."""
        return hashlib.sha256(self.to_bytes()).hexdigest()[:16]

    def compose(self, other: "GateRights") -> "GateRights":
        """AND-compose: the effective rights when both grants apply.

        Bit-wise AND of every permission flag.  The strictest source
        wins on every axis; a False bit in either operand stays False
        in the result.  The returned GateRights is unsigned; sign it
        with an issuer that has the authority to express the
        composed permission, or use it ephemerally inside a frame.

        Raises ValueError if the operands' regime_ids differ.
        """
        if self.regime_id != other.regime_id:
            raise ValueError(
                f"cannot AND-compose GateRights for different regimes: "
                f"{self.regime_id} vs {other.regime_id}"
            )
        if self.gate_release != other.gate_release:
            raise ValueError("cannot AND-compose GateRights from different releases")
        a_exp, b_exp = self.expires_epoch, other.expires_epoch
        if a_exp is None:
            new_exp = b_exp
        elif b_exp is None:
            new_exp = a_exp
        else:
            new_exp = min(a_exp, b_exp)
        meta = dict(self.metadata)
        meta.update(other.metadata)
        meta["compose_of"] = [self._fingerprint_short(), other._fingerprint_short()]
        return GateRights(
            regime_id=self.regime_id,
            version=max(self.version, other.version),
            can_use=self.can_use and other.can_use,
            can_update=self.can_update and other.can_update,
            can_delegate=self.can_delegate and other.can_delegate,
            can_create_subordinate=(self.can_create_subordinate and other.can_create_subordinate),
            can_destroy=self.can_destroy and other.can_destroy,
            can_inspect=self.can_inspect and other.can_inspect,
            issuer=f"compose({self.issuer},{other.issuer})",
            parent_regime=self.parent_regime if self.parent_regime == other.parent_regime else None,
            expires_epoch=new_exp,
            metadata=meta,
            gate_release=self.gate_release,
        )

    def union(self, other: "GateRights") -> "GateRights":
        """OR-compose: the effective rights when either grant applies.

        Bit-wise OR of every permission flag.  Useful for coalition
        rights (a frame opened by a multi-principal authority where
        the union is sufficient) and for the de Morgan side of the
        Boolean algebra.

        Equivalent to ``self.complement().compose(other.complement()).complement()``
        (de Morgan), authenticated by the test suite.

        Raises ValueError if regime_ids differ.
        """
        if self.regime_id != other.regime_id:
            raise ValueError(
                f"cannot OR-compose GateRights for different regimes: "
                f"{self.regime_id} vs {other.regime_id}"
            )
        if self.gate_release != other.gate_release:
            raise ValueError("cannot OR-compose GateRights from different releases")
        a_exp, b_exp = self.expires_epoch, other.expires_epoch
        # OR takes the LATEST expiry (least restrictive in the time
        # axis, parallel to OR being the least restrictive in the
        # permission axis).
        if a_exp is None or b_exp is None:
            new_exp = None  # never-expires wins
        else:
            new_exp = max(a_exp, b_exp)
        meta = dict(self.metadata)
        meta.update(other.metadata)
        meta["union_of"] = [self._fingerprint_short(), other._fingerprint_short()]
        return GateRights(
            regime_id=self.regime_id,
            version=max(self.version, other.version),
            can_use=self.can_use or other.can_use,
            can_update=self.can_update or other.can_update,
            can_delegate=self.can_delegate or other.can_delegate,
            can_create_subordinate=(self.can_create_subordinate or other.can_create_subordinate),
            can_destroy=self.can_destroy or other.can_destroy,
            can_inspect=self.can_inspect or other.can_inspect,
            issuer=f"union({self.issuer},{other.issuer})",
            parent_regime=self.parent_regime if self.parent_regime == other.parent_regime else None,
            expires_epoch=new_exp,
            metadata=meta,
            gate_release=self.gate_release,
        )

    def complement(self) -> "GateRights":
        """NOT-complement: bit-wise inversion of the permission vector.

        Useful for expressing "what you may NOT do" (revocation
        masks), for negative authorization, and for deriving OR from
        AND via de Morgan.  Together with ``compose`` and ``union``
        this completes the Boolean algebra over GateRights.

        ``complement`` is its own inverse: ``a.complement().complement()``
        produces a GateRights with the original permission bits
        (double negation).  Auxiliary fields are preserved unchanged.
        """
        meta = dict(self.metadata)
        meta["complement_of"] = self._fingerprint_short()
        return GateRights(
            regime_id=self.regime_id,
            version=self.version,
            can_use=not self.can_use,
            can_update=not self.can_update,
            can_delegate=not self.can_delegate,
            can_create_subordinate=not self.can_create_subordinate,
            can_destroy=not self.can_destroy,
            can_inspect=not self.can_inspect,
            issuer=f"complement({self.issuer})",
            parent_regime=self.parent_regime,
            expires_epoch=self.expires_epoch,
            metadata=meta,
            gate_release=self.gate_release,
        )

    def __and__(self, other: "GateRights") -> "GateRights":
        return self.compose(other)

    def __or__(self, other: "GateRights") -> "GateRights":
        return self.union(other)

    def __invert__(self) -> "GateRights":
        return self.complement()
