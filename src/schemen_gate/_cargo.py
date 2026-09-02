"""Cargo Mode transactional protocol for partitioned inference data.

A workload presents an authenticated manifest to a Regime bus, exchanges
bounded RAG packets under the admitted operation, and receives a receipt. The
AES-GCM Additional Authenticated Data binds the embedding model, dimensions,
vocabulary hash, pooling rule, operation, subject, and partition. A mismatch
fails authentication before the storage operation is invoked.

Requires ``pip install schemen-gate[crypto]``.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import json
import math
import struct
import time as _time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Any, Dict, List, Optional, Protocol, Sequence

import numpy as np

from schemen_gate._release import GateReleaseIdentity, current_release_identity

_HEX_DIGITS = frozenset("0123456789abcdef")
_MAX_FRAME_BYTES = 256 * 1024 * 1024
_MAX_TEXT_BYTES = 16_384
_MAX_CANONICAL_JSON_BYTES = 16 * 1024 * 1024
_MAX_ITEMS = 100_000


def _verification_epoch(now: object | None) -> float:
    """Resolve a verifier clock override without permitting fail-open values."""

    if now is None:
        return _time.time()
    if isinstance(now, bool) or not isinstance(now, Real):
        raise ValueError("now must be a finite non-negative number")
    try:
        resolved = float(now)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("now must be a finite non-negative number") from exc
    if not math.isfinite(resolved) or resolved < 0:
        raise ValueError("now must be a finite non-negative number")
    return resolved


def _canonical_json(value: Any) -> bytes:
    """Strict canonical JSON for authenticated protocol fields."""
    _validate_json_value(value)
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("Cargo protocol values must be canonical JSON") from exc
    if len(encoded) > _MAX_CANONICAL_JSON_BYTES:
        raise ValueError("Cargo protocol JSON exceeds the maximum encoded size")
    return encoded


def _validate_json_value(
    value: Any,
    *,
    path: str = "$",
    _depth: int = 0,
    _seen: set[int] | None = None,
) -> None:
    """Require an exact JSON data model before canonicalization.

    ``json.dumps`` silently normalizes tuples to arrays and non-string mapping
    keys to strings.  Those coercions create multiple Python inputs with the
    same authenticated bytes.  Protocol values therefore use only the native
    JSON types and finite numbers.
    """
    if _depth > 64:
        raise ValueError("Cargo protocol JSON exceeds the maximum nesting depth")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Cargo protocol number at {path} must be finite")
        return
    if isinstance(value, list):
        seen = set() if _seen is None else _seen
        identity = id(value)
        if identity in seen:
            raise ValueError(f"Cargo protocol JSON at {path} contains a cycle")
        seen.add(identity)
        try:
            for index, item in enumerate(value):
                _validate_json_value(
                    item,
                    path=f"{path}[{index}]",
                    _depth=_depth + 1,
                    _seen=seen,
                )
        finally:
            seen.remove(identity)
        return
    if isinstance(value, dict):
        seen = set() if _seen is None else _seen
        identity = id(value)
        if identity in seen:
            raise ValueError(f"Cargo protocol JSON at {path} contains a cycle")
        seen.add(identity)
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError(f"Cargo protocol object key at {path} must be a string")
                _validate_json_value(
                    item,
                    path=f"{path}.{key}",
                    _depth=_depth + 1,
                    _seen=seen,
                )
        finally:
            seen.remove(identity)
        return
    raise ValueError(f"Cargo protocol value at {path} has non-JSON type {type(value).__name__}")


def _clone_json_object(value: Any, *, name: str) -> Dict[str, Any]:
    """Validate and deep-copy a JSON object through its canonical bytes."""
    encoded = _canonical_json(value)
    cloned = json.loads(encoded)
    if not isinstance(cloned, dict):
        raise ValueError(f"{name} must be a JSON object")
    return cloned


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX_DIGITS for character in value)
    )


def _text(value: object, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a bounded exact string without NUL")
    try:
        encoded_length = len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be a bounded exact string without NUL") from exc
    if "\x00" in value or (not allow_empty and not value) or encoded_length > _MAX_TEXT_BYTES:
        raise ValueError(f"{name} must be a bounded exact string without NUL")
    return value


def _strict_json_loads(value: object, *, name: str) -> Any:
    """Parse JSON while rejecting duplicate object keys and non-JSON values."""
    source = _text(value, name)

    def object_from_pairs(pairs: list[tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate object key {key!r}")
            result[key] = item
        return result

    def reject_constant(constant: str) -> None:
        raise ValueError(f"{name} contains non-finite constant {constant}")

    try:
        decoded = json.loads(
            source,
            object_pairs_hook=object_from_pairs,
            parse_constant=reject_constant,
        )
        _validate_json_value(decoded)
        return decoded
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {name}") from exc


def _update_length_prefixed(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


# ---------------------------------------------------------------------------
# EmbeddingSpec — the cryptographic identity of a workload's encoding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingSpec:
    """The embedding protocol of a workload.

    This is not metadata.  It IS the cryptographic identity.  Two parties
    that agree on an EmbeddingSpec can exchange inference.  Two that don't
    cannot — the gate enforces mutual intelligibility, not application code.
    """

    model_id: str
    """Embedding model identifier, e.g. ``"nomic-ai/nomic-embed-text-v1"``."""

    dimensions: int
    """Embedding dimensionality, e.g. 768."""

    vocabulary_hash: str
    """SHA-256 hex digest of the tokenizer vocabulary (sorted token list)."""

    pooling: str
    """Pooling strategy: ``"cls"``, ``"mean"``, ``"last"``."""

    def __post_init__(self) -> None:
        _text(self.model_id, "model_id")
        if (
            isinstance(self.dimensions, bool)
            or not isinstance(self.dimensions, int)
            or self.dimensions <= 0
        ):
            raise ValueError("dimensions must be a positive integer")
        if not _is_sha256(self.vocabulary_hash):
            raise ValueError("vocabulary_hash must be a lowercase SHA-256 digest")
        _text(self.pooling, "pooling")

    def to_canonical(self) -> str:
        """Deterministic string for AAD binding."""
        return _canonical_json(
            {
                "dimensions": self.dimensions,
                "model_id": self.model_id,
                "pooling": self.pooling,
                "schema": "schemen/embedding-spec-v2",
                "vocabulary_hash": self.vocabulary_hash,
            }
        ).decode("ascii")

    @classmethod
    def from_canonical(cls, s: str) -> EmbeddingSpec:
        try:
            value = _strict_json_loads(s, name="EmbeddingSpec canonical string")
            if not isinstance(value, dict) or set(value) != {
                "dimensions",
                "model_id",
                "pooling",
                "schema",
                "vocabulary_hash",
            }:
                raise ValueError("EmbeddingSpec fields do not match the schema")
            if value.get("schema") != "schemen/embedding-spec-v2":
                raise ValueError("unsupported embedding spec schema")
            return cls(
                model_id=value["model_id"],
                dimensions=value["dimensions"],
                vocabulary_hash=value["vocabulary_hash"],
                pooling=value["pooling"],
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid EmbeddingSpec canonical string: {s!r}") from exc

    def fingerprint(self) -> str:
        """SHA-256 of the canonical form."""
        return hashlib.sha256(self.to_canonical().encode()).hexdigest()


# ---------------------------------------------------------------------------
# CompletionCondition — declared, verifier-owned completion state
# ---------------------------------------------------------------------------


class CompletionKind(str, Enum):
    """How a cargo transaction can complete."""

    TTL_EXPIRED = "ttl_expired"
    PARTITION_DESTROYED = "partition_destroyed"
    RECEIPT_ACKNOWLEDGED = "receipt_acknowledged"
    ITEMS_EXHAUSTED = "items_exhausted"


class CargoManifestOperation(str, Enum):
    """Finite operations a Cargo manifest may authorize."""

    LOAD = "load"
    RETRIEVE = "retrieve"
    LOAD_AND_RETRIEVE = "load_and_retrieve"


class CargoPayloadKind(str, Enum):
    """Finite payload families and their allowed item kind."""

    RAG_DOCUMENTS = "rag_documents"
    ADAPTER_WEIGHTS = "adapter_weights"
    WEIGHT_SNAPSHOTS = "weight_snapshots"


_PAYLOAD_ITEM_KIND = {
    CargoPayloadKind.RAG_DOCUMENTS: "document",
    CargoPayloadKind.ADAPTER_WEIGHTS: "adapter_weight",
    CargoPayloadKind.WEIGHT_SNAPSHOTS: "weight_snapshot",
}


def _item_kind_for_payload(payload_kind: str) -> str:
    """Return the one item kind authorized by a signed payload kind."""
    try:
        kind = CargoPayloadKind(payload_kind)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in CargoPayloadKind)
        raise ValueError(f"payload_kind must be one of: {allowed}") from exc
    return _PAYLOAD_ITEM_KIND[kind]


@dataclass(frozen=True)
class CompletionCondition:
    """A condition a bus may evaluate to complete a cargo transaction.

    The default bus can independently evaluate only TTL expiry and rejects
    every externally asserted condition. Other kinds are reserved for a bus
    implementation with a trusted, condition-specific evidence provider.
    """

    kind: CompletionKind
    target: str
    """Partition key, receipt ID, or empty for TTL."""

    destruction_aad: Optional[str] = None
    """Reserved evidence context authenticated by the manifest."""

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CompletionKind):
            raise ValueError("completion condition kind must be a CompletionKind")
        _text(self.target, "completion condition target", allow_empty=True)
        if self.destruction_aad is not None:
            _text(self.destruction_aad, "destruction_aad")

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"kind": self.kind.value, "target": self.target}
        if self.destruction_aad is not None:
            d["destruction_aad"] = self.destruction_aad
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CompletionCondition:
        if (
            not isinstance(d, dict)
            or not {"kind", "target"}.issubset(d)
            or set(d) - {"kind", "target", "destruction_aad"}
        ):
            raise ValueError("completion condition has missing or unknown fields")
        return cls(
            kind=CompletionKind(d["kind"]),
            target=d["target"],
            destruction_aad=d.get("destruction_aad"),
        )


# ---------------------------------------------------------------------------
# CargoManifest — the bill of goods, AAD-bound
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CargoManifest:
    """AAD-bound bill of goods for a regime bus transaction.

    The manifest IS the AAD.  Every field participates in the AES-GCM
    authentication tag.  If any field is wrong, decryption is a
    cryptographic impossibility — the channel does not exist.
    """

    manifest_id: str
    tenant_id: str
    regime_id: int
    embedding_spec: EmbeddingSpec
    architecture: str
    """``ArchitectureSpec.to_topology()`` or a topology string."""

    payload_hash: str
    """SHA-256 of the serialized payload."""

    payload_kind: str
    """``"rag_documents"``, ``"adapter_weights"``, ``"weight_snapshots"``."""

    item_count: int
    parent_lockbox_hash: str
    issued_at_epoch: int
    subject_id: str
    model_digest: str
    operation: str
    policy_version: str
    partition_key: str
    gate_embeddings_at_rest: bool
    expires_epoch: Optional[int] = field(default_factory=lambda: int(_time.time()) + 3600)
    completion_conditions: tuple[CompletionCondition, ...] = ()
    max_reissues: int = 0
    """How many times tokens may be reissued under this manifest."""
    gate_release: GateReleaseIdentity = field(default_factory=current_release_identity)

    def __post_init__(self) -> None:
        _text(self.manifest_id, "manifest_id")
        _text(self.tenant_id, "tenant_id")
        _text(self.subject_id, "subject_id")
        _text(self.model_digest, "model_digest")
        operation = _text(self.operation, "operation")
        try:
            CargoManifestOperation(operation)
        except ValueError as exc:
            allowed = ", ".join(member.value for member in CargoManifestOperation)
            raise ValueError(f"operation must be one of: {allowed}") from exc
        _text(self.policy_version, "policy_version")
        _text(self.partition_key, "partition_key")
        if type(self.gate_embeddings_at_rest) is not bool:
            raise ValueError("gate_embeddings_at_rest must be an exact boolean")
        if (
            isinstance(self.regime_id, bool)
            or not isinstance(self.regime_id, int)
            or self.regime_id < 0
        ):
            raise ValueError("regime_id must be an integer >= 0")
        if not isinstance(self.embedding_spec, EmbeddingSpec):
            raise ValueError("embedding_spec must be an EmbeddingSpec")
        _text(self.architecture, "architecture", allow_empty=True)
        if not _is_sha256(self.payload_hash):
            raise ValueError("payload_hash must be a lowercase SHA-256 digest")
        payload_kind = _text(self.payload_kind, "payload_kind")
        _item_kind_for_payload(payload_kind)
        if (
            isinstance(self.item_count, bool)
            or not isinstance(self.item_count, int)
            or not 0 <= self.item_count <= _MAX_ITEMS
        ):
            raise ValueError(f"item_count must be an integer in [0, {_MAX_ITEMS}]")
        _text(self.parent_lockbox_hash, "parent_lockbox_hash", allow_empty=True)
        if self.parent_lockbox_hash and not _is_sha256(self.parent_lockbox_hash):
            raise ValueError("parent_lockbox_hash must be empty or a lowercase SHA-256 digest")
        for name, value in (
            ("issued_at_epoch", self.issued_at_epoch),
            ("max_reissues", self.max_reissues),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be an integer >= 0")
        if self.expires_epoch is not None and (
            isinstance(self.expires_epoch, bool)
            or not isinstance(self.expires_epoch, int)
            or self.expires_epoch < 0
        ):
            raise ValueError("expires_epoch must be an integer >= 0 or None")
        if not isinstance(self.completion_conditions, tuple) or not all(
            isinstance(condition, CompletionCondition) for condition in self.completion_conditions
        ):
            raise ValueError("completion_conditions must be a tuple of CompletionCondition")
        if not isinstance(self.gate_release, GateReleaseIdentity):
            raise ValueError("gate_release must be a validated GateReleaseIdentity")

    def to_aad(self) -> bytes:
        """Serialize every manifest field to unambiguous canonical AAD."""
        return _canonical_json(
            {
                "architecture": self.architecture,
                "completion_conditions": [
                    condition.to_dict() for condition in self.completion_conditions
                ],
                "embedding_spec": json.loads(self.embedding_spec.to_canonical()),
                "expires_epoch": self.expires_epoch,
                "gate_release": self.gate_release.to_dict(),
                "gate_embeddings_at_rest": self.gate_embeddings_at_rest,
                "issued_at_epoch": self.issued_at_epoch,
                "item_count": self.item_count,
                "manifest_id": self.manifest_id,
                "max_reissues": self.max_reissues,
                "parent_lockbox_hash": self.parent_lockbox_hash,
                "payload_hash": self.payload_hash,
                "payload_kind": self.payload_kind,
                "partition_key": self.partition_key,
                "policy_version": self.policy_version,
                "regime_id": self.regime_id,
                "schema": "schemen/cargo-manifest-v7",
                "model_digest": self.model_digest,
                "operation": self.operation,
                "subject_id": self.subject_id,
                "tenant_id": self.tenant_id,
            }
        )

    def is_expired(self, now: Optional[float] = None) -> bool:
        verification_epoch = _verification_epoch(now)
        if self.expires_epoch is None:
            return False
        return verification_epoch >= self.expires_epoch

    def fingerprint(self) -> str:
        """SHA-256 of the AAD bytes."""
        return hashlib.sha256(self.to_aad()).hexdigest()

    @staticmethod
    def generate_id() -> str:
        return f"cargo-{uuid.uuid4().hex[:16]}"


# ---------------------------------------------------------------------------
# CargoItem — a single item in a cargo payload
# ---------------------------------------------------------------------------


@dataclass
class CargoItem:
    """A single document/vector/weight in a cargo payload."""

    content: str
    """Text content or label."""

    embedding: np.ndarray
    """Embedding vector."""

    kind: str = "document"
    """``"document"``, ``"weight_snapshot"``, ``"adapter_weight"``."""

    metadata: Dict[str, Any] = field(default_factory=dict)

    doc_id: Optional[str] = None
    """Caller-specified ID; auto-generated if ``None``."""


# ---------------------------------------------------------------------------
# CargoOperation — record of a single load/unload within a session
# ---------------------------------------------------------------------------


class OperationKind(str, Enum):
    LOAD = "load"
    UNLOAD = "unload"


@dataclass(frozen=True)
class CargoOperation:
    """Record of a single operation within a docking session."""

    kind: OperationKind
    timestamp_epoch: float
    item_count: int
    items_hash: str
    """SHA-256 over the canonical operation contract and material items."""

    def __post_init__(self) -> None:
        if not isinstance(self.kind, OperationKind):
            raise ValueError("kind must be an OperationKind")
        if (
            isinstance(self.timestamp_epoch, bool)
            or not isinstance(self.timestamp_epoch, (int, float))
            or not math.isfinite(self.timestamp_epoch)
            or self.timestamp_epoch < 0
        ):
            raise ValueError("timestamp_epoch must be a finite non-negative number")
        if (
            isinstance(self.item_count, bool)
            or not isinstance(self.item_count, int)
            or not 0 <= self.item_count <= _MAX_ITEMS
        ):
            raise ValueError(f"item_count must be an integer in [0, {_MAX_ITEMS}]")
        if not _is_sha256(self.items_hash):
            raise ValueError("items_hash must be a lowercase SHA-256 digest")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "timestamp_epoch": self.timestamp_epoch,
            "item_count": self.item_count,
            "items_hash": self.items_hash,
        }


# ---------------------------------------------------------------------------
# LoadResult / UnloadResult — operation return types
# ---------------------------------------------------------------------------


@dataclass
class LoadResult:
    """Result of loading cargo into a regime bus."""

    doc_ids: List[str]
    items_hash: str
    item_count: int


@dataclass
class UnloadResult:
    """Result of unloading cargo from a regime bus."""

    docs: List[Any]
    """``RetrievedDoc`` instances from the underlying store."""

    gate_mask: Any
    """``GateMask`` for the regime."""

    regime_id: int
    items_hash: str
    item_count: int


# ---------------------------------------------------------------------------
# CargoReceipt — signed, manifest-scoped transaction evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CargoReceipt:
    """Authenticated record of a Cargo transaction.

    Signed by a key derived for the exact manifest scope. Records every
    operation, canonical input/output hashes, the embedding spec, scope, and
    timestamps. Verification requires the expected manifest; the signature
    does not establish that an operator's policy or external system was sound.
    """

    receipt_id: str
    manifest_id: str
    tenant_id: str
    regime_id: int
    subject_id: str
    model_digest: str
    operation: str
    policy_version: str
    partition_key: str
    embedding_spec: EmbeddingSpec
    operations: tuple[CargoOperation, ...]
    cargo_in_hash: str
    """SHA-256 chain over canonical load-operation hashes, in order."""

    cargo_out_hash: str
    """SHA-256 chain over canonical retrieval-operation hashes, in order."""

    docked_at_epoch: float
    departed_at_epoch: float
    reissue_count: int
    """How many token reissues occurred during this session."""

    manifest_fingerprint: str
    """SHA-256 of the exact authenticated manifest."""

    gate_release: GateReleaseIdentity = field(default_factory=current_release_identity)

    completion_conditions_met: tuple[str, ...] = ()
    """Condition kinds that fired during the session."""

    gate_signature: bytes = b""
    """HMAC-SHA256 over the receipt body, keyed by the tenant's derived key."""

    def body_bytes(self) -> bytes:
        """Canonical bytes for signing (everything except the signature)."""
        return _canonical_json(
            {
                "cargo_in_hash": self.cargo_in_hash,
                "cargo_out_hash": self.cargo_out_hash,
                "completion_conditions_met": sorted(self.completion_conditions_met),
                "departed_at_epoch": self.departed_at_epoch,
                "docked_at_epoch": self.docked_at_epoch,
                "embedding_spec": json.loads(self.embedding_spec.to_canonical()),
                "gate_release": self.gate_release.to_dict(),
                "manifest_fingerprint": self.manifest_fingerprint,
                "manifest_id": self.manifest_id,
                "model_digest": self.model_digest,
                "operation": self.operation,
                "operations": [operation.to_dict() for operation in self.operations],
                "partition_key": self.partition_key,
                "policy_version": self.policy_version,
                "receipt_id": self.receipt_id,
                "regime_id": self.regime_id,
                "reissue_count": self.reissue_count,
                "schema": "schemen/cargo-receipt-v6",
                "subject_id": self.subject_id,
                "tenant_id": self.tenant_id,
            }
        )

    def verify(self, tenant_key_secret: bytes) -> bool:
        """Verify only the receipt's HMAC signature.

        Security decisions should use ``DefaultRegimeBus.verify_receipt`` with
        an explicit expected manifest so signature and scope are checked
        together.
        """
        if not isinstance(tenant_key_secret, bytes) or len(tenant_key_secret) != 32:
            return False
        try:
            expected = _hmac.new(
                tenant_key_secret,
                self.body_bytes(),
                hashlib.sha256,
            ).digest()
            return _hmac.compare_digest(expected, self.gate_signature)
        except (AttributeError, TypeError, ValueError):
            return False

    @staticmethod
    def generate_id() -> str:
        return f"rcpt-{uuid.uuid4().hex[:16]}"


# ---------------------------------------------------------------------------
# RegimeBus protocol — the dock
# ---------------------------------------------------------------------------


class DockingSession(Protocol):
    """A live transaction at a regime bus.

    Created by ``RegimeBus.dock()``, ended by ``depart()``.
    """

    @property
    def manifest(self) -> CargoManifest: ...

    @property
    def regime_id(self) -> int: ...

    def load_cargo(
        self,
        items: Sequence[CargoItem],
    ) -> LoadResult:
        """Ingest items into the regime's partition."""
        ...

    def unload_cargo(
        self,
        query: Any,
        top_k: int = 10,
        *,
        kind: Optional[str] = None,
    ) -> UnloadResult:
        """Retrieve items from the regime's partition."""
        ...

    def depart(self) -> CargoReceipt:
        """Finalize the session and produce a signed receipt."""
        ...


class RegimeBus(Protocol):
    """Transactional surface for a regime partition.

    One bus per regime.  The bus is the thing you plug in to for a while.
    """

    @property
    def regime_id(self) -> int: ...

    @property
    def partition_key(self) -> str: ...

    def dock(
        self,
        manifest: CargoManifest,
        gate_key_secret: bytes,
    ) -> DockingSession:
        """Validate the manifest and open a docking session.

        Raises ``CargoAuthenticationError`` if the manifest AAD cannot
        be validated against the gate key and regime.
        """
        ...


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CargoError(Exception):
    """Base for all cargo protocol errors."""


class CargoAuthenticationError(CargoError):
    """The manifest AAD does not match the gate key / regime.

    The cryptographic channel does not exist.
    """


class CargoExpiredError(CargoError):
    """The manifest has expired."""


class CargoSessionClosedError(CargoError):
    """Operation attempted on a session that has already departed."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_items_hash(doc_ids: Sequence[str]) -> str:
    """SHA-256 over length-prefixed identifiers (deterministic order)."""
    h = hashlib.sha256()
    for did in doc_ids:
        _update_length_prefixed(h, _text(did, "doc_id").encode("utf-8"))
    return h.hexdigest()


def _canonical_array(value: Any, name: str) -> bytes:
    """Canonical big-endian float64 array bytes with shape and rank."""
    array = np.ascontiguousarray(np.asarray(value, dtype=">f8"))
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return b"".join(
        (
            len(array.shape).to_bytes(8, "big"),
            *(dimension.to_bytes(8, "big") for dimension in array.shape),
            array.tobytes(),
        )
    )


def compute_load_hash(
    manifest_fingerprint: str,
    partition_key: str,
    doc_ids: Sequence[str],
) -> str:
    """Bind a load result to the authenticated manifest and assigned IDs."""
    if not _is_sha256(manifest_fingerprint):
        raise ValueError("manifest_fingerprint must be a lowercase SHA-256 digest")
    h = hashlib.sha256()
    h.update(b"schemen/cargo-load-v1")
    for field_value in (
        manifest_fingerprint.encode("ascii"),
        _text(partition_key, "partition_key").encode("utf-8"),
    ):
        _update_length_prefixed(h, field_value)
    for doc_id in doc_ids:
        _update_length_prefixed(h, _text(doc_id, "doc_id").encode("utf-8"))
    return h.hexdigest()


def compute_retrieval_hash(
    *,
    query_hash: str,
    partition_key: str,
    top_k: int,
    kind: Optional[str],
    docs: Sequence[Any],
    output_vectors: Any | None = None,
    bridge_hash: str | None = None,
) -> str:
    """Bind a retrieval's exact request contract and returned material."""
    if not _is_sha256(query_hash):
        raise ValueError("query_hash must be a lowercase SHA-256 digest")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    h = hashlib.sha256()
    h.update(b"schemen/cargo-retrieval-v1")
    contract = _canonical_json(
        {
            "bridge_hash": bridge_hash,
            "kind": kind,
            "partition_key": _text(partition_key, "partition_key"),
            "query_hash": query_hash,
            "top_k": top_k,
        }
    )
    _update_length_prefixed(h, contract)
    for doc in docs:
        score = float(doc.score)
        if not math.isfinite(score):
            raise ValueError("retrieval scores must be finite")
        fields = (
            _text(doc.doc_id, "doc_id").encode("utf-8"),
            _text(doc.content, "content", allow_empty=True).encode("utf-8"),
            _canonical_array(doc.embedding, "retrieved embedding"),
            _text(doc.partition_key, "partition_key").encode("utf-8"),
            _text(doc.kind, "kind").encode("utf-8"),
            _canonical_json(doc.metadata or {}),
            struct.pack(">d", score),
        )
        for field_value in fields:
            _update_length_prefixed(h, field_value)
    if output_vectors is None:
        _update_length_prefixed(h, b"no-output-vectors")
    else:
        _update_length_prefixed(
            h,
            b"output-vectors" + _canonical_array(output_vectors, "output vectors"),
        )
    return h.hexdigest()


def compute_payload_hash(items: Sequence[CargoItem]) -> str:
    """Bind every field that can affect Cargo ingestion.

    Embeddings are normalized exactly as the bundled stores ingest them:
    contiguous one-dimensional float64 bytes. Metadata must be strict JSON so the
    same logical payload has one portable representation.
    """
    h = hashlib.sha256()
    for item in items:
        if not isinstance(item, CargoItem):
            raise ValueError("Cargo payload entries must be CargoItem instances")
        content = _text(item.content, "content", allow_empty=True).encode("utf-8")
        kind = _text(item.kind, "kind").encode("utf-8")
        doc_id = b"" if item.doc_id is None else _text(item.doc_id, "doc_id").encode("utf-8")
        source_embedding = np.asarray(item.embedding, dtype=np.float64)
        if source_embedding.ndim != 1 or source_embedding.size == 0:
            raise ValueError("Cargo embeddings must be non-empty one-dimensional vectors")
        embedding = np.ascontiguousarray(source_embedding)
        if not np.all(np.isfinite(embedding)):
            raise ValueError("Cargo embeddings must contain only finite values")
        metadata = _canonical_json(item.metadata if item.metadata is not None else {})
        for field_value in (
            content,
            kind,
            b"none" if item.doc_id is None else b"some" + doc_id,
            embedding.shape[0].to_bytes(8, "big") + embedding.tobytes(),
            metadata,
        ):
            _update_length_prefixed(h, field_value)
    return h.hexdigest()


def hash_vocabulary(vocab: Sequence[str]) -> str:
    """SHA-256 of sorted vocabulary tokens, for EmbeddingSpec binding."""
    h = hashlib.sha256()
    for token in sorted(vocab):
        _update_length_prefixed(
            h, _text(token, "vocabulary token", allow_empty=True).encode("utf-8")
        )
    return h.hexdigest()


# ---------------------------------------------------------------------------
# VectorBridge — dimensional step-up / step-down for direct vector comms
# ---------------------------------------------------------------------------


class VectorBridge:
    """Project vectors between embedding spaces for vector-to-vector inference.

    When the RAG store's embedding dimension differs from the model's hidden
    dimension, a learned (or random) linear projection bridges the gap.
    This cuts out the text -> vector -> text -> vector round-trip: the
    orchestrator retrieves vectors from the RAG store and injects them
    directly into the model's representational space.

    The projection matrix is part of the bridge's identity — changing it
    changes the bridge.  The matrix hash is available for AAD binding so
    both parties agree on the alignment.
    """

    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        projection: Optional[np.ndarray] = None,
    ) -> None:
        if (
            isinstance(source_dim, bool)
            or not isinstance(source_dim, int)
            or source_dim <= 0
            or isinstance(target_dim, bool)
            or not isinstance(target_dim, int)
            or target_dim <= 0
        ):
            raise ValueError("source_dim and target_dim must be positive integers")
        self.source_dim = source_dim
        self.target_dim = target_dim
        if projection is not None:
            if projection.shape != (source_dim, target_dim):
                raise ValueError(
                    f"Projection shape {projection.shape} does not match "
                    f"({source_dim}, {target_dim})"
                )
            if not np.all(np.isfinite(projection)):
                raise ValueError("Projection must contain only finite values")
            self._projection: Optional[np.ndarray] = projection.astype(
                np.float64,
                copy=True,
            )
        elif source_dim == target_dim:
            # An implicit identity bridge needs neither an O(d^2) matrix nor
            # an O(d^3) SVD.  Keep it symbolic so large same-space bridges
            # remain cheap to construct.
            self._projection = None
        else:
            rng = np.random.default_rng(
                seed=int.from_bytes(
                    hashlib.sha256(f"bridge:{source_dim}:{target_dim}".encode()).digest()[:8],
                    "big",
                )
            )
            raw = rng.standard_normal((source_dim, target_dim))
            u, _, vt = np.linalg.svd(raw, full_matrices=False)
            if source_dim <= target_dim:
                self._projection = (u @ vt).astype(np.float64)
            else:
                self._projection = raw.astype(np.float64)
                self._projection /= (
                    np.linalg.norm(
                        self._projection,
                        axis=1,
                        keepdims=True,
                    )
                    + 1e-12
                )

    @property
    def is_identity(self) -> bool:
        return self._projection is None

    def project(self, vectors: np.ndarray) -> np.ndarray:
        """Project vectors from source space to target space.

        Input: ``(N, source_dim)`` or ``(source_dim,)``
        Output: same leading dims, last dim = ``target_dim``
        """
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim not in (1, 2):
            raise ValueError("vectors must be a 1D vector or 2D batch")
        if vectors.shape[-1] != self.source_dim:
            raise ValueError(
                f"vector dimension {vectors.shape[-1]} does not match "
                f"bridge source dimension {self.source_dim}"
            )
        if not np.all(np.isfinite(vectors)):
            raise ValueError("vectors must contain only finite values")
        if self.is_identity:
            return vectors
        squeeze = vectors.ndim == 1
        if squeeze:
            vectors = vectors[np.newaxis, :]
        result = vectors @ self._projection
        return np.asarray(result[0] if squeeze else result)

    def projection_hash(self) -> str:
        """SHA-256 of the projection matrix bytes, for AAD binding."""
        if self._projection is None:
            identity = f"schemen/vector-bridge/identity-v1:{self.source_dim}".encode()
            return hashlib.sha256(identity).hexdigest()
        return hashlib.sha256(self._projection.tobytes()).hexdigest()

    def to_embedding_spec_pair(
        self,
        source_spec: EmbeddingSpec,
        target_spec: EmbeddingSpec,
    ) -> Dict[str, Any]:
        """Serialize the bridge for manifest binding."""
        return {
            "source": source_spec.to_canonical(),
            "target": target_spec.to_canonical(),
            "projection_hash": self.projection_hash(),
            "source_dim": self.source_dim,
            "target_dim": self.target_dim,
        }


class VectorPayload:
    """A batch of vectors ready for direct injection into a model.

    Produced by ``unload_cargo()`` when the caller requests vector-native
    results.  Contains the projected vectors (in the model's hidden
    dimension), the gate mask, and provenance metadata.
    """

    def __init__(
        self,
        vectors: np.ndarray,
        gate_mask: Any,
        regime_id: int,
        source_doc_ids: List[str],
        bridge: Optional[VectorBridge] = None,
        embedding_spec: Optional[EmbeddingSpec] = None,
    ) -> None:
        source = np.asarray(vectors, dtype=np.float64)
        if (
            source.ndim not in {1, 2}
            or source.shape[-1] <= 0
            or (source.ndim == 1 and source.size == 0)
        ):
            raise ValueError("vectors must be a 1D vector or a 2D batch with dimensions")
        if source.nbytes > _MAX_FRAME_BYTES:
            raise ValueError("Vector payload exceeds the byte limit")
        if not np.all(np.isfinite(source)):
            raise ValueError("Vector payload must contain only finite values")
        if isinstance(regime_id, bool) or not isinstance(regime_id, int) or regime_id < 0:
            raise ValueError("regime_id must be an integer >= 0")
        if not isinstance(source_doc_ids, list) or any(
            not isinstance(doc_id, str) or not doc_id or "\x00" in doc_id
            for doc_id in source_doc_ids
        ):
            raise ValueError("source_doc_ids must be a list of non-empty strings without NUL")
        expected_count = source.shape[0] if source.ndim == 2 else 1
        if len(source_doc_ids) != expected_count:
            raise ValueError("source_doc_ids must identify every vector exactly once")
        try:
            invalid_doc_ids = len(set(source_doc_ids)) != len(source_doc_ids) or any(
                len(doc_id.encode("utf-8")) > 4096 for doc_id in source_doc_ids
            )
        except UnicodeEncodeError as exc:
            raise ValueError("source_doc_ids exceed the payload metadata limit") from exc
        if len(source_doc_ids) > _MAX_ITEMS or invalid_doc_ids:
            raise ValueError("source_doc_ids exceed the payload metadata limit")
        if embedding_spec is not None and not isinstance(embedding_spec, EmbeddingSpec):
            raise ValueError("embedding_spec must be an EmbeddingSpec or None")
        if embedding_spec is not None and embedding_spec.dimensions != source.shape[-1]:
            raise ValueError("Vector payload does not match the embedding spec")
        if bridge is not None and not isinstance(bridge, VectorBridge):
            raise ValueError("bridge must be a VectorBridge or None")
        if bridge is not None and bridge.source_dim != source.shape[-1]:
            raise ValueError("Vector payload does not match the bridge source dimension")

        mask_array = (
            gate_mask.to_numpy()
            if hasattr(gate_mask, "to_numpy")
            else np.asarray(gate_mask, dtype=np.float64)
        )
        mask_array = np.asarray(mask_array, dtype=np.float64)
        if (
            mask_array.ndim != 1
            or mask_array.shape[0] != source.shape[-1]
            or not np.all(np.isfinite(mask_array))
            or not np.all((mask_array == 0.0) | (mask_array == 1.0))
        ):
            raise ValueError("gate_mask must be a matching finite binary vector")
        mask_regime = getattr(gate_mask, "regime_id", regime_id)
        if mask_regime != regime_id:
            raise ValueError("gate_mask regime does not match the vector payload")

        detached_source = np.array(source, dtype=np.float64, copy=True, order="C")
        detached_source.setflags(write=False)
        gated_source = detached_source * mask_array
        output = bridge.project(detached_source) if bridge is not None else detached_source
        gated_output = bridge.project(gated_source) if bridge is not None else gated_source
        if output.nbytes > _MAX_FRAME_BYTES or gated_output.nbytes > _MAX_FRAME_BYTES:
            raise ValueError("Vector payload output exceeds the byte limit")
        detached_output = np.array(output, dtype=np.float64, copy=True, order="C")
        detached_output.setflags(write=False)
        detached_gated = np.array(
            gated_output,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        detached_gated.setflags(write=False)

        self._source_vectors = detached_source
        self._vectors = detached_output
        self._gated_vectors = detached_gated
        self.gate_mask = gate_mask
        self.regime_id = regime_id
        self._source_doc_ids = tuple(source_doc_ids)
        self.bridge = bridge
        self._bridge_hash = bridge.projection_hash() if bridge is not None else None
        self.embedding_spec = embedding_spec

    @property
    def vectors(self) -> np.ndarray:
        """Return detached vectors in the bridge target space, if any."""
        vectors = self._vectors.copy()
        vectors.setflags(write=False)
        return vectors

    @property
    def source_doc_ids(self) -> List[str]:
        """Return detached provenance identifiers in vector order."""
        return list(self._source_doc_ids)

    @property
    def count(self) -> int:
        return int(self._vectors.shape[0]) if self._vectors.ndim == 2 else 1

    @property
    def dim(self) -> int:
        return int(self._vectors.shape[-1])

    def gated(self) -> np.ndarray:
        """Return a detached source-gated vector batch in the target space."""
        vectors = self._gated_vectors.copy()
        vectors.setflags(write=False)
        return vectors

    def to_stream_frame(self, *, gate: bool = True) -> InferenceStreamFrame:
        """Package as an inference stream frame for wire transport."""
        vecs = self.gated() if gate else self.vectors
        return InferenceStreamFrame(
            vectors=vecs,
            regime_id=self.regime_id,
            source_doc_ids=self.source_doc_ids,
            embedding_spec=self.embedding_spec,
            bridge_hash=self._bridge_hash,
        )


# ---------------------------------------------------------------------------
# InferenceStream — wire protocol for vector transport to GPU backbones
# ---------------------------------------------------------------------------


@dataclass
class InferenceStreamFrame:
    """A single frame in the inference stream: a matrix of vectors ready
    for the GPU backbone.

    The frame is the unit of transport between the orchestrator (which
    retrieves and gates vectors) and the GPU runtime (which injects them
    into a model's hidden state).  Frames travel as compact float arrays
    -- no tokenization, no text encoding, no prompt construction.

    Wire format: the ``to_wire()`` / ``from_wire()`` methods serialize
    to a dict with base64-encoded float32 arrays.  This is ~10x smaller
    than the equivalent text chunks and parseable by any language.
    """

    vectors: np.ndarray
    """Shape ``(N, D)`` or ``(D,)`` -- the payload."""

    regime_id: int
    """Which regime these vectors belong to."""

    source_doc_ids: List[str]
    """Provenance: which documents contributed these vectors."""

    embedding_spec: Optional[EmbeddingSpec] = None
    """The embedding identity, for compatibility verification."""

    bridge_hash: Optional[str] = None
    """SHA-256 of the projection matrix, if a bridge was applied."""

    sequence_id: Optional[str] = None
    """Stream sequence identifier for multi-frame exchanges."""

    frame_index: int = 0
    """Position in a multi-frame stream (0-indexed)."""

    is_final: bool = True
    """Whether this is the last frame in the stream."""

    def __post_init__(self) -> None:
        self.vectors = self._validated_vectors()
        self.source_doc_ids = list(self.source_doc_ids)

    def _validated_vectors(self) -> np.ndarray:
        """Validate all mutable frame fields and return detached vector bytes."""
        vectors = np.asarray(self.vectors, dtype=np.float32)
        if (
            vectors.ndim not in {1, 2}
            or vectors.size == 0
            or any(dimension <= 0 for dimension in vectors.shape)
        ):
            raise ValueError("vectors must be a non-empty 1D vector or 2D batch")
        if vectors.nbytes > _MAX_FRAME_BYTES:
            raise ValueError("Inference stream frame exceeds the byte limit")
        if not np.all(np.isfinite(vectors)):
            raise ValueError("Inference stream vectors must contain only finite values")
        if (
            isinstance(self.regime_id, bool)
            or not isinstance(self.regime_id, int)
            or self.regime_id < 0
        ):
            raise ValueError("regime_id must be an integer >= 0")
        if not isinstance(self.source_doc_ids, list) or any(
            not isinstance(doc_id, str) or not doc_id or "\x00" in doc_id
            for doc_id in self.source_doc_ids
        ):
            raise ValueError("source_doc_ids must be a list of non-empty strings without NUL")
        try:
            invalid_doc_ids = len(set(self.source_doc_ids)) != len(self.source_doc_ids) or any(
                len(doc_id.encode("utf-8")) > 4096 for doc_id in self.source_doc_ids
            )
        except UnicodeEncodeError as exc:
            raise ValueError("source_doc_ids exceed the frame metadata limit") from exc
        if len(self.source_doc_ids) > _MAX_ITEMS or invalid_doc_ids:
            raise ValueError("source_doc_ids exceed the frame metadata limit")
        if self.embedding_spec is not None and not isinstance(self.embedding_spec, EmbeddingSpec):
            raise ValueError("embedding_spec must be an EmbeddingSpec or None")
        if self.bridge_hash is not None and not _is_sha256(self.bridge_hash):
            raise ValueError("bridge_hash must be a lowercase SHA-256 digest or None")
        if (
            self.embedding_spec is not None
            and self.bridge_hash is None
            and self.embedding_spec.dimensions != vectors.shape[-1]
        ):
            raise ValueError("vector dimension does not match the unbridged embedding spec")
        if self.sequence_id is not None:
            try:
                invalid_sequence_id = (
                    not isinstance(self.sequence_id, str)
                    or not self.sequence_id
                    or "\x00" in self.sequence_id
                    or len(self.sequence_id.encode("utf-8")) > 4096
                )
            except UnicodeEncodeError:
                invalid_sequence_id = True
            if invalid_sequence_id:
                raise ValueError(
                    "sequence_id must be a bounded non-empty string without NUL or None"
                )
        if (
            isinstance(self.frame_index, bool)
            or not isinstance(self.frame_index, int)
            or self.frame_index < 0
        ):
            raise ValueError("frame_index must be an integer >= 0")
        if not isinstance(self.is_final, bool):
            raise ValueError("is_final must be a boolean")
        detached = np.array(vectors, dtype=np.float32, copy=True, order="C")
        detached.setflags(write=False)
        return detached

    @property
    def count(self) -> int:
        return int(self.vectors.shape[0]) if self.vectors.ndim == 2 else 1

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[-1])

    def to_wire(self) -> Dict[str, Any]:
        """Serialize for wire transport.

        Vectors are encoded as base64 float32 for compactness.
        A 768-dim x 8-vector payload is ~24KB base64 vs ~200KB+ as JSON
        text chunks with formatting.
        """
        import base64

        vecs = self._validated_vectors()
        return {
            "v": 1,
            "vectors_b64": base64.b64encode(vecs.tobytes()).decode("ascii"),
            "shape": list(vecs.shape),
            "dtype": "float32",
            "regime_id": self.regime_id,
            "source_doc_ids": list(self.source_doc_ids),
            "embedding_spec": self.embedding_spec.to_canonical() if self.embedding_spec else None,
            "bridge_hash": self.bridge_hash,
            "sequence_id": self.sequence_id,
            "frame_index": self.frame_index,
            "is_final": self.is_final,
        }

    @classmethod
    def from_wire(cls, data: Dict[str, Any]) -> InferenceStreamFrame:
        """Deserialize from wire format."""
        import base64

        required_fields = {
            "dtype",
            "regime_id",
            "shape",
            "source_doc_ids",
            "v",
            "vectors_b64",
            "frame_index",
            "is_final",
        }
        allowed_fields = required_fields | {
            "bridge_hash",
            "embedding_spec",
            "sequence_id",
        }
        if not isinstance(data, dict):
            raise ValueError("Inference stream frame must be an object")
        if not required_fields.issubset(data) or set(data) - allowed_fields:
            raise ValueError("Inference stream frame has missing or unknown fields")
        if data.get("v") != 1 or data.get("dtype") != "float32":
            raise ValueError("Unsupported inference stream frame version or dtype")
        shape_value = data.get("shape")
        if (
            not isinstance(shape_value, list)
            or len(shape_value) not in {1, 2}
            or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in shape_value
            )
        ):
            raise ValueError("Frame shape must contain one or two positive integers")
        shape = tuple(shape_value)
        expected_bytes = math.prod(shape) * np.dtype(np.float32).itemsize
        if expected_bytes > _MAX_FRAME_BYTES:
            raise ValueError("Inference stream frame exceeds the byte limit")

        encoded = data.get("vectors_b64")
        expected_encoded_bytes = 4 * ((expected_bytes + 2) // 3)
        if not isinstance(encoded, str):
            raise ValueError("Invalid base64 vector payload")
        if len(encoded) != expected_encoded_bytes:
            raise ValueError("Frame shape does not match the encoded vector payload length")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid base64 vector payload") from exc
        if len(raw) > _MAX_FRAME_BYTES:
            raise ValueError("Inference stream frame exceeds the byte limit")
        if expected_bytes != len(raw):
            raise ValueError("Frame shape does not match the vector payload length")
        vecs = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
        if not np.all(np.isfinite(vecs)):
            raise ValueError("Inference stream vectors must contain only finite values")

        spec = None
        if data.get("embedding_spec"):
            spec = EmbeddingSpec.from_canonical(data["embedding_spec"])

        return cls(
            vectors=vecs,
            regime_id=data["regime_id"],
            source_doc_ids=data["source_doc_ids"],
            embedding_spec=spec,
            bridge_hash=data.get("bridge_hash"),
            sequence_id=data.get("sequence_id"),
            frame_index=data["frame_index"],
            is_final=data["is_final"],
        )

    def payload_hash(self) -> str:
        """SHA-256 of the vector bytes for receipt binding."""
        return hashlib.sha256(self._validated_vectors().tobytes()).hexdigest()


class InferenceStream(Protocol):
    """Protocol for streaming vector frames to/from a GPU backbone.

    The stream is the transport layer between the orchestrator (which
    holds gated vectors from Cargo Mode) and the inference runtime
    (which injects them into a model).  Implementations may use HTTP,
    gRPC, shared memory, or any transport.

    A stream session looks like:

        stream = connect_inference_stream(runtime_url)
        stream.send_frame(payload.to_stream_frame())
        result = stream.receive_frame()  # model's response vectors
        stream.close()
    """

    def send_frame(self, frame: InferenceStreamFrame) -> None:
        """Send a vector frame to the backbone."""
        ...

    def receive_frame(self) -> InferenceStreamFrame:
        """Receive a response frame from the backbone."""
        ...

    def close(self) -> None:
        """Close the stream."""
        ...
