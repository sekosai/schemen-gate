"""Schemen RAG Adapter — no-migration gated retrieval.

The vector DB is the durable store. The gated model is the hot cache.
The gate acts at inference time, not storage time. Existing corpora
require zero re-embedding. The partition mapping bridges external keys
(Context GUIDs, namespaces) to Schemen regime IDs.

Everything in the store is a vector with metadata. Documents, attention
heads, weight snapshots — all the same type, distinguished by ``kind``.
The store doesn't know or care what the vectors represent.
"""

from __future__ import annotations

import hashlib
import math
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    cast,
    runtime_checkable,
)

import numpy as np

from schemen_gate._cargo import _clone_json_object
from schemen_gate._mask import GateMask

_MAX_RESULTS = 100_000


def _bounded_text(
    value: object,
    name: str,
    *,
    allow_empty: bool = False,
    allow_none: bool = False,
) -> str | None:
    if allow_none and value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a bounded non-empty string without NUL")
    try:
        encoded_length = len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be a bounded non-empty string without NUL") from exc
    if (not allow_empty and not value) or "\x00" in value or encoded_length > 16_384:
        raise ValueError(f"{name} must be a bounded non-empty string without NUL")
    return value


def _positive_top_k(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= _MAX_RESULTS:
        raise ValueError(f"top_k must be an integer in [1, {_MAX_RESULTS}]")
    return value


def _vector_copy(
    value: Any,
    name: str,
    *,
    expected_dimensions: int | None = None,
) -> np.ndarray:
    source = np.asarray(value, dtype=np.float64)
    if source.ndim != 1 or source.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if expected_dimensions is not None and source.shape[0] != expected_dimensions:
        raise ValueError(f"{name} has {source.shape[0]} dimensions; expected {expected_dimensions}")
    if not np.all(np.isfinite(source)):
        raise ValueError(f"{name} must contain only finite values")
    copied = np.array(source, dtype=np.float64, copy=True, order="C")
    copied.setflags(write=False)
    return copied


# ------------------------------------------------------------------
# Errors
# ------------------------------------------------------------------


class PartitionModeError(Exception):
    """Raised when an operation violates the partition's mode constraints."""


class PartitionNotFoundError(KeyError):
    """Raised when a partition key is not registered."""


# ------------------------------------------------------------------
# Enums
# ------------------------------------------------------------------


class PartitionMode(Enum):
    """Controls the read/write behavior of a partition.

    FRESH      — empty partition, start from scratch, allows reads and writes.
    IMMUTABLE  — read-only source of truth. Writes raise PartitionModeError.
    READ_WRITE — bidirectional: model and store co-evolve. Supports the
                 compression/suggestion cycle.
    """

    FRESH = "fresh"
    IMMUTABLE = "immutable"
    READ_WRITE = "read_write"


class CachePolicy(Enum):
    """Model-cache policy declaration.

    Only ``NONE`` is supported by :class:`GatedRAGAdapter`. The training
    policies are reserved enum values and fail closed because the adapter
    cannot prove support confinement for an arbitrary model and optimizer.
    """

    NONE = "none"
    WRITE_THROUGH = "write_through"
    LAZY = "lazy"


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass
class RetrievedDoc:
    """A single result from the vector store."""

    doc_id: str
    content: str
    embedding: np.ndarray
    score: float
    partition_key: str
    kind: str = "document"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GatedRetrievalResult:
    """Result of a gated query: retrieved docs + the regime mask."""

    docs: List[RetrievedDoc]
    gate_mask: GateMask
    regime_id: int
    partition_key: str
    query_hash: str
    """SHA-256 of the exact normalized query vector sent to the store."""

    architecture: Any = None  # ArchitectureSpec when bound


# ------------------------------------------------------------------
# VectorStore protocol
# ------------------------------------------------------------------


@runtime_checkable
class VectorStore(Protocol):
    """Abstract interface for any vector store. Schemen does not own
    or modify the store — it just needs the store to understand
    partition keys.
    """

    def retrieve(
        self,
        query_embedding: np.ndarray,
        partition_key: str,
        top_k: int = 10,
        *,
        kind: Optional[str] = None,
    ) -> List[RetrievedDoc]: ...

    def insert(
        self,
        embedding: np.ndarray,
        document: str,
        partition_key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str: ...

    def insert_many(
        self,
        items: Sequence[tuple[np.ndarray, str, Optional[Dict[str, Any]]]],
        partition_key: str,
    ) -> List[str]:
        """Atomically insert every item or leave the store unchanged."""
        ...

    def list_by_kind(
        self,
        partition_key: str,
        kind: str,
    ) -> List[RetrievedDoc]: ...

    def count(
        self,
        partition_key: str,
        *,
        kind: Optional[str] = None,
    ) -> int: ...

    def delete_partition(self, partition_key: str) -> int:
        """Atomically delete a partition and return the number removed."""
        ...


# ------------------------------------------------------------------
# InMemoryVectorStore
# ------------------------------------------------------------------


@dataclass(frozen=True)
class _StoredVector:
    doc_id: str
    content: str
    embedding: np.ndarray
    partition_key: str
    kind: str
    metadata: Dict[str, Any]


class InMemoryVectorStore:
    """Simple numpy-based vector store for POC and testing.

    Cosine similarity search, keyed by partition. No external deps
    beyond numpy.
    """

    def __init__(self) -> None:
        self._vectors: Dict[str, List[_StoredVector]] = {}
        self._lock = threading.RLock()

    def retrieve(
        self,
        query_embedding: np.ndarray,
        partition_key: str,
        top_k: int = 10,
        *,
        kind: Optional[str] = None,
    ) -> List[RetrievedDoc]:
        _bounded_text(partition_key, "partition_key")
        _positive_top_k(top_k)
        _bounded_text(kind, "kind", allow_none=True)
        q = _vector_copy(query_embedding, "query embedding")
        with self._lock:
            candidates = list(self._vectors.get(partition_key, ()))
        if kind is not None:
            candidates = [v for v in candidates if v.kind == kind]
        if not candidates:
            return []

        q_norm = np.linalg.norm(q)
        if q_norm < 1e-12:
            return []
        q = q / q_norm

        scored: List[tuple[float, _StoredVector]] = []
        for sv in candidates:
            e = sv.embedding.ravel()
            if e.shape[0] != q.shape[0]:
                continue
            e_norm = np.linalg.norm(e)
            if e_norm < 1e-12:
                continue
            score = float(np.dot(q, e / e_norm))
            scored.append((score, sv))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, sv in scored[:top_k]:
            results.append(
                RetrievedDoc(
                    doc_id=sv.doc_id,
                    content=sv.content,
                    embedding=sv.embedding.copy(),
                    score=score,
                    partition_key=sv.partition_key,
                    kind=sv.kind,
                    metadata=_clone_json_object(sv.metadata, name="stored metadata"),
                )
            )
        return results

    def insert(
        self,
        embedding: np.ndarray,
        document: str,
        partition_key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self.insert_many(
            [(embedding, document, metadata)],
            partition_key,
        )[0]

    def insert_many(
        self,
        items: Sequence[tuple[np.ndarray, str, Optional[Dict[str, Any]]]],
        partition_key: str,
    ) -> List[str]:
        """Prepare a complete batch, then commit it under one store lock."""
        _bounded_text(partition_key, "partition_key")
        prepared: list[_StoredVector] = []
        prepared_ids: set[str] = set()
        for embedding, document, metadata in items:
            _bounded_text(document, "document", allow_empty=True)
            meta = _clone_json_object(
                metadata if metadata is not None else {},
                name="metadata",
            )
            doc_id = meta.get("doc_id", str(uuid.uuid4()))
            kind = meta.get("kind", "document")
            _bounded_text(doc_id, "doc_id")
            _bounded_text(kind, "kind")
            if doc_id in prepared_ids:
                raise ValueError("document ids must be unique within an atomic batch")
            prepared_ids.add(doc_id)
            stored_embedding = _vector_copy(embedding, "embedding")
            prepared.append(
                _StoredVector(
                    doc_id=doc_id,
                    content=document,
                    embedding=stored_embedding,
                    partition_key=partition_key,
                    kind=kind,
                    metadata=meta,
                )
            )
        with self._lock:
            self._vectors.setdefault(partition_key, []).extend(prepared)
        return [item.doc_id for item in prepared]

    def list_by_kind(
        self,
        partition_key: str,
        kind: str,
    ) -> List[RetrievedDoc]:
        _bounded_text(partition_key, "partition_key")
        _bounded_text(kind, "kind")
        with self._lock:
            candidates = tuple(self._vectors.get(partition_key, ()))
        return [
            RetrievedDoc(
                doc_id=sv.doc_id,
                content=sv.content,
                embedding=sv.embedding.copy(),
                score=0.0,
                partition_key=sv.partition_key,
                kind=sv.kind,
                metadata=_clone_json_object(sv.metadata, name="stored metadata"),
            )
            for sv in candidates
            if sv.kind == kind
        ]

    def count(
        self,
        partition_key: str,
        *,
        kind: Optional[str] = None,
    ) -> int:
        _bounded_text(partition_key, "partition_key")
        _bounded_text(kind, "kind", allow_none=True)
        with self._lock:
            candidates = tuple(self._vectors.get(partition_key, ()))
        if kind is not None:
            return sum(1 for v in candidates if v.kind == kind)
        return len(candidates)

    def delete_partition(self, partition_key: str) -> int:
        """Remove all vectors for a partition. Returns count deleted."""
        _bounded_text(partition_key, "partition_key")
        with self._lock:
            removed = self._vectors.pop(partition_key, [])
        return len(removed)


# ------------------------------------------------------------------
# PartitionMap
# ------------------------------------------------------------------


@dataclass
class _PartitionEntry:
    regime_id: int
    mode: PartitionMode
    mask: Optional[GateMask] = None


class PartitionMap:
    """Maps external partition keys to Schemen regime IDs.

    Owns the gate key and geometry so downstream code only needs
    a partition key string. Masks are derived lazily and cached.
    """

    def __init__(
        self,
        gate_key: bytes,
        n_dims: int,
        n_regimes: int,
    ) -> None:
        if not isinstance(gate_key, bytes) or not gate_key:
            raise ValueError("gate_key must be non-empty bytes")
        if isinstance(n_dims, bool) or not isinstance(n_dims, int) or n_dims <= 0:
            raise ValueError("n_dims must be a positive integer")
        if (
            isinstance(n_regimes, bool)
            or not isinstance(n_regimes, int)
            or n_regimes <= 0
            or n_regimes > n_dims
        ):
            raise ValueError("n_regimes must be a positive integer not exceeding n_dims")
        self._gate_key = gate_key
        self._n_dims = n_dims
        self._n_regimes = n_regimes
        self._entries: Dict[str, _PartitionEntry] = {}
        self._next_regime: int = 0
        self._lock = threading.RLock()

    @property
    def gate_key(self) -> bytes:
        return self._gate_key

    @property
    def n_dims(self) -> int:
        return self._n_dims

    @property
    def n_regimes(self) -> int:
        return self._n_regimes

    def register(
        self,
        partition_key: str,
        *,
        regime_id: Optional[int] = None,
        mode: PartitionMode = PartitionMode.IMMUTABLE,
    ) -> int:
        """Register a partition key. Auto-assigns regime_id if not given."""
        if not isinstance(partition_key, str) or not partition_key or "\x00" in partition_key:
            raise ValueError("partition_key must be a non-empty string without NUL")
        if not isinstance(mode, PartitionMode):
            raise ValueError("mode must be a PartitionMode")
        with self._lock:
            if partition_key in self._entries:
                existing = self._entries[partition_key].regime_id
                if regime_id is not None and regime_id != existing:
                    raise ValueError(
                        f"partition {partition_key!r} is already bound to regime {existing}"
                    )
                return existing

            if regime_id is None:
                assigned = {entry.regime_id for entry in self._entries.values()}
                available = next(
                    (
                        candidate
                        for candidate in range(self._n_regimes)
                        if candidate not in assigned
                    ),
                    None,
                )
                if available is None:
                    raise ValueError("no unassigned regimes remain")
                regime_id = available
                self._next_regime = max(self._next_regime, regime_id + 1)
            else:
                if (
                    isinstance(regime_id, bool)
                    or not isinstance(regime_id, int)
                    or not 0 <= regime_id < self._n_regimes
                ):
                    raise ValueError(f"regime_id must be an integer in [0, {self._n_regimes})")
                self._next_regime = max(self._next_regime, regime_id + 1)

            self._entries[partition_key] = _PartitionEntry(
                regime_id=regime_id,
                mode=mode,
            )
            return regime_id

    def get_regime_id(self, partition_key: str) -> int:
        with self._lock:
            entry = self._entries.get(partition_key)
            if entry is None:
                raise PartitionNotFoundError(partition_key)
            return entry.regime_id

    def get_mode(self, partition_key: str) -> PartitionMode:
        with self._lock:
            entry = self._entries.get(partition_key)
            if entry is None:
                raise PartitionNotFoundError(partition_key)
            return entry.mode

    def mask_for(self, partition_key: str) -> GateMask:
        """Derive (or return cached) mask for a partition."""
        with self._lock:
            entry = self._entries.get(partition_key)
            if entry is None:
                raise PartitionNotFoundError(partition_key)
            if entry.mask is None:
                entry.mask = GateMask.derive(
                    self._gate_key,
                    entry.regime_id,
                    self._n_dims,
                    self._n_regimes,
                )
            return entry.mask

    def __contains__(self, partition_key: str) -> bool:
        with self._lock:
            return partition_key in self._entries

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._entries.keys())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        with self._lock:
            return {
                "n_dims": self._n_dims,
                "n_regimes": self._n_regimes,
                "gate_key_hash": hashlib.sha256(self._gate_key).hexdigest(),
                "partitions": {
                    k: {"regime_id": e.regime_id, "mode": e.mode.value}
                    for k, e in self._entries.items()
                },
            }


# ------------------------------------------------------------------
# GatedRAGAdapter
# ------------------------------------------------------------------


class GatedRAGAdapter:
    """Orchestrates retrieval through a gated runtime.

    The vector DB stays as-is. The gate acts on the runtime that
    processes retrieved context, not on the storage layer. For new
    documents only, embedding-time gating is opt-in.
    """

    def __init__(
        self,
        store: VectorStore,
        partition_map: PartitionMap,
        *,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        cache_policy: CachePolicy = CachePolicy.NONE,
        architecture: Any = None,
    ) -> None:
        if cache_policy is not CachePolicy.NONE:
            raise ValueError(
                "GatedRAGAdapter supports only CachePolicy.NONE; model training "
                "requires an external, audited support-restricted optimizer loop"
            )
        self.store = store
        self.partition_map = partition_map
        self.embed_fn = embed_fn
        self.cache_policy = cache_policy
        self.architecture = architecture

    # ---- helpers ----

    def _require_writable(self, partition_key: str) -> None:
        mode = self.partition_map.get_mode(partition_key)
        if mode == PartitionMode.IMMUTABLE:
            raise PartitionModeError(
                f"Partition {partition_key!r} is IMMUTABLE — writes are rejected"
            )

    def _embed(self, content: Any) -> np.ndarray:
        """Resolve content to an embedding vector."""
        if isinstance(content, np.ndarray):
            return content
        if self.embed_fn is None:
            raise ValueError("embed_fn required when content is not a numpy array")
        return self.embed_fn(content)

    def _embedding(self, content: Any, name: str) -> np.ndarray:
        return _vector_copy(
            self._embed(content),
            name,
            expected_dimensions=self.partition_map.n_dims,
        )

    def _snapshot_docs(
        self,
        docs: Any,
        partition_key: str,
        *,
        top_k: int | None = None,
        expected_kind: str | None = None,
    ) -> list[RetrievedDoc]:
        """Validate and detach every untrusted vector-store result."""
        if not isinstance(docs, list):
            raise PartitionModeError("vector store must return a list of documents")
        if len(docs) > _MAX_RESULTS:
            raise PartitionModeError("vector store returned too many documents")
        if top_k is not None and len(docs) > top_k:
            raise PartitionModeError("vector store returned more than top_k documents")
        snapshots: list[RetrievedDoc] = []
        seen_ids: set[str] = set()
        try:
            for doc in docs:
                if not isinstance(doc, RetrievedDoc):
                    raise ValueError("result is not a RetrievedDoc")
                _bounded_text(doc.doc_id, "retrieved doc_id")
                _bounded_text(doc.content, "retrieved content", allow_empty=True)
                _bounded_text(doc.partition_key, "retrieved partition_key")
                _bounded_text(doc.kind, "retrieved kind")
                if doc.partition_key != partition_key:
                    raise ValueError("document belongs to a different partition")
                if expected_kind is not None and doc.kind != expected_kind:
                    raise ValueError("vector store returned a document outside the requested kind")
                if doc.doc_id in seen_ids:
                    raise ValueError("retrieval returned a duplicate document id")
                seen_ids.add(doc.doc_id)
                if (
                    isinstance(doc.score, bool)
                    or not isinstance(doc.score, (int, float))
                    or not math.isfinite(doc.score)
                ):
                    raise ValueError("retrieval score must be finite")
                embedding = _vector_copy(
                    doc.embedding,
                    "retrieved embedding",
                    expected_dimensions=self.partition_map.n_dims,
                )
                metadata = _clone_json_object(
                    doc.metadata if doc.metadata is not None else {},
                    name="retrieved metadata",
                )
                snapshots.append(
                    RetrievedDoc(
                        doc_id=doc.doc_id,
                        content=doc.content,
                        embedding=embedding,
                        score=float(doc.score),
                        partition_key=doc.partition_key,
                        kind=doc.kind,
                        metadata=metadata,
                    )
                )
        except (TypeError, ValueError) as exc:
            raise PartitionModeError(f"invalid vector-store result: {exc}") from exc
        return snapshots

    # ---- core operations ----

    def query(
        self,
        content: Any,
        partition_key: str,
        top_k: int = 10,
        *,
        kind: Optional[str] = None,
    ) -> GatedRetrievalResult:
        """Retrieve docs from the store and return with the regime's gate mask.

        The caller applies the mask to their model's inference.
        """
        _positive_top_k(top_k)
        _bounded_text(kind, "kind", allow_none=True)
        regime_id = self.partition_map.get_regime_id(partition_key)
        mask = self.partition_map.mask_for(partition_key)
        query_emb = self._embedding(content, "query embedding")
        normalized_query = np.ascontiguousarray(np.asarray(query_emb, dtype=">f8"))
        query_digest = hashlib.sha256()
        query_digest.update(b"schemen/gated-rag-query-v1")
        query_digest.update(normalized_query.shape[0].to_bytes(8, "big"))
        query_digest.update(normalized_query.tobytes())
        docs = self._snapshot_docs(
            self.store.retrieve(query_emb, partition_key, top_k, kind=kind),
            partition_key,
            top_k=top_k,
            expected_kind=kind,
        )
        return GatedRetrievalResult(
            docs=docs,
            gate_mask=mask,
            regime_id=regime_id,
            partition_key=partition_key,
            query_hash=query_digest.hexdigest(),
            architecture=self.architecture,
        )

    def ingest(
        self,
        content: Any,
        document: str,
        partition_key: str,
        *,
        gate_embedding: bool | None = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Insert a document into the store.

        If gate_embedding is True, the regime mask is applied to the
        embedding before storage (for new docs only — existing data
        is never touched).
        """
        if gate_embedding is None:
            raise ValueError(
                "gate_embedding must be explicitly True or False; inference gating "
                "does not imply storage gating"
            )
        self._require_writable(partition_key)
        emb = self._embedding(content, "ingest embedding")
        if gate_embedding:
            mask = self.partition_map.mask_for(partition_key)
            emb = mask.apply(emb)
        return self.store.insert(emb, document, partition_key, metadata)

    def ingest_many(
        self,
        items: Sequence[tuple[Any, str, Optional[Dict[str, Any]]]],
        partition_key: str,
        *,
        gate_embedding: bool | None = None,
    ) -> list[str]:
        """Atomically insert a prepared batch for Cargo-style exact loads.

        A store must expose ``insert_many`` with an all-or-nothing contract.
        Cargo does not fall back to sequential writes because their partial
        failure cannot be reconciled with an exact receipt.
        """
        if gate_embedding is None:
            raise ValueError(
                "gate_embedding must be explicitly True or False; inference gating "
                "does not imply storage gating"
            )
        self._require_writable(partition_key)
        insert_many = getattr(self.store, "insert_many", None)
        if not callable(insert_many):
            raise PartitionModeError("Cargo loads require a store with atomic insert_many support")
        prepared = []
        mask = self.partition_map.mask_for(partition_key) if gate_embedding else None
        for content, document, metadata in items:
            embedding = self._embedding(content, "ingest embedding")
            if mask is not None:
                embedding = mask.apply(embedding)
            prepared.append((embedding, document, metadata))
        return cast(list[str], insert_many(prepared, partition_key))

    def ingest_weights(
        self,
        weights: np.ndarray,
        label: str,
        partition_key: str,
        *,
        kind: str = "weight_snapshot",
    ) -> str:
        """Store model weight cross-sections as vectors in the partition.

        Convenience method for attention heads, FFN snapshots, etc.
        Enforces READ_WRITE mode.
        """
        mode = self.partition_map.get_mode(partition_key)
        if mode != PartitionMode.READ_WRITE:
            raise PartitionModeError(
                f"Weight ingestion requires READ_WRITE mode, "
                f"got {mode.value!r} for {partition_key!r}"
            )
        flat = np.asarray(weights, dtype=np.float64).ravel()
        return self.store.insert(
            flat,
            label,
            partition_key,
            metadata={"kind": kind, "shape": list(weights.shape)},
        )

    def transfer(
        self,
        content: Any,
        document: str,
        source_key: str,
        dest_key: str,
        *,
        gate_embedding: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Bidirectional gated write: read through source gate, write
        through destination gate.

        Requires the source partition to be readable and the destination
        to be writable. The embedding is gated to the destination
        regime's dimensions before insertion.
        """
        self.partition_map.get_regime_id(source_key)
        self._require_writable(dest_key)

        emb = self._embedding(content, "transfer embedding")
        if gate_embedding:
            dest_mask = self.partition_map.mask_for(dest_key)
            emb = dest_mask.apply(emb)
        return self.store.insert(emb, document, dest_key, metadata)

    def compose(self, *partition_keys: str) -> GateMask:
        """Union (OR) masks for multiple partitions.

        Returns a composed mask for cross-partition queries.
        """
        if not partition_keys:
            raise ValueError("At least one partition key required")
        masks = [self.partition_map.mask_for(k) for k in partition_keys]
        result = masks[0]
        for m in masks[1:]:
            result = result | m
        return result

    def destroy(self, partition_key: str) -> GateMask:
        """Delete a partition's current contents and return its mask.

        This does not revoke future authorization or prevent later writes;
        those are separate policy and store-lifecycle operations.
        """
        mask = self.partition_map.mask_for(partition_key)
        delete_partition = getattr(self.store, "delete_partition", None)
        if not callable(delete_partition):
            raise PartitionModeError(
                "Partition destruction requires an atomic delete_partition store operation"
            )
        deleted = delete_partition(partition_key)
        if isinstance(deleted, bool) or not isinstance(deleted, int) or deleted < 0:
            raise PartitionModeError("delete_partition must return a non-negative integer")
        if self.store.count(partition_key) != 0:
            raise PartitionModeError(
                "Partition destruction did not establish an empty postcondition"
            )
        return mask

    def bind_architecture(self, architecture: Any) -> None:
        """Bind an ArchitectureSpec for AAD certification."""
        self.architecture = architecture
