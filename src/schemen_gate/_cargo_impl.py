"""Concrete Cargo Mode implementation.

``DefaultRegimeBus`` wraps a ``GatedRAGAdapter`` partition and adds the
dock / load / unload / depart transactional lifecycle.

``LiveDockingSession`` accumulates operations during a session and
produces a signed ``CargoReceipt`` on ``depart()``.

Requires ``pip install schemen-gate[crypto]``.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import math
import threading
import time as _time
from typing import Any, Optional, Sequence

import numpy as np

from schemen_gate._cargo import (
    CargoAuthenticationError,
    CargoExpiredError,
    CargoItem,
    CargoManifest,
    CargoManifestOperation,
    CargoOperation,
    CargoReceipt,
    CargoSessionClosedError,
    CompletionCondition,
    CompletionKind,
    EmbeddingSpec,
    LoadResult,
    OperationKind,
    UnloadResult,
    VectorBridge,
    VectorPayload,
    _clone_json_object,
    _item_kind_for_payload,
    _text,
    compute_items_hash,
    compute_load_hash,
    compute_payload_hash,
    compute_retrieval_hash,
)
from schemen_gate._rag import GatedRAGAdapter, GatedRetrievalResult
from schemen_gate._release import (
    GateReleaseIdentity,
    current_release_identity,
    release_identity_matches,
)


def _receipt_fulfills_manifest(
    receipt: CargoReceipt,
    manifest: CargoManifest,
) -> bool:
    """Validate operation obligations and internal receipt hash consistency."""
    try:
        if (
            not isinstance(receipt.operations, tuple)
            or not isinstance(receipt.completion_conditions_met, tuple)
            or tuple(sorted(receipt.completion_conditions_met)) != receipt.completion_conditions_met
            or len(set(receipt.completion_conditions_met)) != len(receipt.completion_conditions_met)
            or any(
                not isinstance(condition, str) or not condition
                for condition in receipt.completion_conditions_met
            )
        ):
            return False
        if (
            isinstance(receipt.docked_at_epoch, bool)
            or not isinstance(receipt.docked_at_epoch, (int, float))
            or not math.isfinite(receipt.docked_at_epoch)
            or receipt.docked_at_epoch < 0
            or isinstance(receipt.departed_at_epoch, bool)
            or not isinstance(receipt.departed_at_epoch, (int, float))
            or not math.isfinite(receipt.departed_at_epoch)
            or receipt.departed_at_epoch < 0
            or receipt.docked_at_epoch > receipt.departed_at_epoch
            or isinstance(receipt.reissue_count, bool)
            or not isinstance(receipt.reissue_count, int)
            or not 0 <= receipt.reissue_count <= manifest.max_reissues
        ):
            return False
        loads: list[CargoOperation] = []
        unloads: list[CargoOperation] = []
        previous_timestamp = receipt.docked_at_epoch
        for operation in receipt.operations:
            if not isinstance(operation, CargoOperation):
                return False
            if (
                not isinstance(operation.kind, OperationKind)
                or isinstance(operation.timestamp_epoch, bool)
                or not isinstance(operation.timestamp_epoch, (int, float))
                or not math.isfinite(operation.timestamp_epoch)
                or not receipt.docked_at_epoch
                <= operation.timestamp_epoch
                <= receipt.departed_at_epoch
                or operation.timestamp_epoch < previous_timestamp
                or isinstance(operation.item_count, bool)
                or not isinstance(operation.item_count, int)
                or operation.item_count < 0
                or not isinstance(operation.items_hash, str)
                or len(operation.items_hash) != 64
                or any(character not in "0123456789abcdef" for character in operation.items_hash)
            ):
                return False
            previous_timestamp = operation.timestamp_epoch
            (loads if operation.kind == OperationKind.LOAD else unloads).append(operation)

        if receipt.cargo_in_hash != compute_items_hash(
            [operation.items_hash for operation in loads]
        ):
            return False
        if receipt.cargo_out_hash != compute_items_hash(
            [operation.items_hash for operation in unloads]
        ):
            return False

        granted = CargoManifestOperation(manifest.operation)
        if granted in {
            CargoManifestOperation.LOAD,
            CargoManifestOperation.LOAD_AND_RETRIEVE,
        } and (len(loads) != 1 or loads[0].item_count != manifest.item_count):
            return False
        if granted == CargoManifestOperation.LOAD and unloads:
            return False
        if granted == CargoManifestOperation.RETRIEVE and loads:
            return False

        expected_conditions: set[str] = set()
        for condition in manifest.completion_conditions:
            if (
                condition.kind == CompletionKind.TTL_EXPIRED
                and manifest.expires_epoch is not None
                and receipt.departed_at_epoch >= manifest.expires_epoch
            ):
                expected_conditions.add(condition.kind.value)
        if tuple(sorted(set(receipt.completion_conditions_met))) != tuple(
            sorted(expected_conditions)
        ):
            return False
        return True
    except (AttributeError, TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# LiveDockingSession
# ---------------------------------------------------------------------------


class LiveDockingSession:
    """A live transaction at a regime bus.

    Created by ``DefaultRegimeBus.dock()``.  Accumulates load/unload
    operations and finalizes into a ``CargoReceipt`` on ``depart()``.
    """

    def __init__(
        self,
        manifest: CargoManifest,
        adapter: GatedRAGAdapter,
        partition_key: str,
        tenant_key_secret: bytes,
    ) -> None:
        self._manifest = manifest
        self._adapter = adapter
        self._partition_key = partition_key
        self._tenant_key_secret = tenant_key_secret
        self._docked_at = _time.time()
        self._operations: list[CargoOperation] = []
        self._loaded_ids: list[str] = []
        self._load_completed = False
        self._unloaded_ids: list[str] = []
        self._loaded_operation_hashes: list[str] = []
        self._unloaded_operation_hashes: list[str] = []
        self._closed = False
        self._reissue_count = 0
        self._conditions_met: list[str] = []
        self._state_lock = threading.RLock()

    @property
    def manifest(self) -> CargoManifest:
        return self._manifest

    @property
    def regime_id(self) -> int:
        return self._manifest.regime_id

    def _require_open(self) -> None:
        if self._closed:
            raise CargoSessionClosedError(
                "Session has already departed. Cannot perform further operations."
            )
        if self._manifest.is_expired():
            raise CargoExpiredError(
                f"Manifest {self._manifest.manifest_id} has expired "
                f"(epoch {self._manifest.expires_epoch})."
            )

    def _require_operation(self, required: CargoManifestOperation) -> None:
        """Fail closed unless the signed manifest authorizes this method."""
        granted = CargoManifestOperation(self._manifest.operation)
        allowed = {
            CargoManifestOperation.LOAD: {
                CargoManifestOperation.LOAD,
                CargoManifestOperation.LOAD_AND_RETRIEVE,
            },
            CargoManifestOperation.RETRIEVE: {
                CargoManifestOperation.RETRIEVE,
                CargoManifestOperation.LOAD_AND_RETRIEVE,
            },
        }
        if granted not in allowed[required]:
            raise CargoAuthenticationError(
                f"Manifest operation {granted.value!r} does not authorize {required.value}."
            )

    def _require_partition_binding(self) -> None:
        """Prove the adapter still maps this partition to the signed regime."""
        try:
            actual_regime = self._adapter.partition_map.get_regime_id(self._partition_key)
        except Exception as exc:
            raise CargoAuthenticationError(
                "Cargo partition is not registered in the adapter partition map"
            ) from exc
        if actual_regime != self._manifest.regime_id:
            raise CargoAuthenticationError(
                f"Cargo partition {self._partition_key!r} maps to regime "
                f"{actual_regime}, not signed regime {self._manifest.regime_id}."
            )
        if self._adapter.partition_map.n_dims != self._manifest.embedding_spec.dimensions:
            raise CargoAuthenticationError(
                "Adapter partition dimensions do not match the signed embedding spec"
            )

    def _snapshot_items(self, items: Sequence[CargoItem]) -> list[CargoItem]:
        """Validate and detach the exact payload before hashing or storage."""
        expected_kind = _item_kind_for_payload(self._manifest.payload_kind)
        snapshots: list[CargoItem] = []
        explicit_doc_ids: set[str] = set()
        for index, item in enumerate(items):
            if not isinstance(item, CargoItem):
                raise CargoAuthenticationError(f"Cargo item {index} is not a CargoItem")
            try:
                _text(item.content, f"Cargo item {index} content", allow_empty=True)
                _text(item.kind, f"Cargo item {index} kind")
                if item.doc_id is not None:
                    _text(item.doc_id, f"Cargo item {index} doc_id")
            except ValueError as exc:
                raise CargoAuthenticationError(str(exc)) from exc
            if item.kind != expected_kind:
                raise CargoAuthenticationError(
                    f"Cargo payload kind {self._manifest.payload_kind!r} requires "
                    f"item kind {expected_kind!r}, got {item.kind!r}."
                )
            if item.doc_id is not None:
                if item.doc_id in explicit_doc_ids:
                    raise CargoAuthenticationError(
                        "Cargo item doc_ids must be unique within one payload"
                    )
                explicit_doc_ids.add(item.doc_id)
            source_embedding = np.asarray(item.embedding, dtype=np.float64)
            if source_embedding.ndim != 1 or source_embedding.size == 0:
                raise CargoAuthenticationError(
                    f"Cargo item {index} embedding must be a non-empty 1-D vector"
                )
            if source_embedding.shape[0] != self._manifest.embedding_spec.dimensions:
                raise CargoAuthenticationError(
                    f"Cargo item {index} has {source_embedding.shape[0]} dimensions; "
                    f"the signed embedding spec requires "
                    f"{self._manifest.embedding_spec.dimensions}."
                )
            if not np.all(np.isfinite(source_embedding)):
                raise CargoAuthenticationError(
                    f"Cargo item {index} embedding must contain only finite values"
                )
            embedding = np.array(
                source_embedding,
                dtype=np.float64,
                copy=True,
                order="C",
            )
            embedding.setflags(write=False)
            try:
                metadata = _clone_json_object(
                    item.metadata if item.metadata is not None else {},
                    name=f"Cargo item {index} metadata",
                )
            except ValueError as exc:
                raise CargoAuthenticationError(str(exc)) from exc
            snapshots.append(
                CargoItem(
                    content=item.content,
                    embedding=embedding,
                    kind=item.kind,
                    metadata=metadata,
                    doc_id=item.doc_id,
                )
            )
        return snapshots

    def _require_retrieval_scope(self, result: GatedRetrievalResult) -> None:
        """Reject any adapter result that crosses the authenticated scope."""
        if (
            result.partition_key != self._partition_key
            or result.regime_id != self._manifest.regime_id
            or result.gate_mask.regime_id != self._manifest.regime_id
            or result.gate_mask.n_dims != self._manifest.embedding_spec.dimensions
            or any(doc.partition_key != self._partition_key for doc in result.docs)
        ):
            raise CargoAuthenticationError(
                "Adapter retrieval result does not match the authenticated "
                "partition, regime, and embedding scope"
            )

    def _require_manifest_obligations(self) -> None:
        """Require every declared load to complete before a success receipt."""
        granted = CargoManifestOperation(self._manifest.operation)
        loads = [op for op in self._operations if op.kind == OperationKind.LOAD]
        unloads = [op for op in self._operations if op.kind == OperationKind.UNLOAD]
        if granted in {
            CargoManifestOperation.LOAD,
            CargoManifestOperation.LOAD_AND_RETRIEVE,
        } and (len(loads) != 1 or loads[0].item_count != self._manifest.item_count):
            raise CargoAuthenticationError(
                "Cannot issue a success receipt before the signed load payload "
                "has completed exactly once"
            )
        if granted == CargoManifestOperation.LOAD and unloads:
            raise CargoAuthenticationError("A load-only receipt contains retrieval work")
        if granted == CargoManifestOperation.RETRIEVE and loads:
            raise CargoAuthenticationError("A retrieve-only receipt contains load work")

    def load_cargo(
        self,
        items: Sequence[CargoItem],
    ) -> LoadResult:
        """Atomically ingest the manifest payload exactly once."""
        with self._state_lock:
            return self._load_cargo_unlocked(items)

    def _load_cargo_unlocked(
        self,
        items: Sequence[CargoItem],
    ) -> LoadResult:
        """Ingest items into the regime's partition via ``GatedRAGAdapter``."""
        self._require_open()
        self._require_operation(CargoManifestOperation.LOAD)
        self._require_partition_binding()
        if self._load_completed:
            raise CargoAuthenticationError("Manifest payload may be loaded only once.")
        if len(items) != self._manifest.item_count:
            raise CargoAuthenticationError(
                f"Manifest declares {self._manifest.item_count} items, "
                f"but caller supplied {len(items)}."
            )
        snapshots = self._snapshot_items(items)
        supplied_hash = compute_payload_hash(snapshots)
        if not _hmac.compare_digest(supplied_hash, self._manifest.payload_hash):
            raise CargoAuthenticationError(
                "Cargo payload hash does not match the authenticated manifest."
            )
        doc_ids = self._adapter.ingest_many(
            [
                (
                    item.embedding,
                    item.content,
                    {
                        **(item.metadata or {}),
                        "kind": item.kind,
                        **({"doc_id": item.doc_id} if item.doc_id else {}),
                    },
                )
                for item in snapshots
            ],
            self._partition_key,
            gate_embedding=self._manifest.gate_embeddings_at_rest,
        )
        if (
            len(doc_ids) != len(snapshots)
            or len(set(doc_ids)) != len(doc_ids)
            or any(
                not isinstance(doc_id, str) or not doc_id or "\x00" in doc_id for doc_id in doc_ids
            )
        ):
            raise CargoAuthenticationError(
                "Atomic store returned an incomplete or duplicate load result"
            )
        if any(
            item.doc_id is not None and returned_id != item.doc_id
            for item, returned_id in zip(snapshots, doc_ids, strict=True)
        ):
            raise CargoAuthenticationError("Atomic store replaced an explicitly signed document id")
        self._loaded_ids.extend(doc_ids)
        self._load_completed = True

        items_hash = compute_load_hash(
            self._manifest.fingerprint(),
            self._partition_key,
            doc_ids,
        )
        self._loaded_operation_hashes.append(items_hash)
        op = CargoOperation(
            kind=OperationKind.LOAD,
            timestamp_epoch=_time.time(),
            item_count=len(doc_ids),
            items_hash=items_hash,
        )
        self._operations.append(op)

        return LoadResult(
            doc_ids=doc_ids,
            items_hash=items_hash,
            item_count=len(doc_ids),
        )

    def unload_cargo(
        self,
        query: Any,
        top_k: int = 10,
        *,
        kind: Optional[str] = None,
    ) -> UnloadResult:
        """Atomically retrieve items and record the operation."""
        with self._state_lock:
            return self._unload_cargo_unlocked(query, top_k, kind=kind)

    def _unload_cargo_unlocked(
        self,
        query: Any,
        top_k: int = 10,
        *,
        kind: Optional[str] = None,
    ) -> UnloadResult:
        """Retrieve items from the regime's partition via ``GatedRAGAdapter``."""
        self._require_open()
        self._require_operation(CargoManifestOperation.RETRIEVE)
        self._require_partition_binding()
        result: GatedRetrievalResult = self._adapter.query(
            query,
            self._partition_key,
            top_k,
            kind=kind,
        )
        self._require_retrieval_scope(result)
        unloaded_ids = [doc.doc_id for doc in result.docs]
        self._unloaded_ids.extend(unloaded_ids)

        items_hash = compute_retrieval_hash(
            query_hash=result.query_hash,
            partition_key=self._partition_key,
            top_k=top_k,
            kind=kind,
            docs=result.docs,
        )
        self._unloaded_operation_hashes.append(items_hash)
        op = CargoOperation(
            kind=OperationKind.UNLOAD,
            timestamp_epoch=_time.time(),
            item_count=len(result.docs),
            items_hash=items_hash,
        )
        self._operations.append(op)

        return UnloadResult(
            docs=result.docs,
            gate_mask=result.gate_mask,
            regime_id=result.regime_id,
            items_hash=items_hash,
            item_count=len(result.docs),
        )

    def unload_vectors(
        self,
        query: Any,
        top_k: int = 10,
        *,
        kind: Optional[str] = None,
        bridge: Optional[VectorBridge] = None,
    ) -> VectorPayload:
        """Atomically retrieve vectors and record the operation."""
        with self._state_lock:
            return self._unload_vectors_unlocked(
                query,
                top_k,
                kind=kind,
                bridge=bridge,
            )

    def _unload_vectors_unlocked(
        self,
        query: Any,
        top_k: int = 10,
        *,
        kind: Optional[str] = None,
        bridge: Optional[VectorBridge] = None,
    ) -> VectorPayload:
        """Retrieve vectors for direct injection into a model.

        Bypasses the text -> vector -> text -> vector round-trip.  Returns
        a ``VectorPayload`` with raw vectors (optionally projected via a
        ``VectorBridge``) and the regime's gate mask.
        """
        self._require_open()
        self._require_operation(CargoManifestOperation.RETRIEVE)
        self._require_partition_binding()
        result: GatedRetrievalResult = self._adapter.query(
            query,
            self._partition_key,
            top_k,
            kind=kind,
        )
        self._require_retrieval_scope(result)
        doc_ids = [doc.doc_id for doc in result.docs]
        self._unloaded_ids.extend(doc_ids)

        source_vectors = (
            np.stack([doc.embedding for doc in result.docs])
            if result.docs
            else np.empty((0, self._manifest.embedding_spec.dimensions))
        )
        output_vectors = bridge.project(source_vectors) if bridge is not None else source_vectors

        items_hash = compute_retrieval_hash(
            query_hash=result.query_hash,
            partition_key=self._partition_key,
            top_k=top_k,
            kind=kind,
            docs=result.docs,
            output_vectors=output_vectors,
            bridge_hash=None if bridge is None else bridge.projection_hash(),
        )
        self._unloaded_operation_hashes.append(items_hash)
        op = CargoOperation(
            kind=OperationKind.UNLOAD,
            timestamp_epoch=_time.time(),
            item_count=len(result.docs),
            items_hash=items_hash,
        )
        self._operations.append(op)

        return VectorPayload(
            vectors=source_vectors,
            gate_mask=result.gate_mask,
            regime_id=result.regime_id,
            source_doc_ids=doc_ids,
            bridge=bridge,
            embedding_spec=self._manifest.embedding_spec,
        )

    def record_reissue(self) -> None:
        """Record that a token reissue occurred during this session."""
        with self._state_lock:
            self._require_open()
            if self._reissue_count >= self._manifest.max_reissues:
                raise CargoAuthenticationError("Manifest reissue allowance has been exhausted.")
            self._reissue_count += 1

    def signal_condition(self, kind: CompletionKind, target: str = "") -> None:
        """Reject unverified external assertions in the signed receipt path."""
        raise CargoAuthenticationError(
            "External completion signals are not accepted as signed evidence"
        )

    def depart(self) -> CargoReceipt:
        """Atomically close the session and emit exactly one receipt."""
        with self._state_lock:
            return self._depart_unlocked()

    def _depart_unlocked(self) -> CargoReceipt:
        """Finalize the session and produce a signed receipt.

        Departure is always permitted even if the manifest has expired --
        the session is concluding, not starting.  Expiry is recorded as a
        fired completion condition.
        """
        if self._closed:
            raise CargoSessionClosedError(
                "Session has already departed. Cannot perform further operations."
            )
        self._require_partition_binding()
        self._require_manifest_obligations()
        departed_at = _time.time()

        if self._manifest.completion_conditions:
            for cond in self._manifest.completion_conditions:
                if cond.kind == CompletionKind.TTL_EXPIRED:
                    if self._manifest.is_expired(departed_at):
                        self._conditions_met.append(cond.kind.value)

        cargo_in_hash = compute_items_hash(self._loaded_operation_hashes)
        cargo_out_hash = compute_items_hash(self._unloaded_operation_hashes)

        receipt = CargoReceipt(
            receipt_id=CargoReceipt.generate_id(),
            manifest_id=self._manifest.manifest_id,
            tenant_id=self._manifest.tenant_id,
            regime_id=self._manifest.regime_id,
            subject_id=self._manifest.subject_id,
            model_digest=self._manifest.model_digest,
            operation=self._manifest.operation,
            policy_version=self._manifest.policy_version,
            partition_key=self._manifest.partition_key,
            embedding_spec=self._manifest.embedding_spec,
            operations=tuple(self._operations),
            cargo_in_hash=cargo_in_hash,
            cargo_out_hash=cargo_out_hash,
            docked_at_epoch=self._docked_at,
            departed_at_epoch=departed_at,
            reissue_count=self._reissue_count,
            manifest_fingerprint=self._manifest.fingerprint(),
            gate_release=self._manifest.gate_release,
            completion_conditions_met=tuple(sorted(set(self._conditions_met))),
        )

        signature = _hmac.new(
            self._tenant_key_secret,
            receipt.body_bytes(),
            hashlib.sha256,
        ).digest()

        signed_receipt = CargoReceipt(
            receipt_id=receipt.receipt_id,
            manifest_id=receipt.manifest_id,
            tenant_id=receipt.tenant_id,
            regime_id=receipt.regime_id,
            subject_id=receipt.subject_id,
            model_digest=receipt.model_digest,
            operation=receipt.operation,
            policy_version=receipt.policy_version,
            partition_key=receipt.partition_key,
            embedding_spec=receipt.embedding_spec,
            operations=receipt.operations,
            cargo_in_hash=receipt.cargo_in_hash,
            cargo_out_hash=receipt.cargo_out_hash,
            docked_at_epoch=receipt.docked_at_epoch,
            departed_at_epoch=receipt.departed_at_epoch,
            reissue_count=receipt.reissue_count,
            manifest_fingerprint=receipt.manifest_fingerprint,
            gate_release=receipt.gate_release,
            completion_conditions_met=receipt.completion_conditions_met,
            gate_signature=signature,
        )
        self._closed = True
        return signed_receipt


# ---------------------------------------------------------------------------
# DefaultRegimeBus
# ---------------------------------------------------------------------------


class DefaultRegimeBus:
    """Concrete ``RegimeBus`` backed by a ``GatedRAGAdapter``.

    One bus per regime partition.  ``dock()`` validates the manifest
    AAD against the gate key and opens a ``LiveDockingSession``.
    """

    def __init__(
        self,
        adapter: GatedRAGAdapter,
        partition_key: str,
        regime_id: int,
        gate_key_secret: bytes,
        embedding_spec: EmbeddingSpec,
        release_identity: GateReleaseIdentity | None = None,
    ) -> None:
        self._adapter = adapter
        self._partition_key = partition_key
        self._regime_id = regime_id
        self._gate_key_secret = gate_key_secret
        self._embedding_spec = embedding_spec
        self._release_identity = release_identity or current_release_identity()
        if not isinstance(adapter, GatedRAGAdapter):
            raise ValueError("adapter must be a GatedRAGAdapter")
        if not isinstance(partition_key, str) or not partition_key or "\x00" in partition_key:
            raise ValueError("partition_key must be a non-empty string without NUL")
        if isinstance(regime_id, bool) or not isinstance(regime_id, int) or regime_id < 0:
            raise ValueError("regime_id must be an integer >= 0")
        if not isinstance(gate_key_secret, bytes) or len(gate_key_secret) != 32:
            raise ValueError("gate_key_secret must be exactly 32 bytes")
        if not isinstance(embedding_spec, EmbeddingSpec):
            raise ValueError("embedding_spec must be an EmbeddingSpec")
        self._require_partition_binding()
        self._manifest_lock = threading.Lock()
        self._docked_manifest_ids: set[str] = set()

    def _require_partition_binding(self) -> None:
        """Fail closed on configuration drift between bus and adapter."""
        try:
            actual_regime = self._adapter.partition_map.get_regime_id(self._partition_key)
        except Exception as exc:
            raise CargoAuthenticationError(
                f"Bus partition {self._partition_key!r} is not registered"
            ) from exc
        if actual_regime != self._regime_id:
            raise CargoAuthenticationError(
                f"Bus regime {self._regime_id} does not match partition-map "
                f"regime {actual_regime} for {self._partition_key!r}"
            )
        if self._adapter.partition_map.n_dims != self._embedding_spec.dimensions:
            raise CargoAuthenticationError(
                "Bus embedding dimensions do not match the partition map"
            )

    @property
    def regime_id(self) -> int:
        return self._regime_id

    @property
    def partition_key(self) -> str:
        return self._partition_key

    def dock(
        self,
        manifest: CargoManifest,
        gate_key_secret: bytes,
    ) -> LiveDockingSession:
        """Validate the manifest and open a docking session.

        Validation:
        1. Manifest regime matches bus regime
        2. Manifest partition matches the exact bus partition
        3. Embedding spec matches bus embedding spec
        4. Manifest is not expired
        5. The complete manifest scope is authenticated via HMAC

        Raises ``CargoAuthenticationError`` on any mismatch.
        """
        if not isinstance(manifest, CargoManifest):
            raise CargoAuthenticationError("manifest must be a CargoManifest")
        if not release_identity_matches(
            manifest.gate_release,
            self._release_identity,
            require_source_commit=True,
        ):
            raise CargoAuthenticationError("Manifest Gate release differs from the bus runtime")
        if not isinstance(gate_key_secret, bytes) or len(gate_key_secret) != 32:
            raise CargoAuthenticationError("gate_key_secret must be exactly 32 bytes")
        self._require_partition_binding()
        if manifest.regime_id != self._regime_id:
            raise CargoAuthenticationError(
                f"Manifest regime {manifest.regime_id} does not match "
                f"bus regime {self._regime_id}. The channel does not exist."
            )

        if manifest.partition_key != self._partition_key:
            raise CargoAuthenticationError(
                f"Manifest partition {manifest.partition_key!r} does not match "
                f"bus partition {self._partition_key!r}. The channel does not exist."
            )

        if manifest.embedding_spec != self._embedding_spec:
            raise CargoAuthenticationError(
                f"Embedding spec mismatch. "
                f"Manifest: {manifest.embedding_spec.to_canonical()}, "
                f"Bus: {self._embedding_spec.to_canonical()}. "
                "Mutual intelligibility requires agreement on the encoding protocol."
            )

        if manifest.is_expired():
            raise CargoExpiredError(
                f"Manifest {manifest.manifest_id} has expired (epoch {manifest.expires_epoch})."
            )

        unsupported_conditions = {
            condition.kind
            for condition in manifest.completion_conditions
            if condition.kind != CompletionKind.TTL_EXPIRED
        }
        if unsupported_conditions:
            unsupported = ", ".join(sorted(kind.value for kind in unsupported_conditions))
            raise ValueError(f"DefaultRegimeBus unsupported completion condition: {unsupported}")
        if (
            any(
                condition.kind == CompletionKind.TTL_EXPIRED
                for condition in manifest.completion_conditions
            )
            and manifest.expires_epoch is None
        ):
            raise ValueError("TTL completion condition requires a manifest expiry")

        manifest_aad = manifest.to_aad()
        access_key = derive_cargo_access_key(
            self._gate_key_secret,
            manifest.regime_id,
            manifest.tenant_id,
            subject_id=manifest.subject_id,
            model_digest=manifest.model_digest,
            operation=manifest.operation,
            policy_version=manifest.policy_version,
            partition_key=manifest.partition_key,
        )
        expected_tag = _hmac.new(access_key, manifest_aad, hashlib.sha256).digest()
        root_tag = _hmac.new(self._gate_key_secret, manifest_aad, hashlib.sha256).digest()

        caller_tag = _hmac.new(
            gate_key_secret,
            manifest_aad,
            hashlib.sha256,
        ).digest()

        if not (
            _hmac.compare_digest(expected_tag, caller_tag)
            or _hmac.compare_digest(root_tag, caller_tag)
        ):
            raise CargoAuthenticationError(
                "Manifest AAD authentication failed. The cryptographic channel does not exist."
            )

        with self._manifest_lock:
            if manifest.manifest_id in self._docked_manifest_ids:
                raise CargoAuthenticationError("Cargo manifest has already been docked.")
            self._docked_manifest_ids.add(manifest.manifest_id)

        tenant_key_secret = _derive_cargo_receipt_key(
            self._gate_key_secret,
            manifest.regime_id,
            manifest.tenant_id,
            subject_id=manifest.subject_id,
            model_digest=manifest.model_digest,
            operation=manifest.operation,
            policy_version=manifest.policy_version,
            partition_key=manifest.partition_key,
        )

        return LiveDockingSession(
            manifest=manifest,
            adapter=self._adapter,
            partition_key=self._partition_key,
            tenant_key_secret=tenant_key_secret,
        )

    def verify_receipt(
        self,
        receipt: CargoReceipt,
        *,
        expected_manifest: CargoManifest,
    ) -> bool:
        """Verify receipt signature and its complete expected manifest scope."""
        if not isinstance(receipt, CargoReceipt) or not isinstance(
            expected_manifest, CargoManifest
        ):
            return False
        try:
            self._require_partition_binding()
            if (
                expected_manifest.regime_id != self._regime_id
                or expected_manifest.partition_key != self._partition_key
                or receipt.manifest_id != expected_manifest.manifest_id
                or receipt.manifest_fingerprint != expected_manifest.fingerprint()
                or receipt.tenant_id != expected_manifest.tenant_id
                or receipt.regime_id != expected_manifest.regime_id
                or receipt.subject_id != expected_manifest.subject_id
                or receipt.model_digest != expected_manifest.model_digest
                or receipt.operation != expected_manifest.operation
                or receipt.policy_version != expected_manifest.policy_version
                or receipt.partition_key != expected_manifest.partition_key
                or receipt.embedding_spec != expected_manifest.embedding_spec
                or receipt.gate_release != expected_manifest.gate_release
                or not release_identity_matches(
                    expected_manifest.gate_release,
                    self._release_identity,
                    require_source_commit=True,
                )
            ):
                return False
            key = _derive_cargo_receipt_key(
                self._gate_key_secret,
                expected_manifest.regime_id,
                expected_manifest.tenant_id,
                subject_id=expected_manifest.subject_id,
                model_digest=expected_manifest.model_digest,
                operation=expected_manifest.operation,
                policy_version=expected_manifest.policy_version,
                partition_key=self._partition_key,
            )
            return _receipt_fulfills_manifest(receipt, expected_manifest) and receipt.verify(key)
        except (AttributeError, CargoAuthenticationError, TypeError, ValueError):
            return False


# ---------------------------------------------------------------------------
# BusTerminal — manages multiple regime buses
# ---------------------------------------------------------------------------


class BusTerminal:
    """A terminal managing multiple regime buses.

    Wraps a single ``GatedRAGAdapter`` and exposes one bus per
    registered partition.
    """

    def __init__(
        self,
        adapter: GatedRAGAdapter,
        gate_key_secret: bytes,
        embedding_spec: EmbeddingSpec,
    ) -> None:
        if not isinstance(adapter, GatedRAGAdapter):
            raise ValueError("adapter must be a GatedRAGAdapter")
        if not isinstance(gate_key_secret, bytes) or len(gate_key_secret) != 32:
            raise ValueError("gate_key_secret must be exactly 32 bytes")
        if not isinstance(embedding_spec, EmbeddingSpec):
            raise ValueError("embedding_spec must be an EmbeddingSpec")
        if adapter.partition_map.n_dims != embedding_spec.dimensions:
            raise CargoAuthenticationError(
                "Terminal embedding dimensions do not match the partition map"
            )
        self._adapter = adapter
        self._gate_key_secret = gate_key_secret
        self._embedding_spec = embedding_spec
        self._buses: dict[str, DefaultRegimeBus] = {}
        self._bus_lock = threading.RLock()

    def register_bus(self, partition_key: str, regime_id: int) -> DefaultRegimeBus:
        """Register a regime bus for a partition."""
        with self._bus_lock:
            existing = self._buses.get(partition_key)
            if existing is not None:
                if existing.regime_id != regime_id:
                    raise CargoAuthenticationError(
                        f"Bus {partition_key!r} is already bound to regime {existing.regime_id}"
                    )
                return existing
            bus = DefaultRegimeBus(
                adapter=self._adapter,
                partition_key=partition_key,
                regime_id=regime_id,
                gate_key_secret=self._gate_key_secret,
                embedding_spec=self._embedding_spec,
            )
            self._buses[partition_key] = bus
            return bus

    def get_bus(self, partition_key: str) -> DefaultRegimeBus:
        with self._bus_lock:
            bus = self._buses.get(partition_key)
            if bus is None:
                raise KeyError(f"No bus registered for partition {partition_key!r}")
            return bus

    def list_buses(self) -> list[str]:
        with self._bus_lock:
            return list(self._buses.keys())

    @property
    def embedding_spec(self) -> EmbeddingSpec:
        return self._embedding_spec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _derive_cargo_key(
    gate_key_secret: bytes,
    regime_id: int,
    tenant_id: str,
    purpose: bytes,
    *,
    subject_id: str = "",
    model_digest: str = "",
    operation: str = "",
    policy_version: str = "",
    partition_key: str = "",
) -> bytes:
    if not isinstance(gate_key_secret, bytes) or len(gate_key_secret) != 32:
        raise ValueError("gate_key_secret must be exactly 32 bytes")
    if not isinstance(purpose, bytes) or not purpose:
        raise ValueError("purpose must be non-empty bytes")
    if isinstance(regime_id, bool) or not isinstance(regime_id, int) or regime_id < 0:
        raise ValueError("regime_id must be an integer >= 0")
    for name, value in (
        ("tenant_id", tenant_id),
        ("subject_id", subject_id),
        ("model_digest", model_digest),
        ("operation", operation),
        ("policy_version", policy_version),
        ("partition_key", partition_key),
    ):
        _text(value, name)
    scope = {
        "model_digest": model_digest,
        "operation": operation,
        "partition_key": partition_key,
        "policy_version": policy_version,
        "regime_id": regime_id,
        "subject_id": subject_id,
        "tenant_id": tenant_id,
    }
    info = b"schemen:v3:cargo:" + purpose + b":" + _canonical_scope(scope)
    return _hmac.new(gate_key_secret, info + b"\x01", hashlib.sha256).digest()


def _canonical_scope(scope: dict[str, object]) -> bytes:
    """Encode a key-derivation scope without delimiter ambiguity."""
    import json

    return json.dumps(
        scope,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def derive_cargo_access_key(
    gate_key_secret: bytes,
    regime_id: int,
    tenant_id: str,
    *,
    subject_id: str,
    model_digest: str,
    operation: str,
    policy_version: str,
    partition_key: str,
) -> bytes:
    """Derive a client key for one exact Cargo authorization scope."""
    return _derive_cargo_key(
        gate_key_secret,
        regime_id,
        tenant_id,
        b"access",
        subject_id=subject_id,
        model_digest=model_digest,
        operation=operation,
        policy_version=policy_version,
        partition_key=partition_key,
    )


def _derive_cargo_receipt_key(
    gate_key_secret: bytes,
    regime_id: int,
    tenant_id: str,
    *,
    subject_id: str,
    model_digest: str,
    operation: str,
    policy_version: str,
    partition_key: str,
) -> bytes:
    """Derive a bus-only receipt key for one exact manifest scope."""
    return _derive_cargo_key(
        gate_key_secret,
        regime_id,
        tenant_id,
        b"receipt-v2",
        subject_id=subject_id,
        model_digest=model_digest,
        operation=operation,
        policy_version=policy_version,
        partition_key=partition_key,
    )


def create_manifest(
    *,
    tenant_id: str,
    regime_id: int,
    embedding_spec: EmbeddingSpec,
    payload_hash: str,
    payload_kind: str,
    item_count: int,
    subject_id: str,
    model_digest: str,
    operation: str,
    policy_version: str,
    partition_key: str,
    gate_embeddings_at_rest: bool,
    architecture: str = "",
    parent_lockbox_hash: str = "",
    ttl_seconds: Optional[int] = 3600,
    completion_conditions: Sequence[CompletionCondition] = (),
    max_reissues: int = 0,
    release_identity: GateReleaseIdentity | None = None,
) -> CargoManifest:
    """Factory for ``CargoManifest`` with sane defaults."""
    if type(gate_embeddings_at_rest) is not bool:
        raise ValueError("gate_embeddings_at_rest must be an exact boolean")
    if ttl_seconds is not None and (
        isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int) or ttl_seconds < 0
    ):
        raise ValueError("ttl_seconds must be an integer >= 0 or None")
    now = int(_time.time())
    expires = (now + ttl_seconds) if ttl_seconds is not None else None
    return CargoManifest(
        manifest_id=CargoManifest.generate_id(),
        tenant_id=tenant_id,
        regime_id=regime_id,
        embedding_spec=embedding_spec,
        architecture=architecture,
        payload_hash=payload_hash,
        payload_kind=payload_kind,
        item_count=item_count,
        parent_lockbox_hash=parent_lockbox_hash,
        issued_at_epoch=now,
        subject_id=subject_id,
        model_digest=model_digest,
        operation=operation,
        policy_version=policy_version,
        partition_key=partition_key,
        gate_embeddings_at_rest=gate_embeddings_at_rest,
        expires_epoch=expires,
        completion_conditions=tuple(completion_conditions),
        max_reissues=max_reissues,
        gate_release=release_identity or current_release_identity(),
    )
