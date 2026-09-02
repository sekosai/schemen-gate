"""Authority and Cargo integrity regressions."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from typing import Callable

import numpy as np
import pytest

from schemen_gate import CargoManifestOperation, GateKey
from schemen_gate._cargo import (
    CargoAuthenticationError,
    CargoItem,
    EmbeddingSpec,
    compute_payload_hash,
)
from schemen_gate._cargo_impl import DefaultRegimeBus, create_manifest
from schemen_gate._lockbox import (
    Grant,
    HierarchyDef,
    Lockbox,
    RegimeCapability,
    RevocationCheck,
    create_lockbox,
    fingerprint_from_x509,
    generate_authority_cert,
    generate_self_signed_x25519_cert,
    seal_grant,
    sign_lockbox,
    verify_grant_provenance,
)
from schemen_gate._rag import (
    GatedRAGAdapter,
    InMemoryVectorStore,
    PartitionMap,
    PartitionMode,
)

GATE_KEY = b"g" * 32
PARTITION_KEY = "audit-partition"
SPEC = EmbeddingSpec(
    model_id="audit-model",
    dimensions=4,
    vocabulary_hash=hashlib.sha256(b"audit-vocabulary").hexdigest(),
    pooling="mean",
)


class _CountingVectorStore(InMemoryVectorStore):
    def __init__(self) -> None:
        super().__init__()
        self.retrieve_calls = 0

    def retrieve(self, *args, **kwargs):
        self.retrieve_calls += 1
        return super().retrieve(*args, **kwargs)


def _cargo_fixture() -> tuple[_CountingVectorStore, DefaultRegimeBus]:
    store = _CountingVectorStore()
    partition_map = PartitionMap(GATE_KEY, n_dims=4, n_regimes=1)
    partition_map.register(PARTITION_KEY, regime_id=0, mode=PartitionMode.READ_WRITE)
    adapter = GatedRAGAdapter(store, partition_map, embed_fn=lambda value: value)
    return store, DefaultRegimeBus(
        adapter,
        PARTITION_KEY,
        0,
        GATE_KEY,
        SPEC,
    )


def _item() -> CargoItem:
    return CargoItem(
        content="authorized cargo",
        embedding=np.array([1.0, 0.0, 0.0, 0.0]),
        doc_id="authorized-item",
    )


def _manifest(
    operation: CargoManifestOperation,
    items: tuple[CargoItem, ...] = (),
):
    return create_manifest(
        tenant_id="tenant",
        regime_id=0,
        embedding_spec=SPEC,
        payload_hash=compute_payload_hash(items),
        payload_kind="rag_documents",
        item_count=len(items),
        subject_id="subject",
        model_digest="model-digest",
        operation=operation.value,
        policy_version="v1",
        partition_key=PARTITION_KEY,
        gate_embeddings_at_rest=False,
    )


def test_retrieve_only_manifest_cannot_load_or_mutate_the_store() -> None:
    store, bus = _cargo_fixture()
    items = (_item(),)
    session = bus.dock(
        _manifest(CargoManifestOperation.RETRIEVE, items),
        GATE_KEY,
    )

    with pytest.raises(CargoAuthenticationError, match="does not authorize load"):
        session.load_cargo(items)

    assert store.count(PARTITION_KEY) == 0


@pytest.mark.parametrize("method", ["unload_cargo", "unload_vectors"])
def test_load_only_manifest_cannot_call_retrieval_adapter(method: str) -> None:
    store, bus = _cargo_fixture()
    store.insert(
        np.array([1.0, 0.0, 0.0, 0.0]),
        "existing",
        PARTITION_KEY,
        {"doc_id": "existing-item"},
    )
    session = bus.dock(_manifest(CargoManifestOperation.LOAD), GATE_KEY)

    with pytest.raises(CargoAuthenticationError, match="does not authorize retrieve"):
        getattr(session, method)(np.array([1.0, 0.0, 0.0, 0.0]))

    assert store.retrieve_calls == 0


def test_combined_manifest_explicitly_authorizes_load_and_retrieve() -> None:
    store, bus = _cargo_fixture()
    items = (_item(),)
    session = bus.dock(
        _manifest(CargoManifestOperation.LOAD_AND_RETRIEVE, items),
        GATE_KEY,
    )

    assert session.load_cargo(items).item_count == 1
    assert session.unload_cargo(np.array([1.0, 0.0, 0.0, 0.0])).item_count == 1
    assert store.retrieve_calls == 1


def test_manifest_rejects_an_unknown_operation() -> None:
    with pytest.raises(ValueError, match="operation must be one of"):
        replace(_manifest(CargoManifestOperation.LOAD), operation="admin")


def _signed_grant_fixture(
    recipient_id: str = "recipient",
) -> tuple[Lockbox, Grant, bytes, list[str]]:
    lockbox = create_lockbox(
        GateKey(b"m" * 32),
        chain_hash=hashlib.sha256(b"chain").hexdigest(),
        chain_name="audit-chain",
        n_dims=4,
        n_regimes=1,
        hierarchy_def=[
            HierarchyDef(
                "reader",
                "read only",
                [0],
                [RegimeCapability(0, "read", ["value"])],
            )
        ],
    )
    consumer_cert, _, consumer_public_key = generate_self_signed_x25519_cert(recipient_id)
    grant = seal_grant(
        lockbox,
        GateKey(b"m" * 32),
        recipient_id,
        lockbox.hierarchy[0],
        consumer_public_key,
        recipient_cert_pem=consumer_cert,
    )
    authority_cert, authority_key = generate_authority_cert("audit-authority")
    lockbox.authority = sign_lockbox(
        lockbox,
        signing_key=authority_key,
        signing_cert_pem=authority_cert,
        ca_root_pem=authority_cert,
    )
    return (
        lockbox,
        grant,
        consumer_cert,
        [fingerprint_from_x509(authority_cert)],
    )


GrantMutation = Callable[[Grant], None]


def _mutate_recipient_id(grant: Grant) -> None:
    grant.recipient_id += "-forged"


def _mutate_access_fingerprint(grant: Grant) -> None:
    grant.access_fingerprint = "0" * 64


def _mutate_algorithm(grant: Grant) -> None:
    grant.algorithm += "-forged"


def _mutate_recipient_fingerprint(grant: Grant) -> None:
    grant.recipient_fingerprint = "0" * 64


def _mutate_sealed_key(grant: Grant) -> None:
    sealed_key = grant.sealed_keys[0]
    grant.sealed_keys[0] = replace(
        sealed_key,
        ciphertext=sealed_key.ciphertext + b"forged",
    )


def _mutate_sealed_key_regime(grant: Grant) -> None:
    grant.sealed_keys[0] = replace(grant.sealed_keys[0], regime_id=1)


def _mutate_sealed_key_ephemeral_key(grant: Grant) -> None:
    sealed_key = grant.sealed_keys[0]
    grant.sealed_keys[0] = replace(
        sealed_key,
        ephemeral_public_key=sealed_key.ephemeral_public_key + b"forged",
    )


def _mutate_sealed_key_nonce(grant: Grant) -> None:
    sealed_key = grant.sealed_keys[0]
    grant.sealed_keys[0] = replace(
        sealed_key,
        nonce=sealed_key.nonce + b"forged",
    )


def _remove_sealed_key(grant: Grant) -> None:
    grant.sealed_keys.clear()


def _mutate_mask_token(grant: Grant) -> None:
    token = grant.mask_tokens[0]
    grant.mask_tokens[0] = replace(
        token,
        ciphertext=token.ciphertext + b"forged",
    )


def _mutate_mask_token_regime(grant: Grant) -> None:
    grant.mask_tokens[0] = replace(grant.mask_tokens[0], regime_id=1)


def _mutate_mask_token_tenant(grant: Grant) -> None:
    grant.mask_tokens[0] = replace(
        grant.mask_tokens[0],
        tenant_id="forged-tenant",
    )


def _mutate_mask_token_nonce(grant: Grant) -> None:
    token = grant.mask_tokens[0]
    grant.mask_tokens[0] = replace(token, nonce=token.nonce + b"forged")


def _mutate_mask_token_dimensions(grant: Grant) -> None:
    grant.mask_tokens[0] = replace(grant.mask_tokens[0], n_dims=5)


def _mutate_mask_token_regime_count(grant: Grant) -> None:
    grant.mask_tokens[0] = replace(grant.mask_tokens[0], n_regimes=2)


def _remove_mask_token(grant: Grant) -> None:
    grant.mask_tokens.clear()


@pytest.mark.parametrize(
    "mutate",
    [
        _mutate_recipient_id,
        _mutate_access_fingerprint,
        _mutate_algorithm,
        _mutate_recipient_fingerprint,
        _mutate_sealed_key_regime,
        _mutate_sealed_key_ephemeral_key,
        _mutate_sealed_key_nonce,
        _mutate_sealed_key,
        _remove_sealed_key,
        _mutate_mask_token_regime,
        _mutate_mask_token_tenant,
        _mutate_mask_token_nonce,
        _mutate_mask_token,
        _mutate_mask_token_dimensions,
        _mutate_mask_token_regime_count,
        _remove_mask_token,
    ],
    ids=[
        "recipient-id",
        "access-fingerprint",
        "algorithm",
        "recipient-fingerprint",
        "sealed-key-regime",
        "sealed-key-ephemeral-key",
        "sealed-key-nonce",
        "sealed-key-ciphertext",
        "sealed-key-list",
        "mask-token-regime",
        "mask-token-tenant",
        "mask-token-nonce",
        "mask-token-ciphertext",
        "mask-token-dimensions",
        "mask-token-regime-count",
        "mask-token-list",
    ],
)
def test_grant_provenance_rejects_every_mutated_signed_field(
    mutate: GrantMutation,
) -> None:
    lockbox, grant, consumer_cert, trusted_authorities = _signed_grant_fixture()
    detached_grant = copy.deepcopy(grant)
    mutate(detached_grant)

    result = verify_grant_provenance(
        lockbox,
        detached_grant,
        consumer_cert,
        trusted_authorities,
        revocation=RevocationCheck.SKIP,
    )

    assert not result.trusted
    assert result.reasons == ["Grant is not an exact member of the signed lockbox"]


def test_grant_provenance_accepts_an_exact_detached_copy() -> None:
    lockbox, grant, consumer_cert, trusted_authorities = _signed_grant_fixture()

    result = verify_grant_provenance(
        lockbox,
        copy.deepcopy(grant),
        consumer_cert,
        trusted_authorities,
        revocation=RevocationCheck.SKIP,
    )

    assert result.trusted
    assert result.reasons == []


def test_grant_provenance_rejects_a_grant_from_another_signed_lockbox() -> None:
    lockbox, _, _, trusted_authorities = _signed_grant_fixture("recipient-one")
    _, other_grant, other_cert, _ = _signed_grant_fixture("recipient-two")

    result = verify_grant_provenance(
        lockbox,
        other_grant,
        other_cert,
        trusted_authorities,
        revocation=RevocationCheck.SKIP,
    )

    assert not result.trusted
    assert result.reasons == ["Grant is not an exact member of the signed lockbox"]
