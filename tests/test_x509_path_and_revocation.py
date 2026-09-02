"""X.509 path-building and revocation-boundary regressions."""

from __future__ import annotations

import ast
import datetime
import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from cryptography import x509
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.x509.oid import NameOID

from schemen_gate import GateKey
from schemen_gate._cargo import (
    CargoItem,
    CompletionCondition,
    CompletionKind,
    EmbeddingSpec,
    compute_payload_hash,
)
from schemen_gate._cargo_impl import DefaultRegimeBus, create_manifest
from schemen_gate._lockbox import (
    HierarchyDef,
    RegimeCapability,
    RevocationCheck,
    check_certificate_revocation,
    create_lockbox,
    sign_lockbox,
    verify_authority,
)
from schemen_gate._rag import (
    GatedRAGAdapter,
    InMemoryVectorStore,
    PartitionMap,
    PartitionMode,
    PartitionModeError,
)

GATE_KEY = b"g" * 32
SPEC = EmbeddingSpec(
    model_id="audit-model",
    dimensions=4,
    vocabulary_hash=hashlib.sha256(b"vocab").hexdigest(),
    pooling="mean",
)


def _cargo_fixture(partitions: tuple[str, ...] = ("partition-a",)):
    partition_map = PartitionMap(GATE_KEY, n_dims=4, n_regimes=1)
    for partition in partitions:
        partition_map.register(partition, regime_id=0, mode=PartitionMode.READ_WRITE)
    store = InMemoryVectorStore()
    adapter = GatedRAGAdapter(store, partition_map, embed_fn=lambda value: value)
    buses = {
        partition: DefaultRegimeBus(adapter, partition, 0, GATE_KEY, SPEC)
        for partition in partitions
    }
    return store, buses


def _manifest(
    partition: str,
    *,
    items: tuple[CargoItem, ...] = (),
    completion_conditions: tuple[CompletionCondition, ...] = (),
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
        operation="retrieve",
        policy_version="v1",
        partition_key=partition,
        gate_embeddings_at_rest=False,
        completion_conditions=completion_conditions,
    )


def test_cargo_output_digest_binds_returned_content_and_embedding() -> None:
    store, buses = _cargo_fixture()
    bus = buses["partition-a"]

    def retrieve_once():
        manifest = _manifest("partition-a")
        session = bus.dock(manifest, GATE_KEY)
        result = session.unload_cargo(np.array([1.0, 0.0, 0.0, 0.0]), top_k=1)
        return result, session.depart(), manifest

    store.insert(
        np.array([1.0, 0.0, 0.0, 0.0]),
        "first content",
        "partition-a",
        {"doc_id": "stable-id"},
    )
    first_result, first_receipt, first_manifest = retrieve_once()

    store.delete_partition("partition-a")
    store.insert(
        np.array([1.0, 9.0, 9.0, 9.0]),
        "different content",
        "partition-a",
        {"doc_id": "stable-id"},
    )
    second_result, second_receipt, second_manifest = retrieve_once()

    assert first_result.docs[0].content != second_result.docs[0].content
    assert not np.array_equal(
        first_result.docs[0].embedding,
        second_result.docs[0].embedding,
    )
    assert first_receipt.cargo_out_hash != second_receipt.cargo_out_hash
    assert bus.verify_receipt(first_receipt, expected_manifest=first_manifest)
    assert bus.verify_receipt(second_receipt, expected_manifest=second_manifest)


def test_cargo_receipt_verification_rejects_a_different_partition_bus() -> None:
    _, buses = _cargo_fixture(("partition-a", "partition-b"))
    manifest = _manifest("partition-a")
    session = buses["partition-a"].dock(manifest, GATE_KEY)
    receipt = session.depart()

    assert buses["partition-a"].verify_receipt(
        receipt,
        expected_manifest=manifest,
    )
    assert not buses["partition-b"].verify_receipt(
        receipt,
        expected_manifest=manifest,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("manifest_id", "cargo-different"),
        ("tenant_id", "other-tenant"),
        ("regime_id", 1),
        ("subject_id", "other-subject"),
        ("model_digest", "other-model"),
        ("operation", "load"),
        ("policy_version", "other-policy"),
        ("partition_key", "partition-b"),
    ],
)
def test_cargo_receipt_requires_the_exact_expected_manifest(field, value) -> None:
    _, buses = _cargo_fixture()
    manifest = _manifest("partition-a")
    session = buses["partition-a"].dock(manifest, GATE_KEY)
    receipt = session.depart()

    assert not buses["partition-a"].verify_receipt(
        receipt,
        expected_manifest=replace(manifest, **{field: value}),
    )


def test_cargo_output_digest_binds_the_resolved_query() -> None:
    store, buses = _cargo_fixture()
    bus = buses["partition-a"]
    store.insert(
        np.array([1.0, 0.0, 0.0, 0.0]),
        "stable content",
        "partition-a",
        {"doc_id": "stable-id"},
    )

    manifest_one = _manifest("partition-a")
    session_one = bus.dock(manifest_one, GATE_KEY)
    session_one.unload_cargo(np.array([1.0, 0.0, 0.0, 0.0]), top_k=1)
    receipt_one = session_one.depart()

    manifest_two = _manifest("partition-a")
    session_two = bus.dock(manifest_two, GATE_KEY)
    session_two.unload_cargo(np.array([2.0, 0.0, 0.0, 0.0]), top_k=1)
    receipt_two = session_two.depart()

    assert receipt_one.cargo_out_hash != receipt_two.cargo_out_hash


def test_default_bus_rejects_unverifiable_external_completion_conditions() -> None:
    _, buses = _cargo_fixture()
    condition = CompletionCondition(
        CompletionKind.PARTITION_DESTROYED,
        "partition-a",
        destruction_aad="caller assertion is not evidence",
    )
    manifest = _manifest("partition-a", completion_conditions=(condition,))

    with pytest.raises(ValueError, match="unsupported completion condition"):
        buses["partition-a"].dock(manifest, GATE_KEY)


@pytest.mark.parametrize(
    "kind",
    [
        CompletionKind.PARTITION_DESTROYED,
        CompletionKind.RECEIPT_ACKNOWLEDGED,
        CompletionKind.ITEMS_EXHAUSTED,
    ],
)
def test_default_bus_only_accepts_completion_conditions_it_can_verify(kind) -> None:
    _, buses = _cargo_fixture()
    manifest = _manifest(
        "partition-a",
        completion_conditions=(CompletionCondition(kind, "partition-a"),),
    )
    with pytest.raises(ValueError, match="unsupported completion condition"):
        buses["partition-a"].dock(manifest, GATE_KEY)


def test_partition_destruction_requires_a_conforming_store() -> None:
    class StoreWithoutDelete:
        def count(self, partition_key: str, *, kind=None) -> int:
            return 0

    partition_map = PartitionMap(GATE_KEY, n_dims=4, n_regimes=1)
    partition_map.register("partition-a", regime_id=0)
    adapter = GatedRAGAdapter(StoreWithoutDelete(), partition_map)

    with pytest.raises(PartitionModeError, match="delete_partition"):
        adapter.destroy("partition-a")


def _name(common_name: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])


def _certificate(
    *,
    subject_name: x509.Name,
    issuer_name: x509.Name,
    public_key: object,
    issuer_key: Ed25519PrivateKey,
    ca: bool,
    path_length: int | None,
    digital_signature: bool = True,
) -> x509.Certificate:
    now = datetime.datetime.now(datetime.timezone.utc)
    return (
        x509.CertificateBuilder()
        .subject_name(subject_name)
        .issuer_name(issuer_name)
        .public_key(public_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=ca, path_length=path_length), True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=digital_signature,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=ca,
                crl_sign=ca,
                encipher_only=None,
                decipher_only=None,
            ),
            True,
        )
        .sign(issuer_key, None)
    )


def _lockbox():
    return create_lockbox(
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


def _signing_lockbox(
    leaf: x509.Certificate,
    leaf_key: Ed25519PrivateKey,
    root: x509.Certificate,
    chain: list[x509.Certificate],
):
    box = _lockbox()
    box.authority = sign_lockbox(
        box,
        signing_key=leaf_key,
        signing_cert_pem=leaf.public_bytes(Encoding.PEM),
        ca_root_pem=root.public_bytes(Encoding.PEM),
        cert_chain_pems=[cert.public_bytes(Encoding.PEM) for cert in chain],
    )
    return box


def test_certificate_path_length_constraint_is_enforced() -> None:
    root_key = Ed25519PrivateKey.generate()
    intermediate_key = Ed25519PrivateKey.generate()
    leaf_key = Ed25519PrivateKey.generate()
    root = _certificate(
        subject_name=_name("root"),
        issuer_name=_name("root"),
        public_key=root_key.public_key(),
        issuer_key=root_key,
        ca=True,
        path_length=0,
    )
    intermediate = _certificate(
        subject_name=_name("intermediate"),
        issuer_name=root.subject,
        public_key=intermediate_key.public_key(),
        issuer_key=root_key,
        ca=True,
        path_length=0,
    )
    leaf = _certificate(
        subject_name=_name("leaf"),
        issuer_name=intermediate.subject,
        public_key=leaf_key.public_key(),
        issuer_key=intermediate_key,
        ca=False,
        path_length=None,
    )
    box = _signing_lockbox(leaf, leaf_key, root, [intermediate, root])

    with pytest.raises(ValueError, match="path length"):
        verify_authority(
            box,
            [root.fingerprint(SHA256()).hex()],
            revocation=RevocationCheck.SKIP,
        )


def test_certificate_leaf_must_allow_digital_signatures() -> None:
    root_key = Ed25519PrivateKey.generate()
    leaf_key = Ed25519PrivateKey.generate()
    root = _certificate(
        subject_name=_name("root"),
        issuer_name=_name("root"),
        public_key=root_key.public_key(),
        issuer_key=root_key,
        ca=True,
        path_length=1,
    )
    leaf = _certificate(
        subject_name=_name("leaf"),
        issuer_name=root.subject,
        public_key=leaf_key.public_key(),
        issuer_key=root_key,
        ca=False,
        path_length=None,
        digital_signature=False,
    )
    box = _signing_lockbox(leaf, leaf_key, root, [root])

    with pytest.raises(ValueError, match="digital signatures"):
        verify_authority(
            box,
            [root.fingerprint(SHA256()).hex()],
            revocation=RevocationCheck.SKIP,
        )


def test_certificate_with_unknown_critical_extension_is_rejected() -> None:
    root_key = Ed25519PrivateKey.generate()
    leaf_key = Ed25519PrivateKey.generate()
    root = _certificate(
        subject_name=_name("root"),
        issuer_name=_name("root"),
        public_key=root_key.public_key(),
        issuer_key=root_key,
        ca=True,
        path_length=1,
    )
    now = datetime.datetime.now(datetime.timezone.utc)
    leaf = (
        x509.CertificateBuilder()
        .subject_name(_name("leaf"))
        .issuer_name(root.subject)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=None,
                decipher_only=None,
            ),
            True,
        )
        .add_extension(
            x509.UnrecognizedExtension(
                x509.ObjectIdentifier("1.3.6.1.4.1.57264.99.1"),
                b"unsupported",
            ),
            True,
        )
        .sign(root_key, None)
    )
    box = _signing_lockbox(leaf, leaf_key, root, [root])

    with pytest.raises(ValueError, match="unsupported critical extensions"):
        verify_authority(
            box,
            [root.fingerprint(SHA256()).hex()],
            revocation=RevocationCheck.SKIP,
        )


def _certificate_with_crl_url(url: str):
    ca_key = Ed25519PrivateKey.generate()
    leaf_key = Ed25519PrivateKey.generate()
    now = datetime.datetime.now(datetime.timezone.utc)
    ca_name = _name("scope-ca")
    ca = _certificate(
        subject_name=ca_name,
        issuer_name=ca_name,
        public_key=ca_key.public_key(),
        issuer_key=ca_key,
        ca=True,
        path_length=1,
    )
    leaf = (
        x509.CertificateBuilder()
        .subject_name(_name("leaf"))
        .issuer_name(ca_name)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), True)
        .add_extension(
            x509.CRLDistributionPoints(
                [
                    x509.DistributionPoint(
                        full_name=[x509.UniformResourceIdentifier(url)],
                        relative_name=None,
                        reasons=None,
                        crl_issuer=None,
                    )
                ]
            ),
            False,
        )
        .sign(ca_key, None)
    )
    return ca, ca_key, leaf


def test_wrong_scope_crl_cannot_establish_good_status() -> None:
    leaf_url = "https://revocation.example.test/leaf-scope.crl"
    other_url = "https://revocation.example.test/other-scope.crl"
    ca, ca_key, leaf = _certificate_with_crl_url(leaf_url)
    now = datetime.datetime.now(datetime.timezone.utc)
    wrong_scope_crl = (
        x509.CertificateRevocationListBuilder()
        .issuer_name(ca.subject)
        .last_update(now - datetime.timedelta(minutes=1))
        .next_update(now + datetime.timedelta(hours=1))
        .add_extension(
            x509.IssuingDistributionPoint(
                full_name=[x509.UniformResourceIdentifier(other_url)],
                relative_name=None,
                only_contains_user_certs=False,
                only_contains_ca_certs=False,
                only_some_reasons=None,
                indirect_crl=False,
                only_contains_attribute_certs=False,
            ),
            True,
        )
        .sign(ca_key, None)
    )

    with pytest.raises(ValueError, match="distribution point"):
        check_certificate_revocation(
            leaf,
            ca,
            mode=RevocationCheck.ENFORCE,
            fetcher=lambda **kwargs: wrong_scope_crl.public_bytes(Encoding.DER),
        )


def test_direct_revocation_egress_requires_an_operator_fetcher(monkeypatch) -> None:
    ca, _, leaf = _certificate_with_crl_url("https://revocation.example.test/leaf.crl")

    def public_resolution(*args, **kwargs):
        return [(2, 1, 6, "", ("93.184.216.34", 443))]

    monkeypatch.setattr("socket.getaddrinfo", public_resolution)

    def connection_attempt(*args, **kwargs):
        raise AssertionError("Gate attempted direct revocation egress")

    monkeypatch.setattr("urllib.request.build_opener", connection_attempt)
    with pytest.raises(ValueError, match="operator-supplied revocation fetcher"):
        check_certificate_revocation(
            leaf,
            ca,
            mode=RevocationCheck.ENFORCE,
        )


def test_release_examples_do_not_depend_on_optimized_away_assertions() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "examples" / "quickstart.py",
        root / "examples" / "cotrained_shard_lockbox" / "demo.py",
        root / "scripts" / "release_check.py",
    ]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree)), path
