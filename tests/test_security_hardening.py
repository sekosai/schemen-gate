"""Regression proofs for security-sensitive trust boundaries."""

from __future__ import annotations

import datetime
import hashlib
import sys
import types
from typing import Any, ClassVar

import pytest

from schemen_gate import GateKey
from schemen_gate._cargo_impl import derive_cargo_access_key
from schemen_gate._lockbox import (
    HierarchyDef,
    RegimeCapability,
    RevocationCheck,
    RevocationPolicy,
    SpiffeWorkloadClient,
    WorkloadIdentity,
    attest_model,
    bind_model_artifact,
    check_certificate_revocation,
    compute_model_graph_hash,
    create_lockbox,
    fingerprint_from_x509,
    generate_authority_cert,
    generate_self_signed_x25519_cert,
    generate_svid,
    load_lockbox,
    save_lockbox,
    seal_grant,
    seal_grant_spiffe,
    sign_lockbox,
    unseal_grant,
    verify_authority,
    verify_grant_provenance,
    verify_model_attestation,
    verify_provenance_spiffe,
)
from schemen_gate._rag_arch import _hash_state_dict
from schemen_gate._rag_stores import PgVectorStore


def _lockbox():
    capability = RegimeCapability(0, "read", ["value"])
    lockbox = create_lockbox(
        GateKey(b"m" * 32),
        chain_hash=hashlib.sha256(b"chain").hexdigest(),
        chain_name="audit-chain",
        n_dims=4,
        n_regimes=1,
        hierarchy_def=[HierarchyDef("reader", "read only", [0], [capability])],
    )
    return lockbox


def test_authority_verification_requires_external_trust_anchor() -> None:
    lockbox = _lockbox()
    cert_pem, private_key = generate_authority_cert()
    lockbox.authority = sign_lockbox(
        lockbox,
        signing_key=private_key,
        signing_cert_pem=cert_pem,
        ca_root_pem=cert_pem,
    )

    trusted = fingerprint_from_x509(cert_pem)
    assert verify_authority(lockbox, [trusted], revocation=RevocationCheck.SKIP)
    with pytest.raises(ValueError, match="trust store"):
        verify_authority(
            lockbox,
            ["0" * 64],
            revocation=RevocationCheck.SKIP,
        )
    with pytest.raises(ValueError, match="external trusted"):
        verify_authority(lockbox, [], revocation=RevocationCheck.SKIP)


def test_authority_verification_requires_an_explicit_revocation_policy() -> None:
    lockbox = _lockbox()
    cert_pem, private_key = generate_authority_cert()
    lockbox.authority = sign_lockbox(
        lockbox,
        signing_key=private_key,
        signing_cert_pem=cert_pem,
        ca_root_pem=cert_pem,
    )

    with pytest.raises(ValueError, match="revocation policy"):
        verify_authority(lockbox, [fingerprint_from_x509(cert_pem)])


def test_grant_cannot_name_an_out_of_hierarchy_access_level() -> None:
    from dataclasses import replace

    from schemen_gate._key_wrapping import generate_x25519_keypair

    lockbox = _lockbox()
    _, public_key = generate_x25519_keypair()
    forged = replace(lockbox.hierarchy[0], regimes=[0, 1])

    with pytest.raises(ValueError, match="exact member"):
        seal_grant(
            lockbox,
            GateKey(b"m" * 32),
            "tenant",
            forged,
            public_key,
        )


def test_spiffe_svid_rejects_an_unbound_recipient_key() -> None:
    from schemen_gate._key_wrapping import generate_x25519_keypair

    lockbox = _lockbox()
    ca_pem, ca_key = generate_authority_cert()
    svid_pem, _, _ = generate_svid("spiffe://example.org/workload/a", ca_pem, ca_key)
    _, attacker_public_key = generate_x25519_keypair()
    lockbox.trusted_recipient_cas = [fingerprint_from_x509(ca_pem)]

    with pytest.raises(ValueError, match="not bound"):
        seal_grant_spiffe(
            lockbox,
            GateKey(b"m" * 32),
            svid_pem,
            lockbox.hierarchy[0],
            attacker_public_key,
            trust_bundle=[ca_pem],
        )


def test_spiffe_provenance_requires_and_validates_the_committed_recipient_root() -> None:
    recipient_root_pem, recipient_root_key = generate_authority_cert("spiffe-root")
    svid_pem, _, recipient_public_key = generate_svid(
        "spiffe://example.org/workload/recipient",
        recipient_root_pem,
        recipient_root_key,
    )
    lockbox = _lockbox()
    lockbox.trusted_recipient_cas = [fingerprint_from_x509(recipient_root_pem)]
    grant = seal_grant_spiffe(
        lockbox,
        GateKey(b"m" * 32),
        svid_pem,
        lockbox.hierarchy[0],
        recipient_public_key,
        trust_bundle=[recipient_root_pem],
    )
    authority_pem, authority_key = generate_authority_cert("lockbox-authority")
    lockbox.authority = sign_lockbox(
        lockbox,
        signing_key=authority_key,
        signing_cert_pem=authority_pem,
        ca_root_pem=authority_pem,
    )
    trusted_authorities = [fingerprint_from_x509(authority_pem)]

    result = verify_provenance_spiffe(
        lockbox,
        grant,
        svid_pem,
        [recipient_root_pem],
        trusted_authority_cas=trusted_authorities,
        revocation=RevocationCheck.SKIP,
    )
    assert result.trusted

    wrong_root_pem, _ = generate_authority_cert("wrong-spiffe-root")
    denied = verify_provenance_spiffe(
        lockbox,
        grant,
        svid_pem,
        [wrong_root_pem],
        trusted_authority_cas=trusted_authorities,
        revocation=RevocationCheck.SKIP,
    )
    assert not denied.trusted
    assert "Recipient certificate validation failed" in denied.reasons[0]


def test_recipient_revocation_enforce_requires_an_authenticated_issuer_path() -> None:
    lockbox = _lockbox()
    recipient_pem, _, recipient_public_key = generate_self_signed_x25519_cert("recipient")
    grant = seal_grant(
        lockbox,
        GateKey(b"m" * 32),
        "recipient",
        lockbox.hierarchy[0],
        recipient_public_key,
        recipient_cert_pem=recipient_pem,
    )
    authority_pem, authority_key = generate_authority_cert("lockbox-authority")
    lockbox.authority = sign_lockbox(
        lockbox,
        signing_key=authority_key,
        signing_cert_pem=authority_pem,
        ca_root_pem=authority_pem,
    )

    result = verify_grant_provenance(
        lockbox,
        grant,
        recipient_pem,
        [fingerprint_from_x509(authority_pem)],
        revocation=RevocationCheck.ENFORCE,
    )

    assert not result.trusted
    assert "no trusted recipient CA fingerprints" in result.reasons[0]


def test_recipient_revocation_warn_reports_indeterminate_path(
    caplog: pytest.LogCaptureFixture,
) -> None:
    lockbox = _lockbox()
    recipient_pem, _, recipient_public_key = generate_self_signed_x25519_cert("recipient")
    grant = seal_grant(
        lockbox,
        GateKey(b"m" * 32),
        "recipient",
        lockbox.hierarchy[0],
        recipient_public_key,
        recipient_cert_pem=recipient_pem,
    )
    authority_pem, authority_key = generate_authority_cert("lockbox-authority")
    lockbox.authority = sign_lockbox(
        lockbox,
        signing_key=authority_key,
        signing_cert_pem=authority_pem,
        ca_root_pem=authority_pem,
    )

    result = verify_grant_provenance(
        lockbox,
        grant,
        recipient_pem,
        [fingerprint_from_x509(authority_pem)],
        revocation=RevocationCheck.WARN,
    )

    assert result.trusted
    assert "no trusted recipient CA fingerprints" in caplog.text


@pytest.mark.parametrize(
    "consumer_cert_pem",
    [
        b"",
        b"-----BEGIN CERTIFICATE-----\ntruncated\n",
        b"x" * (64 * 1024 + 1),
        "not-pem-bytes",
        bytearray(b"not-pem-bytes"),
    ],
    ids=["empty", "truncated", "oversized", "string", "bytearray"],
)
def test_grant_provenance_denies_malformed_recipient_certificates(
    consumer_cert_pem,
) -> None:
    lockbox = _lockbox()
    recipient_pem, _, recipient_public_key = generate_self_signed_x25519_cert("recipient")
    grant = seal_grant(
        lockbox,
        GateKey(b"m" * 32),
        "recipient",
        lockbox.hierarchy[0],
        recipient_public_key,
        recipient_cert_pem=recipient_pem,
    )
    authority_pem, authority_key = generate_authority_cert("lockbox-authority")
    lockbox.authority = sign_lockbox(
        lockbox,
        signing_key=authority_key,
        signing_cert_pem=authority_pem,
        ca_root_pem=authority_pem,
    )

    result = verify_grant_provenance(
        lockbox,
        grant,
        consumer_cert_pem,
        [fingerprint_from_x509(authority_pem)],
        revocation=RevocationCheck.SKIP,
    )

    assert not result.trusted
    assert result.reasons
    assert result.reasons[0].startswith("Recipient certificate validation failed:")


def test_recipient_certificate_cannot_be_paired_with_an_attacker_key() -> None:
    from schemen_gate._key_wrapping import generate_x25519_keypair

    root_pem, root_key = generate_authority_cert("recipient-root")
    svid_pem, recipient_private_key, recipient_public_key = generate_svid(
        "spiffe://example.org/workload/recipient",
        root_pem,
        root_key,
    )
    _, attacker_public_key = generate_x25519_keypair()
    lockbox = _lockbox()
    lockbox.trusted_recipient_cas = [fingerprint_from_x509(root_pem)]

    with pytest.raises(ValueError, match="not bound"):
        seal_grant(
            lockbox,
            GateKey(b"m" * 32),
            "recipient",
            lockbox.hierarchy[0],
            attacker_public_key,
            recipient_cert_pem=svid_pem,
            recipient_cert_chain_pems=[root_pem],
        )

    grant = seal_grant(
        lockbox,
        GateKey(b"m" * 32),
        "recipient",
        lockbox.hierarchy[0],
        recipient_public_key,
        recipient_cert_pem=svid_pem,
        recipient_cert_chain_pems=[root_pem],
    )
    assert unseal_grant(grant, recipient_private_key)[0].secret


def test_recipient_ca_policy_requires_a_validated_chain() -> None:
    root_pem, root_key = generate_authority_cert("recipient-root")
    svid_pem, _, recipient_public_key = generate_svid(
        "spiffe://example.org/workload/recipient",
        root_pem,
        root_key,
    )
    lockbox = _lockbox()
    lockbox.trusted_recipient_cas = [fingerprint_from_x509(root_pem)]

    with pytest.raises(ValueError, match="certificate chain"):
        seal_grant(
            lockbox,
            GateKey(b"m" * 32),
            "recipient",
            lockbox.hierarchy[0],
            recipient_public_key,
            recipient_cert_pem=svid_pem,
        )


def test_spiffe_workload_client_supports_current_object_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cryptography import x509

    cert_pem, private_key = generate_authority_cert()
    certificate = x509.load_pem_x509_certificate(cert_pem)
    svid_private_key = private_key

    class FakeSvid:
        cert_chain: ClassVar[list[Any]] = [certificate]
        private_key = svid_private_key
        spiffe_id = "spiffe://example.org/workload/a"

    class FakeBundle:
        x509_authorities: ClassVar[set[Any]] = {certificate}

    class FakeBundleSet:
        bundles: ClassVar[set[Any]] = {FakeBundle()}

    class FakeClient:
        closed = False

        def __init__(self, socket_path, *, default_timeout):
            assert socket_path == "unix:///tmp/spire.sock"
            assert default_timeout == 5.0

        def fetch_x509_svid(self):
            return FakeSvid()

        def fetch_x509_bundles(self):
            return FakeBundleSet()

        def close(self):
            self.closed = True

    monkeypatch.setitem(
        sys.modules,
        "spiffe",
        types.SimpleNamespace(WorkloadApiClient=FakeClient),
    )
    identity = SpiffeWorkloadClient(socket_path="unix:///tmp/spire.sock").fetch_identity(
        validate_spiffe_id_syntax=False
    )

    assert identity.spiffe_id == "spiffe://example.org/workload/a"
    assert b"BEGIN CERTIFICATE" in identity.svid_pem
    assert b"BEGIN PRIVATE KEY" in identity.private_key_pem
    assert len(identity.trust_bundle) == 1


def test_workload_identity_repr_redacts_private_key() -> None:
    identity = WorkloadIdentity(
        spiffe_id="spiffe://example.org/workload/a",
        svid_pem=b"certificate",
        private_key_pem=b"PRIVATE-KEY-MATERIAL",
        trust_bundle=[],
    )

    assert "PRIVATE-KEY-MATERIAL" not in repr(identity)
    assert "PRIVATE-KEY-MATERIAL" not in str(identity)


def test_security_boundary_operations_are_public_api() -> None:
    import schemen_gate

    for name in (
        "check_certificate_revocation",
        "RevocationPolicy",
        "compute_model_artifact_hash",
        "compute_model_graph_hash",
        "bind_model_artifact",
        "attest_model",
        "verify_model_attestation",
        "seal_grant_spiffe",
        "verify_provenance_spiffe",
        "verify_attestation_spiffe",
        "WorkloadIdentity",
        "SpiffeWorkloadClient",
    ):
        assert name in schemen_gate.__all__
        assert getattr(schemen_gate, name) is not None


def test_unsigned_lockbox_is_rejected_by_default(tmp_path) -> None:
    path = tmp_path / "unsigned.yaml"
    save_lockbox(_lockbox(), path)

    with pytest.raises(ValueError, match="unsigned"):
        load_lockbox(
            path,
            trusted_authority_cas=["0" * 64],
            revocation=RevocationCheck.SKIP,
        )

    loaded = load_lockbox(path, require_authority=False)
    assert loaded.authority is None


@pytest.mark.parametrize(
    "document",
    [
        "schemen_lockbox:\n  version: '2'\n  version: '2'\n",
        "schemen_lockbox: &box\n  version: '2'\ncopy: *box\n",
    ],
    ids=["duplicate-key", "alias"],
)
def test_lockbox_yaml_rejects_ambiguous_documents(tmp_path, document: str) -> None:
    path = tmp_path / "ambiguous.lockbox.yaml"
    path.write_text(document, encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid lockbox YAML"):
        load_lockbox(path, require_authority=False)


def _certificate_with_crl_url(url: str):
    from cryptography import x509
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.x509.oid import NameOID

    root_pem, root_key = generate_authority_cert("revocation-root")
    root = x509.load_pem_x509_certificate(root_pem)
    leaf_key = Ed25519PrivateKey.generate()
    now = datetime.datetime.now(datetime.timezone.utc)
    leaf = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "leaf")]))
        .issuer_name(root.subject)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
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
            critical=False,
        )
        .sign(root_key, None)
    )
    return root, root_key, leaf


def _recipient_certificate_with_crl_url(
    ca_cert_pem: bytes,
    ca_key,
    url: str,
):
    from cryptography import x509
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding
    from cryptography.x509.oid import NameOID

    ca_cert = x509.load_pem_x509_certificate(ca_cert_pem)
    signing_key = Ed25519PrivateKey.generate()
    wrapping_key = X25519PrivateKey.generate()
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "recipient")]))
        .issuer_name(ca_cert.subject)
        .public_key(signing_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.UnrecognizedExtension(
                x509.ObjectIdentifier("1.3.6.1.4.1.57264.1.1"),
                wrapping_key.public_key().public_bytes_raw(),
            ),
            critical=False,
        )
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
            critical=False,
        )
        .sign(ca_key, None)
    )
    return (
        certificate.public_bytes(Encoding.PEM),
        wrapping_key.public_key().public_bytes_raw(),
        certificate,
    )


def _certificate_with_ocsp_url(url: str):
    from cryptography import x509
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.x509.oid import AuthorityInformationAccessOID, NameOID

    root_pem, root_key = generate_authority_cert("ocsp-root")
    root = x509.load_pem_x509_certificate(root_pem)
    leaf_key = Ed25519PrivateKey.generate()
    now = datetime.datetime.now(datetime.timezone.utc)
    leaf = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "leaf")]))
        .issuer_name(root.subject)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(
            x509.AuthorityInformationAccess(
                [
                    x509.AccessDescription(
                        AuthorityInformationAccessOID.OCSP,
                        x509.UniformResourceIdentifier(url),
                    )
                ]
            ),
            critical=False,
        )
        .sign(root_key, None)
    )
    return root, root_key, leaf


def _crl_bytes(
    root,
    signing_key,
    *,
    revoked_serial: int | None = None,
    stale: bool = False,
) -> bytes:
    from cryptography import x509
    from cryptography.hazmat.primitives.serialization import Encoding

    now = datetime.datetime.now(datetime.timezone.utc)
    builder = (
        x509.CertificateRevocationListBuilder()
        .issuer_name(root.subject)
        .last_update(now - datetime.timedelta(hours=2 if stale else 0, minutes=1))
        .next_update(now + (-datetime.timedelta(hours=1) if stale else datetime.timedelta(hours=1)))
    )
    if revoked_serial is not None:
        revoked = (
            x509.RevokedCertificateBuilder()
            .serial_number(revoked_serial)
            .revocation_date(now - datetime.timedelta(seconds=1))
            .build()
        )
        builder = builder.add_revoked_certificate(revoked)
    return builder.sign(signing_key, None).public_bytes(Encoding.DER)


def test_revocation_rejects_a_crl_signed_by_an_unrelated_key() -> None:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    root, _, leaf = _certificate_with_crl_url("https://revocation.example/crl")
    attacker_key = Ed25519PrivateKey.generate()

    with pytest.raises(ValueError, match="CRL signature"):
        check_certificate_revocation(
            leaf,
            root,
            mode=RevocationCheck.ENFORCE,
            fetcher=lambda *args, **kwargs: _crl_bytes(root, attacker_key),
        )


def test_revocation_accepts_a_fresh_ca_signed_crl_and_rejects_revoked_leaf() -> None:
    root, root_key, leaf = _certificate_with_crl_url("https://revocation.example/crl")

    assert (
        check_certificate_revocation(
            leaf,
            root,
            mode=RevocationCheck.ENFORCE,
            fetcher=lambda *args, **kwargs: _crl_bytes(root, root_key),
        )
        == []
    )
    with pytest.raises(ValueError, match="REVOKED"):
        check_certificate_revocation(
            leaf,
            root,
            mode=RevocationCheck.ENFORCE,
            fetcher=lambda *args, **kwargs: _crl_bytes(
                root,
                root_key,
                revoked_serial=leaf.serial_number,
            ),
        )


def test_revocation_rejects_stale_and_wrong_issuer_crls() -> None:
    from cryptography import x509

    root, root_key, leaf = _certificate_with_crl_url("https://revocation.example/crl")
    other_root_pem, other_key = generate_authority_cert("other-revocation-root")
    other_root = x509.load_pem_x509_certificate(other_root_pem)

    with pytest.raises(ValueError, match="CRL is stale"):
        check_certificate_revocation(
            leaf,
            root,
            mode=RevocationCheck.ENFORCE,
            fetcher=lambda *args, **kwargs: _crl_bytes(
                root,
                root_key,
                stale=True,
            ),
        )
    with pytest.raises(ValueError, match="CRL issuer"):
        check_certificate_revocation(
            leaf,
            root,
            mode=RevocationCheck.ENFORCE,
            fetcher=lambda *args, **kwargs: _crl_bytes(other_root, other_key),
        )


def test_revocation_blocks_private_endpoints_before_network_access() -> None:
    _, _, leaf = _certificate_with_crl_url("http://127.0.0.1/crl")
    root, _, _ = _certificate_with_crl_url("https://revocation.example/crl")

    with pytest.raises(ValueError, match="prohibited address"):
        check_certificate_revocation(
            leaf,
            root,
            mode=RevocationCheck.ENFORCE,
        )


def test_revocation_policy_enforces_exact_hosts_and_supports_local_fetcher() -> None:
    root, root_key, leaf = _certificate_with_crl_url("https://revocation.example/crl")
    fetch_calls = 0

    def fetcher(*args, **kwargs):
        nonlocal fetch_calls
        fetch_calls += 1
        return _crl_bytes(root, root_key)

    denied = RevocationPolicy(
        mode=RevocationCheck.ENFORCE,
        allowed_hosts=("approved.example",),
        fetcher=fetcher,
    )
    with pytest.raises(ValueError, match="not allowlisted"):
        denied.check(leaf, root)
    assert fetch_calls == 0

    approved = RevocationPolicy(
        mode=RevocationCheck.ENFORCE,
        allowed_hosts=("REVOCATION.EXAMPLE.",),
        fetcher=fetcher,
    )
    assert approved.check(leaf, root) == []
    assert fetch_calls == 1


def test_network_revocation_policy_requires_exact_host_allowlist() -> None:
    with pytest.raises(ValueError, match="exact-host allowlist"):
        RevocationPolicy(
            mode=RevocationCheck.ENFORCE,
            fetcher=lambda *args, **kwargs: b"unreachable",
        )


@pytest.mark.parametrize(
    "timeout",
    [True, "1", 0, -1, float("nan"), float("inf"), float("-inf")],
)
def test_revocation_policy_rejects_invalid_timeouts(timeout) -> None:
    with pytest.raises(ValueError, match="timeout"):
        RevocationPolicy(mode=RevocationCheck.ENFORCE, timeout_seconds=timeout)


@pytest.mark.parametrize(
    "max_response_bytes",
    [True, "1", 1.0, 0, -1, float("nan"), float("inf")],
)
def test_revocation_policy_rejects_non_positive_exact_integer_byte_limits(
    max_response_bytes,
) -> None:
    with pytest.raises(ValueError, match="byte limit"):
        RevocationPolicy(
            mode=RevocationCheck.ENFORCE,
            max_response_bytes=max_response_bytes,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("timeout", True),
        ("timeout", "1"),
        ("timeout", float("nan")),
        ("timeout", float("inf")),
        ("timeout", 0),
        ("max_response_bytes", True),
        ("max_response_bytes", 1.0),
        ("max_response_bytes", float("nan")),
        ("max_response_bytes", 0),
    ],
)
def test_direct_revocation_api_rejects_invalid_transport_bounds(field, value) -> None:
    kwargs = {field: value}
    with pytest.raises(ValueError):
        check_certificate_revocation(
            object(),
            object(),
            mode=RevocationCheck.SKIP,
            **kwargs,
        )


def test_grant_provenance_accepts_fresh_and_denies_revoked_recipient_path() -> None:
    from cryptography import x509

    recipient_root_pem, recipient_root_key = generate_authority_cert("recipient-root")
    recipient_pem, recipient_public_key, recipient_cert = _recipient_certificate_with_crl_url(
        recipient_root_pem,
        recipient_root_key,
        "https://revocation.example/recipient.crl",
    )
    recipient_root = x509.load_pem_x509_certificate(recipient_root_pem)

    lockbox = _lockbox()
    lockbox.trusted_recipient_cas = [fingerprint_from_x509(recipient_root_pem)]
    grant = seal_grant(
        lockbox,
        GateKey(b"m" * 32),
        "recipient",
        lockbox.hierarchy[0],
        recipient_public_key,
        recipient_cert_pem=recipient_pem,
        recipient_cert_chain_pems=[recipient_root_pem],
    )
    authority_pem, authority_key = generate_authority_cert("lockbox-authority")
    lockbox.authority = sign_lockbox(
        lockbox,
        signing_key=authority_key,
        signing_cert_pem=authority_pem,
        ca_root_pem=authority_pem,
    )
    trusted_authorities = [fingerprint_from_x509(authority_pem)]

    fresh = verify_grant_provenance(
        lockbox,
        grant,
        recipient_pem,
        trusted_authorities,
        consumer_cert_chain_pems=[recipient_root_pem],
        revocation=RevocationPolicy(
            mode=RevocationCheck.ENFORCE,
            allowed_hosts=("revocation.example",),
            fetcher=lambda *args, **kwargs: _crl_bytes(
                recipient_root,
                recipient_root_key,
            ),
        ),
    )
    assert fresh.trusted, fresh.reasons

    revoked = verify_grant_provenance(
        lockbox,
        grant,
        recipient_pem,
        trusted_authorities,
        consumer_cert_chain_pems=[recipient_root_pem],
        revocation=RevocationPolicy(
            mode=RevocationCheck.ENFORCE,
            allowed_hosts=("revocation.example",),
            fetcher=lambda *args, **kwargs: _crl_bytes(
                recipient_root,
                recipient_root_key,
                revoked_serial=recipient_cert.serial_number,
            ),
        ),
    )
    assert not revoked.trusted
    assert any("REVOKED" in reason for reason in revoked.reasons)


def _ocsp_bytes(
    root,
    leaf,
    signing_key,
    *,
    revoked: bool = False,
    stale: bool = False,
) -> bytes:
    from cryptography import x509
    from cryptography.hazmat.primitives.hashes import SHA256
    from cryptography.hazmat.primitives.serialization import Encoding
    from cryptography.x509 import ocsp

    now = datetime.datetime.now(datetime.timezone.utc)
    status = ocsp.OCSPCertStatus.REVOKED if revoked else ocsp.OCSPCertStatus.GOOD
    response = (
        ocsp.OCSPResponseBuilder()
        .add_response(
            cert=leaf,
            issuer=root,
            algorithm=SHA256(),
            cert_status=status,
            this_update=now - datetime.timedelta(hours=2 if stale else 0, minutes=1),
            next_update=(
                now - datetime.timedelta(hours=1) if stale else now + datetime.timedelta(hours=1)
            ),
            revocation_time=(now - datetime.timedelta(seconds=1) if revoked else None),
            revocation_reason=(x509.ReasonFlags.key_compromise if revoked else None),
        )
        .responder_id(ocsp.OCSPResponderEncoding.HASH, root)
        .sign(signing_key, None)
    )
    return response.public_bytes(Encoding.DER)


def test_ocsp_response_requires_an_authorized_signature() -> None:
    root, root_key, leaf = _certificate_with_ocsp_url("https://revocation.example/ocsp")
    forged_response = bytearray(_ocsp_bytes(root, leaf, root_key))
    forged_response[-1] ^= 1

    with pytest.raises(ValueError, match="OCSP response signature"):
        check_certificate_revocation(
            leaf,
            root,
            mode=RevocationCheck.ENFORCE,
            fetcher=lambda *args, **kwargs: bytes(forged_response),
        )


def test_ocsp_accepts_fresh_issuer_response_and_rejects_revocation() -> None:
    root, root_key, leaf = _certificate_with_ocsp_url("https://revocation.example/ocsp")

    assert (
        check_certificate_revocation(
            leaf,
            root,
            mode=RevocationCheck.ENFORCE,
            fetcher=lambda *args, **kwargs: _ocsp_bytes(root, leaf, root_key),
        )
        == []
    )
    with pytest.raises(ValueError, match="REVOKED"):
        check_certificate_revocation(
            leaf,
            root,
            mode=RevocationCheck.ENFORCE,
            fetcher=lambda *args, **kwargs: _ocsp_bytes(
                root,
                leaf,
                root_key,
                revoked=True,
            ),
        )


def test_ocsp_rejects_stale_and_wrong_serial_responses() -> None:
    from cryptography import x509
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.x509.oid import NameOID

    root, root_key, leaf = _certificate_with_ocsp_url("https://revocation.example/ocsp")
    now = datetime.datetime.now(datetime.timezone.utc)
    other_leaf = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "other")]))
        .issuer_name(root.subject)
        .public_key(Ed25519PrivateKey.generate().public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .sign(root_key, None)
    )

    with pytest.raises(ValueError, match="OCSP response is stale"):
        check_certificate_revocation(
            leaf,
            root,
            mode=RevocationCheck.ENFORCE,
            fetcher=lambda *args, **kwargs: _ocsp_bytes(
                root,
                leaf,
                root_key,
                stale=True,
            ),
        )
    with pytest.raises(ValueError, match="serial does not match"):
        check_certificate_revocation(
            leaf,
            root,
            mode=RevocationCheck.ENFORCE,
            fetcher=lambda *args, **kwargs: _ocsp_bytes(
                root,
                other_leaf,
                root_key,
            ),
        )


def test_ocsp_rejects_delegated_responder_without_signing_usage() -> None:
    from cryptography import x509
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.hashes import SHA256
    from cryptography.hazmat.primitives.serialization import Encoding
    from cryptography.x509 import ocsp
    from cryptography.x509.oid import NameOID

    root, root_key, leaf = _certificate_with_ocsp_url("https://revocation.example/ocsp")
    responder_key = Ed25519PrivateKey.generate()
    now = datetime.datetime.now(datetime.timezone.utc)
    responder = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "responder")]))
        .issuer_name(root.subject)
        .public_key(responder_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(root_key, None)
    )
    response = (
        ocsp.OCSPResponseBuilder()
        .add_response(
            cert=leaf,
            issuer=root,
            algorithm=SHA256(),
            cert_status=ocsp.OCSPCertStatus.GOOD,
            this_update=now - datetime.timedelta(minutes=1),
            next_update=now + datetime.timedelta(hours=1),
            revocation_time=None,
            revocation_reason=None,
        )
        .responder_id(ocsp.OCSPResponderEncoding.HASH, responder)
        .certificates([responder])
        .sign(responder_key, None)
        .public_bytes(Encoding.DER)
    )

    with pytest.raises(ValueError, match="lacks OCSPSigning usage"):
        check_certificate_revocation(
            leaf,
            root,
            mode=RevocationCheck.ENFORCE,
            fetcher=lambda *args, **kwargs: response,
        )


def test_ocsp_rejects_responder_with_contradictory_key_usage() -> None:
    from cryptography import x509
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.hashes import SHA256
    from cryptography.hazmat.primitives.serialization import Encoding
    from cryptography.x509 import ocsp
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    root, root_key, leaf = _certificate_with_ocsp_url("https://revocation.example/ocsp")
    responder_key = Ed25519PrivateKey.generate()
    now = datetime.datetime.now(datetime.timezone.utc)
    responder = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "responder")]))
        .issuer_name(root.subject)
        .public_key(responder_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.OCSP_SIGNING]),
            critical=False,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=None,
                decipher_only=None,
            ),
            critical=True,
        )
        .sign(root_key, None)
    )
    response = (
        ocsp.OCSPResponseBuilder()
        .add_response(
            cert=leaf,
            issuer=root,
            algorithm=SHA256(),
            cert_status=ocsp.OCSPCertStatus.GOOD,
            this_update=now - datetime.timedelta(minutes=1),
            next_update=now + datetime.timedelta(hours=1),
            revocation_time=None,
            revocation_reason=None,
        )
        .responder_id(ocsp.OCSPResponderEncoding.HASH, responder)
        .certificates([responder])
        .sign(responder_key, None)
        .public_bytes(Encoding.DER)
    )

    with pytest.raises(ValueError, match="does not allow digital signatures"):
        check_certificate_revocation(
            leaf,
            root,
            mode=RevocationCheck.ENFORCE,
            fetcher=lambda *args, **kwargs: response,
        )


def _delegated_ocsp_response(*, include_no_check: bool) -> tuple[object, object, bytes]:
    from cryptography import x509
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.hashes import SHA256
    from cryptography.hazmat.primitives.serialization import Encoding
    from cryptography.x509 import ocsp
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    root, root_key, leaf = _certificate_with_ocsp_url("https://revocation.example/ocsp")
    responder_key = Ed25519PrivateKey.generate()
    now = datetime.datetime.now(datetime.timezone.utc)
    builder = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "responder")]))
        .issuer_name(root.subject)
        .public_key(responder_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.OCSP_SIGNING]),
            critical=False,
        )
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
            critical=True,
        )
    )
    if include_no_check:
        builder = builder.add_extension(x509.OCSPNoCheck(), critical=False)
    responder = builder.sign(root_key, None)
    response = (
        ocsp.OCSPResponseBuilder()
        .add_response(
            cert=leaf,
            issuer=root,
            algorithm=SHA256(),
            cert_status=ocsp.OCSPCertStatus.GOOD,
            this_update=now - datetime.timedelta(minutes=1),
            next_update=now + datetime.timedelta(hours=1),
            revocation_time=None,
            revocation_reason=None,
        )
        .responder_id(ocsp.OCSPResponderEncoding.HASH, responder)
        .certificates([responder])
        .sign(responder_key, None)
        .public_bytes(Encoding.DER)
    )
    return root, leaf, response


def test_ocsp_rejects_delegated_responder_without_revocation_contract() -> None:
    root, leaf, response = _delegated_ocsp_response(include_no_check=False)

    with pytest.raises(ValueError, match="lacks OCSP No Check"):
        check_certificate_revocation(
            leaf,
            root,
            mode=RevocationCheck.ENFORCE,
            fetcher=lambda *args, **kwargs: response,
        )


def test_ocsp_accepts_delegated_responder_with_noncritical_no_check() -> None:
    root, leaf, response = _delegated_ocsp_response(include_no_check=True)

    assert (
        check_certificate_revocation(
            leaf,
            root,
            mode=RevocationCheck.ENFORCE,
            fetcher=lambda *args, **kwargs: response,
        )
        == []
    )


def test_model_hash_binds_opset_and_complete_model(tmp_path) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    input_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    output_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "graph",
        [input_info],
        [output_info],
    )
    paths = []
    for opset in (13, 21):
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", opset)],
        )
        path = tmp_path / f"opset-{opset}.onnx"
        onnx.save(model, path)
        paths.append(path)

    assert compute_model_graph_hash(paths[0]) != compute_model_graph_hash(paths[1])

    base = onnx.load(paths[0])
    ir_changed = type(base)()
    ir_changed.CopyFrom(base)
    ir_changed.ir_version += 1
    ir_path = tmp_path / "ir-version.onnx"
    onnx.save(ir_changed, ir_path)

    function_changed = type(base)()
    function_changed.CopyFrom(base)
    function_changed.functions.extend(
        [
            helper.make_function(
                "audit.local",
                "IdentityFunction",
                ["input"],
                ["output"],
                [helper.make_node("Identity", ["input"], ["output"])],
                [helper.make_opsetid("", 13)],
            )
        ]
    )
    function_path = tmp_path / "function.onnx"
    onnx.save(function_changed, function_path)

    graph_changed = type(base)()
    graph_changed.CopyFrom(base)
    graph_changed.graph.node[0].name = "named-identity"
    graph_path = tmp_path / "graph.onnx"
    onnx.save(graph_changed, graph_path)

    base_hash = compute_model_graph_hash(paths[0])
    assert compute_model_graph_hash(ir_path) != base_hash
    assert compute_model_graph_hash(function_path) != base_hash
    assert compute_model_graph_hash(graph_path) != base_hash


def test_model_hash_rejects_external_tensor_path_escape(tmp_path) -> None:
    pytest.importorskip("onnx")
    from onnx import TensorProto, external_data_helper, helper

    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"\x00\x00\x80?")
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    weight = helper.make_tensor(
        "weight",
        TensorProto.FLOAT,
        [1],
        b"\x00\x00\x80?",
        raw=True,
    )
    external_data_helper.set_external_data(
        weight,
        location="../outside.bin",
        offset=0,
        length=4,
    )
    weight.data_location = TensorProto.EXTERNAL
    graph = helper.make_graph([], "graph", [], [], initializer=[weight])
    model = helper.make_model(graph)
    path = model_dir / "escaped.onnx"
    path.write_bytes(model.SerializeToString())

    with pytest.raises(ValueError, match="escapes the model directory"):
        compute_model_graph_hash(path)


def test_model_hash_binds_external_tensor_bytes(tmp_path) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    weight = helper.make_tensor(
        "weight",
        TensorProto.FLOAT,
        [1],
        b"\x00\x00\x80?",
        raw=True,
    )
    graph = helper.make_graph([], "graph", [], [], initializer=[weight])
    model = helper.make_model(graph)
    path = tmp_path / "external.onnx"
    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.bin",
        size_threshold=0,
    )

    before = compute_model_graph_hash(path)
    weights_path = tmp_path / "weights.bin"
    weights_path.write_bytes(b"\x00\x00\x00@")
    after = compute_model_graph_hash(path)
    assert before != after


def test_model_attestation_rejects_opset_tampering(tmp_path) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    input_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    output_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "graph",
        [input_info],
        [output_info],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path = tmp_path / "model.onnx"
    onnx.save(model, path)

    cert_pem, private_key = generate_authority_cert("attestation-root")
    lockbox = _lockbox()
    bind_model_artifact(lockbox, path)
    lockbox.authority = sign_lockbox(
        lockbox,
        signing_key=private_key,
        signing_cert_pem=cert_pem,
        ca_root_pem=cert_pem,
    )
    attest_model(
        path,
        private_key,
        cert_pem,
        cert_pem,
        lockbox.chain_hash,
        lockbox,
    )
    trusted_root = fingerprint_from_x509(cert_pem)
    expected_lockbox_hash = lockbox.authority.lockbox_hash
    assert verify_model_attestation(
        path,
        [trusted_root],
        expected_lockbox_hash=expected_lockbox_hash,
        revocation=RevocationCheck.SKIP,
    ).trusted

    tampered = onnx.load(path)
    tampered.opset_import[0].version = 21
    onnx.save(tampered, path)
    assert not verify_model_attestation(
        path,
        [trusted_root],
        expected_lockbox_hash=expected_lockbox_hash,
        revocation=RevocationCheck.SKIP,
    ).trusted


def test_model_attestation_rejects_duplicate_onnx_metadata_keys(tmp_path) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    input_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    output_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "graph",
        [input_info],
        [output_info],
    )
    path = tmp_path / "duplicate-metadata.onnx"
    onnx.save(helper.make_model(graph), path)

    cert_pem, private_key = generate_authority_cert("attestation-root")
    lockbox = _lockbox()
    bind_model_artifact(lockbox, path)
    lockbox.authority = sign_lockbox(
        lockbox,
        signing_key=private_key,
        signing_cert_pem=cert_pem,
        ca_root_pem=cert_pem,
    )
    attest_model(
        path,
        private_key,
        cert_pem,
        cert_pem,
        lockbox.chain_hash,
        lockbox,
    )

    model = onnx.load(path)
    duplicate = model.metadata_props.add()
    duplicate.key = "schemen.chain_hash"
    duplicate.value = lockbox.chain_hash
    onnx.save(model, path)

    result = verify_model_attestation(
        path,
        [fingerprint_from_x509(cert_pem)],
        expected_lockbox_hash=lockbox.authority.lockbox_hash,
        revocation=RevocationCheck.SKIP,
    )
    assert not result.trusted
    assert result.reasons == ["Duplicate ONNX metadata keys are not permitted: schemen.chain_hash"]


def test_model_attestation_rejects_chain_hash_outside_signed_lockbox(tmp_path) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    input_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    output_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "graph",
        [input_info],
        [output_info],
    )
    path = tmp_path / "model.onnx"
    onnx.save(helper.make_model(graph), path)

    cert_pem, private_key = generate_authority_cert("attestation-root")
    lockbox = _lockbox()
    bind_model_artifact(lockbox, path)
    lockbox.authority = sign_lockbox(
        lockbox,
        signing_key=private_key,
        signing_cert_pem=cert_pem,
        ca_root_pem=cert_pem,
    )

    with pytest.raises(ValueError, match="does not match the signed lockbox"):
        attest_model(
            path,
            private_key,
            cert_pem,
            cert_pem,
            "0" * 64,
            lockbox,
        )


def test_postgres_identifier_injection_is_rejected_before_connect() -> None:
    with pytest.raises(ValueError, match="identifier"):
        PgVectorStore("postgresql://unused", table="vectors; DROP TABLE users")


def test_postgres_vectors_are_validated_before_connect() -> None:
    import numpy as np

    # Exercise the pre-connection guards without requiring the optional
    # psycopg extra in the core test environment.
    store = object.__new__(PgVectorStore)
    store._dim = 3
    with pytest.raises(ValueError, match="dimension"):
        store.retrieve(np.ones(2), "regime-0")
    with pytest.raises(ValueError, match="finite"):
        store.insert(np.array([1.0, np.nan, 3.0]), "doc", "regime-0")
    with pytest.raises(ValueError, match="top_k"):
        store.retrieve(np.ones(3), "regime-0", top_k=0)


def test_postgres_rows_fail_closed_instead_of_synthesizing_vectors() -> None:
    store = object.__new__(PgVectorStore)
    store._dim = 3

    with pytest.raises(ValueError, match="invalid embedding"):
        store._embedding_from_row(object())
    with pytest.raises(ValueError, match="invalid embedding"):
        store._embedding_from_row("[1.0, 2.0]")
    with pytest.raises(ValueError, match="invalid embedding"):
        store._embedding_from_row("[1.0, NaN, 3.0]")

    metadata = {"nested": {"labels": ["original"]}}
    detached = store._metadata_from_row(metadata)
    metadata["nested"]["labels"][0] = "mutated"
    assert detached["nested"]["labels"] == ["original"]
    with pytest.raises(ValueError, match="invalid metadata"):
        store._metadata_from_row(["not", "an", "object"])


def test_postgres_upsert_cannot_move_an_id_between_partitions() -> None:
    import numpy as np

    pytest.importorskip("psycopg")

    class ConflictResult:
        def fetchone(self):
            return None

    class FakeConnection:
        closed = False

        def execute(self, statement, parameters):
            assert 'WHERE "schemen_vectors".regime_id = EXCLUDED.regime_id' in statement.as_string()
            return ConflictResult()

    store = PgVectorStore("postgresql://unused")
    store._dim = 3
    store._conn = FakeConnection()
    with pytest.raises(ValueError, match="another partition"):
        store.insert(
            np.ones(3),
            "doc",
            "regime-b",
            {"doc_id": "owned-by-regime-a"},
        )


def test_architecture_certification_fails_closed_when_state_export_fails() -> None:
    from schemen_gate._rag_arch import ArchitectureSpec

    class BrokenModel:
        def state_dict(self):
            raise RuntimeError("state export failed")

    with pytest.raises(RuntimeError, match="state export failed"):
        ArchitectureSpec.from_model(BrokenModel())


def test_cargo_access_keys_are_bound_to_the_complete_authorization_scope() -> None:
    root = b"g" * 32
    scope = {
        "subject_id": "subject-a",
        "model_digest": "model-a",
        "operation": "retrieve",
        "policy_version": "policy-v1",
        "partition_key": "partition-a",
    }
    key = derive_cargo_access_key(root, 0, "tenant-a", **scope)
    assert key != root
    assert key != derive_cargo_access_key(root, 0, "tenant-b", **scope)
    assert key != derive_cargo_access_key(root, 1, "tenant-a", **scope)
    for field, value in (
        ("subject_id", "subject-b"),
        ("model_digest", "model-b"),
        ("operation", "load"),
        ("policy_version", "policy-v2"),
        ("partition_key", "partition-b"),
    ):
        changed = {**scope, field: value}
        assert key != derive_cargo_access_key(root, 0, "tenant-a", **changed)


def test_architecture_hash_binds_exact_parameter_values() -> None:
    import numpy as np

    first = {"layer.weight": np.array([1.0, 2.0])}
    second = {"layer.weight": np.array([1.0, 3.0])}
    assert _hash_state_dict(first) != _hash_state_dict(second)
