"""PKCS#12 authority-credential contract tests."""

from __future__ import annotations

import datetime
import hashlib
import ipaddress
from dataclasses import replace
from typing import Any, Sequence

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed448, ed25519, padding, rsa
from cryptography.hazmat.primitives.serialization import pkcs12
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from schemen_gate import (
    GateKey,
    HierarchyDef,
    Pkcs12KeyProvider,
    RegimeCapability,
    RevocationCheck,
    create_lockbox,
    fingerprint_from_x509,
    verify_authority,
)
from schemen_gate._lockbox import _name_within_constraint, _verify_cert_chain, _verify_signature

PASSWORD = b"fixture-password"


def test_pkcs12_provider_is_a_public_lazy_export() -> None:
    from schemen_gate import Pkcs12KeyProvider as PublicProvider

    assert PublicProvider is Pkcs12KeyProvider


def test_pkcs12_provider_rejects_unbounded_or_mutable_input_before_parsing() -> None:
    with pytest.raises(ValueError, match="non-empty bytes"):
        Pkcs12KeyProvider.from_bytes(bytearray(b"not-a-pkcs12"), PASSWORD)
    with pytest.raises(ValueError, match="no larger"):
        Pkcs12KeyProvider.from_bytes(bytes(8 * 1024 * 1024 + 1), PASSWORD)
    with pytest.raises(ValueError, match="password must be bytes"):
        Pkcs12KeyProvider.from_bytes(b"not-a-pkcs12", bytearray(PASSWORD))


def _certificate(
    *,
    subject_name: str,
    subject_key: Any,
    issuer_name: x509.Name,
    issuer_key: Any,
    is_ca: bool,
    extra_extensions: Sequence[tuple[x509.ExtensionType, bool]] = (),
    rsa_padding: padding.AsymmetricPadding | None = None,
) -> x509.Certificate:
    now = datetime.datetime.now(datetime.timezone.utc)
    builder = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject_name)]))
        .issuer_name(issuer_name)
        .public_key(subject_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=30))
        .add_extension(x509.BasicConstraints(ca=is_ca, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=is_ca,
                crl_sign=is_ca,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
    )
    for extension, critical in extra_extensions:
        builder = builder.add_extension(extension, critical=critical)
    algorithm = (
        None
        if isinstance(
            issuer_key,
            (ed25519.Ed25519PrivateKey, ed448.Ed448PrivateKey),
        )
        else hashes.SHA256()
    )
    if rsa_padding is not None:
        return builder.sign(issuer_key, algorithm, rsa_padding=rsa_padding)
    return builder.sign(issuer_key, algorithm)


def _private_key(kind: str) -> Any:
    if kind == "ed25519":
        return ed25519.Ed25519PrivateKey.generate()
    if kind == "ed448":
        return ed448.Ed448PrivateKey.generate()
    if kind == "ecdsa":
        return ec.generate_private_key(ec.SECP256R1())
    if kind == "rsa":
        return rsa.generate_private_key(public_exponent=65537, key_size=2048)
    raise ValueError(f"unsupported fixture key kind: {kind}")


def _credential(
    root_label: str = "test-machine-root",
    *,
    key_kind: str = "ed25519",
    root_extensions: Sequence[tuple[x509.ExtensionType, bool]] = (),
    leaf_extensions: Sequence[tuple[x509.ExtensionType, bool]] = (),
) -> tuple[bytes, bytes]:
    root_key = _private_key(key_kind)
    root_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, root_label)])
    root = _certificate(
        subject_name=root_label,
        subject_key=root_key,
        issuer_name=root_name,
        issuer_key=root_key,
        is_ca=True,
        extra_extensions=root_extensions,
    )
    leaf_key = _private_key(key_kind)
    leaf = _certificate(
        subject_name="test-machine",
        subject_key=leaf_key,
        issuer_name=root.subject,
        issuer_key=root_key,
        is_ca=False,
        extra_extensions=leaf_extensions,
    )
    data = pkcs12.serialize_key_and_certificates(
        name=b"schemen-gate-machine",
        key=leaf_key,
        cert=leaf,
        cas=[root],
        encryption_algorithm=serialization.BestAvailableEncryption(PASSWORD),
    )
    return data, root.public_bytes(serialization.Encoding.PEM)


def _lockbox():
    capability = RegimeCapability(0, "read", ["value"])
    return create_lockbox(
        GateKey(b"m" * 32),
        chain_hash=hashlib.sha256(b"chain").hexdigest(),
        chain_name="pkcs12-chain",
        n_dims=4,
        n_regimes=1,
        hierarchy_def=[HierarchyDef("reader", "read only", [0], [capability])],
    )


def test_pkcs12_machine_identity_signs_to_an_external_root_pin() -> None:
    data, root_pem = _credential()
    provider = Pkcs12KeyProvider.from_bytes(data, PASSWORD)
    lockbox = _lockbox()
    lockbox.authority = provider.sign_lockbox(lockbox)

    assert provider.root_certificate_pem == root_pem
    assert len(provider.certificate_chain_pems) == 1
    assert verify_authority(
        lockbox,
        [fingerprint_from_x509(root_pem)],
        revocation=RevocationCheck.SKIP,
    )


@pytest.mark.parametrize(
    ("key_kind", "expected_algorithm"),
    [
        ("ed25519", "Ed25519"),
        ("ed448", "Ed448"),
        ("ecdsa", "ECDSA-SHA256"),
        ("rsa", "RSA-PSS-SHA256"),
    ],
)
def test_pkcs12_common_authority_key_families_are_supported(
    key_kind: str,
    expected_algorithm: str,
) -> None:
    data, root_pem = _credential(key_kind=key_kind)
    provider = Pkcs12KeyProvider.from_bytes(data, PASSWORD)
    lockbox = _lockbox()
    lockbox.authority = provider.sign_lockbox(lockbox)

    assert provider.signature_algorithm == expected_algorithm
    assert lockbox.authority.signature_algorithm == expected_algorithm
    assert verify_authority(
        lockbox,
        [fingerprint_from_x509(root_pem)],
        revocation=RevocationCheck.SKIP,
    )


def test_pkcs12_accepts_common_noncritical_machine_certificate_metadata() -> None:
    policies = x509.CertificatePolicies(
        [x509.PolicyInformation(x509.ObjectIdentifier("1.3.6.1.4.1.57264.1"), None)]
    )
    eku = x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH])
    data, root_pem = _credential(
        leaf_extensions=((policies, False), (eku, False)),
    )
    provider = Pkcs12KeyProvider.from_bytes(data, PASSWORD)
    lockbox = _lockbox()
    lockbox.authority = provider.sign_lockbox(lockbox)

    assert verify_authority(
        lockbox,
        [fingerprint_from_x509(root_pem)],
        revocation=RevocationCheck.SKIP,
    )


def test_pkcs12_rejects_leaf_eku_that_does_not_authorize_machine_signing() -> None:
    data, root_pem = _credential(
        leaf_extensions=((x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), False),),
    )
    provider = Pkcs12KeyProvider.from_bytes(data, PASSWORD)
    lockbox = _lockbox()
    lockbox.authority = provider.sign_lockbox(lockbox)

    with pytest.raises(ValueError, match="Extended Key Usage"):
        verify_authority(
            lockbox,
            [fingerprint_from_x509(root_pem)],
            revocation=RevocationCheck.SKIP,
        )


@pytest.mark.parametrize(
    ("dns_name", "accepted"),
    [
        ("machine.example.com", True),
        ("nested.machine.example.com", True),
        ("machine.example.net", False),
    ],
)
def test_pkcs12_enforces_ca_dns_name_constraints(
    dns_name: str,
    accepted: bool,
) -> None:
    data, root_pem = _credential(
        root_extensions=(
            (
                x509.NameConstraints(
                    permitted_subtrees=[x509.DNSName(".example.com")],
                    excluded_subtrees=None,
                ),
                True,
            ),
        ),
        leaf_extensions=((x509.SubjectAlternativeName([x509.DNSName(dns_name)]), False),),
    )
    provider = Pkcs12KeyProvider.from_bytes(data, PASSWORD)
    lockbox = _lockbox()
    lockbox.authority = provider.sign_lockbox(lockbox)

    def verify() -> bool:
        return verify_authority(
            lockbox,
            [fingerprint_from_x509(root_pem)],
            revocation=RevocationCheck.SKIP,
        )

    if accepted:
        assert verify()
    else:
        with pytest.raises(ValueError, match="permitted name subtrees"):
            verify()


def test_pkcs12_enforces_excluded_ca_dns_name_constraint() -> None:
    data, root_pem = _credential(
        root_extensions=(
            (
                x509.NameConstraints(
                    permitted_subtrees=None,
                    excluded_subtrees=[x509.DNSName(".blocked.example.com")],
                ),
                True,
            ),
        ),
        leaf_extensions=(
            (
                x509.SubjectAlternativeName([x509.DNSName("machine.blocked.example.com")]),
                False,
            ),
        ),
    )
    provider = Pkcs12KeyProvider.from_bytes(data, PASSWORD)
    lockbox = _lockbox()
    lockbox.authority = provider.sign_lockbox(lockbox)

    with pytest.raises(ValueError, match="excluded name subtree"):
        verify_authority(
            lockbox,
            [fingerprint_from_x509(root_pem)],
            revocation=RevocationCheck.SKIP,
        )


def test_pkcs12_rejects_unsupported_name_constraint_form() -> None:
    oid = x509.ObjectIdentifier("1.3.6.1.4.1.57264.99")
    other_name = x509.OtherName(oid, b"\x0c\x03abc")
    data, root_pem = _credential(
        root_extensions=(
            (
                x509.NameConstraints(
                    permitted_subtrees=[other_name],
                    excluded_subtrees=None,
                ),
                True,
            ),
        ),
        leaf_extensions=((x509.SubjectAlternativeName([other_name]), False),),
    )
    provider = Pkcs12KeyProvider.from_bytes(data, PASSWORD)
    lockbox = _lockbox()
    lockbox.authority = provider.sign_lockbox(lockbox)

    with pytest.raises(ValueError, match="Unsupported NameConstraints GeneralName"):
        verify_authority(
            lockbox,
            [fingerprint_from_x509(root_pem)],
            revocation=RevocationCheck.SKIP,
        )


@pytest.mark.parametrize(
    ("name", "constraint", "accepted"),
    [
        (x509.RFC822Name("user@example.com"), x509.RFC822Name("example.com"), True),
        (x509.RFC822Name("user@example.net"), x509.RFC822Name("example.com"), False),
        (
            x509.UniformResourceIdentifier("spiffe://worker.example.com/service"),
            x509.UniformResourceIdentifier(".example.com"),
            True,
        ),
        (
            x509.UniformResourceIdentifier("spiffe://worker.example.net/service"),
            x509.UniformResourceIdentifier(".example.com"),
            False,
        ),
        (
            x509.IPAddress(ipaddress.ip_address("192.0.2.10")),
            x509.IPAddress(ipaddress.ip_network("192.0.2.0/24")),
            True,
        ),
        (
            x509.IPAddress(ipaddress.ip_address("198.51.100.10")),
            x509.IPAddress(ipaddress.ip_network("192.0.2.0/24")),
            False,
        ),
        (
            x509.DirectoryName(
                x509.Name(
                    [
                        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Example"),
                        x509.NameAttribute(NameOID.COMMON_NAME, "machine"),
                    ]
                )
            ),
            x509.DirectoryName(
                x509.Name(
                    [
                        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Example"),
                    ]
                )
            ),
            True,
        ),
    ],
)
def test_name_constraint_matching_standard_forms(
    name: x509.GeneralName,
    constraint: x509.GeneralName,
    accepted: bool,
) -> None:
    assert _name_within_constraint(name, constraint) is accepted


@pytest.mark.parametrize(
    ("ca_purpose", "accepted"),
    [
        (ExtendedKeyUsageOID.CLIENT_AUTH, True),
        (ExtendedKeyUsageOID.CODE_SIGNING, True),
        (ExtendedKeyUsageOID.SERVER_AUTH, False),
    ],
)
def test_pkcs12_enforces_ca_extended_key_usage_constraints(
    ca_purpose: x509.ObjectIdentifier,
    accepted: bool,
) -> None:
    data, root_pem = _credential(
        root_extensions=((x509.ExtendedKeyUsage([ca_purpose]), False),),
    )
    provider = Pkcs12KeyProvider.from_bytes(data, PASSWORD)
    lockbox = _lockbox()
    lockbox.authority = provider.sign_lockbox(lockbox)

    def verify() -> bool:
        return verify_authority(
            lockbox,
            [fingerprint_from_x509(root_pem)],
            revocation=RevocationCheck.SKIP,
        )

    if accepted:
        assert verify()
    else:
        with pytest.raises(ValueError, match="no common Extended Key Usage"):
            verify()


def test_pkcs12_rejects_disjoint_leaf_and_ca_extended_key_usage() -> None:
    data, root_pem = _credential(
        root_extensions=((x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]), False),),
        leaf_extensions=((x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CODE_SIGNING]), False),),
    )
    provider = Pkcs12KeyProvider.from_bytes(data, PASSWORD)
    lockbox = _lockbox()
    lockbox.authority = provider.sign_lockbox(lockbox)

    with pytest.raises(ValueError, match="no common Extended Key Usage"):
        verify_authority(
            lockbox,
            [fingerprint_from_x509(root_pem)],
            revocation=RevocationCheck.SKIP,
        )


def test_pkcs12_can_end_at_an_explicitly_pinned_non_self_signed_anchor() -> None:
    root_key = _private_key("rsa")
    root_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "offline-root")])
    root = _certificate(
        subject_name="offline-root",
        subject_key=root_key,
        issuer_name=root_name,
        issuer_key=root_key,
        is_ca=True,
    )
    anchor_key = _private_key("ecdsa")
    anchor = _certificate(
        subject_name="pinned-enterprise-anchor",
        subject_key=anchor_key,
        issuer_name=root.subject,
        issuer_key=root_key,
        is_ca=True,
    )
    leaf_key = _private_key("ed25519")
    leaf = _certificate(
        subject_name="machine",
        subject_key=leaf_key,
        issuer_name=anchor.subject,
        issuer_key=anchor_key,
        is_ca=False,
        extra_extensions=((x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]), False),),
    )
    data = pkcs12.serialize_key_and_certificates(
        name=b"machine",
        key=leaf_key,
        cert=leaf,
        cas=[anchor],
        encryption_algorithm=serialization.BestAvailableEncryption(PASSWORD),
    )
    provider = Pkcs12KeyProvider.from_bytes(data, PASSWORD)
    lockbox = _lockbox()
    lockbox.authority = provider.sign_lockbox(lockbox)
    anchor_pem = anchor.public_bytes(serialization.Encoding.PEM)

    assert verify_authority(
        lockbox,
        [fingerprint_from_x509(anchor_pem)],
        revocation=RevocationCheck.SKIP,
    )


def test_pkcs12_accepts_rsa_pss_signed_certificate_paths() -> None:
    root_key = _private_key("rsa")
    root_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "pss-root")])
    pss = padding.PSS(
        mgf=padding.MGF1(hashes.SHA256()),
        salt_length=hashes.SHA256().digest_size,
    )
    root = _certificate(
        subject_name="pss-root",
        subject_key=root_key,
        issuer_name=root_name,
        issuer_key=root_key,
        is_ca=True,
        rsa_padding=pss,
    )
    leaf_key = _private_key("ecdsa")
    leaf = _certificate(
        subject_name="pss-issued-machine",
        subject_key=leaf_key,
        issuer_name=root.subject,
        issuer_key=root_key,
        is_ca=False,
        rsa_padding=pss,
    )
    data = pkcs12.serialize_key_and_certificates(
        name=b"pss-issued-machine",
        key=leaf_key,
        cert=leaf,
        cas=[root],
        encryption_algorithm=serialization.BestAvailableEncryption(PASSWORD),
    )
    provider = Pkcs12KeyProvider.from_bytes(data, PASSWORD)
    lockbox = _lockbox()
    lockbox.authority = provider.sign_lockbox(lockbox)
    root_pem = root.public_bytes(serialization.Encoding.PEM)

    assert verify_authority(
        lockbox,
        [fingerprint_from_x509(root_pem)],
        revocation=RevocationCheck.SKIP,
    )


def test_rsa_pss_certificate_signature_rejects_weak_mgf_hash() -> None:
    key = _private_key("rsa")
    payload = b"certificate-tbs-fixture"
    parameters = padding.PSS(
        # Negative test only: prove the verifier rejects a weak MGF hash.
        mgf=padding.MGF1(hashes.SHA1()),  # nosec B303
        salt_length=hashes.SHA256().digest_size,
    )
    signature = key.sign(payload, parameters, hashes.SHA256())

    with pytest.raises(ValueError, match="mask-generation hash"):
        _verify_signature(
            key.public_key(),
            signature,
            payload,
            hashes.SHA256(),
            parameters,
        )


def test_recipient_path_rejects_server_only_leaf_eku() -> None:
    data, root_pem = _credential(
        leaf_extensions=((x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), False),),
    )
    provider = Pkcs12KeyProvider.from_bytes(data, PASSWORD)

    with pytest.raises(ValueError, match="recipient identity"):
        _verify_cert_chain(
            provider.leaf_certificate,
            list(provider.certificate_chain_pems),
            hashes.SHA256,
            require_leaf_digital_signature=False,
            trusted_root_fingerprints=[fingerprint_from_x509(root_pem)],
        )


def test_pkcs12_wrong_password_is_rejected() -> None:
    data, _ = _credential()
    with pytest.raises(ValueError):
        Pkcs12KeyProvider.from_bytes(data, b"wrong-password")


def test_pkcs12_trust_is_external_not_bundle_selected() -> None:
    data, _ = _credential()
    provider = Pkcs12KeyProvider.from_bytes(data, PASSWORD)
    lockbox = _lockbox()
    lockbox.authority = provider.sign_lockbox(lockbox)

    with pytest.raises(ValueError, match="trust store"):
        verify_authority(
            lockbox,
            ["0" * 64],
            revocation=RevocationCheck.SKIP,
        )


def test_authority_certificate_chain_count_is_bounded_before_path_work() -> None:
    data, root_pem = _credential()
    provider = Pkcs12KeyProvider.from_bytes(data, PASSWORD)
    lockbox = _lockbox()
    authority = provider.sign_lockbox(lockbox)
    lockbox.authority = replace(
        authority,
        cert_chain_pems=[root_pem] * 17,
    )

    with pytest.raises(ValueError, match="maximum of 16 certificates"):
        verify_authority(
            lockbox,
            [fingerprint_from_x509(root_pem)],
            revocation=RevocationCheck.SKIP,
        )


def test_authority_certificate_pem_size_is_bounded_before_parsing() -> None:
    data, root_pem = _credential()
    provider = Pkcs12KeyProvider.from_bytes(data, PASSWORD)
    lockbox = _lockbox()
    authority = provider.sign_lockbox(lockbox)
    lockbox.authority = replace(
        authority,
        signing_cert_pem=b"A" * (64 * 1024 + 1),
    )

    with pytest.raises(ValueError, match="no larger than 65536 bytes"):
        verify_authority(
            lockbox,
            [fingerprint_from_x509(root_pem)],
            revocation=RevocationCheck.SKIP,
        )


def test_verifier_accepts_exactly_the_user_selected_ca_roots() -> None:
    selected_data, selected_root_pem = _credential("selected-machine-root")
    _, unrelated_root_pem = _credential("unrelated-machine-root")
    provider = Pkcs12KeyProvider.from_bytes(selected_data, PASSWORD)
    lockbox = _lockbox()
    lockbox.authority = provider.sign_lockbox(lockbox)

    selected_root = fingerprint_from_x509(selected_root_pem)
    unrelated_root = fingerprint_from_x509(unrelated_root_pem)

    assert verify_authority(
        lockbox,
        [selected_root],
        revocation=RevocationCheck.SKIP,
    )
    assert verify_authority(
        lockbox,
        [unrelated_root, selected_root],
        revocation=RevocationCheck.SKIP,
    )
    with pytest.raises(ValueError, match="trust store"):
        verify_authority(
            lockbox,
            [unrelated_root],
            revocation=RevocationCheck.SKIP,
        )
