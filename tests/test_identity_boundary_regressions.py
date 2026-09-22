"""Synthetic regressions for signed hierarchy and X.509 name boundaries."""

from __future__ import annotations

import datetime
from dataclasses import replace
from typing import Any

import pytest
from cryptography import x509
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.x509 import ocsp
from cryptography.x509.name import _ASN1Type
from cryptography.x509.oid import AuthorityInformationAccessOID, ExtendedKeyUsageOID, NameOID

from schemen_gate import GateKey
from schemen_gate._lockbox import (
    HierarchyDef,
    RegimeCapability,
    RevocationCheck,
    _name_within_constraint,
    check_certificate_revocation,
    create_lockbox,
    sign_lockbox,
    verify_authority,
)
from schemen_gate._release import GateReleaseIdentity

RELEASE = GateReleaseIdentity(
    package="schemen-gate",
    version="1.0.2",
    source_repository="https://github.com/sekosai/schemen-gate",
    source_commit="f" * 40,
)


def _certificate(
    key: Ed25519PrivateKey,
    subject: str,
    *,
    issuer: x509.Certificate | None = None,
    issuer_key: Ed25519PrivateKey | None = None,
    ca: bool = False,
    extensions: tuple[tuple[x509.ExtensionType, bool], ...] = (),
) -> x509.Certificate:
    now = datetime.datetime.now(datetime.timezone.utc)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject)])
    builder = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(issuer.subject if issuer is not None else name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
    )
    for extension, critical in extensions:
        builder = builder.add_extension(extension, critical)
    return builder.sign(issuer_key or key, None)


def _signed_lockbox(
    constraints: x509.NameConstraints | None = None,
    *,
    uri: str | None = None,
):
    root_key = Ed25519PrivateKey.generate()
    root_extensions = () if constraints is None else ((constraints, True),)
    root = _certificate(root_key, "root", ca=True, extensions=root_extensions)
    leaf_key = Ed25519PrivateKey.generate()
    leaf_extensions = (
        ()
        if uri is None
        else ((x509.SubjectAlternativeName([x509.UniformResourceIdentifier(uri)]), False),)
    )
    leaf = _certificate(
        leaf_key,
        "Example Authority",
        issuer=root,
        issuer_key=root_key,
        extensions=leaf_extensions,
    )
    lockbox = create_lockbox(
        GateKey(b"m" * 32),
        "chain",
        "test",
        4,
        1,
        [HierarchyDef("reader", "read only", [0], [RegimeCapability(0, "read", ["value"])])],
        release_identity=RELEASE,
    )
    root_pem = root.public_bytes(Encoding.PEM)
    lockbox.authority = sign_lockbox(
        lockbox,
        signing_key=leaf_key,
        signing_cert_pem=leaf.public_bytes(Encoding.PEM),
        ca_root_pem=root_pem,
        cert_chain_pems=[root_pem],
        expected_release=RELEASE,
    )
    return lockbox, [root.fingerprint(SHA256()).hex()]


def _verify(lockbox: Any, roots: list[str]) -> bool:
    return verify_authority(
        lockbox,
        roots,
        revocation=RevocationCheck.SKIP,
        expected_release=RELEASE,
    )


def test_unchanged_hierarchy_retains_the_existing_signed_wire_hash() -> None:
    lockbox, roots = _signed_lockbox()
    # Captured with the pre-fix hash function and this fixed release fixture.
    assert lockbox.authority.lockbox_hash == (
        "f8aa6660688d0a67c0588ade6382eac732606ae5e8a1f80858018999ae875e84"
    )
    assert _verify(lockbox, roots)


@pytest.mark.parametrize("field", ["name", "regimes", "capabilities", "columns"])
def test_authority_verification_rechecks_current_hierarchy(field: str) -> None:
    lockbox, roots = _signed_lockbox()
    assert _verify(lockbox, roots)
    level = lockbox.hierarchy[0]
    if field == "name":
        lockbox.hierarchy[0] = replace(level, name="administrator")
    elif field == "regimes":
        level.regimes.append(1)
    elif field == "capabilities":
        level.capabilities.append(RegimeCapability(0, "admin", ["secret"]))
    else:
        level.capabilities[0].column_names.append("secret")
    with pytest.raises(ValueError, match="Hierarchy hash mismatch"):
        _verify(lockbox, roots)


@pytest.mark.parametrize(
    ("constraint", "uri", "accepted"),
    [
        ("tenant.example", "spiffe://tenant.example/workload", True),
        ("tenant.example", "spiffe://child.tenant.example/workload", False),
        (".tenant.example", "spiffe://child.tenant.example/workload", True),
        (".tenant.example", "spiffe://tenant.example/workload", False),
        ("TENANT.EXAMPLE", "spiffe://tenant.example/workload", True),
    ],
)
def test_authority_uri_constraints_distinguish_hosts_from_subtrees(
    constraint: str, uri: str, accepted: bool
) -> None:
    lockbox, roots = _signed_lockbox(
        x509.NameConstraints(
            permitted_subtrees=[x509.UniformResourceIdentifier(constraint)],
            excluded_subtrees=None,
        ),
        uri=uri,
    )
    if accepted:
        assert _verify(lockbox, roots)
    else:
        with pytest.raises(ValueError, match="permitted name subtrees"):
            _verify(lockbox, roots)


@pytest.mark.parametrize("uri", ["spiffe://192.0.2.1/workload", "urn:example:workload"])
def test_uri_constraints_reject_non_dns_hosts_even_for_exclusions(uri: str) -> None:
    lockbox, roots = _signed_lockbox(
        x509.NameConstraints(
            permitted_subtrees=None,
            excluded_subtrees=[x509.UniformResourceIdentifier("blocked.example")],
        ),
        uri=uri,
    )
    with pytest.raises(ValueError, match="URI subject name"):
        _verify(lockbox, roots)


@pytest.mark.parametrize("excluded", ["EXAMPLE AUTHORITY", "  Example   Authority  "])
def test_directory_name_exclusions_apply_case_and_space_matching(excluded: str) -> None:
    lockbox, roots = _signed_lockbox(
        x509.NameConstraints(
            permitted_subtrees=None,
            excluded_subtrees=[
                x509.DirectoryName(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, excluded)]))
            ],
        )
    )
    with pytest.raises(ValueError, match="excluded name subtree"):
        _verify(lockbox, roots)


@pytest.mark.parametrize(
    "attribute",
    [
        x509.NameAttribute(NameOID.COMMON_NAME, "caf\N{LATIN SMALL LETTER E WITH ACUTE}"),
        x509.NameAttribute(NameOID.COMMON_NAME, "bad\tname"),
        x509.NameAttribute(x509.ObjectIdentifier("1.2.3.4.5.6"), "unknown"),
        x509.NameAttribute(NameOID.COMMON_NAME, "blocked", _type=_ASN1Type.BMPString),
    ],
)
def test_directory_name_constraints_refuse_unsupported_matching_profiles(
    attribute: x509.NameAttribute[str],
) -> None:
    candidate = x509.DirectoryName(x509.Name([attribute]))
    constraint = x509.DirectoryName(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "blocked")]))
    with pytest.raises(ValueError, match="DirectoryName"):
        _name_within_constraint(candidate, constraint)
    with pytest.raises(ValueError, match="DirectoryName"):
        _name_within_constraint(constraint, candidate)


def test_directory_name_constraints_match_rdn_sets_but_preserve_rdn_order() -> None:
    organization = x509.NameAttribute(NameOID.ORGANIZATION_NAME, " Example   Org ")
    domain = x509.NameAttribute(NameOID.DOMAIN_COMPONENT, "EXAMPLE")
    candidate = x509.DirectoryName(
        x509.Name(
            [
                x509.RelativeDistinguishedName([organization, domain]),
                x509.RelativeDistinguishedName([x509.NameAttribute(NameOID.COMMON_NAME, "worker")]),
            ]
        )
    )
    constraint = x509.DirectoryName(
        x509.Name(
            [
                x509.RelativeDistinguishedName(
                    [
                        x509.NameAttribute(NameOID.DOMAIN_COMPONENT, "example"),
                        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "example org"),
                    ]
                ),
            ]
        )
    )
    assert _name_within_constraint(candidate, constraint)
    reversed_candidate = x509.DirectoryName(x509.Name(list(reversed(candidate.value.rdns))))
    assert not _name_within_constraint(reversed_candidate, constraint)


@pytest.mark.parametrize("critical", [True, False])
def test_delegated_ocsp_responder_rejects_unsupported_extensions(critical: bool) -> None:
    root_key = Ed25519PrivateKey.generate()
    root = _certificate(root_key, "root", ca=True)
    leaf = _certificate(
        Ed25519PrivateKey.generate(),
        "leaf",
        issuer=root,
        issuer_key=root_key,
        extensions=(
            (
                x509.AuthorityInformationAccess(
                    [
                        x509.AccessDescription(
                            AuthorityInformationAccessOID.OCSP,
                            x509.UniformResourceIdentifier("https://revocation.example/ocsp"),
                        )
                    ]
                ),
                False,
            ),
        ),
    )
    responder_key = Ed25519PrivateKey.generate()
    unsupported = (
        x509.UnrecognizedExtension(x509.ObjectIdentifier("1.2.3.4.5.6"), b"\x05\x00")
        if critical
        else x509.PolicyConstraints(require_explicit_policy=0, inhibit_policy_mapping=None)
    )
    responder = _certificate(
        responder_key,
        "responder",
        issuer=root,
        issuer_key=root_key,
        extensions=(
            (x509.ExtendedKeyUsage([ExtendedKeyUsageOID.OCSP_SIGNING]), False),
            (x509.OCSPNoCheck(), False),
            (unsupported, critical),
        ),
    )
    now = datetime.datetime.now(datetime.timezone.utc)
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
    with pytest.raises(ValueError, match="unsupported (critical|path)"):
        check_certificate_revocation(
            leaf,
            root,
            mode=RevocationCheck.ENFORCE,
            fetcher=lambda *args, **kwargs: response,
        )
