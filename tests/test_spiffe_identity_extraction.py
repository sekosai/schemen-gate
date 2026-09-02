"""SPIFFE identity extraction reads only the Subject Alternative Name."""

from __future__ import annotations

import datetime

import pytest
from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.x509.oid import NameOID

from schemen_gate._lockbox import extract_spiffe_id


def _certificate(uris: list[str] | None) -> bytes:
    key = ed25519.Ed25519PrivateKey.generate()
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "workload")])
    now = datetime.datetime.now(datetime.timezone.utc)
    builder = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(hours=1))
    )
    if uris is not None:
        builder = builder.add_extension(
            x509.SubjectAlternativeName([x509.UniformResourceIdentifier(u) for u in uris]),
            critical=False,
        )
    return builder.sign(key, None).public_bytes(Encoding.PEM)


def test_certificate_without_san_has_no_spiffe_id() -> None:
    assert extract_spiffe_id(_certificate(None)) is None


def test_non_spiffe_uri_is_not_an_identity() -> None:
    assert extract_spiffe_id(_certificate(["https://example.org/service"])) is None


def test_exactly_one_spiffe_uri_is_returned() -> None:
    assert extract_spiffe_id(_certificate(["spiffe://example.org/workload/a"])) == (
        "spiffe://example.org/workload/a"
    )


def test_multiple_spiffe_uris_are_rejected() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        extract_spiffe_id(_certificate(["spiffe://example.org/a", "spiffe://example.org/b"]))


def test_malformed_certificate_bytes_fail_closed() -> None:
    with pytest.raises(ValueError):
        extract_spiffe_id(b"not a certificate")
