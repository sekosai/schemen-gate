"""PKCS#12-backed authority credentials for Schemen Gate.

PKCS#12 is a portable credential container, not a hardware-attestation
protocol. This reference provider supports common software signing keys and
loads the private key into process memory. Deployments that require
non-exportable hardware keys should provide a ``KeyProvider`` that signs
through the platform's native key handle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, ed448, ed25519, padding, rsa
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    pkcs12,
)

from schemen_gate._lockbox import Authority, KeyProvider, Lockbox, sign_lockbox
from schemen_gate._tokens import DerivedKey, GateKey, derive_tenant_key

_MAX_PKCS12_BYTES = 8 * 1024 * 1024
_MAX_PKCS12_CHAIN_LENGTH = 16


def _order_certificate_chain(
    leaf: x509.Certificate,
    additional: list[x509.Certificate],
) -> tuple[x509.Certificate, ...]:
    """Order an unordered PKCS#12 CA set from the leaf through its root."""

    if len(additional) > _MAX_PKCS12_CHAIN_LENGTH:
        raise ValueError(
            "PKCS#12 certificate chain exceeds the maximum of "
            f"{_MAX_PKCS12_CHAIN_LENGTH} certificates"
        )

    if leaf.issuer == leaf.subject:
        if additional:
            raise ValueError("PKCS#12 contains certificates outside the leaf chain")
        return ()

    remaining = list(additional)
    ordered: list[x509.Certificate] = []
    child = leaf
    while remaining:
        matches = [certificate for certificate in remaining if child.issuer == certificate.subject]
        if len(matches) != 1:
            raise ValueError("PKCS#12 certificate chain is missing or ambiguous")
        parent = matches[0]
        remaining.remove(parent)
        ordered.append(parent)
        child = parent
        if parent.issuer == parent.subject:
            break

    if not ordered:
        raise ValueError("PKCS#12 certificate chain is missing")
    if remaining:
        raise ValueError("PKCS#12 contains certificates outside the leaf chain")
    return tuple(ordered)


@dataclass(frozen=True)
class Pkcs12KeyProvider(KeyProvider):
    """Portable authority credential loaded from PKCS#12 bytes.

    The provider validates that the private key matches the leaf certificate
    and that the included certificates form one unambiguous leaf-to-anchor
    path. Ed25519, Ed448, ECDSA (at least 256 bits), and RSA (at least 2048
    bits) authority keys are supported. Certificate signatures, validity, and
    the verifier's independent trust-anchor pin are enforced by
    :func:`schemen_gate.verify_authority`.
    """

    _private_key: Any = field(repr=False)
    leaf_certificate: x509.Certificate
    certificate_chain: tuple[x509.Certificate, ...]

    @classmethod
    def from_bytes(cls, data: bytes, password: bytes | None) -> "Pkcs12KeyProvider":
        """Load one already-hydrated PKCS#12 credential."""

        if not isinstance(data, bytes) or not data or len(data) > _MAX_PKCS12_BYTES:
            raise ValueError(
                "PKCS#12 credential must be non-empty bytes no larger than "
                f"{_MAX_PKCS12_BYTES} bytes"
            )
        if password is not None and (not isinstance(password, bytes) or len(password) > 1024):
            raise ValueError("PKCS#12 password must be bytes no larger than 1024 bytes")
        private_key, certificate, additional = pkcs12.load_key_and_certificates(data, password)
        if certificate is None:
            raise ValueError("PKCS#12 must contain a leaf certificate")
        if not isinstance(
            private_key,
            (
                ed25519.Ed25519PrivateKey,
                ed448.Ed448PrivateKey,
                ec.EllipticCurvePrivateKey,
                rsa.RSAPrivateKey,
            ),
        ):
            raise ValueError("PKCS#12 private key must be Ed25519, Ed448, ECDSA, or RSA")
        if isinstance(private_key, ec.EllipticCurvePrivateKey) and private_key.key_size < 256:
            raise ValueError("PKCS#12 EC private key must be at least 256 bits")
        if isinstance(private_key, rsa.RSAPrivateKey) and private_key.key_size < 2048:
            raise ValueError("PKCS#12 RSA private key must be at least 2048 bits")
        public_key = certificate.public_key()
        private_spki = private_key.public_key().public_bytes(
            Encoding.DER,
            PublicFormat.SubjectPublicKeyInfo,
        )
        certificate_spki = public_key.public_bytes(
            Encoding.DER,
            PublicFormat.SubjectPublicKeyInfo,
        )
        if private_spki != certificate_spki:
            raise ValueError("PKCS#12 private key does not match the leaf certificate")
        chain = _order_certificate_chain(certificate, list(additional or ()))
        return cls(private_key, certificate, chain)

    @property
    def root_certificate(self) -> x509.Certificate:
        """Return the terminal trust anchor in the packaged chain."""

        return self.certificate_chain[-1] if self.certificate_chain else self.leaf_certificate

    @property
    def root_certificate_pem(self) -> bytes:
        return self.root_certificate.public_bytes(Encoding.PEM)

    @property
    def certificate_chain_pems(self) -> tuple[bytes, ...]:
        return tuple(
            certificate.public_bytes(Encoding.PEM) for certificate in self.certificate_chain
        )

    @property
    def signature_algorithm(self) -> str:
        if isinstance(self._private_key, ed25519.Ed25519PrivateKey):
            return "Ed25519"
        if isinstance(self._private_key, ed448.Ed448PrivateKey):
            return "Ed448"
        if isinstance(self._private_key, ec.EllipticCurvePrivateKey):
            return "ECDSA-SHA256"
        if isinstance(self._private_key, rsa.RSAPrivateKey):
            return "RSA-PSS-SHA256"
        raise TypeError("unsupported PKCS#12 authority key")

    def sign(self, data: bytes) -> bytes:
        if isinstance(
            self._private_key,
            (ed25519.Ed25519PrivateKey, ed448.Ed448PrivateKey),
        ):
            return self._private_key.sign(data)
        if isinstance(self._private_key, ec.EllipticCurvePrivateKey):
            return self._private_key.sign(data, ec.ECDSA(hashes.SHA256()))
        if isinstance(self._private_key, rsa.RSAPrivateKey):
            return self._private_key.sign(
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=hashes.SHA256().digest_size,
                ),
                hashes.SHA256(),
            )
        raise TypeError("unsupported PKCS#12 authority key")

    def public_key_pem(self) -> bytes:
        return self.leaf_certificate.public_bytes(Encoding.PEM)

    def derive_tenant_key(
        self,
        master_key: GateKey,
        regime_id: int,
        tenant_id: str,
    ) -> DerivedKey:
        return derive_tenant_key(master_key, regime_id, tenant_id)

    def sign_lockbox(self, lockbox: Lockbox) -> Authority:
        """Sign a lockbox with this credential and attach its public chain."""

        return sign_lockbox(
            lockbox,
            key_provider=self,
            ca_root_pem=self.root_certificate_pem,
            cert_chain_pems=list(self.certificate_chain_pems),
        )
