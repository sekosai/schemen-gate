"""Schemen Lockbox — zero-trust key distribution for gated models.

Consolidated from poc/lockbox.py into the schemen-gate package.

A lockbox is a publicly distributable artifact with three trust layers:

    1. Cleartext manifest — the hierarchy of access levels, regime-to-
       capability mapping, and chain binding.  Anyone can read this to
       understand what capabilities exist.

    2. Encrypted mask tokens — already AES-256-GCM encrypted under each
       tenant's derived key (from gate_crypto).  Safe to include in
       cleartext; useless without the tenant key.

    3. Sealed tenant keys — wrapped under the recipient's public key via
       X25519 ECDH + HKDF + AES-256-GCM (ECIES pattern).  Only the
       recipient's private key can unseal them.

The hierarchy enforces strict winnowing: each access level's regime set
is a strict superset of the next level's.  The format rejects non-
hierarchical topologies at creation time.

Each access level carries a canonical fingerprint derived from the chain
hash and sorted regime IDs — usable directly as an OAuth scope, X.509
extension, or SAML attribute by upstream PKI systems.

File format: .schemen.lockbox.yaml
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import math
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence, cast

import yaml

from schemen_gate._key_wrapping import KeyWrapper, WrappedKey, X25519AESGCMWrapper
from schemen_gate._release import (
    GateReleaseIdentity,
    ReleaseIdentityError,
    current_release_identity,
    release_identity_matches,
)
from schemen_gate._tokens import (
    DerivedKey,
    GateKey,
    MaskToken,
    derive_tenant_key,
    issue_mask_token,
)

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


_X25519_BINDING_OID = "1.3.6.1.4.1.57264.1.1"
_MAX_CERTIFICATE_CHAIN_LENGTH = 16
_MAX_CERTIFICATE_PEM_BYTES = 64 * 1024
_MAX_REVOCATION_RESPONSE_BYTES = 4 * 1024 * 1024
_REVOCATION_CLOCK_SKEW_SECONDS = 300
_LOGGER = logging.getLogger(__name__)
_MAX_LOCKBOX_YAML_DEPTH = 64


class _CertificateRevokedError(ValueError):
    """A validated revocation source reported the certificate as REVOKED."""


def _validate_revocation_timeout(value: object) -> None:
    """Require a finite, positive real-valued network timeout."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("revocation timeout must be a finite positive number")
    try:
        resolved = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("revocation timeout must be a finite positive number") from exc
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError("revocation timeout must be a finite positive number")


def _validate_revocation_byte_limit(value: object) -> None:
    """Require an exact positive integer response-size limit."""

    if type(value) is not int or value <= 0:
        raise ValueError("revocation byte limit must be a positive integer")


class _PlainSafeDumper(yaml.SafeDumper):
    """Safe YAML dumper that never emits anchors or aliases.

    Shared Python objects (for example one capability list reused by two
    hierarchy levels) would otherwise serialize as ``&id`` anchors, which the
    strict loader deliberately rejects.
    """

    def ignore_aliases(self, data: Any) -> bool:
        return True


class _StrictSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects aliases, duplicate keys, and deep trees."""

    def __init__(self, stream: Any) -> None:
        super().__init__(stream)
        self._gate_depth = 0

    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(yaml.AliasEvent):
            raise yaml.YAMLError("YAML aliases are not permitted in lockboxes")
        self._gate_depth += 1
        try:
            if self._gate_depth > _MAX_LOCKBOX_YAML_DEPTH:
                raise yaml.YAMLError(
                    f"Lockbox YAML exceeds maximum depth {_MAX_LOCKBOX_YAML_DEPTH}"
                )
            return super().compose_node(parent, index)
        finally:
            self._gate_depth -= 1

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[Any, Any]:
        if not isinstance(node, yaml.MappingNode):
            raise yaml.constructor.ConstructorError(
                None,
                None,
                "expected a mapping node",
                node.start_mark,
            )
        self.flatten_mapping(node)
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as exc:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable key",
                    key_node.start_mark,
                ) from exc
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def _load_bounded_pem_certificate(pem: bytes, label: str) -> Any:
    """Parse one bounded PEM certificate from an untrusted protocol object."""

    from cryptography.x509 import load_pem_x509_certificate

    if not isinstance(pem, bytes) or not pem or len(pem) > _MAX_CERTIFICATE_PEM_BYTES:
        raise ValueError(
            f"{label} must be non-empty PEM bytes no larger than {_MAX_CERTIFICATE_PEM_BYTES} bytes"
        )
    return load_pem_x509_certificate(pem)


class RevocationCheck(Enum):
    """Controls certificate revocation checking behavior.

    ENFORCE: Revoked certificates cause verification to fail.
             Unreachable endpoints also cause failure (fail-closed).
    WARN:    Revoked certificates cause verification to fail.
             Unreachable endpoints are logged but do not cause failure.
    SKIP:    No revocation checking is performed. Callers must select this
             explicitly for an offline fixture or a separately managed trust
             anchor; it is not a production default.
    """

    ENFORCE = "enforce"
    WARN = "warn"
    SKIP = "skip"


@dataclass(frozen=True)
class RevocationPolicy:
    """Complete retrieval and validation policy for certificate revocation.

    ``allowed_hosts`` is an exact hostname/IP allowlist applied to the
    certificate endpoint before dispatch. Network-capable ENFORCE and WARN
    policies require an operator-supplied ``fetcher`` so DNS resolution,
    redirects, proxies, peer-address validation, and egress remain inside the
    operator's controlled transport. Gate never performs certificate-directed
    network egress itself.
    """

    mode: RevocationCheck
    allowed_hosts: tuple[str, ...] | None = None
    timeout_seconds: float = 5.0
    max_response_bytes: int = _MAX_REVOCATION_RESPONSE_BYTES
    fetcher: Callable[..., bytes] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.mode, RevocationCheck):
            raise ValueError("revocation mode must be a RevocationCheck value")
        _validate_revocation_timeout(self.timeout_seconds)
        _validate_revocation_byte_limit(self.max_response_bytes)
        if self.allowed_hosts is not None:
            normalized = tuple(sorted({host.rstrip(".").lower() for host in self.allowed_hosts}))
            if not normalized or any(
                not host or "://" in host or "/" in host or "@" in host for host in normalized
            ):
                raise ValueError("revocation allowed_hosts must contain exact hostnames or IPs")
            object.__setattr__(self, "allowed_hosts", normalized)
        if self.fetcher is not None and self.mode is not RevocationCheck.SKIP:
            if self.allowed_hosts is None:
                raise ValueError(
                    "network-capable revocation policies require a nonempty exact-host allowlist"
                )
        if self.mode is RevocationCheck.SKIP and self.fetcher is not None:
            raise ValueError("SKIP revocation policy cannot configure a network fetcher")

    def check(self, cert: Any, issuer_cert: Any) -> list[str]:
        return check_certificate_revocation(
            cert,
            issuer_cert,
            mode=self.mode,
            timeout=self.timeout_seconds,
            max_response_bytes=self.max_response_bytes,
            fetcher=self.fetcher,
            allowed_hosts=self.allowed_hosts,
        )


def _revocation_policy(
    value: RevocationCheck | RevocationPolicy | None,
) -> RevocationPolicy:
    if value is None:
        raise ValueError("An explicit revocation policy is required")
    if isinstance(value, RevocationPolicy):
        return value
    if isinstance(value, RevocationCheck):
        return RevocationPolicy(mode=value)
    raise ValueError("revocation policy must be a RevocationCheck or RevocationPolicy")


# ---------------------------------------------------------------------------
# KeyProvider — abstract interface for HSM/KMS integration
# ---------------------------------------------------------------------------


class KeyProvider(ABC):
    """Abstract interface for cryptographic key operations.

    Allows signing and key derivation to be delegated to an HSM, cloud
    KMS, or other hardware-backed backend.  The default implementation
    (:class:`InMemoryKeyProvider`) holds keys in Python memory.

    Organizations that require the gate key to never leave an HSM
    boundary can implement this interface with a PKCS#11 or cloud KMS
    backend.
    """

    @property
    def signature_algorithm(self) -> str:
        """Authority signature identifier used by :func:`sign_lockbox`.

        Existing providers remain Ed25519 by default. Portable PKCS#12 and
        native HSM/KMS providers may override this with another algorithm that
        Gate verifies explicitly.
        """

        return "Ed25519"

    @abstractmethod
    def sign(self, data: bytes) -> bytes:
        """Sign *data* with the provider's declared signature algorithm."""

    @abstractmethod
    def public_key_pem(self) -> bytes:
        """Return the PEM-encoded public key or certificate."""

    @abstractmethod
    def derive_tenant_key(self, master_key: GateKey, regime_id: int, tenant_id: str) -> DerivedKey:
        """Derive a tenant key from the master key.

        Implementations may perform the HKDF locally (if the master key
        is in memory) or delegate to an HSM that supports HKDF.
        """


class InMemoryKeyProvider(KeyProvider):
    """Default KeyProvider that holds keys in Python process memory."""

    def __init__(
        self,
        signing_key: Any,
        signing_cert_pem: bytes,
    ) -> None:
        self._signing_key = signing_key
        self._signing_cert_pem = signing_cert_pem

    def sign(self, data: bytes) -> bytes:
        return cast(bytes, self._signing_key.sign(data))

    def public_key_pem(self) -> bytes:
        return self._signing_cert_pem

    def derive_tenant_key(self, master_key: GateKey, regime_id: int, tenant_id: str) -> DerivedKey:
        return derive_tenant_key(master_key, regime_id, tenant_id)


# ---------------------------------------------------------------------------
# X.509 adapter
# ---------------------------------------------------------------------------


def public_key_from_x509(cert_pem: bytes) -> bytes:
    """Extract the raw public key bytes from a PEM-encoded X.509 certificate.

    X25519 certificate keys are returned as 32 raw bytes. Certificates with a
    signing-only key must carry the signed Gate X25519 binding extension; that
    extension is handled by :func:`_recipient_wrapping_key`.
    """
    from cryptography.hazmat.primitives.asymmetric import x25519

    cert = _load_bounded_pem_certificate(cert_pem, "public-key certificate")
    pub = cert.public_key()

    if isinstance(pub, x25519.X25519PublicKey):
        return pub.public_bytes_raw()
    raise TypeError(
        f"Unsupported wrapping key type in certificate: {type(pub).__name__}. "
        "Lockbox requires X25519 or a signed Gate X25519 binding extension."
    )


def fingerprint_from_x509(cert_pem: bytes) -> str:
    """SHA-256 fingerprint of a PEM-encoded X.509 certificate."""
    from cryptography.hazmat.primitives.hashes import SHA256 as _SHA256

    cert = _load_bounded_pem_certificate(cert_pem, "fingerprint certificate")
    return str(cert.fingerprint(_SHA256()).hex())


def generate_authority_cert(
    common_name: str = "schemen-authority",
) -> tuple[bytes, "Ed25519PrivateKey"]:
    """Generate a self-signed Ed25519 CA certificate for lockbox signing.

    Returns (cert_pem, private_key).  The private key is used to sign
    lockbox hierarchy hashes.  The cert PEM is distributed as the CA root.
    """
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding
    from cryptography.x509.oid import NameOID

    signing_key = Ed25519PrivateKey.generate()

    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ]
    )

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(signing_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(
            datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=3650)
        )
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True,
        )
        .sign(signing_key, None)
    )

    return cert.public_bytes(Encoding.PEM), signing_key


def sign_lockbox(
    lockbox: "Lockbox",
    signing_key: "Ed25519PrivateKey | None" = None,
    signing_cert_pem: bytes | None = None,
    ca_root_pem: bytes | None = None,
    cert_chain_pems: list[bytes] | None = None,
    *,
    key_provider: KeyProvider | None = None,
    expected_release: GateReleaseIdentity | None = None,
) -> Authority:
    """Sign the complete lockbox with an Ed25519 key and attach authority.

    Must be called AFTER all grants are sealed — the signature covers the
    hierarchy, all grants, and the trust policy (via ``lockbox_hash``).
    The lockbox becomes a write-once artifact after signing.

    Can be called in two ways:

    1. **Direct keys** (backward compatible): pass ``signing_key``,
       ``signing_cert_pem``, and ``ca_root_pem`` directly.
    2. **KeyProvider**: pass a ``key_provider`` and ``ca_root_pem``.
       The provider handles signing and supplies the certificate PEM.

    Returns an Authority object.  Caller should set ``lockbox.authority``.
    """
    from cryptography.hazmat.primitives.hashes import SHA256 as _SHA256

    release = expected_release or current_release_identity()
    if not release_identity_matches(
        lockbox.gate_release,
        release,
        require_source_commit=True,
    ):
        raise ValueError("Lockbox Gate release differs from the signing runtime")
    lb_hash = compute_lockbox_hash(lockbox)

    if key_provider is not None:
        signature = key_provider.sign(lb_hash.encode("utf-8"))
        signing_cert_pem = key_provider.public_key_pem()
        if ca_root_pem is None:
            raise ValueError("ca_root_pem is required even when using a KeyProvider")
    elif signing_key is not None and signing_cert_pem is not None and ca_root_pem is not None:
        signature = signing_key.sign(lb_hash.encode("utf-8"))
    else:
        raise ValueError(
            "Either provide (signing_key, signing_cert_pem, ca_root_pem) "
            "or (key_provider, ca_root_pem)"
        )

    root_cert = _load_bounded_pem_certificate(ca_root_pem, "authority root certificate")
    ca_fp = root_cert.fingerprint(_SHA256()).hex()

    signature_algorithm = (
        key_provider.signature_algorithm if key_provider is not None else "Ed25519"
    )
    return Authority(
        ca_root_fingerprint=ca_fp,
        signing_cert_pem=signing_cert_pem,
        cert_chain_pems=cert_chain_pems or [],
        signature=signature,
        signature_algorithm=signature_algorithm,
        lockbox_hash=lb_hash,
    )


def _verify_signature(
    public_key: Any,
    signature: bytes,
    signed_bytes: bytes,
    hash_algorithm: Any,
    signature_parameters: Any = None,
) -> None:
    """Verify a certificate-family signature without algorithm guessing."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec, ed448, ed25519, padding, rsa

    approved_hashes = (
        hashes.SHA256,
        hashes.SHA384,
        hashes.SHA512,
        hashes.SHA3_256,
        hashes.SHA3_384,
        hashes.SHA3_512,
    )

    if hash_algorithm is not None and hash_algorithm.name not in {
        "sha256",
        "sha384",
        "sha512",
        "sha3-256",
        "sha3-384",
        "sha3-512",
    }:
        raise ValueError(f"Unsupported certificate signature hash: {hash_algorithm.name}")

    if isinstance(public_key, (ed25519.Ed25519PublicKey, ed448.Ed448PublicKey)):
        public_key.verify(signature, signed_bytes)
    elif isinstance(public_key, ec.EllipticCurvePublicKey):
        if public_key.key_size < 256:
            raise ValueError("EC certificate signing key must be at least 256 bits")
        if not isinstance(hash_algorithm, approved_hashes):
            raise ValueError("Unsupported EC certificate signature hash")
        parameters = signature_parameters or ec.ECDSA(hash_algorithm)
        if not isinstance(parameters, ec.ECDSA):
            raise ValueError("Unsupported EC certificate signature parameters")
        public_key.verify(signature, signed_bytes, parameters)
    elif isinstance(public_key, rsa.RSAPublicKey):
        if public_key.key_size < 2048:
            raise ValueError("RSA certificate signing key must be at least 2048 bits")
        if not isinstance(hash_algorithm, approved_hashes):
            raise ValueError("Unsupported RSA certificate signature hash")
        parameters = signature_parameters or padding.PKCS1v15()
        if not isinstance(parameters, (padding.PKCS1v15, padding.PSS)):
            raise ValueError("Unsupported RSA certificate signature parameters")
        if isinstance(parameters, padding.PSS):
            mgf = getattr(parameters, "mgf", getattr(parameters, "_mgf", None))
            mgf_hash = getattr(mgf, "_algorithm", None)
            if not isinstance(mgf, padding.MGF1) or not isinstance(mgf_hash, approved_hashes):
                raise ValueError("Unsupported RSA-PSS mask-generation hash")
        public_key.verify(
            signature,
            signed_bytes,
            parameters,
            hash_algorithm,
        )
    else:
        raise ValueError("Unsupported certificate signature key type")


def _verify_authority_signature(
    public_key: Any,
    algorithm: str,
    signature: bytes,
    signed_bytes: bytes,
) -> None:
    """Verify a Gate authority signature using its exact protocol identifier."""

    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec, ed448, ed25519, padding, rsa

    if algorithm == "Ed25519" and isinstance(public_key, ed25519.Ed25519PublicKey):
        public_key.verify(signature, signed_bytes)
        return
    if algorithm == "Ed448" and isinstance(public_key, ed448.Ed448PublicKey):
        public_key.verify(signature, signed_bytes)
        return
    if algorithm == "ECDSA-SHA256" and isinstance(public_key, ec.EllipticCurvePublicKey):
        if public_key.key_size < 256:
            raise ValueError("EC authority key must be at least 256 bits")
        public_key.verify(signature, signed_bytes, ec.ECDSA(hashes.SHA256()))
        return
    if algorithm == "RSA-PSS-SHA256" and isinstance(public_key, rsa.RSAPublicKey):
        if public_key.key_size < 2048:
            raise ValueError("RSA authority key must be at least 2048 bits")
        public_key.verify(
            signature,
            signed_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=hashes.SHA256().digest_size,
            ),
            hashes.SHA256(),
        )
        return
    raise ValueError(
        f"Authority signature algorithm {algorithm!r} does not match the signing certificate key"
    )


def _name_within_constraint(name: Any, constraint: Any) -> bool:
    """Return whether one RFC 5280 GeneralName is inside a name subtree."""

    import ipaddress
    from urllib.parse import urlsplit

    from cryptography import x509

    def _dns_match(candidate: str, permitted: str) -> bool:
        candidate = candidate.rstrip(".").lower()
        permitted = permitted.rstrip(".").lower()
        if permitted.startswith("."):
            return candidate.endswith(permitted) and candidate != permitted[1:]
        return candidate == permitted or candidate.endswith(f".{permitted}")

    if isinstance(name, x509.DNSName) and isinstance(constraint, x509.DNSName):
        return _dns_match(name.value, constraint.value)
    if isinstance(name, x509.RFC822Name) and isinstance(constraint, x509.RFC822Name):
        mailbox = name.value.lower()
        rule = constraint.value.lower()
        if "@" in rule:
            return hmac.compare_digest(mailbox, rule)
        domain = mailbox.rsplit("@", 1)[-1]
        if rule.startswith("."):
            return domain.endswith(rule) and domain != rule[1:]
        return domain == rule
    if isinstance(name, x509.UniformResourceIdentifier) and isinstance(
        constraint, x509.UniformResourceIdentifier
    ):
        host = urlsplit(name.value).hostname
        if host is None:
            raise ValueError("URI subject name has no host for NameConstraints")
        return _dns_match(host, constraint.value)
    if isinstance(name, x509.IPAddress) and isinstance(constraint, x509.IPAddress):
        candidate = name.value
        permitted = constraint.value
        if not isinstance(permitted, (ipaddress.IPv4Network, ipaddress.IPv6Network)):
            raise ValueError("IP NameConstraints value must be an address network")
        if isinstance(candidate, ipaddress.IPv4Address) and isinstance(
            permitted, ipaddress.IPv4Network
        ):
            return candidate in permitted
        if isinstance(candidate, ipaddress.IPv6Address) and isinstance(
            permitted, ipaddress.IPv6Network
        ):
            return candidate in permitted
        if isinstance(candidate, ipaddress.IPv4Network) and isinstance(
            permitted, ipaddress.IPv4Network
        ):
            return candidate.subnet_of(permitted)
        if isinstance(candidate, ipaddress.IPv6Network) and isinstance(
            permitted, ipaddress.IPv6Network
        ):
            return candidate.subnet_of(permitted)
        return False
    if isinstance(name, x509.DirectoryName) and isinstance(constraint, x509.DirectoryName):
        candidate_rdns = tuple(name.value.rdns)
        constraint_rdns = tuple(constraint.value.rdns)
        return candidate_rdns[: len(constraint_rdns)] == constraint_rdns
    if type(name) is type(constraint):
        raise ValueError(f"Unsupported NameConstraints GeneralName type: {type(name).__name__}")
    return False


def _certificate_general_names(cert: Any) -> list[Any]:
    """Collect subject names to which RFC 5280 name constraints apply."""

    from cryptography import x509
    from cryptography.x509.oid import ExtensionOID, NameOID

    names: list[Any] = [x509.DirectoryName(cert.subject)]
    try:
        names.extend(
            cert.extensions.get_extension_for_oid(ExtensionOID.SUBJECT_ALTERNATIVE_NAME).value
        )
    except x509.ExtensionNotFound:
        pass
    names.extend(
        x509.RFC822Name(attribute.value)
        for attribute in cert.subject.get_attributes_for_oid(NameOID.EMAIL_ADDRESS)
    )
    return names


def _enforce_name_constraints(ca_cert: Any, subordinate: Any, position: int) -> None:
    """Apply one CA's permitted and excluded RFC 5280 name subtrees."""

    from cryptography import x509
    from cryptography.x509.oid import ExtensionOID

    try:
        constraints = ca_cert.extensions.get_extension_for_oid(ExtensionOID.NAME_CONSTRAINTS).value
    except x509.ExtensionNotFound:
        return

    names = _certificate_general_names(subordinate)
    excluded = list(constraints.excluded_subtrees or ())
    permitted = list(constraints.permitted_subtrees or ())
    for name in names:
        if any(_name_within_constraint(name, rule) for rule in excluded):
            raise ValueError(
                f"Certificate at chain position {position} violates an excluded name subtree"
            )
        applicable = [rule for rule in permitted if type(rule) is type(name)]
        if applicable and not any(_name_within_constraint(name, rule) for rule in applicable):
            raise ValueError(
                f"Certificate at chain position {position} is outside permitted name subtrees"
            )


def _validate_revocation_url(
    url: str,
    *,
    allowed_hosts: Sequence[str] | None = None,
) -> None:
    """Validate a certificate endpoint before an operator fetcher receives it."""
    import ipaddress
    from urllib.parse import urlsplit

    if not isinstance(url, str):
        raise ValueError("revocation endpoint must be a string")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("revocation endpoint must use http or https")
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise ValueError("revocation endpoint must have a host and no user information")
    hostname = parsed.hostname.rstrip(".").lower()
    if allowed_hosts is not None and hostname not in allowed_hosts:
        raise ValueError("revocation endpoint host is not allowlisted")
    if hostname == "localhost" or hostname.endswith((".localhost", ".local")):
        raise ValueError("revocation endpoint resolves to a prohibited address")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("revocation endpoint has an invalid port") from exc

    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ValueError("revocation endpoint is a prohibited address")


def _fetch_revocation_bytes(
    fetcher: Callable[..., bytes] | None,
    *,
    url: str,
    method: str,
    data: bytes | None,
    headers: Mapping[str, str],
    timeout: float,
    max_bytes: int,
    allowed_hosts: Sequence[str] | None,
) -> bytes:
    _validate_revocation_url(
        url,
        allowed_hosts=allowed_hosts,
    )
    if fetcher is None:
        raise ValueError(
            "An operator-supplied revocation fetcher is required; "
            "Gate does not perform certificate-directed network egress"
        )
    payload = fetcher(
        url=url,
        method=method,
        data=data,
        headers=headers,
        timeout=timeout,
        max_bytes=max_bytes,
    )
    if not isinstance(payload, bytes):
        raise ValueError("revocation fetcher must return bytes")
    if len(payload) > max_bytes:
        raise ValueError("revocation response exceeds the byte limit")
    return payload


def _reject_unsupported_crl_extensions(crl: Any) -> None:
    from cryptography.x509.oid import CRLEntryExtensionOID, ExtensionOID

    supported_critical_extensions = {
        ExtensionOID.AUTHORITY_KEY_IDENTIFIER,
        ExtensionOID.CRL_NUMBER,
        ExtensionOID.ISSUING_DISTRIBUTION_POINT,
    }
    unsupported = sorted(
        extension.oid.dotted_string
        for extension in crl.extensions
        if extension.critical and extension.oid not in supported_critical_extensions
    )
    if unsupported:
        raise ValueError(f"CRL contains unsupported critical extensions: {', '.join(unsupported)}")
    supported_entry_extensions = {
        CRLEntryExtensionOID.CRL_REASON,
        CRLEntryExtensionOID.INVALIDITY_DATE,
    }
    for entry in crl:
        unsupported_entry = sorted(
            extension.oid.dotted_string
            for extension in entry.extensions
            if extension.critical and extension.oid not in supported_entry_extensions
        )
        if unsupported_entry:
            raise ValueError(
                "CRL entry contains unsupported critical extensions: "
                f"{', '.join(unsupported_entry)}"
            )


def _verify_crl_issuer(crl: Any, issuer_cert: Any) -> None:
    from cryptography import x509
    from cryptography.x509.oid import ExtensionOID

    if crl.issuer != issuer_cert.subject:
        raise ValueError("CRL issuer does not match the certificate issuer")
    try:
        _verify_signature(
            issuer_cert.public_key(),
            crl.signature,
            crl.tbs_certlist_bytes,
            crl.signature_hash_algorithm,
            getattr(crl, "signature_algorithm_parameters", None),
        )
    except Exception as exc:
        raise ValueError("CRL signature verification failed") from exc

    try:
        usage = issuer_cert.extensions.get_extension_for_oid(ExtensionOID.KEY_USAGE).value
        if not usage.crl_sign:
            raise ValueError("CRL issuer certificate is not authorized for CRL signing")
    except x509.ExtensionNotFound:
        pass

    try:
        authority_key = crl.extensions.get_extension_for_oid(
            ExtensionOID.AUTHORITY_KEY_IDENTIFIER
        ).value.key_identifier
    except x509.ExtensionNotFound:
        authority_key = None
    if authority_key is not None:
        try:
            issuer_key = issuer_cert.extensions.get_extension_for_oid(
                ExtensionOID.SUBJECT_KEY_IDENTIFIER
            ).value.digest
        except x509.ExtensionNotFound:
            issuer_key = x509.SubjectKeyIdentifier.from_public_key(issuer_cert.public_key()).digest
        if not hmac.compare_digest(authority_key, issuer_key):
            raise ValueError("CRL authority key does not match the issuer certificate")


def _crl_issuing_point(crl: Any) -> Any:
    from cryptography import x509
    from cryptography.x509.oid import ExtensionOID

    try:
        return crl.extensions.get_extension_for_oid(ExtensionOID.ISSUING_DISTRIBUTION_POINT).value
    except x509.ExtensionNotFound:
        return None


def _validate_crl_scope(cert: Any, distribution_point: Any, issuing_point: Any) -> None:
    from cryptography import x509
    from cryptography.x509.oid import ExtensionOID

    if distribution_point.relative_name is not None:
        raise ValueError("relative-name CRL distribution points are unsupported")
    if distribution_point.crl_issuer is not None:
        raise ValueError("indirect CRL issuers are unsupported")

    if issuing_point is not None:
        if issuing_point.relative_name is not None:
            raise ValueError("relative-name CRL issuing distribution points are unsupported")
        if issuing_point.indirect_crl:
            raise ValueError("indirect CRLs are unsupported")
        if issuing_point.only_contains_attribute_certs:
            raise ValueError("attribute-certificate CRL cannot cover this certificate")

        try:
            cert_is_ca = cert.extensions.get_extension_for_oid(
                ExtensionOID.BASIC_CONSTRAINTS
            ).value.ca
        except x509.ExtensionNotFound:
            cert_is_ca = False
        if issuing_point.only_contains_user_certs and cert_is_ca:
            raise ValueError("user-certificate CRL cannot cover a CA certificate")
        if issuing_point.only_contains_ca_certs and not cert_is_ca:
            raise ValueError("CA-certificate CRL cannot cover an end-entity certificate")

        if issuing_point.full_name is not None:
            cert_names = {
                (type(name).__name__, name.value) for name in distribution_point.full_name or ()
            }
            issuing_names = {(type(name).__name__, name.value) for name in issuing_point.full_name}
            if not cert_names.intersection(issuing_names):
                raise ValueError(
                    "CRL issuing distribution point does not match certificate distribution point"
                )


def _crl_covered_reasons(distribution_point: Any, issuing_point: Any) -> frozenset[Any]:
    from cryptography import x509

    all_reasons = frozenset(
        reason for reason in x509.ReasonFlags if reason != x509.ReasonFlags.remove_from_crl
    )
    point_reasons = (
        all_reasons if distribution_point.reasons is None else frozenset(distribution_point.reasons)
    )
    issuing_reasons = (
        all_reasons
        if issuing_point is None or issuing_point.only_some_reasons is None
        else frozenset(issuing_point.only_some_reasons)
    )
    covered_reasons = point_reasons.intersection(issuing_reasons)
    if not covered_reasons:
        raise ValueError("CRL distribution point covers no applicable reasons")
    return frozenset(covered_reasons)


def _validate_crl_freshness(crl: Any, *, now: Any) -> None:
    import datetime

    skew = datetime.timedelta(seconds=_REVOCATION_CLOCK_SKEW_SECONDS)
    if crl.last_update_utc > now + skew:
        raise ValueError("CRL thisUpdate is in the future")
    if crl.next_update_utc is None:
        raise ValueError("CRL has no nextUpdate")
    if crl.next_update_utc < now - skew:
        raise ValueError("CRL is stale")


def _validate_crl(
    crl: Any,
    issuer_cert: Any,
    cert: Any,
    distribution_point: Any,
    *,
    now: Any,
) -> frozenset[Any]:
    _reject_unsupported_crl_extensions(crl)
    _verify_crl_issuer(crl, issuer_cert)
    issuing_point = _crl_issuing_point(crl)
    _validate_crl_scope(cert, distribution_point, issuing_point)
    covered_reasons = _crl_covered_reasons(distribution_point, issuing_point)
    _validate_crl_freshness(crl, now=now)
    return covered_reasons


def _ocsp_responder_matches(response: Any, certificate: Any) -> bool:
    from cryptography import x509

    if response.responder_name is not None:
        return bool(response.responder_name == certificate.subject)
    if response.responder_key_hash is not None:
        digest = x509.SubjectKeyIdentifier.from_public_key(certificate.public_key()).digest
        return hmac.compare_digest(response.responder_key_hash, digest)
    return False


def _validate_ocsp_identity(response: Any, request: Any, cert: Any) -> None:
    from cryptography.x509 import ocsp

    if response.response_status != ocsp.OCSPResponseStatus.SUCCESSFUL:
        raise ValueError(f"OCSP responder returned {response.response_status}")
    if response.serial_number != cert.serial_number:
        raise ValueError("OCSP response serial does not match the certificate")
    if not hmac.compare_digest(response.issuer_name_hash, request.issuer_name_hash):
        raise ValueError("OCSP issuer name hash does not match the request")
    if not hmac.compare_digest(response.issuer_key_hash, request.issuer_key_hash):
        raise ValueError("OCSP issuer key hash does not match the request")


def _select_ocsp_responder(response: Any, issuer_cert: Any) -> Any:
    from cryptography.hazmat.primitives.hashes import SHA256

    candidates = [
        candidate
        for candidate in response.certificates
        if _ocsp_responder_matches(response, candidate)
    ]
    if _ocsp_responder_matches(response, issuer_cert):
        candidates.append(issuer_cert)
    unique = {candidate.fingerprint(SHA256()): candidate for candidate in candidates}
    if len(unique) != 1:
        raise ValueError("OCSP response does not identify one authorized responder")
    return next(iter(unique.values()))


def _validate_delegated_ocsp_responder(responder: Any, issuer_cert: Any) -> None:
    from cryptography import x509
    from cryptography.x509.oid import ExtendedKeyUsageOID, ExtensionOID

    if responder.issuer != issuer_cert.subject:
        raise ValueError("OCSP delegated responder was not issued by the certificate issuer")
    try:
        _verify_signature(
            issuer_cert.public_key(),
            responder.signature,
            responder.tbs_certificate_bytes,
            responder.signature_hash_algorithm,
            getattr(responder, "signature_algorithm_parameters", None),
        )
    except Exception as exc:
        raise ValueError("OCSP delegated responder certificate is invalid") from exc
    try:
        usages = responder.extensions.get_extension_for_oid(ExtensionOID.EXTENDED_KEY_USAGE).value
    except x509.ExtensionNotFound as exc:
        raise ValueError("OCSP delegated responder lacks OCSPSigning usage") from exc
    if ExtendedKeyUsageOID.OCSP_SIGNING not in usages:
        raise ValueError("OCSP delegated responder lacks OCSPSigning usage")
    try:
        key_usage = responder.extensions.get_extension_for_oid(ExtensionOID.KEY_USAGE).value
    except x509.ExtensionNotFound:
        key_usage = None
    if key_usage is not None and not key_usage.digital_signature:
        raise ValueError("OCSP delegated responder KeyUsage does not allow digital signatures")
    try:
        no_check = responder.extensions.get_extension_for_oid(ExtensionOID.OCSP_NO_CHECK)
    except x509.ExtensionNotFound as exc:
        raise ValueError(
            "OCSP delegated responder lacks OCSP No Check; responder revocation is not established"
        ) from exc
    if no_check.critical or not isinstance(no_check.value, x509.OCSPNoCheck):
        raise ValueError("OCSP delegated responder has an invalid OCSP No Check extension")


def _validate_ocsp_responder_signature(response: Any, responder: Any, *, now: Any) -> None:

    if now < responder.not_valid_before_utc or now > responder.not_valid_after_utc:
        raise ValueError("OCSP responder certificate is not currently valid")
    try:
        _verify_signature(
            responder.public_key(),
            response.signature,
            response.tbs_response_bytes,
            response.signature_hash_algorithm,
            getattr(response, "signature_algorithm_parameters", None),
        )
    except Exception as exc:
        raise ValueError("OCSP response signature verification failed") from exc


def _validate_ocsp_freshness(response: Any, *, now: Any) -> None:
    import datetime

    skew = datetime.timedelta(seconds=_REVOCATION_CLOCK_SKEW_SECONDS)
    if response.this_update_utc > now + skew:
        raise ValueError("OCSP thisUpdate is in the future")
    if response.next_update_utc is None:
        raise ValueError("OCSP response has no nextUpdate")
    if response.next_update_utc < now - skew:
        raise ValueError("OCSP response is stale")
    if response.produced_at_utc > now + skew:
        raise ValueError("OCSP producedAt is in the future")


def _validate_ocsp_response(
    response: Any,
    request: Any,
    cert: Any,
    issuer_cert: Any,
    *,
    now: Any,
) -> None:
    from cryptography.hazmat.primitives.hashes import SHA256

    _validate_ocsp_identity(response, request, cert)
    responder = _select_ocsp_responder(response, issuer_cert)
    if responder.fingerprint(SHA256()) != issuer_cert.fingerprint(SHA256()):
        _validate_delegated_ocsp_responder(responder, issuer_cert)
    _validate_ocsp_responder_signature(response, responder, now=now)
    _validate_ocsp_freshness(response, now=now)


def _normalize_revocation_hosts(allowed_hosts: Sequence[str] | None) -> tuple[str, ...] | None:
    if allowed_hosts is None:
        return None
    normalized = tuple(sorted({host.rstrip(".").lower() for host in allowed_hosts}))
    if not normalized:
        raise ValueError("revocation allowed_hosts cannot be empty")
    return normalized


def _certificate_crl_distribution_points(cert: Any) -> tuple[Any, ...]:
    from cryptography import x509
    from cryptography.x509.oid import ExtensionOID

    try:
        extension = cert.extensions.get_extension_for_oid(ExtensionOID.CRL_DISTRIBUTION_POINTS)
    except x509.ExtensionNotFound:
        return ()
    return tuple(extension.value)


def _check_crl_endpoints(
    cert: Any,
    issuer_cert: Any,
    *,
    now: Any,
    timeout: float,
    max_response_bytes: int,
    fetcher: Callable[..., bytes] | None,
    allowed_hosts: Sequence[str] | None,
) -> tuple[bool, bool, list[str]]:
    from cryptography import x509
    from cryptography.x509 import load_der_x509_crl

    failures: list[str] = []
    endpoint_seen = False
    covered_reasons: set[Any] = set()
    all_reasons = {
        reason for reason in x509.ReasonFlags if reason != x509.ReasonFlags.remove_from_crl
    }
    for distribution_point in _certificate_crl_distribution_points(cert):
        for name in distribution_point.full_name or ():
            url = name.value
            if not isinstance(url, str):
                continue
            endpoint_seen = True
            try:
                payload = _fetch_revocation_bytes(
                    fetcher,
                    url=url,
                    method="GET",
                    data=None,
                    headers={"Accept": "application/pkix-crl"},
                    timeout=timeout,
                    max_bytes=max_response_bytes,
                    allowed_hosts=allowed_hosts,
                )
                crl = load_der_x509_crl(payload)
                coverage = _validate_crl(
                    crl,
                    issuer_cert,
                    cert,
                    distribution_point,
                    now=now,
                )
                revoked = crl.get_revoked_certificate_by_serial_number(cert.serial_number)
                if revoked is not None:
                    raise _CertificateRevokedError(
                        f"Certificate serial {cert.serial_number} is REVOKED "
                        f"(CRL from {url}, revoked on {revoked.revocation_date_utc})"
                    )
                covered_reasons.update(coverage)
                if covered_reasons.issuperset(all_reasons):
                    return endpoint_seen, True, []
            except _CertificateRevokedError:
                raise
            except Exception as exc:
                failures.append(f"CRL validation failed for {url}: {exc}")

    if covered_reasons and not covered_reasons.issuperset(all_reasons):
        missing = sorted(reason.value for reason in all_reasons - covered_reasons)
        failures.append(
            "Applicable CRLs do not cover every revocation reason; missing " + ", ".join(missing)
        )
    return endpoint_seen, False, failures


def _certificate_ocsp_urls(cert: Any) -> tuple[str, ...]:
    from cryptography import x509
    from cryptography.x509.oid import AuthorityInformationAccessOID, ExtensionOID

    try:
        extension = cert.extensions.get_extension_for_oid(ExtensionOID.AUTHORITY_INFORMATION_ACCESS)
    except x509.ExtensionNotFound:
        return ()
    return tuple(
        description.access_location.value
        for description in extension.value
        if description.access_method == AuthorityInformationAccessOID.OCSP
        and isinstance(description.access_location.value, str)
    )


def _check_ocsp_endpoints(
    cert: Any,
    issuer_cert: Any,
    urls: Sequence[str],
    *,
    now: Any,
    timeout: float,
    max_response_bytes: int,
    fetcher: Callable[..., bytes] | None,
    allowed_hosts: Sequence[str] | None,
) -> tuple[bool, list[str]]:
    from cryptography.hazmat.primitives.hashes import SHA256
    from cryptography.hazmat.primitives.serialization import Encoding
    from cryptography.x509 import ocsp

    failures: list[str] = []
    for url in urls:
        try:
            request = ocsp.OCSPRequestBuilder().add_certificate(cert, issuer_cert, SHA256()).build()
            payload = _fetch_revocation_bytes(
                fetcher,
                url=url,
                method="POST",
                data=request.public_bytes(Encoding.DER),
                headers={
                    "Accept": "application/ocsp-response",
                    "Content-Type": "application/ocsp-request",
                },
                timeout=timeout,
                max_bytes=max_response_bytes,
                allowed_hosts=allowed_hosts,
            )
            response = ocsp.load_der_ocsp_response(payload)
            _validate_ocsp_response(response, request, cert, issuer_cert, now=now)
            if response.certificate_status == ocsp.OCSPCertStatus.REVOKED:
                raise _CertificateRevokedError(
                    f"Certificate serial {cert.serial_number} is REVOKED (OCSP from {url})"
                )
            if response.certificate_status != ocsp.OCSPCertStatus.GOOD:
                raise ValueError("OCSP responder returned UNKNOWN status")
            return True, []
        except _CertificateRevokedError:
            raise
        except Exception as exc:
            failures.append(f"OCSP validation failed for {url}: {exc}")
    return False, failures


def check_certificate_revocation(
    cert: Any,
    issuer_cert: Any,
    *,
    mode: RevocationCheck = RevocationCheck.ENFORCE,
    timeout: float = 5.0,
    max_response_bytes: int = _MAX_REVOCATION_RESPONSE_BYTES,
    fetcher: Callable[..., bytes] | None = None,
    allowed_hosts: Sequence[str] | None = None,
) -> list[str]:
    """Validate a certificate's revocation status via authenticated CRL/OCSP.

    A revoked certificate always fails in both WARN and ENFORCE modes. WARN
    permits an indeterminate status but returns structured warning strings;
    ENFORCE requires at least one fresh, issuer-authenticated answer.
    """
    import datetime

    if not isinstance(mode, RevocationCheck):
        raise ValueError("revocation mode must be a RevocationCheck value")
    _validate_revocation_timeout(timeout)
    _validate_revocation_byte_limit(max_response_bytes)
    normalized_hosts = _normalize_revocation_hosts(allowed_hosts)
    if mode == RevocationCheck.SKIP:
        return []

    now = datetime.datetime.now(datetime.timezone.utc)
    crl_seen, crl_valid, failures = _check_crl_endpoints(
        cert,
        issuer_cert,
        now=now,
        timeout=timeout,
        max_response_bytes=max_response_bytes,
        fetcher=fetcher,
        allowed_hosts=normalized_hosts,
    )
    if crl_valid:
        return []

    ocsp_urls = _certificate_ocsp_urls(cert)
    ocsp_valid, ocsp_failures = _check_ocsp_endpoints(
        cert,
        issuer_cert,
        ocsp_urls,
        now=now,
        timeout=timeout,
        max_response_bytes=max_response_bytes,
        fetcher=fetcher,
        allowed_hosts=normalized_hosts,
    )
    if ocsp_valid:
        return []
    failures.extend(ocsp_failures)

    if not crl_seen and not ocsp_urls:
        failures.append("No CRL or OCSP endpoint available on certificate")
    message = "; ".join(failures) or "No authenticated revocation answer was available"
    if mode == RevocationCheck.ENFORCE:
        raise ValueError(message)
    return failures


def verify_authority(
    lockbox: "Lockbox",
    trusted_ca_fingerprints: list[str],
    *,
    revocation: RevocationCheck | RevocationPolicy | None = None,
    expected_release: GateReleaseIdentity | None = None,
) -> bool:
    """Verify the lockbox's authority signature.

    Recomputes the lockbox_hash from the current state (hierarchy + grants +
    trust policy) and verifies the declared Gate authority signature against
    the signing certificate. Also validates the CA trust-anchor fingerprint.

    Parameters
    ----------
    revocation : RevocationCheck | RevocationPolicy
        Required explicit policy. WARN logs an indeterminate status but always
        rejects an authenticated revoked result. ENFORCE requires a fresh,
        issuer-authenticated CRL or OCSP result. SKIP is intended only for an
        explicit offline or self-signed-root policy.

    Returns True if valid.  Raises ValueError on failure.
    """
    from cryptography.hazmat.primitives.hashes import SHA256 as _SHA256

    if not trusted_ca_fingerprints:
        raise ValueError("At least one external trusted CA fingerprint is required")
    release = expected_release or current_release_identity()
    if not release_identity_matches(
        lockbox.gate_release,
        release,
        require_source_commit=True,
    ):
        raise ValueError("Lockbox Gate release differs from the verifying runtime")
    if any(
        token.gate_release != lockbox.gate_release
        for grant in lockbox.grants
        for token in grant.mask_tokens
    ):
        raise ValueError("Lockbox contains a mask token from a different Gate release")
    if lockbox.authority is None:
        raise ValueError("Lockbox has no authority section")
    policy = _revocation_policy(revocation)

    auth = lockbox.authority

    expected_hash = compute_lockbox_hash(lockbox)
    if not hmac.compare_digest(auth.lockbox_hash, expected_hash):
        raise ValueError(
            "Lockbox hash mismatch — lockbox may have been tampered with "
            "(hierarchy, grants, or trust policy modified after signing)"
        )

    signing_cert = _load_bounded_pem_certificate(
        auth.signing_cert_pem,
        "authority signing certificate",
    )
    pub = signing_cert.public_key()

    try:
        _verify_authority_signature(
            pub,
            auth.signature_algorithm,
            auth.signature,
            expected_hash.encode("utf-8"),
        )
    except Exception as exc:
        raise ValueError(
            "Authority signature verification failed — lockbox may have been tampered with"
        ) from exc

    if auth.cert_chain_pems:
        root_fp = _verify_cert_chain(
            signing_cert,
            auth.cert_chain_pems,
            _SHA256,
            trusted_root_fingerprints=trusted_ca_fingerprints,
        )
    else:
        root_fp = _verify_cert_chain(
            signing_cert,
            [],
            _SHA256,
            trusted_root_fingerprints=trusted_ca_fingerprints,
        )

    if not hmac.compare_digest(root_fp, auth.ca_root_fingerprint):
        raise ValueError(
            "CA root fingerprint mismatch — signing certificate does "
            "not match the declared authority"
        )
    if not any(hmac.compare_digest(root_fp, trusted) for trusted in trusted_ca_fingerprints):
        raise ValueError("Lockbox authority is not anchored in the verifier trust store")

    if policy.mode != RevocationCheck.SKIP:
        if auth.cert_chain_pems:
            chain_certs = [
                _load_bounded_pem_certificate(pem, "authority chain certificate")
                for pem in auth.cert_chain_pems
            ]
            all_certs = [signing_cert] + chain_certs
            for i in range(len(all_certs) - 1):
                issuer = all_certs[i + 1]
                warnings = policy.check(all_certs[i], issuer)
                for warning in warnings:
                    _LOGGER.warning("Certificate revocation warning: %s", warning)

    return True


def _reject_unsupported_certificate_extensions(cert: Any, position: int) -> None:
    from cryptography import x509
    from cryptography.x509.oid import ExtensionOID

    supported_critical_extensions = {
        ExtensionOID.BASIC_CONSTRAINTS,
        ExtensionOID.EXTENDED_KEY_USAGE,
        ExtensionOID.KEY_USAGE,
        ExtensionOID.NAME_CONSTRAINTS,
        ExtensionOID.SUBJECT_ALTERNATIVE_NAME,
        x509.ObjectIdentifier(_X25519_BINDING_OID),
    }
    unsupported_path_extensions = {
        ExtensionOID.INHIBIT_ANY_POLICY,
        ExtensionOID.POLICY_CONSTRAINTS,
        ExtensionOID.POLICY_MAPPINGS,
    }
    unsupported_path = sorted(
        extension.oid.dotted_string
        for extension in cert.extensions
        if extension.oid in unsupported_path_extensions
    )
    if unsupported_path:
        raise ValueError(
            "Certificate at chain position "
            f"{position} contains unsupported path or purpose constraints: "
            f"{', '.join(unsupported_path)}"
        )
    unsupported = sorted(
        extension.oid.dotted_string
        for extension in cert.extensions
        if extension.critical and extension.oid not in supported_critical_extensions
    )
    if unsupported:
        raise ValueError(
            "Certificate at chain position "
            f"{position} contains unsupported critical extensions: "
            f"{', '.join(unsupported)}"
        )


def _require_ca_certificate(cert: Any, position: int, full_chain: Sequence[Any]) -> None:
    from cryptography import x509
    from cryptography.x509.oid import ExtensionOID

    try:
        constraints = cert.extensions.get_extension_for_oid(ExtensionOID.BASIC_CONSTRAINTS).value
    except x509.ExtensionNotFound as exc:
        raise ValueError(
            f"Certificate at chain position {position} lacks BasicConstraints"
        ) from exc
    if not constraints.ca:
        raise ValueError(f"Certificate at chain position {position} is not a CA")
    subordinate_ca_count = sum(
        1 for subordinate in full_chain[1:position] if subordinate.subject != subordinate.issuer
    )
    if constraints.path_length is not None and subordinate_ca_count > constraints.path_length:
        raise ValueError(
            f"Certificate path length constraint violated at position {position}: "
            f"allows {constraints.path_length}, found {subordinate_ca_count}"
        )
    try:
        usage = cert.extensions.get_extension_for_oid(ExtensionOID.KEY_USAGE).value
    except x509.ExtensionNotFound:
        return
    if not usage.key_cert_sign:
        raise ValueError(f"Certificate at chain position {position} cannot sign certificates")


def _validate_certificate_path_purpose(
    full_chain: Sequence[Any], *, require_leaf_digital_signature: bool
) -> None:
    from cryptography import x509
    from cryptography.x509.oid import ExtendedKeyUsageOID, ExtensionOID

    accepted_eku = (
        {
            ExtendedKeyUsageOID.ANY_EXTENDED_KEY_USAGE,
            ExtendedKeyUsageOID.CLIENT_AUTH,
            ExtendedKeyUsageOID.CODE_SIGNING,
        }
        if require_leaf_digital_signature
        else {
            ExtendedKeyUsageOID.ANY_EXTENDED_KEY_USAGE,
            ExtendedKeyUsageOID.CLIENT_AUTH,
        }
    )
    if require_leaf_digital_signature:
        try:
            leaf_usage = (
                full_chain[0].extensions.get_extension_for_oid(ExtensionOID.KEY_USAGE).value
            )
            if not leaf_usage.digital_signature:
                raise ValueError("Signing certificate Key Usage does not permit digital signatures")
        except x509.ExtensionNotFound:
            pass
    effective_eku = set(accepted_eku)
    for position, certificate in enumerate(full_chain):
        try:
            certificate_eku = certificate.extensions.get_extension_for_oid(
                ExtensionOID.EXTENDED_KEY_USAGE
            ).value
        except x509.ExtensionNotFound:
            continue
        if ExtendedKeyUsageOID.ANY_EXTENDED_KEY_USAGE not in certificate_eku:
            effective_eku.intersection_update(certificate_eku)
        if not effective_eku:
            purpose = (
                "authority signing" if require_leaf_digital_signature else "recipient identity"
            )
            raise ValueError(
                "Certificate at chain position "
                f"{position} leaves no common Extended Key Usage for Gate {purpose}"
            )


def _validate_certificate_path_link(full_chain: Sequence[Any], position: int, *, now: Any) -> None:
    child = full_chain[position]
    parent = full_chain[position + 1]

    if child.issuer != parent.subject:
        raise ValueError(f"Certificate chain issuer mismatch at position {position}")
    _require_ca_certificate(parent, position + 1, full_chain)

    for constraint_position in range(position + 1, len(full_chain)):
        _enforce_name_constraints(full_chain[constraint_position], child, position)

    if now < child.not_valid_before_utc or now > child.not_valid_after_utc:
        raise ValueError(
            f"Certificate at chain position {position} is not currently valid "
            f"(valid {child.not_valid_before_utc} to {child.not_valid_after_utc})"
        )

    try:
        _verify_signature(
            parent.public_key(),
            child.signature,
            child.tbs_certificate_bytes,
            child.signature_hash_algorithm,
            getattr(child, "signature_algorithm_parameters", None),
        )
    except Exception as exc:
        raise ValueError(
            "Certificate chain validation failed: certificate at "
            f"position {position} was not signed by certificate at position {position + 1}"
        ) from exc


def _validate_certificate_trust_anchor(
    root: Any,
    full_chain: Sequence[Any],
    hash_cls: Any,
    trusted_root_fingerprints: Sequence[str],
    *,
    now: Any,
) -> str:
    root_fingerprint = root.fingerprint(hash_cls()).hex()
    root_is_pinned = any(
        hmac.compare_digest(root_fingerprint, trusted) for trusted in trusted_root_fingerprints
    )
    if root.issuer != root.subject and not root_is_pinned:
        raise ValueError(
            "Certificate chain does not end in a self-signed or verifier-pinned trust anchor"
        )
    _require_ca_certificate(root, len(full_chain) - 1, full_chain)

    if now < root.not_valid_before_utc or now > root.not_valid_after_utc:
        raise ValueError(
            f"Root CA certificate is not currently valid "
            f"(valid {root.not_valid_before_utc} to {root.not_valid_after_utc})"
        )

    if root.issuer == root.subject:
        try:
            _verify_signature(
                root.public_key(),
                root.signature,
                root.tbs_certificate_bytes,
                root.signature_hash_algorithm,
                getattr(root, "signature_algorithm_parameters", None),
            )
        except Exception as exc:
            raise ValueError(
                "Root CA certificate is not self-signed — chain does not terminate at a trust anchor"
            ) from exc

    return cast(str, root_fingerprint)


def _verify_cert_chain(
    signing_cert: Any,
    chain_pems: list[bytes],
    hash_cls: Any,
    *,
    require_leaf_digital_signature: bool = True,
    trusted_root_fingerprints: Sequence[str] = (),
) -> str:
    """Walk a leaf-to-root chain and return the validated root fingerprint."""
    import datetime

    if len(chain_pems) > _MAX_CERTIFICATE_CHAIN_LENGTH:
        raise ValueError(
            f"Certificate chain exceeds the maximum of {_MAX_CERTIFICATE_CHAIN_LENGTH} certificates"
        )
    chain_certs = [
        _load_bounded_pem_certificate(pem, "certificate chain member") for pem in chain_pems
    ]
    full_chain = [signing_cert] + chain_certs
    now = datetime.datetime.now(datetime.timezone.utc)

    for position, certificate in enumerate(full_chain):
        _reject_unsupported_certificate_extensions(certificate, position)
    _validate_certificate_path_purpose(
        full_chain,
        require_leaf_digital_signature=require_leaf_digital_signature,
    )
    for position in range(len(full_chain) - 1):
        _validate_certificate_path_link(full_chain, position, now=now)
    return _validate_certificate_trust_anchor(
        full_chain[-1],
        full_chain,
        hash_cls,
        trusted_root_fingerprints,
        now=now,
    )


def _recipient_wrapping_key(cert_pem: bytes, supplied_key: bytes) -> bytes:
    """Return only a wrapping key cryptographically bound to ``cert_pem``."""
    from cryptography import x509

    certificate = _load_bounded_pem_certificate(cert_pem, "recipient certificate")
    try:
        certificate_key = public_key_from_x509(cert_pem)
    except TypeError:
        try:
            binding = certificate.extensions.get_extension_for_oid(
                x509.ObjectIdentifier(_X25519_BINDING_OID)
            ).value
            if not isinstance(binding, x509.UnrecognizedExtension):
                raise ValueError("Certificate X25519 binding has the wrong extension type")
            certificate_key = binding.value
        except x509.ExtensionNotFound as exc:
            raise ValueError(
                "Certificate has no ECDH key or cryptographically bound X25519 key"
            ) from exc
        if not isinstance(certificate_key, bytes) or len(certificate_key) != 32:
            raise ValueError("Certificate X25519 binding must contain exactly 32 bytes") from None

    if supplied_key and not hmac.compare_digest(supplied_key, certificate_key):
        raise ValueError("recipient_public_key is not bound to the recipient certificate")
    return certificate_key


def _validate_recipient_certificate(
    cert_pem: bytes,
    chain_pems: Sequence[bytes],
    trusted_root_fingerprints: Sequence[str],
) -> Any:
    """Validate a recipient leaf and its path to an independent root pin."""
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives.hashes import SHA256
    from cryptography.x509.oid import ExtensionOID

    certificate = _load_bounded_pem_certificate(cert_pem, "recipient certificate")
    now = datetime.datetime.now(datetime.timezone.utc)
    if now < certificate.not_valid_before_utc or now > certificate.not_valid_after_utc:
        raise ValueError("Recipient certificate is not currently valid")
    try:
        constraints_extension = certificate.extensions.get_extension_for_oid(
            ExtensionOID.BASIC_CONSTRAINTS
        ).value
        if not isinstance(constraints_extension, x509.BasicConstraints):
            raise ValueError("Recipient BasicConstraints has the wrong extension type")
        constraints = constraints_extension
        if constraints.ca:
            raise ValueError("Recipient certificate must be an end-entity certificate")
    except x509.ExtensionNotFound:
        pass

    try:
        from cryptography.hazmat.primitives.asymmetric import x25519

        usage_extension = certificate.extensions.get_extension_for_oid(ExtensionOID.KEY_USAGE).value
        if not isinstance(usage_extension, x509.KeyUsage):
            raise ValueError("Recipient KeyUsage has the wrong extension type")
        usage = usage_extension
        if isinstance(certificate.public_key(), x25519.X25519PublicKey) and not (
            usage.key_agreement
        ):
            raise ValueError("Recipient certificate is not authorized for key agreement")
    except x509.ExtensionNotFound:
        pass

    if not trusted_root_fingerprints:
        return certificate
    if not chain_pems:
        raise ValueError("Recipient certificate chain is required by trusted_recipient_cas")
    root_fingerprint = _verify_cert_chain(
        certificate,
        list(chain_pems),
        SHA256,
        require_leaf_digital_signature=False,
        trusted_root_fingerprints=trusted_root_fingerprints,
    )
    if not any(
        hmac.compare_digest(root_fingerprint, trusted) for trusted in trusted_root_fingerprints
    ):
        raise ValueError("Recipient certificate chain is not anchored in a trusted CA")
    return certificate


def verify_grant_provenance(
    lockbox: "Lockbox",
    grant: "Grant",
    consumer_cert_pem: bytes,
    trusted_authority_cas: list[str],
    *,
    consumer_cert_chain_pems: Sequence[bytes] | None = None,
    revocation: RevocationCheck | RevocationPolicy | None = None,
    expected_release: GateReleaseIdentity | None = None,
) -> "ProvenanceResult":
    """Consumer-side provenance check before unsealing a grant.

    Verifies:
    (a) The lockbox has an authority section.
    (b) The authority's CA root fingerprint is in ``trusted_authority_cas``.
    (c) The grant's ``recipient_fingerprint`` matches the consumer's cert.
    (d) Certificate revocation status (if ``revocation`` is not SKIP).

    Returns a :class:`ProvenanceResult` with ``trusted=True`` on success,
    or ``trusted=False`` with a list of reasons on failure.
    """
    reasons: list[str] = []
    try:
        revocation_policy = _revocation_policy(revocation)
    except ValueError as exc:
        return ProvenanceResult(trusted=False, reasons=[str(exc)])

    if lockbox.authority is None:
        reasons.append("Lockbox has no authority section")
        return ProvenanceResult(trusted=False, reasons=reasons)

    try:
        verify_authority(
            lockbox,
            trusted_authority_cas,
            revocation=revocation_policy,
            expected_release=expected_release,
        )
    except ValueError as e:
        reasons.append(f"Authority verification failed: {e}")
        return ProvenanceResult(trusted=False, reasons=reasons)

    try:
        candidate_digest = _compute_grant_digest(grant)
        is_signed_member = any(
            hmac.compare_digest(candidate_digest, _compute_grant_digest(member))
            for member in lockbox.grants
        )
    except (AttributeError, TypeError, ValueError, OverflowError):
        return ProvenanceResult(trusted=False, reasons=["Grant is malformed"])
    if not is_signed_member:
        return ProvenanceResult(
            trusted=False,
            reasons=["Grant is not an exact member of the signed lockbox"],
        )

    ca_fp = lockbox.authority.ca_root_fingerprint
    if not any(hmac.compare_digest(ca_fp, trusted) for trusted in trusted_authority_cas):
        reasons.append(f"Authority CA {ca_fp[:16]}... is not in the consumer's trust store")

    try:
        consumer_fp = fingerprint_from_x509(consumer_cert_pem)
        consumer_cert = _validate_recipient_certificate(
            consumer_cert_pem,
            list(consumer_cert_chain_pems or ()),
            lockbox.trusted_recipient_cas,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        return ProvenanceResult(
            trusted=False,
            reasons=[f"Recipient certificate validation failed: {exc}"],
        )

    if not hmac.compare_digest(grant.recipient_fingerprint, consumer_fp):
        reasons.append(
            f"Grant recipient fingerprint {grant.recipient_fingerprint[:16]}... "
            f"does not match consumer cert {consumer_fp[:16]}..."
        )

    try:
        if revocation_policy.mode != RevocationCheck.SKIP:
            if not lockbox.trusted_recipient_cas:
                message = (
                    "Recipient revocation is indeterminate because no trusted "
                    "recipient CA fingerprints are configured"
                )
                if revocation_policy.mode == RevocationCheck.ENFORCE:
                    raise ValueError(message)
                _LOGGER.warning("%s", message)
            elif not consumer_cert_chain_pems:
                raise ValueError("Recipient certificate chain is required for revocation checking")
            else:
                chain = [
                    _load_bounded_pem_certificate(pem, "consumer chain certificate")
                    for pem in consumer_cert_chain_pems
                ]
                revocation_chain = [consumer_cert] + chain
                for index in range(len(revocation_chain) - 1):
                    for warning in revocation_policy.check(
                        revocation_chain[index],
                        revocation_chain[index + 1],
                    ):
                        _LOGGER.warning("Certificate revocation warning: %s", warning)
    except ValueError as exc:
        reasons.append(f"Recipient certificate validation failed: {exc}")

    if reasons:
        return ProvenanceResult(trusted=False, reasons=reasons)
    return ProvenanceResult(trusted=True, reasons=[])


def generate_self_signed_x25519_cert(
    common_name: str = "lockbox-recipient",
) -> tuple[bytes, bytes, bytes]:
    """Generate a self-signed certificate with an X25519 public key.

    Returns (cert_pem, private_key_bytes).

    NOTE: X.509 certificates don't natively support X25519 keys in most
    CAs, but the ``cryptography`` library can construct them.  For the PoC
    we use an Ed25519 signing key and embed the X25519 public key via a
    paired keypair approach: we generate an Ed25519 key for signing and
    an X25519 key for the actual key exchange.  The certificate carries
    the Ed25519 key (for signing validity) and the X25519 raw public key
    is returned separately.

    A production CA should issue or sign the binding extension under its normal
    certificate policy and lifecycle controls.
    """
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat
    from cryptography.x509.oid import NameOID

    signing_key = Ed25519PrivateKey.generate()
    x25519_priv = X25519PrivateKey.generate()

    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ]
    )

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(signing_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(
            datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365)
        )
        .add_extension(
            x509.UnrecognizedExtension(
                x509.ObjectIdentifier(_X25519_BINDING_OID),
                x25519_priv.public_key().public_bytes_raw(),
            ),
            critical=False,
        )
        .sign(signing_key, None)
    )

    cert_pem = cert.public_bytes(Encoding.PEM)
    x25519_priv_bytes = x25519_priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    x25519_pub_bytes = x25519_priv.public_key().public_bytes_raw()

    return cert_pem, x25519_priv_bytes, x25519_pub_bytes


# ---------------------------------------------------------------------------
# SPIFFE adapter
# ---------------------------------------------------------------------------


def extract_spiffe_id(cert_pem: bytes) -> str | None:
    """Extract the SPIFFE ID from a certificate's Subject Alternative Name.

    A SPIFFE ID is a ``spiffe://`` URI embedded as a SAN of type
    ``uniformResourceIdentifier``.  Returns ``None`` if no SPIFFE ID is
    found.  Raises ``ValueError`` if more than one SPIFFE URI is present
    (the SPIFFE spec requires exactly one).
    """
    from cryptography import x509
    from cryptography.x509.oid import ExtensionOID

    cert = _load_bounded_pem_certificate(cert_pem, "SPIFFE certificate")
    try:
        san = cert.extensions.get_extension_for_oid(ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
    except x509.ExtensionNotFound:
        return None

    from cryptography.x509 import UniformResourceIdentifier

    if not isinstance(san.value, x509.SubjectAlternativeName):
        raise ValueError("Subject Alternative Name has the wrong extension type")
    uris = san.value.get_values_for_type(UniformResourceIdentifier)
    spiffe_uris = [u for u in uris if u.startswith("spiffe://")]
    if not spiffe_uris:
        return None
    if len(spiffe_uris) > 1:
        raise ValueError(
            f"Certificate has {len(spiffe_uris)} SPIFFE URIs; the SPIFFE spec requires exactly one"
        )
    return spiffe_uris[0]


def validate_svid(cert_pem: bytes) -> tuple[bool, str]:
    """Validate that a certificate is a well-formed SPIFFE SVID.

    Returns ``(True, spiffe_id)`` on success, or ``(False, reason)`` on
    failure.  Checks:

    - Certificate has a SAN extension with exactly one ``spiffe://`` URI
    - The URI has a non-empty trust domain and path
    """
    try:
        spiffe_id = extract_spiffe_id(cert_pem)
    except ValueError as exc:
        return False, str(exc)

    if spiffe_id is None:
        return False, "No spiffe:// URI in Subject Alternative Name"

    parts = spiffe_id.split("/", 3)
    if len(parts) < 4 or not parts[2]:
        return False, f"Malformed SPIFFE ID (missing trust domain): {spiffe_id}"
    trust_domain = parts[2]
    path = "/" + parts[3] if len(parts) > 3 and parts[3] else ""
    if not path:
        return False, f"Malformed SPIFFE ID (empty path): {spiffe_id}"
    if not all(c.isalnum() or c in ".-_" for c in trust_domain):
        return False, f"Invalid trust domain characters: {trust_domain}"

    return True, spiffe_id


def generate_svid(
    spiffe_id: str,
    ca_cert_pem: bytes,
    ca_key: "Ed25519PrivateKey",
) -> tuple[bytes, bytes, bytes]:
    """Generate a SPIFFE SVID (X.509-SVID) signed by the given CA.

    The SVID carries an Ed25519 signing key (for X.509 validity) and an
    X25519 key pair is generated for ECDH key exchange.  The SPIFFE ID
    is embedded as a URI SAN.

    Parameters
    ----------
    spiffe_id : str
        Full SPIFFE ID, e.g. ``spiffe://example.org/workload/inference``.
    ca_cert_pem : bytes
        PEM-encoded CA certificate (the trust bundle root).
    ca_key : Ed25519PrivateKey
        CA's private key for signing the SVID.

    Returns ``(svid_pem, x25519_private_key_bytes, x25519_public_key_bytes)``.
    """
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat
    from cryptography.x509.oid import NameOID

    parts = spiffe_id.split("/", 3)
    if len(parts) < 4 or not parts[2]:
        raise ValueError(f"Invalid SPIFFE ID: {spiffe_id}")
    trust_domain = parts[2]
    path = "/" + parts[3] if len(parts) > 3 and parts[3] else ""
    if not path:
        raise ValueError(f"Malformed SPIFFE ID (empty path): {spiffe_id}")

    signing_key = Ed25519PrivateKey.generate()
    x25519_priv = X25519PrivateKey.generate()

    ca_cert = _load_bounded_pem_certificate(ca_cert_pem, "SVID issuer certificate")

    subject = x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, trust_domain),
        ]
    )

    san = x509.SubjectAlternativeName(
        [
            x509.UniformResourceIdentifier(spiffe_id),
        ]
    )

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(signing_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1))
        .add_extension(san, critical=False)
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        .add_extension(
            x509.UnrecognizedExtension(
                x509.ObjectIdentifier(_X25519_BINDING_OID),
                x25519_priv.public_key().public_bytes_raw(),
            ),
            critical=False,
        )
        .sign(ca_key, None)
    )

    svid_pem = cert.public_bytes(Encoding.PEM)
    x25519_priv_bytes = x25519_priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    x25519_pub_bytes = x25519_priv.public_key().public_bytes_raw()

    return svid_pem, x25519_priv_bytes, x25519_pub_bytes


def trust_bundle_fingerprints(ca_pems: list[bytes]) -> list[str]:
    """Convert a SPIFFE trust bundle to a list of CA fingerprints.

    A trust bundle is a list of PEM-encoded CA certificates (one per
    trust domain).  Returns SHA-256 fingerprints compatible with
    ``trusted_recipient_cas``, ``trusted_authority_cas``, and
    ``trusted_ca_fingerprints`` parameters throughout the lockbox API.
    """
    return [fingerprint_from_x509(pem) for pem in ca_pems]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeCapability:
    """Description of what a single regime grants access to."""

    regime_id: int
    auth_label: str
    column_names: list[str]


@dataclass(frozen=True)
class AccessLevel:
    """An ordered access tier in the lockbox hierarchy."""

    level: int
    name: str
    description: str
    regimes: list[int]
    capabilities: list[RegimeCapability]
    fingerprint: str


@dataclass(frozen=True)
class SealedKey:
    """A single regime's tenant key, wrapped for a specific recipient."""

    regime_id: int
    ephemeral_public_key: bytes
    nonce: bytes
    ciphertext: bytes


@dataclass(frozen=True)
class SerializedToken:
    """MaskToken fields suitable for YAML serialization."""

    regime_id: int
    tenant_id: str
    nonce: bytes
    ciphertext: bytes
    n_dims: int
    n_regimes: int
    gate_release: GateReleaseIdentity = field(default_factory=current_release_identity)

    def to_mask_token(self) -> MaskToken:
        return MaskToken(
            tenant_id=self.tenant_id,
            regime_id=self.regime_id,
            nonce=self.nonce,
            ciphertext=self.ciphertext,
            n_dims=self.n_dims,
            n_regimes=self.n_regimes,
            gate_release=self.gate_release,
        )

    @classmethod
    def from_mask_token(cls, token: MaskToken) -> SerializedToken:
        return cls(
            regime_id=token.regime_id,
            tenant_id=token.tenant_id,
            nonce=token.nonce,
            ciphertext=token.ciphertext,
            n_dims=token.n_dims,
            n_regimes=token.n_regimes,
            gate_release=token.gate_release,
        )


@dataclass
class Grant:
    """A per-recipient sealed key bundle."""

    recipient_id: str
    access_fingerprint: str
    algorithm: str
    sealed_keys: list[SealedKey]
    mask_tokens: list[SerializedToken]
    recipient_fingerprint: str


@dataclass(frozen=True)
class Authority:
    """Cryptographic identity of the training authority.

    Binds the lockbox to a CA root certificate, proving who created it.
    The signing cert (leaf) signs the lockbox_hash; the cert chain
    validates back to the root CA fingerprint.  The lockbox_hash covers
    the hierarchy, all grants, and the trust policy — making the signed
    lockbox a write-once artifact.
    """

    ca_root_fingerprint: str
    signing_cert_pem: bytes
    cert_chain_pems: list[bytes]
    signature: bytes
    signature_algorithm: str
    lockbox_hash: str


@dataclass(frozen=True)
class ProvenanceResult:
    """Result of a consumer-side grant provenance check."""

    trusted: bool
    reasons: list[str]


@dataclass
class Lockbox:
    """The complete lockbox: hierarchy + sealed grants + trust policy."""

    version: str
    chain_hash: str
    chain_name: str
    n_dims: int
    n_regimes: int
    hierarchy: list[AccessLevel]
    hierarchy_hash: str
    grants: list[Grant] = field(default_factory=list)
    trusted_recipient_cas: list[str] = field(default_factory=list)
    model_artifact_hash: str = ""
    authority: Authority | None = None
    gate_release: GateReleaseIdentity = field(default_factory=current_release_identity)


# ---------------------------------------------------------------------------
# Shamir Secret Sharing for gate key recovery
# ---------------------------------------------------------------------------


def _gf256_add(a: int, b: int) -> int:
    return a ^ b


def _gf256_mul(a: int, b: int) -> int:
    """Multiply two elements in GF(2^8) with irreducible polynomial x^8+x^4+x^3+x+1."""
    p = 0
    for _ in range(8):
        if b & 1:
            p ^= a
        hi = a & 0x80
        a = (a << 1) & 0xFF
        if hi:
            a ^= 0x1B
        b >>= 1
    return p


def _gf256_inv(a: int) -> int:
    """Compute a^254 = a^(-1) in GF(2^8) via repeated squaring."""
    if a == 0:
        raise ValueError("Cannot invert zero in GF(256)")
    result = a
    for _ in range(6):
        result = _gf256_mul(result, result)
        result = _gf256_mul(result, a)
    result = _gf256_mul(result, result)
    return result


def _gf256_div(a: int, b: int) -> int:
    return _gf256_mul(a, _gf256_inv(b))


def _eval_poly(coeffs: list[int], x: int) -> int:
    """Evaluate polynomial at x in GF(256). coeffs[0] is the secret."""
    result = 0
    for c in reversed(coeffs):
        result = _gf256_add(_gf256_mul(result, x), c)
    return result


def split_gate_key(
    key: GateKey,
    threshold: int,
    num_shares: int,
) -> list[tuple[int, bytes]]:
    """Split a gate key into Shamir shares over GF(256).

    Parameters
    ----------
    key : GateKey
        The 32-byte gate key to split.
    threshold : int
        Minimum number of shares needed to reconstruct (k).
    num_shares : int
        Total number of shares to generate (n).  Must satisfy
        2 <= threshold <= num_shares <= 255.

    Returns
    -------
    list[tuple[int, bytes]]
        Each share is ``(index, share_bytes)`` where ``index`` is in
        ``[1, num_shares]`` and ``share_bytes`` is 32 bytes.
    """
    if not (2 <= threshold <= num_shares <= 255):
        raise ValueError(
            f"Invalid parameters: need 2 <= threshold <= num_shares <= 255, "
            f"got threshold={threshold}, num_shares={num_shares}"
        )

    secret = key.secret

    polys: list[list[int]] = []
    for byte_idx in range(32):
        coeffs = [secret[byte_idx]]
        coeffs.extend(os.urandom(threshold - 1))
        polys.append(coeffs)

    shares: list[tuple[int, bytes]] = []
    for x in range(1, num_shares + 1):
        share_bytes = bytearray(32)
        for byte_idx in range(32):
            share_bytes[byte_idx] = _eval_poly(polys[byte_idx], x)
        shares.append((x, bytes(share_bytes)))

    return shares


def reconstitute_gate_key(
    shares: list[tuple[int, bytes]],
) -> GateKey:
    """Reconstruct a gate key from Shamir shares via Lagrange interpolation.

    Parameters
    ----------
    shares : list[tuple[int, bytes]]
        At least ``threshold`` shares from ``split_gate_key()``.

    Returns
    -------
    GateKey
        The reconstructed gate key.
    """
    if len(shares) < 2:
        raise ValueError("Need at least 2 shares to reconstruct")
    share_len = len(shares[0][1])
    if any(len(s[1]) != share_len for s in shares):
        raise ValueError("All shares must be the same length")

    xs = [s[0] for s in shares]
    if any(isinstance(x, bool) or not isinstance(x, int) or not 1 <= x <= 255 for x in xs):
        raise ValueError("Share indices must be unique integers in [1, 255]")
    if len(set(xs)) != len(xs):
        raise ValueError("Duplicate share indices")

    secret = bytearray(share_len)
    for byte_idx in range(share_len):
        for i, (xi, si) in enumerate(shares):
            yi = si[byte_idx]
            numerator = 1
            denominator = 1
            for j, (xj, _) in enumerate(shares):
                if i == j:
                    continue
                numerator = _gf256_mul(numerator, xj)
                denominator = _gf256_mul(denominator, _gf256_add(xj, xi))
            lagrange = _gf256_div(numerator, denominator)
            secret[byte_idx] = _gf256_add(secret[byte_idx], _gf256_mul(yi, lagrange))

    return GateKey(secret=bytes(secret))


# ---------------------------------------------------------------------------
# Canonical access fingerprint
# ---------------------------------------------------------------------------


def compute_access_fingerprint(chain_hash: str, regime_ids: list[int]) -> str:
    """Deterministic SHA-256 fingerprint for an access level.

    Preimage: "schemen:v1:access:{chain_hash}:{sorted_comma_separated_regime_ids}"
    """
    sorted_ids = ",".join(str(r) for r in sorted(regime_ids))
    preimage = f"schemen:v1:access:{chain_hash}:{sorted_ids}"
    return hashlib.sha256(preimage.encode()).hexdigest()


def _compute_hierarchy_hash(hierarchy: list["AccessLevel"], chain_hash: str) -> str:
    """SHA-256 over the canonical representation of the hierarchy.

    Uses length-prefixed encoding for all variable-length fields and
    includes capabilities (auth_label, column_names) and descriptions.
    """
    h = hashlib.sha256()
    _lp(h, chain_hash.encode())
    h.update(len(hierarchy).to_bytes(4, "big"))
    for level in hierarchy:
        h.update(level.level.to_bytes(4, "big"))
        _lp(h, level.name.encode())
        _lp(h, level.description.encode())
        h.update(len(level.regimes).to_bytes(4, "big"))
        for r in sorted(level.regimes):
            h.update(r.to_bytes(4, "big"))
        _lp(h, level.fingerprint.encode())
        h.update(len(level.capabilities).to_bytes(4, "big"))
        for cap in sorted(level.capabilities, key=lambda c: c.regime_id):
            h.update(cap.regime_id.to_bytes(4, "big"))
            _lp(h, cap.auth_label.encode())
            h.update(len(cap.column_names).to_bytes(4, "big"))
            for col in sorted(cap.column_names):
                _lp(h, col.encode())
    return h.hexdigest()


def _lp(h: "hashlib._Hash", data: bytes) -> None:
    """Length-prefixed update: feeds 4-byte big-endian length then data."""
    h.update(len(data).to_bytes(4, "big"))
    h.update(data)


def _update_grant_hash(h: "hashlib._Hash", grant: "Grant") -> None:
    """Feed every security-relevant grant field to ``h`` canonically."""
    _lp(h, grant.recipient_id.encode())
    _lp(h, grant.access_fingerprint.encode())
    _lp(h, grant.algorithm.encode())
    _lp(h, grant.recipient_fingerprint.encode())
    h.update(len(grant.sealed_keys).to_bytes(4, "big"))
    for sealed_key in grant.sealed_keys:
        h.update(sealed_key.regime_id.to_bytes(4, "big"))
        _lp(h, sealed_key.ephemeral_public_key)
        _lp(h, sealed_key.nonce)
        _lp(h, sealed_key.ciphertext)
    h.update(len(grant.mask_tokens).to_bytes(4, "big"))
    for token in grant.mask_tokens:
        h.update(token.regime_id.to_bytes(4, "big"))
        _lp(h, token.tenant_id.encode())
        _lp(h, token.nonce)
        _lp(h, token.ciphertext)
        h.update(token.n_dims.to_bytes(4, "big"))
        h.update(token.n_regimes.to_bytes(4, "big"))
        _lp(
            h,
            json.dumps(
                token.gate_release.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii"),
        )


def _compute_grant_digest(grant: "Grant") -> bytes:
    """Return a domain-separated digest for exact signed membership checks."""
    if not isinstance(grant, Grant):
        raise TypeError("grant must be a Grant")
    h = hashlib.sha256()
    _lp(h, b"schemen/grant-v1")
    _update_grant_hash(h, grant)
    return h.digest()


def compute_lockbox_hash(lockbox: "Lockbox") -> str:
    """SHA-256 over the entire lockbox excluding the authority section.

    Every variable-length field is length-prefixed (4-byte big-endian)
    to prevent boundary-confusion collisions.  Covers: version,
    chain_name, hierarchy_hash, all grant data, and the trust policy.
    """
    h = hashlib.sha256()
    _lp(h, lockbox.version.encode())
    _lp(
        h,
        json.dumps(
            lockbox.gate_release.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii"),
    )
    _lp(h, lockbox.chain_name.encode())
    _lp(h, lockbox.chain_hash.encode())
    h.update(lockbox.n_dims.to_bytes(4, "big"))
    h.update(lockbox.n_regimes.to_bytes(4, "big"))
    _lp(h, lockbox.hierarchy_hash.encode())

    h.update(len(lockbox.grants).to_bytes(4, "big"))
    for g in lockbox.grants:
        _update_grant_hash(h, g)

    h.update(len(lockbox.trusted_recipient_cas).to_bytes(4, "big"))
    for ca_fp in sorted(lockbox.trusted_recipient_cas):
        _lp(h, ca_fp.encode())

    _lp(h, lockbox.model_artifact_hash.encode())

    return h.hexdigest()


# ---------------------------------------------------------------------------
# Hierarchy validation
# ---------------------------------------------------------------------------


def validate_hierarchy(hierarchy: list[AccessLevel]) -> list[str]:
    """Validate strict winnowing: each level's regime set is a strict superset of the next.

    Returns a list of error messages (empty if valid).
    """
    errors: list[str] = []

    if not hierarchy:
        errors.append("Hierarchy is empty")
        return errors

    for i, level in enumerate(hierarchy):
        if level.level != i:
            errors.append(f"Level {i} has ordinal {level.level}; expected sequential from 0")

    for i in range(len(hierarchy) - 1):
        current_set = set(hierarchy[i].regimes)
        next_set = set(hierarchy[i + 1].regimes)
        if not next_set < current_set:
            errors.append(
                f"Level {i} regimes {sorted(current_set)} must be a strict "
                f"superset of level {i + 1} regimes {sorted(next_set)}"
            )

    return errors


# ---------------------------------------------------------------------------
# Regime inventory (extract capability menu from a chain)
# ---------------------------------------------------------------------------


def regime_inventory(chain: Any) -> dict[str, list[str]]:
    """Extract the capability menu from a chain's auth tags.

    Returns a dict mapping auth labels to lists of column names.
    """
    inventory: dict[str, list[str]] = {}
    for step in chain.steps:
        for col in step.adds:
            if col.auth:
                inventory.setdefault(col.auth, []).append(col.name)
    return inventory


# ---------------------------------------------------------------------------
# Lockbox creation
# ---------------------------------------------------------------------------


@dataclass
class HierarchyDef:
    """User-provided definition for one access level (before fingerprinting)."""

    name: str
    description: str
    regimes: list[int]
    capabilities: list[RegimeCapability]


def create_lockbox(
    master_key: GateKey,
    chain_hash: str,
    chain_name: str,
    n_dims: int,
    n_regimes: int,
    hierarchy_def: list[HierarchyDef],
    trusted_recipient_cas: list[str] | None = None,
    release_identity: GateReleaseIdentity | None = None,
) -> Lockbox:
    """Build a lockbox from a master key and hierarchy definition.

    Validates winnowing, computes fingerprints, and seals the hierarchy hash.
    ``trusted_recipient_cas`` is an optional list of CA root fingerprints;
    when non-empty, ``seal_grant`` will require recipient certs issued by
    one of these CAs.
    """
    access_levels: list[AccessLevel] = []
    for i, hdef in enumerate(hierarchy_def):
        fp = compute_access_fingerprint(chain_hash, hdef.regimes)
        access_levels.append(
            AccessLevel(
                level=i,
                name=hdef.name,
                description=hdef.description,
                regimes=list(hdef.regimes),
                capabilities=list(hdef.capabilities),
                fingerprint=fp,
            )
        )

    errors = validate_hierarchy(access_levels)
    if errors:
        raise ValueError(f"Invalid hierarchy: {'; '.join(errors)}")

    hierarchy_hash = _compute_hierarchy_hash(access_levels, chain_hash)

    return Lockbox(
        version="2",
        chain_hash=chain_hash,
        chain_name=chain_name,
        n_dims=n_dims,
        n_regimes=n_regimes,
        hierarchy=access_levels,
        hierarchy_hash=hierarchy_hash,
        trusted_recipient_cas=list(trusted_recipient_cas or []),
        gate_release=release_identity or current_release_identity(),
    )


# ---------------------------------------------------------------------------
# Lockbox reissuance (rotation / revocation)
# ---------------------------------------------------------------------------


@dataclass
class GrantSpec:
    """Specification for a grant to include in a reissued lockbox."""

    recipient_id: str
    access_level_name: str
    recipient_public_key: bytes
    recipient_cert_pem: bytes | None = None
    recipient_cert_chain_pems: tuple[bytes, ...] = ()


def reissue_lockbox(
    source: Lockbox,
    master_key: GateKey,
    revoke: set[str] | None = None,
    add: list[GrantSpec] | None = None,
    wrapper: KeyWrapper | None = None,
) -> Lockbox:
    """Create a new unsigned lockbox by applying a rotation delta to an existing one.

    Carries forward all grants from *source* except those whose
    ``recipient_id`` is in *revoke*, then seals new grants from *add*.
    The hierarchy, chain binding, trust policy, and model artifact hash
    are preserved.  The result is unsigned — the caller must sign it.

    Parameters
    ----------
    source : Lockbox
        The existing (typically signed) lockbox to base the reissue on.
    master_key : GateKey
        The gate key for deriving tenant keys and issuing mask tokens.
    revoke : set[str] | None
        Recipient IDs to drop from the new lockbox.
    add : list[GrantSpec] | None
        New grants to seal into the new lockbox.
    wrapper : KeyWrapper | None
        Key wrapper (defaults to X25519AESGCMWrapper).

    Returns
    -------
    Lockbox
        A new, unsigned lockbox with the updated grant set.
    """
    revoke = revoke or set()
    add = add or []
    if wrapper is None:
        wrapper = X25519AESGCMWrapper()

    level_by_name = {lv.name: lv for lv in source.hierarchy}
    new_lb = Lockbox(
        version=source.version,
        chain_hash=source.chain_hash,
        chain_name=source.chain_name,
        n_dims=source.n_dims,
        n_regimes=source.n_regimes,
        hierarchy=list(source.hierarchy),
        hierarchy_hash=source.hierarchy_hash,
        grants=[],
        trusted_recipient_cas=list(source.trusted_recipient_cas),
        model_artifact_hash=source.model_artifact_hash,
        authority=None,
        gate_release=source.gate_release,
    )

    for grant in source.grants:
        if grant.recipient_id in revoke:
            continue
        new_lb.grants.append(grant)

    for spec in add:
        access_level = level_by_name.get(spec.access_level_name)
        if access_level is None:
            raise ValueError(
                f"Access level '{spec.access_level_name}' not found in hierarchy. "
                f"Available: {sorted(level_by_name.keys())}"
            )
        seal_grant(
            new_lb,
            master_key,
            spec.recipient_id,
            access_level,
            spec.recipient_public_key,
            wrapper=wrapper,
            recipient_cert_pem=spec.recipient_cert_pem,
            recipient_cert_chain_pems=spec.recipient_cert_chain_pems,
        )

    return new_lb


# ---------------------------------------------------------------------------
# Access resolution
# ---------------------------------------------------------------------------


def resolve_access(lockbox: Lockbox, fingerprint: str) -> AccessLevel:
    """Look up an access level by its canonical fingerprint.

    Uses constant-time comparison to prevent timing side-channels.
    Raises KeyError if the fingerprint doesn't match any level.
    """
    for level in lockbox.hierarchy:
        if hmac.compare_digest(level.fingerprint, fingerprint):
            return level
    raise KeyError(f"No access level with fingerprint {fingerprint!r}")


def resolve_access_by_scopes(lockbox: Lockbox, requested_scopes: set[str]) -> AccessLevel:
    """Find the minimum (least-privilege) access level covering all requested scopes.

    Scopes are matched against auth_labels in each level's capabilities.
    Returns the level with the fewest regimes that still covers every
    requested scope.

    Raises ValueError if no single access level covers all requested scopes.
    """
    candidates: list[AccessLevel] = []
    for level in lockbox.hierarchy:
        level_scopes = {cap.auth_label for cap in level.capabilities}
        if requested_scopes <= level_scopes:
            candidates.append(level)

    if not candidates:
        raise ValueError(f"No access level covers all requested scopes: {sorted(requested_scopes)}")

    return min(candidates, key=lambda lv: len(lv.regimes))


# ---------------------------------------------------------------------------
# Grant sealing / unsealing
# ---------------------------------------------------------------------------


def seal_grant(
    lockbox: Lockbox,
    master_key: GateKey,
    recipient_id: str,
    access_level: AccessLevel,
    recipient_public_key: bytes,
    wrapper: KeyWrapper | None = None,
    recipient_fingerprint: str = "",
    recipient_cert_pem: bytes | None = None,
    recipient_cert_chain_pems: Sequence[bytes] | None = None,
) -> Grant:
    """Seal tenant keys and pre-issue mask tokens for a recipient.

    Derives a tenant key per regime in the access level, wraps each under
    the recipient's public key, and pre-issues mask tokens.  Appends the
    grant to the lockbox and returns it.

    When ``recipient_cert_pem`` is provided, the wrapping key must either be
    the certificate's X25519 key or an X25519 key carried in the Gate binding
    extension. An independently supplied key is accepted only when it exactly
    matches that certificate-bound value.

    When ``lockbox.trusted_recipient_cas`` is non-empty,
    ``recipient_cert_pem`` and its complete leaf-to-root chain are required;
    the validated root fingerprint must match the trust policy.
    """
    if lockbox.authority is not None:
        raise ValueError(
            "Cannot seal a grant after the lockbox has been signed. "
            "The lockbox is a write-once artifact after sign_lockbox()."
        )

    canonical_level = next(
        (
            level
            for level in lockbox.hierarchy
            if hmac.compare_digest(level.fingerprint, access_level.fingerprint)
        ),
        None,
    )
    if canonical_level is None or canonical_level != access_level:
        raise ValueError("access_level is not an exact member of this lockbox hierarchy")

    if wrapper is None:
        wrapper = X25519AESGCMWrapper()

    if recipient_cert_pem is not None:
        recipient_fingerprint = fingerprint_from_x509(recipient_cert_pem)
        recipient_public_key = _recipient_wrapping_key(
            recipient_cert_pem,
            recipient_public_key,
        )
        _validate_recipient_certificate(
            recipient_cert_pem,
            list(recipient_cert_chain_pems or ()),
            lockbox.trusted_recipient_cas,
        )

    if lockbox.trusted_recipient_cas:
        if recipient_cert_pem is None:
            raise ValueError(
                "Lockbox has a trust policy (trusted_recipient_cas) but no "
                "recipient_cert_pem was provided"
            )
    elif recipient_cert_chain_pems:
        raise ValueError(
            "recipient_cert_chain_pems requires a lockbox trusted_recipient_cas policy"
        )

    sealed_keys: list[SealedKey] = []
    mask_tokens: list[SerializedToken] = []

    for regime_id in sorted(access_level.regimes):
        tenant_key = derive_tenant_key(master_key, regime_id, recipient_id)

        wrapped = wrapper.wrap(tenant_key.secret, recipient_public_key)
        sealed_keys.append(
            SealedKey(
                regime_id=regime_id,
                ephemeral_public_key=wrapped.ephemeral_public_key,
                nonce=wrapped.nonce,
                ciphertext=wrapped.ciphertext,
            )
        )

        token = issue_mask_token(
            master_key,
            recipient_id,
            regime_id,
            lockbox.n_dims,
            lockbox.n_regimes,
            release_identity=lockbox.gate_release,
        )
        mask_tokens.append(SerializedToken.from_mask_token(token))

    grant = Grant(
        recipient_id=recipient_id,
        access_fingerprint=access_level.fingerprint,
        algorithm=wrapper.algorithm_id,
        sealed_keys=sealed_keys,
        mask_tokens=mask_tokens,
        recipient_fingerprint=recipient_fingerprint,
    )

    lockbox.grants.append(grant)
    return grant


def unseal_grant(
    grant: Grant,
    recipient_private_key: bytes,
    wrapper: KeyWrapper | None = None,
    expected_recipient_id: str | None = None,
) -> dict[int, DerivedKey]:
    """Unseal tenant keys from a grant using the recipient's private key.

    Returns a dict mapping regime_id -> DerivedKey.
    """
    if wrapper is None:
        wrapper = X25519AESGCMWrapper()

    if grant.algorithm != wrapper.algorithm_id:
        raise ValueError(
            f"Algorithm mismatch: grant uses {grant.algorithm!r}, "
            f"wrapper provides {wrapper.algorithm_id!r}"
        )

    if expected_recipient_id is not None and grant.recipient_id != expected_recipient_id:
        raise ValueError(
            f"Recipient mismatch: grant is for {grant.recipient_id!r}, "
            f"expected {expected_recipient_id!r}"
        )

    keys: dict[int, DerivedKey] = {}
    for sk in grant.sealed_keys:
        wrapped = WrappedKey(
            ephemeral_public_key=sk.ephemeral_public_key,
            nonce=sk.nonce,
            ciphertext=sk.ciphertext,
        )
        secret = wrapper.unwrap(wrapped, recipient_private_key)
        keys[sk.regime_id] = DerivedKey(
            secret=secret,
            context=f"regime:{sk.regime_id}:tenant:{grant.recipient_id}",
        )

    return keys


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def diagnose_grants(lockbox: Lockbox) -> list[str]:
    """Detect issues in the lockbox's grant set.

    Checks for:
    - Redundant grants (same recipient at multiple levels where one subsumes another)
    - Coverage gaps (hierarchy levels with no grants)
    - Duplicate grants (same recipient + same access level)
    """
    warnings: list[str] = []

    recipient_grants: dict[str, list[Grant]] = {}
    for g in lockbox.grants:
        recipient_grants.setdefault(g.recipient_id, []).append(g)

    for recipient_id, grants in recipient_grants.items():
        if len(grants) < 2:
            continue

        fps = [g.access_fingerprint for g in grants]
        if len(fps) != len(set(fps)):
            warnings.append(
                f"Recipient '{recipient_id}' has duplicate grants at the same access level"
            )

        level_by_fp = {lv.fingerprint: lv for lv in lockbox.hierarchy}
        regime_sets = []
        for g in grants:
            lv = level_by_fp.get(g.access_fingerprint)
            if lv:
                regime_sets.append((g.access_fingerprint, set(lv.regimes)))

        for i, (fp_a, rs_a) in enumerate(regime_sets):
            for fp_b, rs_b in regime_sets[i + 1 :]:
                if rs_a <= rs_b:
                    warnings.append(
                        f"Recipient '{recipient_id}': grant at '{fp_a[:12]}...' "
                        f"is redundant (subsumed by '{fp_b[:12]}...')"
                    )
                elif rs_b <= rs_a:
                    warnings.append(
                        f"Recipient '{recipient_id}': grant at '{fp_b[:12]}...' "
                        f"is redundant (subsumed by '{fp_a[:12]}...')"
                    )

    granted_fps = {g.access_fingerprint for g in lockbox.grants}
    for level in lockbox.hierarchy:
        if level.fingerprint not in granted_fps:
            warnings.append(
                f"Access level '{level.name}' (level {level.level}) has no grants issued"
            )

    return warnings


def diagnose_operational(
    lockbox: Lockbox,
    revoked_recipients: set[str] | None = None,
    reference_time: Any | None = None,
    expected_chain_hash: str | None = None,
    *,
    revocation: RevocationCheck | RevocationPolicy = RevocationCheck.SKIP,
) -> list[str]:
    """Detect operational issues in a lockbox's grant set.

    Complements :func:`diagnose_grants` (which checks structural issues)
    with temporal and lifecycle checks.

    Parameters
    ----------
    lockbox : Lockbox
        The lockbox to diagnose.
    revoked_recipients : set[str] | None
        Recipient IDs that should have been revoked.  Grants for these
        recipients are flagged.
    reference_time : datetime | None
        Timestamp for certificate validity checks.  Defaults to now.
        When provided, grants whose ``recipient_fingerprint`` references
        a certificate that has expired by this time are flagged (requires
        the grant to carry a cert — not always available in the lockbox
        format, so this checks the authority signing cert validity).
    expected_chain_hash : str | None
        If provided, flags any lockbox whose ``chain_hash`` does not
        match, indicating a possible key lineage mismatch.
    revocation : RevocationCheck
        Controls certificate revocation checking on the authority signing
        certificate. SKIP is an explicit offline diagnostic policy.
    """
    import datetime as _dt

    warnings: list[str] = []
    now = reference_time or _dt.datetime.now(_dt.timezone.utc)
    revocation_policy = _revocation_policy(revocation)

    if revoked_recipients:
        for g in lockbox.grants:
            if g.recipient_id in revoked_recipients:
                warnings.append(f"Grant for revoked recipient '{g.recipient_id}' is still present")

    if expected_chain_hash is not None:
        if not hmac.compare_digest(lockbox.chain_hash, expected_chain_hash):
            warnings.append(
                f"Chain hash mismatch: lockbox has '{lockbox.chain_hash[:16]}...', "
                f"expected '{expected_chain_hash[:16]}...' — possible key lineage mismatch"
            )

    if lockbox.authority is not None:
        signing_cert = None
        try:
            signing_cert = _load_bounded_pem_certificate(
                lockbox.authority.signing_cert_pem,
                "authority signing certificate",
            )
            if now > signing_cert.not_valid_after_utc:
                warnings.append(
                    f"Authority signing certificate expired on "
                    f"{signing_cert.not_valid_after_utc.isoformat()}"
                )
            if now < signing_cert.not_valid_before_utc:
                warnings.append(
                    f"Authority signing certificate not yet valid "
                    f"(valid from {signing_cert.not_valid_before_utc.isoformat()})"
                )
        except Exception:
            warnings.append("Could not parse authority signing certificate for validity check")

        if revocation_policy.mode != RevocationCheck.SKIP and signing_cert is not None:
            try:
                revoc_warnings = revocation_policy.check(signing_cert, signing_cert)
                warnings.extend(revoc_warnings)
            except ValueError as e:
                warnings.append(f"Revocation check failed: {e}")

    if not lockbox.authority:
        warnings.append("Lockbox is unsigned — no authority section present")

    if lockbox.model_artifact_hash == "":
        warnings.append("Lockbox has no model artifact hash — model binding is missing")

    return warnings


# ---------------------------------------------------------------------------
# YAML serialization
# ---------------------------------------------------------------------------


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(s: str) -> bytes:
    try:
        return base64.b64decode(s, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid base64 in lockbox") from exc


def save_lockbox(lockbox: Lockbox, path: str | Path) -> None:
    """Serialize a lockbox to YAML."""
    hierarchy_data = []
    for lv in lockbox.hierarchy:
        caps = [
            {
                "regime": cap.regime_id,
                "auth": cap.auth_label,
                "columns": cap.column_names,
            }
            for cap in lv.capabilities
        ]
        hierarchy_data.append(
            {
                "level": lv.level,
                "name": lv.name,
                "description": lv.description,
                "fingerprint": lv.fingerprint,
                "regimes": lv.regimes,
                "capabilities": caps,
            }
        )

    grants_data = []
    for g in lockbox.grants:
        sealed = [
            {
                "regime": sk.regime_id,
                "ephemeral_public_key": _b64(sk.ephemeral_public_key),
                "nonce": _b64(sk.nonce),
                "ciphertext": _b64(sk.ciphertext),
            }
            for sk in g.sealed_keys
        ]
        tokens = [
            {
                "regime": t.regime_id,
                "tenant_id": t.tenant_id,
                "nonce": _b64(t.nonce),
                "ciphertext": _b64(t.ciphertext),
                "n_dims": t.n_dims,
                "n_regimes": t.n_regimes,
                "gate_release": t.gate_release.to_dict(),
            }
            for t in g.mask_tokens
        ]
        grants_data.append(
            {
                "recipient": g.recipient_id,
                "access_fingerprint": g.access_fingerprint,
                "algorithm": g.algorithm,
                "recipient_fingerprint": g.recipient_fingerprint,
                "sealed_keys": sealed,
                "mask_tokens": tokens,
            }
        )

    authority_data = None
    if lockbox.authority is not None:
        auth = lockbox.authority
        authority_data = {
            "ca_root_fingerprint": auth.ca_root_fingerprint,
            "signing_cert_pem": auth.signing_cert_pem.decode("ascii"),
            "cert_chain_pems": [p.decode("ascii") for p in auth.cert_chain_pems],
            "signature": _b64(auth.signature),
            "signature_algorithm": auth.signature_algorithm,
            "lockbox_hash": auth.lockbox_hash,
        }

    doc: dict[str, Any] = {
        "schemen_lockbox": {
            "version": lockbox.version,
            "gate_release": lockbox.gate_release.to_dict(),
            "model": {
                "chain_name": lockbox.chain_name,
                "chain_hash": lockbox.chain_hash,
                "n_dims": lockbox.n_dims,
                "n_regimes": lockbox.n_regimes,
            },
            "hierarchy": hierarchy_data,
            "hierarchy_hash": lockbox.hierarchy_hash,
            "grants": grants_data,
        }
    }
    if lockbox.model_artifact_hash:
        doc["schemen_lockbox"]["model_artifact_hash"] = lockbox.model_artifact_hash
    if lockbox.trusted_recipient_cas:
        doc["schemen_lockbox"]["trusted_recipient_cas"] = lockbox.trusted_recipient_cas
    if authority_data is not None:
        doc["schemen_lockbox"]["authority"] = authority_data

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(doc, f, Dumper=_PlainSafeDumper, default_flow_style=False, sort_keys=False)


def load_lockbox(
    path: str | Path,
    trusted_authority_cas: list[str] | None = None,
    *,
    revocation: RevocationCheck | RevocationPolicy | None = None,
    require_authority: bool = True,
    expected_release: GateReleaseIdentity | None = None,
) -> Lockbox:
    """Deserialize a lockbox and require authenticated authority by default.

    ``require_authority=False`` is an explicit unsafe parsing mode for local
    construction workflows. It cannot be combined with a trust store, which
    prevents callers from mistaking unsigned parsing for verification.
    """
    path = Path(path)
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Lockbox file exceeds the 64 MiB safety limit")
    try:
        with open(path, "r", encoding="utf-8") as f:
            # _StrictSafeLoader is a SafeLoader subclass with narrower syntax.
            doc = yaml.load(f, Loader=_StrictSafeLoader)  # nosec B506
    except (UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid lockbox YAML: {exc}") from exc

    if not isinstance(doc, dict) or "schemen_lockbox" not in doc:
        raise ValueError("Invalid lockbox file: missing 'schemen_lockbox' root key")

    lb = doc["schemen_lockbox"]
    if not isinstance(lb, dict):
        raise ValueError("Invalid lockbox file: 'schemen_lockbox' must be a mapping")

    if lb.get("version") != "2":
        raise ValueError(f"Unsupported lockbox version: {lb.get('version')!r} (expected '2')")

    try:
        gate_release = GateReleaseIdentity.from_dict(dict(lb["gate_release"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid lockbox Gate release identity") from exc
    release = expected_release or current_release_identity()
    if not release_identity_matches(
        gate_release,
        release,
        require_source_commit=True,
    ):
        raise ValueError("Lockbox Gate release differs from the verifying runtime")

    model = lb.get("model")
    if not isinstance(model, dict):
        raise ValueError("Invalid lockbox file: missing 'model' section")

    for field_name in ("chain_hash", "chain_name"):
        if not isinstance(model.get(field_name), str):
            raise ValueError(f"Invalid lockbox: model.{field_name} must be a string")
    for field_name in ("n_dims", "n_regimes"):
        val = model.get(field_name)
        if not isinstance(val, int) or val < 1:
            raise ValueError(f"Invalid lockbox: model.{field_name} must be a positive integer")

    try:
        lockbox = _lockbox_from_document(lb, model, gate_release)
    except (KeyError, TypeError, AttributeError, ReleaseIdentityError) as exc:
        raise ValueError(
            "Invalid lockbox structure: a required field is missing or malformed"
        ) from exc

    expected_hash = _compute_hierarchy_hash(lockbox.hierarchy, model["chain_hash"])
    if lockbox.hierarchy_hash != expected_hash:
        raise ValueError("Hierarchy hash mismatch — lockbox may have been tampered with")

    if lockbox.authority is not None:
        verify_authority(
            lockbox,
            trusted_authority_cas or [],
            revocation=revocation,
            expected_release=release,
        )
    elif require_authority:
        raise ValueError("Lockbox is unsigned; an authority section is required")
    elif trusted_authority_cas:
        raise ValueError("trusted_authority_cas cannot authenticate an unsigned lockbox")

    return lockbox


def _lockbox_from_document(
    lb: dict[str, Any],
    model: dict[str, Any],
    gate_release: GateReleaseIdentity,
) -> Lockbox:
    """Build the unverified in-memory lockbox from a parsed YAML mapping."""
    hierarchy: list[AccessLevel] = []
    for lv_data in lb["hierarchy"]:
        caps = [
            RegimeCapability(
                regime_id=c["regime"],
                auth_label=c["auth"],
                column_names=c["columns"],
            )
            for c in lv_data["capabilities"]
        ]
        hierarchy.append(
            AccessLevel(
                level=lv_data["level"],
                name=lv_data["name"],
                description=lv_data["description"],
                regimes=lv_data["regimes"],
                capabilities=caps,
                fingerprint=lv_data["fingerprint"],
            )
        )

    grants: list[Grant] = []
    for g_data in lb.get("grants") or []:
        sealed = [
            SealedKey(
                regime_id=sk["regime"],
                ephemeral_public_key=_unb64(sk["ephemeral_public_key"]),
                nonce=_unb64(sk["nonce"]),
                ciphertext=_unb64(sk["ciphertext"]),
            )
            for sk in g_data["sealed_keys"]
        ]
        tokens = [
            SerializedToken(
                regime_id=t["regime"],
                tenant_id=t["tenant_id"],
                nonce=_unb64(t["nonce"]),
                ciphertext=_unb64(t["ciphertext"]),
                n_dims=t["n_dims"],
                n_regimes=t["n_regimes"],
                gate_release=GateReleaseIdentity.from_dict(t["gate_release"]),
            )
            for t in g_data["mask_tokens"]
        ]
        grants.append(
            Grant(
                recipient_id=g_data["recipient"],
                access_fingerprint=g_data["access_fingerprint"],
                algorithm=g_data["algorithm"],
                recipient_fingerprint=g_data.get("recipient_fingerprint", ""),
                sealed_keys=sealed,
                mask_tokens=tokens,
            )
        )

    trusted_cas: list[str] = lb.get("trusted_recipient_cas", [])

    authority: Authority | None = None
    auth_data = lb.get("authority")
    if auth_data is not None:
        authority = Authority(
            ca_root_fingerprint=auth_data["ca_root_fingerprint"],
            signing_cert_pem=auth_data["signing_cert_pem"].encode("ascii"),
            cert_chain_pems=[p.encode("ascii") for p in auth_data.get("cert_chain_pems", [])],
            signature=_unb64(auth_data["signature"]),
            signature_algorithm=auth_data["signature_algorithm"],
            lockbox_hash=auth_data["lockbox_hash"],
        )

    return Lockbox(
        version=lb["version"],
        chain_hash=model["chain_hash"],
        chain_name=model["chain_name"],
        n_dims=model["n_dims"],
        n_regimes=model["n_regimes"],
        hierarchy=hierarchy,
        hierarchy_hash=lb["hierarchy_hash"],
        grants=grants,
        trusted_recipient_cas=trusted_cas,
        model_artifact_hash=lb.get("model_artifact_hash", ""),
        authority=authority,
        gate_release=gate_release,
    )


# ---------------------------------------------------------------------------
# Model artifact attestation
# ---------------------------------------------------------------------------


_MODEL_ATTESTATION_SCHEMA = "schemen/model-attestation-v3"
_MODEL_ATTESTATION_METADATA_KEYS = frozenset(
    {
        "schemen.attestation_schema",
        "schemen.model_graph_hash",
        "schemen.model_hash",
        "schemen.attestation_signature",
        "schemen.attestation_cert_pem",
        "schemen.attestation_ca_fingerprint",
        "schemen.attestation_ca_cert_pem",
        "schemen.chain_hash",
        "schemen.lockbox_hash",
    }
)


def _load_complete_onnx_model(onnx_path: str | Path) -> Any:
    """Load an ONNX model after confining external tensor paths."""
    import onnx

    model_path = Path(onnx_path).resolve(strict=True)
    model = onnx.load_model(str(model_path), load_external_data=False)

    def iter_tensors(message: Any) -> Iterator[Any]:
        if isinstance(message, onnx.TensorProto):
            yield message
            return
        for field_descriptor, value in message.ListFields():
            if field_descriptor.type != field_descriptor.TYPE_MESSAGE:
                continue
            if field_descriptor.is_repeated:
                for child in value:
                    yield from iter_tensors(child)
            else:
                yield from iter_tensors(value)

    model_dir = model_path.parent
    for tensor in iter_tensors(model):
        if not onnx.external_data_helper.uses_external_data(tensor):
            continue
        external = {entry.key: entry.value for entry in tensor.external_data}
        location = external.get("location")
        if not location or Path(location).is_absolute():
            raise ValueError("ONNX external tensor location must be a relative path")
        tensor_path = (model_dir / location).resolve(strict=True)
        if not tensor_path.is_relative_to(model_dir):
            raise ValueError("ONNX external tensor escapes the model directory")
        try:
            offset = int(external.get("offset", "0"))
            length_value = external.get("length")
            length = None if length_value is None else int(length_value)
        except ValueError as exc:
            raise ValueError("ONNX external tensor has invalid offset or length") from exc
        if offset < 0 or (length is not None and length < 0):
            raise ValueError("ONNX external tensor offset and length must be non-negative")
        file_size = tensor_path.stat().st_size
        if offset > file_size or (length is not None and offset + length > file_size):
            raise ValueError("ONNX external tensor range exceeds its data file")

    onnx.external_data_helper.load_external_data_for_model(model, str(model_dir))
    onnx.external_data_helper.convert_model_from_external_data(model)
    return model


def _canonical_model_bytes(model: Any) -> bytes:
    """Serialize every model field except self-referential Gate metadata."""
    sanitized = type(model)()
    sanitized.CopyFrom(model)
    retained_metadata = sorted(
        (
            (entry.key, entry.value)
            for entry in sanitized.metadata_props
            if entry.key not in _MODEL_ATTESTATION_METADATA_KEYS
        ),
        key=lambda item: (item[0], item[1]),
    )
    sanitized.ClearField("metadata_props")
    for key, value in retained_metadata:
        entry = sanitized.metadata_props.add()
        entry.key = key
        entry.value = value
    return bytes(sanitized.SerializeToString(deterministic=True))


def compute_model_artifact_hash(onnx_path: str | Path) -> str:
    """Hash the complete executable ONNX model, excluding Gate metadata."""
    return hashlib.sha256(_canonical_model_bytes(_load_complete_onnx_model(onnx_path))).hexdigest()


def compute_model_graph_hash(onnx_path: str | Path) -> str:
    """Compatibility alias for the v3 complete-model artifact hash."""
    return compute_model_artifact_hash(onnx_path)


def bind_model_artifact(lockbox: Lockbox, onnx_path: str | Path) -> str:
    """Bind an ONNX model to the lockbox by setting ``model_artifact_hash``.

    Must be called **before** ``sign_lockbox()`` — the hash becomes part
    of ``lockbox_hash`` and is therefore transitively covered by the
    authority signature.

    Returns the computed complete-model artifact hash.
    """
    if lockbox.authority is not None:
        raise ValueError(
            "Cannot bind a model artifact after the lockbox has been signed. "
            "Call bind_model_artifact() before sign_lockbox()."
        )
    artifact_hash = compute_model_artifact_hash(onnx_path)
    lockbox.model_artifact_hash = artifact_hash
    return artifact_hash


def attest_model(
    onnx_path: str | Path,
    signing_key: "Ed25519PrivateKey",
    signing_cert_pem: bytes,
    ca_root_pem: bytes,
    chain_hash: str,
    lockbox: Lockbox,
) -> str:
    """Sign and embed complete-model attestation metadata into an ONNX file.

    The following keys are written
    into ONNX ``metadata_props``:

    - ``schemen.model_hash``
    - ``schemen.attestation_signature`` (base64)
    - ``schemen.attestation_cert_pem``
    - ``schemen.attestation_ca_fingerprint``
    - ``schemen.chain_hash``
    - ``schemen.lockbox_hash``

    Returns the complete model artifact hash.
    """
    import onnx
    from cryptography.hazmat.primitives.hashes import SHA256 as _SHA256

    model = _load_complete_onnx_model(onnx_path)
    model_hash = hashlib.sha256(_canonical_model_bytes(model)).hexdigest()

    if lockbox.authority is None:
        raise ValueError("Model attestation requires a signed lockbox")
    if not hmac.compare_digest(chain_hash, lockbox.chain_hash):
        raise ValueError("Model chain_hash does not match the signed lockbox")
    if not lockbox.model_artifact_hash or not hmac.compare_digest(
        lockbox.model_artifact_hash,
        model_hash,
    ):
        raise ValueError("Model artifact must be bound to the lockbox before it is signed")
    lb_hash = lockbox.authority.lockbox_hash
    payload = json.dumps(
        {
            "chain_hash": chain_hash,
            "lockbox_hash": lb_hash,
            "model_hash": model_hash,
            "schema": _MODEL_ATTESTATION_SCHEMA,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    signature = signing_key.sign(payload)

    root_cert = _load_bounded_pem_certificate(ca_root_pem, "attestation root certificate")
    ca_fp = root_cert.fingerprint(_SHA256()).hex()

    attestation_meta = {
        "schemen.attestation_schema": _MODEL_ATTESTATION_SCHEMA,
        "schemen.model_hash": model_hash,
        "schemen.attestation_signature": base64.b64encode(signature).decode("ascii"),
        "schemen.attestation_cert_pem": signing_cert_pem.decode("ascii"),
        "schemen.attestation_ca_fingerprint": ca_fp,
        "schemen.attestation_ca_cert_pem": ca_root_pem.decode("ascii"),
        "schemen.chain_hash": chain_hash,
        "schemen.lockbox_hash": lb_hash,
    }

    retained_metadata = [
        (entry.key, entry.value)
        for entry in model.metadata_props
        if entry.key not in _MODEL_ATTESTATION_METADATA_KEYS
    ]
    model.ClearField("metadata_props")
    for key, value in retained_metadata:
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    for key, value in attestation_meta.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value

    onnx.save(model, str(onnx_path))
    return model_hash


def verify_model_attestation(
    onnx_path: str | Path,
    trusted_ca_fingerprints: list[str],
    *,
    expected_lockbox_hash: str,
    revocation: RevocationCheck | RevocationPolicy | None = None,
) -> "ProvenanceResult":
    """Verify attestation metadata embedded in an ONNX model file.

    Recomputes the complete-model hash, verifies the Ed25519 signature, and
    checks the CA fingerprint against the caller's trust store.

    Returns a ``ProvenanceResult`` with ``trusted=True`` on success or
    a list of reasons on failure.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.hazmat.primitives.hashes import SHA256 as _SHA256

    reasons: list[str] = []
    try:
        revocation_policy = _revocation_policy(revocation)
    except ValueError as exc:
        return ProvenanceResult(
            trusted=False,
            reasons=[str(exc)],
        )
    model = _load_complete_onnx_model(onnx_path)

    meta: dict[str, str] = {}
    duplicate_keys: set[str] = set()
    for prop in model.metadata_props:
        if prop.key in meta:
            duplicate_keys.add(prop.key)
        else:
            meta[prop.key] = prop.value
    if duplicate_keys:
        return ProvenanceResult(
            trusted=False,
            reasons=[
                "Duplicate ONNX metadata keys are not permitted: "
                + ", ".join(sorted(duplicate_keys))
            ],
        )

    required_keys = [
        "schemen.attestation_schema",
        "schemen.model_hash",
        "schemen.attestation_signature",
        "schemen.attestation_cert_pem",
        "schemen.attestation_ca_fingerprint",
        "schemen.attestation_ca_cert_pem",
        "schemen.chain_hash",
        "schemen.lockbox_hash",
    ]
    missing = [k for k in required_keys if k not in meta]
    if missing:
        reasons.append(f"Missing attestation metadata: {', '.join(missing)}")
        return ProvenanceResult(trusted=False, reasons=reasons)

    actual_hash = hashlib.sha256(_canonical_model_bytes(model)).hexdigest()
    declared_hash = meta["schemen.model_hash"]
    if not hmac.compare_digest(actual_hash, declared_hash):
        reasons.append("Model artifact hash mismatch — model may have been tampered with")
        return ProvenanceResult(trusted=False, reasons=reasons)

    if meta["schemen.attestation_schema"] != _MODEL_ATTESTATION_SCHEMA:
        reasons.append("Unsupported model attestation schema")
        return ProvenanceResult(trusted=False, reasons=reasons)

    ca_fp = meta["schemen.attestation_ca_fingerprint"]
    if not any(hmac.compare_digest(ca_fp, trusted) for trusted in trusted_ca_fingerprints):
        reasons.append(f"Attestation CA {ca_fp[:16]}... is not in the consumer's trust store")

    cert_pem = meta["schemen.attestation_cert_pem"].encode("ascii")
    ca_cert_pem = meta["schemen.attestation_ca_cert_pem"].encode("ascii")
    sig_b64 = meta["schemen.attestation_signature"]
    chain_hash = meta["schemen.chain_hash"]
    lockbox_hash = meta["schemen.lockbox_hash"]

    if not expected_lockbox_hash or not hmac.compare_digest(lockbox_hash, expected_lockbox_hash):
        reasons.append("Model attestation is not bound to the expected lockbox")

    try:
        cert = _load_bounded_pem_certificate(cert_pem, "attestation certificate")
        root = _load_bounded_pem_certificate(ca_cert_pem, "attestation root certificate")
        actual_ca_fp = root.fingerprint(_SHA256()).hex()
        if not hmac.compare_digest(actual_ca_fp, ca_fp):
            reasons.append("Declared CA fingerprint does not match the embedded CA certificate")
        _verify_cert_chain(cert, [ca_cert_pem], _SHA256)
        if revocation_policy.mode != RevocationCheck.SKIP and cert.fingerprint(
            _SHA256()
        ) != root.fingerprint(_SHA256()):
            for warning in revocation_policy.check(cert, root):
                _LOGGER.warning("Certificate revocation warning: %s", warning)
        pub_key = cert.public_key()
        if not isinstance(pub_key, Ed25519PublicKey):
            reasons.append("Attestation certificate does not contain an Ed25519 key")
        else:
            payload = json.dumps(
                {
                    "chain_hash": chain_hash,
                    "lockbox_hash": lockbox_hash,
                    "model_hash": declared_hash,
                    "schema": _MODEL_ATTESTATION_SCHEMA,
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
            signature = base64.b64decode(sig_b64, validate=True)
            pub_key.verify(signature, payload)
    except Exception:
        if not reasons:
            reasons.append("Attestation signature verification failed")

    if reasons:
        return ProvenanceResult(trusted=False, reasons=reasons)
    return ProvenanceResult(trusted=True, reasons=[])


# ---------------------------------------------------------------------------
# SPIFFE convenience wrappers
# ---------------------------------------------------------------------------


def _spiffe_tenant_id(spiffe_id: str) -> str:
    """Derive a colon-free tenant ID from a SPIFFE ID.

    The HKDF info string uses ``:`` as a delimiter, so raw SPIFFE URIs
    (which contain ``://``) cannot be used directly.  We SHA-256 the
    SPIFFE ID and prefix with ``spiffe_`` for debuggability.
    """
    digest = hashlib.sha256(spiffe_id.encode("utf-8")).hexdigest()
    return f"spiffe_{digest}"


def seal_grant_spiffe(
    lockbox: Lockbox,
    master_key: "GateKey",
    svid_pem: bytes,
    access_level: "AccessLevel",
    recipient_public_key: bytes,
    wrapper: "KeyWrapper | None" = None,
    trust_bundle: Sequence[bytes] | None = None,
) -> "Grant":
    """Seal a grant using a SPIFFE SVID as the recipient identity.

    The SPIFFE ID (``spiffe://trust-domain/path``) is extracted from the
    certificate's SAN.  A deterministic, colon-free tenant ID is derived
    from it for HKDF key derivation (since the HKDF info string uses
    ``:`` as a delimiter).  The full SPIFFE URI is stored in the grant's
    ``recipient_id`` for auditability.

    The SVID PEM is forwarded as ``recipient_cert_pem`` for fingerprinting
    and validation against the recipient CA fingerprints committed in the
    lockbox. ``trust_bundle`` must contain the ordered issuer path ending at
    one of those independently configured roots.
    """
    if not lockbox.trusted_recipient_cas:
        raise ValueError("SPIFFE grant sealing requires lockbox trusted_recipient_cas")
    if not trust_bundle:
        raise ValueError("SPIFFE grant sealing requires a non-empty trust bundle")
    valid, id_or_reason = validate_svid(svid_pem)
    if not valid:
        raise ValueError(f"Invalid SPIFFE SVID: {id_or_reason}")
    spiffe_id = id_or_reason
    tenant_id = _spiffe_tenant_id(spiffe_id)

    from cryptography import x509

    cert = _load_bounded_pem_certificate(svid_pem, "SPIFFE SVID")
    try:
        binding = cert.extensions.get_extension_for_oid(
            x509.ObjectIdentifier(_X25519_BINDING_OID)
        ).value
        if not isinstance(binding, x509.UnrecognizedExtension):
            raise ValueError("SPIFFE X25519 binding has the wrong extension type")
        bound_key = binding.value
    except x509.ExtensionNotFound as exc:
        raise ValueError(
            "SPIFFE SVID does not cryptographically bind an X25519 recipient key"
        ) from exc
    if not hmac.compare_digest(bound_key, recipient_public_key):
        raise ValueError("recipient_public_key is not bound to the SPIFFE SVID")

    return seal_grant(
        lockbox,
        master_key,
        recipient_id=tenant_id,
        access_level=access_level,
        recipient_public_key=recipient_public_key,
        wrapper=wrapper,
        recipient_cert_pem=svid_pem,
        recipient_cert_chain_pems=trust_bundle,
    )


def verify_provenance_spiffe(
    lockbox: Lockbox,
    grant: "Grant",
    consumer_svid_pem: bytes,
    trust_bundle: list[bytes],
    *,
    trusted_authority_cas: list[str],
    revocation: RevocationCheck | RevocationPolicy | None = None,
) -> "ProvenanceResult":
    """Verify grant provenance using SPIFFE trust bundle and consumer SVID.

    The ordered SPIFFE issuer path authenticates the recipient SVID only when
    its terminal root matches a recipient CA fingerprint committed in the
    signed lockbox. Lockbox authority roots remain a separate, independently
    configured trust store.
    """
    if not trust_bundle:
        return ProvenanceResult(
            trusted=False,
            reasons=["SPIFFE provenance requires a non-empty trust bundle"],
        )
    if not lockbox.trusted_recipient_cas:
        return ProvenanceResult(
            trusted=False,
            reasons=["SPIFFE provenance requires recipient CA fingerprints in the signed lockbox"],
        )
    return verify_grant_provenance(
        lockbox,
        grant,
        consumer_svid_pem,
        trusted_authority_cas,
        consumer_cert_chain_pems=trust_bundle,
        revocation=revocation,
    )


def verify_attestation_spiffe(
    onnx_path: str | Path,
    trust_bundle: list[bytes],
    *,
    expected_lockbox_hash: str,
    revocation: RevocationCheck | RevocationPolicy | None = None,
) -> "ProvenanceResult":
    """Verify ONNX model attestation using a SPIFFE trust bundle.

    Converts the trust bundle to CA fingerprints and delegates to
    :func:`verify_model_attestation`.
    """
    ca_fps = trust_bundle_fingerprints(trust_bundle)
    return verify_model_attestation(
        onnx_path,
        ca_fps,
        expected_lockbox_hash=expected_lockbox_hash,
        revocation=revocation,
    )


# ---------------------------------------------------------------------------
# SPIFFE Workload API client (optional runtime integration)
# ---------------------------------------------------------------------------


@dataclass
class WorkloadIdentity:
    """A fetched SPIFFE workload identity (SVID + trust bundle)."""

    spiffe_id: str
    svid_pem: bytes
    private_key_pem: bytes = field(repr=False)
    trust_bundle: list[bytes]


class SpiffeWorkloadClient:
    """Client for the SPIFFE Workload API.

    Fetches X.509-SVIDs and trust bundles at runtime from a SPIFFE
    agent (e.g. SPIRE).  This enables automatic SVID rotation for
    lockbox seal/verify operations.

    Usage::

        client = SpiffeWorkloadClient()  # uses SPIFFE_ENDPOINT_SOCKET env
        identity = client.fetch_identity()
        grant = seal_grant_spiffe(
            lockbox, master_key,
            svid_pem=identity.svid_pem,
            access_level=level,
            recipient_public_key=pub,
        )

    The client wraps the ``spiffe`` package when available, or falls back to
    direct file-based SVID loading for environments without the gRPC
    Workload API.

    This is an **optional** integration.  The lockbox works without it
    by accepting static PEM bytes directly.
    """

    def __init__(
        self,
        socket_path: str | None = None,
        svid_path: str | None = None,
        key_path: str | None = None,
        bundle_path: str | None = None,
    ) -> None:
        """Initialize the client.

        Parameters
        ----------
        socket_path : str | None
            SPIFFE Workload API socket path.  If None, reads from the
            ``SPIFFE_ENDPOINT_SOCKET`` environment variable.
        svid_path : str | None
            Path to a PEM-encoded X.509-SVID file (file-based fallback).
        key_path : str | None
            Path to the SVID's private key PEM (file-based fallback).
        bundle_path : str | None
            Path to the trust bundle PEM (file-based fallback).
        """
        self._socket_path = socket_path or os.environ.get("SPIFFE_ENDPOINT_SOCKET")
        self._svid_path = svid_path
        self._key_path = key_path
        self._bundle_path = bundle_path

    def fetch_identity(
        self,
        *,
        validate_spiffe_id_syntax: bool = True,
    ) -> WorkloadIdentity:
        """Fetch the current workload identity.

        Tries file-based loading first (if all three paths are set),
        then the Workload API (via py-spiffe).

        Parameters
        ----------
        validate_spiffe_id_syntax : bool
            If True (default), validate only the SPIFFE URI syntax in the
            fetched SVID. Full certificate-path, key-match, validity, and
            trust-domain verification remains a separate authority step.

        Returns
        -------
        WorkloadIdentity
            The fetched identity with SVID PEM, private key, and trust
            bundle.

        Raises
        ------
        RuntimeError
            If neither the Workload API nor file paths are available.
        ValueError
            If validation is enabled and the fetched SVID is malformed.
        """
        if self._svid_path and self._key_path and self._bundle_path:
            identity = self._fetch_from_files()
        elif self._socket_path:
            identity = self._fetch_from_workload_api()
        else:
            raise RuntimeError(
                "No SPIFFE identity source configured. Provide either "
                "file paths (svid_path, key_path, bundle_path) or set "
                "SPIFFE_ENDPOINT_SOCKET for the Workload API."
            )

        if validate_spiffe_id_syntax:
            ok, detail = validate_svid(identity.svid_pem)
            if not ok:
                raise ValueError(f"Fetched SVID failed validation: {detail}")

        return identity

    def _fetch_from_files(self) -> WorkloadIdentity:
        """Load identity from PEM files on disk."""
        svid_pem = Path(self._svid_path).read_bytes()  # type: ignore[arg-type]
        key_pem = Path(self._key_path).read_bytes()  # type: ignore[arg-type]
        bundle_pem = Path(self._bundle_path).read_bytes()  # type: ignore[arg-type]

        spiffe_id = extract_spiffe_id(svid_pem)
        if spiffe_id is None:
            raise ValueError(f"File {self._svid_path} does not contain a SPIFFE SVID")

        bundle_certs = self._split_pem_bundle(bundle_pem)

        return WorkloadIdentity(
            spiffe_id=spiffe_id,
            svid_pem=svid_pem,
            private_key_pem=key_pem,
            trust_bundle=bundle_certs,
        )

    def _fetch_from_workload_api(self) -> WorkloadIdentity:
        """Fetch identity via the SPIFFE Workload API (requires ``spiffe``)."""
        try:
            from spiffe import WorkloadApiClient
        except ImportError:
            raise RuntimeError(
                "spiffe is not installed. Install it with: "
                "pip install spiffe>=0.3.0\n"
                "Or use file-based identity loading instead."
            ) from None

        try:
            client = WorkloadApiClient(self._socket_path, default_timeout=5.0)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to connect to SPIFFE Workload API at {self._socket_path!r}: {exc}"
            ) from exc

        try:
            svid = client.fetch_x509_svid()
            bundles = client.fetch_x509_bundles()

            from cryptography.hazmat.primitives import serialization

            svid_pem = b"".join(
                certificate.public_bytes(serialization.Encoding.PEM)
                for certificate in svid.cert_chain
            )
            key_pem = svid.private_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
            spiffe_id = str(svid.spiffe_id)

            bundle_certs: list[bytes] = []
            for bundle in bundles.bundles:
                for certificate in bundle.x509_authorities:
                    bundle_certs.append(certificate.public_bytes(serialization.Encoding.PEM))

            return WorkloadIdentity(
                spiffe_id=spiffe_id,
                svid_pem=svid_pem,
                private_key_pem=key_pem,
                trust_bundle=bundle_certs,
            )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"SPIFFE Workload API fetch failed: {exc}") from exc
        finally:
            try:
                client.close()
            except Exception as exc:
                _LOGGER.warning("SPIFFE Workload API client close failed: %s", exc)

    @staticmethod
    def _split_pem_bundle(bundle_pem: bytes) -> list[bytes]:
        """Split a concatenated PEM bundle into individual certificates."""
        certs: list[bytes] = []
        current: list[bytes] = []
        for line in bundle_pem.split(b"\n"):
            current.append(line)
            if line.strip() == b"-----END CERTIFICATE-----":
                certs.append(b"\n".join(current))
                current = []
        return certs
