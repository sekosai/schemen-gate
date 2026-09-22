"""Bounded RFC 5280 name matching for the supported Gate X.509 profile."""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit

from cryptography import x509
from cryptography.x509.oid import NameOID

_DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", re.ASCII)
_CASE_IGNORE_DIRECTORY_OIDS = frozenset(
    {
        NameOID.COMMON_NAME,
        NameOID.SURNAME,
        NameOID.SERIAL_NUMBER,
        NameOID.COUNTRY_NAME,
        NameOID.LOCALITY_NAME,
        NameOID.STATE_OR_PROVINCE_NAME,
        NameOID.STREET_ADDRESS,
        NameOID.ORGANIZATION_NAME,
        NameOID.ORGANIZATIONAL_UNIT_NAME,
        NameOID.TITLE,
        NameOID.GIVEN_NAME,
        NameOID.INITIALS,
        NameOID.GENERATION_QUALIFIER,
        NameOID.DN_QUALIFIER,
        NameOID.PSEUDONYM,
    }
)


def _dns_host(value: str, *, label: str) -> str:
    """Require a DNS host, refusing IP literals and unsupported host syntax."""
    host = value.removesuffix(".").lower()
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ValueError(f"{label} must be a DNS host, not an IP address")
    if len(host) > 253 or any(_DNS_LABEL.fullmatch(part) is None for part in host.split(".")):
        raise ValueError(f"{label} must be a DNS host")
    return host


def uri_within_constraint(uri: str, constraint: str) -> bool:
    """URI constraints without a leading dot denote one exact DNS host."""
    if any(ord(character) <= 32 or ord(character) == 127 for character in uri):
        raise ValueError("URI subject name contains unsupported whitespace or controls")
    host = urlsplit(uri).hostname
    if host is None:
        raise ValueError("URI subject name has no host for NameConstraints")
    host = _dns_host(host, label="URI subject name")
    subtree = constraint.startswith(".")
    rule = _dns_host(constraint[1:] if subtree else constraint, label="URI name constraint")
    return host.endswith(f".{rule}") if subtree else host == rule


def _directory_attribute(attribute: x509.NameAttribute[str | bytes]) -> tuple[str, str]:
    """Apply the RFC 4518 case-ignore ASCII subset, refusing other profiles.

    Unicode StringPrep is deliberately unsupported here. A partial Unicode
    normalization would make excluded subtrees unsafe, so all non-ASCII text
    and encodings or attribute matching rules outside this profile fail closed.
    """
    encoding = attribute._type.name
    if attribute.oid == NameOID.DOMAIN_COMPONENT:
        supported = encoding == "IA5String"
    else:
        supported = attribute.oid in _CASE_IGNORE_DIRECTORY_OIDS and encoding in {
            "PrintableString",
            "UTF8String",
        }
    value = attribute.value
    if not supported or not isinstance(value, str):
        raise ValueError("DirectoryName constraint uses an unsupported attribute matching profile")
    if not value or any(not 32 <= ord(character) <= 126 for character in value):
        raise ValueError("DirectoryName constraints support printable ASCII attribute values only")
    # For this ASCII subset, case folding and insignificant ordinary spaces
    # are the only RFC 4518 transformations that change the comparison value.
    return attribute.oid.dotted_string, " ".join(value.lower().split())


def _directory_rdns(name: x509.Name) -> tuple[tuple[tuple[str, str], ...], ...]:
    return tuple(
        tuple(sorted(_directory_attribute(attribute) for attribute in rdn)) for rdn in name.rdns
    )


def directory_within_constraint(name: x509.Name, constraint: x509.Name) -> bool:
    """Compare ordered RDN prefixes with unordered attributes within each RDN."""
    candidate_rdns = _directory_rdns(name)
    constraint_rdns = _directory_rdns(constraint)
    return candidate_rdns[: len(constraint_rdns)] == constraint_rdns
