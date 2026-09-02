"""AI-PKI quickstart: certificate -> grant -> Regime -> Gate -> denial."""

from __future__ import annotations

import datetime
import hashlib

import numpy as np
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import pkcs12
from cryptography.x509.oid import NameOID

from schemen_gate import (
    GateKey,
    GateMask,
    HierarchyDef,
    Pkcs12KeyProvider,
    RegimeCapability,
    RevocationCheck,
    create_lockbox,
    fingerprint_from_x509,
    generate_authority_cert,
    generate_svid,
    redeem_mask_token,
    seal_grant,
    unseal_grant,
    verify_authority,
    verify_grant_provenance,
)


def _authority_credential(password: bytes) -> tuple[bytes, bytes]:
    """Create an ephemeral two-certificate PKCS#12 authority fixture."""

    root_pem, root_key = generate_authority_cert("example-authority-root")
    root = x509.load_pem_x509_certificate(root_pem)
    leaf_key = Ed25519PrivateKey.generate()
    now = datetime.datetime.now(datetime.timezone.utc)
    leaf = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "example-authority")]))
        .issuer_name(root.subject)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(root_key, None)
    )
    credential = pkcs12.serialize_key_and_certificates(
        name=b"example-authority",
        key=leaf_key,
        cert=leaf,
        cas=[root],
        encryption_algorithm=serialization.BestAvailableEncryption(password),
    )
    return credential, root_pem


def run_demo() -> dict[str, object]:
    """Execute one permitted Gate operation and two cryptographic denials."""

    password = b"ephemeral-example-password"
    authority_bytes, authority_root_pem = _authority_credential(password)
    authority = Pkcs12KeyProvider.from_bytes(authority_bytes, password)

    recipient_root_pem, recipient_root_key = generate_authority_cert("example-recipient-root")
    recipient_cert_pem, recipient_private_key, recipient_public_key = generate_svid(
        "spiffe://example.org/workload/inference-client",
        recipient_root_pem,
        recipient_root_key,
    )

    gate_key = GateKey(b"g" * 32)
    recipient_id = "inference-client"
    capability = RegimeCapability(0, "infer", ["example-model"])
    lockbox = create_lockbox(
        gate_key,
        chain_hash=hashlib.sha256(b"example-model-v1").hexdigest(),
        chain_name="example-model-v1",
        n_dims=8,
        n_regimes=2,
        hierarchy_def=[HierarchyDef("inference", "invoke example model", [0], [capability])],
    )
    lockbox.trusted_recipient_cas = [fingerprint_from_x509(recipient_root_pem)]
    grant = seal_grant(
        lockbox,
        gate_key,
        recipient_id,
        lockbox.hierarchy[0],
        recipient_public_key,
        recipient_cert_pem=recipient_cert_pem,
        recipient_cert_chain_pems=[recipient_root_pem],
    )
    lockbox.authority = authority.sign_lockbox(lockbox)

    authority_roots = [fingerprint_from_x509(authority_root_pem)]
    verify_authority(
        lockbox,
        authority_roots,
        revocation=RevocationCheck.SKIP,
    )
    provenance = verify_grant_provenance(
        lockbox,
        grant,
        recipient_cert_pem,
        authority_roots,
        consumer_cert_chain_pems=[recipient_root_pem],
        revocation=RevocationCheck.SKIP,
    )
    if not provenance.trusted:
        raise AssertionError(f"expected trusted grant: {provenance.reasons}")

    tenant_keys = unseal_grant(
        grant,
        recipient_private_key,
        expected_recipient_id=recipient_id,
    )
    resolved = redeem_mask_token(grant.mask_tokens[0].to_mask_token(), tenant_keys[0])
    mask = GateMask.from_numpy(resolved, regime_id=0)
    hidden = np.arange(8, dtype=np.float64)
    gated = mask.apply(hidden)

    untrusted_root_pem, _ = generate_authority_cert("untrusted-root")
    wrong_root_denied = False
    try:
        verify_authority(
            lockbox,
            [fingerprint_from_x509(untrusted_root_pem)],
            revocation=RevocationCheck.SKIP,
        )
    except ValueError:
        wrong_root_denied = True

    other_cert_pem, _, _ = generate_svid(
        "spiffe://example.org/workload/other-client",
        recipient_root_pem,
        recipient_root_key,
    )
    wrong_recipient = verify_grant_provenance(
        lockbox,
        grant,
        other_cert_pem,
        authority_roots,
        consumer_cert_chain_pems=[recipient_root_pem],
        revocation=RevocationCheck.SKIP,
    )

    if not wrong_root_denied:
        raise RuntimeError("wrong authority root was accepted")
    if wrong_recipient.trusted:
        raise RuntimeError("wrong recipient certificate was accepted")
    if not np.all(gated[mask.mask == 0] == 0):
        raise RuntimeError("inactive Gate coordinates were not zero")
    if not np.array_equal(gated[mask.mask == 1], hidden[mask.mask == 1]):
        raise RuntimeError("active Gate coordinates changed")
    return {
        "authority_verified": True,
        "grant_verified": True,
        "resolved_regime": mask.regime_id,
        "active_dimensions": mask.active_dims,
        "wrong_root_denied": wrong_root_denied,
        "wrong_recipient_denied": not wrong_recipient.trusted,
        "gated": gated.tolist(),
    }


def main() -> None:
    result = run_demo()
    print("PASS: certificate -> signed grant -> resolved Regime -> Gate")
    print("PASS: wrong trust root denied before Gate")
    print("PASS: wrong recipient certificate denied before Gate")
    print(f"resolved regime: {result['resolved_regime']}")
    print(f"gated activation: {result['gated']}")


if __name__ == "__main__":
    main()
