"""End-to-end PKCS#12 machine-identity example with an ephemeral local root."""

from __future__ import annotations

import hashlib

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization import pkcs12

from schemen_gate import (
    GateKey,
    HierarchyDef,
    Pkcs12KeyProvider,
    RegimeCapability,
    RevocationCheck,
    create_lockbox,
    fingerprint_from_x509,
    generate_authority_cert,
    verify_authority,
)


def main() -> None:
    password = b"ephemeral-example-password"
    root_pem, root_key = generate_authority_cert("local-machine-root")
    credential = pkcs12.serialize_key_and_certificates(
        name=b"local-machine",
        key=root_key,
        cert=x509.load_pem_x509_certificate(root_pem),
        cas=None,
        encryption_algorithm=serialization.BestAvailableEncryption(password),
    )
    provider = Pkcs12KeyProvider.from_bytes(credential, password)

    capability = RegimeCapability(0, "invoke", ["model"])
    lockbox = create_lockbox(
        GateKey(b"g" * 32),
        chain_hash=hashlib.sha256(b"example-model").hexdigest(),
        chain_name="example",
        n_dims=4,
        n_regimes=1,
        hierarchy_def=[HierarchyDef("operator", "invoke model", [0], [capability])],
    )
    lockbox.authority = provider.sign_lockbox(lockbox)
    # This fixture uses one ephemeral self-signed root with no revocation
    # service. Production leaf credentials should use ENFORCE.
    verify_authority(
        lockbox,
        [fingerprint_from_x509(root_pem)],
        revocation=RevocationCheck.SKIP,
    )
    print("PASS: PKCS#12 identity signed to the independently pinned root")


if __name__ == "__main__":
    main()
