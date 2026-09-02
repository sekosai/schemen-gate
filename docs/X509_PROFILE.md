# X.509 and PKCS#12 interoperability profile

Schemen Gate does not choose a certificate authority. The verifier supplies the
SHA-256 fingerprints of the trust anchors it accepts. A public CA, enterprise
CA, lab CA, or explicitly pinned self-signed certificate is acceptable when it
conforms to this profile and the operator accepts its identity and custody
practices.

## Portable authority credential

`Pkcs12KeyProvider` loads a PKCS#12/PFX credential using `cryptography`'s
standards-based parser. It requires one private key, one matching leaf
certificate, and an unambiguous ordered path to the terminal trust anchor.

Supported Gate authority keys and signature identifiers:

| PKCS#12 leaf key | Gate authority signature |
|---|---|
| Ed25519 | `Ed25519` |
| Ed448 | `Ed448` |
| ECDSA, curve size at least 256 bits | `ECDSA-SHA256` |
| RSA, modulus at least 2048 bits | `RSA-PSS-SHA256` |

The private key must match the leaf certificate's SubjectPublicKeyInfo exactly.
The reference provider loads it into process memory. A TPM, HSM, Secure
Enclave, CNG, Keychain, PKCS#11, Vault, or cloud-KMS integration should
implement `KeyProvider` and retain its native non-exportable handle.

## Certificate-path signature support

Certificate, CRL, and OCSP signatures accept:

- Ed25519 and Ed448;
- ECDSA with a curve size of at least 256 bits and SHA-256, SHA-384, SHA-512,
  SHA3-256, SHA3-384, or SHA3-512; and
- RSA with a modulus of at least 2048 bits, PKCS#1 v1.5 or RSA-PSS parameters,
  and one of those hash functions. RSA-PSS MGF1 must also use one of those
  approved hash functions.

The terminal certificate may be a self-signed root or a non-self-signed CA
certificate whose exact fingerprint the verifier configured as a trust anchor.
The latter supports enterprise deployments that deliberately pin an issuing or
policy CA without packaging its offline parent.

Untrusted certificate paths are bounded to 16 chain members and 64 KiB per PEM
certificate before parsing. The bound caps signature/path work and makes the
ancestor Name Constraints walk finite.

## Enforced path properties

Gate validates:

- issuer/subject linkage and every certificate signature;
- certificate validity at verification time;
- CA `BasicConstraints`, path length, and `keyCertSign` when Key Usage exists;
- leaf `digitalSignature` when Key Usage exists;
- a leaf Extended Key Usage containing client authentication, code signing, or
  any Extended Key Usage when EKU exists;
- a recipient leaf Extended Key Usage containing client authentication or any
  Extended Key Usage when EKU exists;
- CA Extended Key Usage restrictions by intersecting purposes across the whole
  path, so disjoint leaf/CA purposes cannot pass independently;
- permitted and excluded DNS, email, URI-host, IP, and directory-name subtrees
  from CA Name Constraints;
- supported critical extensions;
- the terminal trust-anchor fingerprint against the verifier-owned set; and
- the selected CRL/OCSP policy when revocation checking is requested.

Noncritical certificate-policy metadata is accepted. Policy mapping, policy
constraints, and inhibit-any-policy remain fail-closed because Gate does not
silently ignore policy-tree semantics it has not processed. Name-constraint
forms without defined Gate matching semantics also fail closed. An unknown
critical extension is always rejected.

This is broad issuer interoperability with an explicit, testable validation
profile. It is not a claim that every certificate ever issued for an unrelated
purpose is automatically a Gate authority credential.

## Revocation and recipient identity profile

An OCSP response signed directly by the certificate issuer needs no separate
responder credential. A delegated responder must have a currently valid
issuer-signed certificate, the `id-kp-OCSPSigning` Extended Key Usage, digital
signature Key Usage when that extension is present, and a noncritical
`id-pkix-ocsp-nocheck` extension. Gate rejects a delegated response without
that explicit issuer contract rather than silently assuming the responder is
not revoked. It does not recursively discover or validate a second revocation
path for the responder.

Recipient-certificate revocation is meaningful only after the recipient path
has been authenticated. `verify_grant_provenance(..., revocation=ENFORCE)`
therefore requires an ordered issuer path anchored in a fingerprint committed
by `lockbox.trusted_recipient_cas`. `WARN` logs an indeterminate result when no
recipient CA policy exists; `SKIP` is the explicit choice for a fingerprint-only
or separately managed recipient identity policy. A caller-supplied chain never
becomes trusted merely because it was supplied.

The SPIFFE convenience wrappers enforce the same boundary. The trust-bundle
argument is treated as an ordered SVID issuer path, and its terminal root must
match a recipient CA fingerprint already committed in the signed lockbox.
Authority-signing roots remain a separate verifier-owned trust store.

## Empirical coverage

`tests/test_pkcs12.py` executes:

- Ed25519, Ed448, ECDSA, and RSA authority credentials;
- mixed-key certificate paths;
- RSA-PSS-signed certificates;
- a verifier-pinned non-self-signed enterprise anchor;
- client-auth EKU and noncritical certificate policies;
- CA EKU restrictions and permitted/excluded Name Constraints;
- wrong-password, wrong-root, and incompatible-EKU denials; and
- complete PKCS#12 signing, lockbox verification, and external root selection.

`tests/test_x509_path_and_revocation.py` and
`tests/test_security_hardening.py` cover path length, Key Usage, critical
extensions, CRL/OCSP authenticity and scope, controlled retrieval, wrong
issuers, delegated-responder authorization, recipient-path authentication,
SPIFFE root mismatch, revocation, and malformed objects.
