# Gate API and identity guide

[Back to Schemen Gate](../README.md)

Run commands from the repository root. Start with the
[certificate-to-Gate quickstart](../README.md#two-minutes-to-working-ai-pki)
before using these individual primitives.

## Use a PKCS#12 machine identity

The lockbox extra includes `Pkcs12KeyProvider`, which consumes an already
hydrated `.p12`/`.pfx` credential containing an Ed25519, Ed448, ECDSA, or RSA
private key, leaf certificate, and certificate chain to the trust anchor:

```python
from schemen_gate import (
    Pkcs12KeyProvider,
    RevocationCheck,
    fingerprint_from_x509,
    verify_authority,
)

provider = Pkcs12KeyProvider.from_bytes(pkcs12_bytes, pkcs12_password)
lockbox.authority = provider.sign_lockbox(lockbox)

# Trust policy is user-owned and independent of the credential being verified.
# Supply fingerprints for whichever CA roots your organization has approved.
trusted_roots = [fingerprint_from_x509(approved_root_certificate_pem)]
verify_authority(
    lockbox,
    trusted_roots,
    revocation=RevocationCheck.ENFORCE,
)
```

Gate has no preferred issuer and no default CA bundle. Changing
`trusted_roots` changes the verifier's trust decision; it does not change the
Gate algorithm or require a vendor integration. The presented certificate
chain is evidence to validate, never authority to extend `trusted_roots`.

The revocation policy is mandatory: verification never silently chooses
`SKIP`. A deliberately pinned self-signed root has no issuer revocation
service, so an offline fixture may explicitly select `RevocationCheck.SKIP`;
production leaf credentials should normally use `ENFORCE` with authenticated,
fresh CRL or OCSP data. Gate performs no certificate-directed network egress
and does not ship a network fetcher. It validates the certificate's initial
endpoint syntax, rejects literal non-global destinations, applies the exact
responder-host allowlist, and rejects an oversized returned body. The required
operator-supplied fetcher owns DNS resolution, every redirect, proxy behavior,
TLS peer and connected-address validation, timeouts, response streaming limits,
and egress policy. A locked-down deployment can call
`check_certificate_revocation` with a bounded fetcher backed by an approved
local revocation service. Pass a
`RevocationPolicy` to `verify_authority`, `load_lockbox`, or the model/grant
verifiers to carry that fetcher, timeout, byte cap, and exact responder-host
allowlist through the complete trusted operation.

Issuer-signed OCSP responses are accepted directly. A delegated OCSP responder
must present an issuer-signed, currently valid OCSPSigning certificate with a
noncritical OCSP No Check extension; otherwise Gate cannot establish the
responder's own revocation contract and fails closed. Recipient revocation also
requires an authenticated issuer path anchored in `trusted_recipient_cas`.
Supplying an unpinned chain never makes it trusted. Use `SKIP` explicitly when
the operator has deliberately chosen a fingerprint-only or separately managed
recipient policy.

Run the self-contained, ephemeral example with:

```bash
python examples/pkcs12_identity.py
```

PKCS#12 is the portable credential interface, not a claim of chip residency.
The reference provider loads the signing key into process memory. An
organization may instead implement `KeyProvider` with its TPM, HSM, native
keystore, or cloud KMS when a non-exportable key handle is required.
The exact algorithms, extension handling, and tested path shapes are listed in
[`docs/X509_PROFILE.md`](../docs/X509_PROFILE.md).

## First working gate

In production, an authority or trusted runtime resolves the regime. The caller
must not be allowed to choose an arbitrary mask or regime identifier.

```python
import numpy as np
from secrets import token_bytes

from schemen_gate import GateMask

# Trusted authority-side derivation. Keep this key out of request payloads,
# logs, model artifacts, and untrusted training workers.
root_key = token_bytes(32)
mask = GateMask.derive(
    key=root_key,
    regime_id=0,
    n_dims=8,
    n_regimes=2,
)

hidden = np.arange(8, dtype=np.float64)
gated = mask.apply(hidden)

assert mask.active_dims == 4
assert np.all(gated[mask.mask == 0] == 0)
assert np.array_equal(gated[mask.mask == 1], hidden[mask.mask == 1])
```

Most training workers should receive an exported mask rather than the root
key:

```python
from schemen_gate import GateMask

mask = GateMask.from_file("masks/regime_0.npy")
gated = mask.apply(hidden)
```

`GateMask` retains a private, read-only copy and rejects non-binary values.

## Other primitives

The package is broader than `GateMask`:

- [Capability delegations](../tests/test_capability.py) bind separate
  policy-signing and runtime-result
  identities. Enforced consumers must use `verify_enforced_delegation`; the
  historical v1 `derive_policy_key(delegation.signature)` path is forgeable
  because the signature is public.
- [Lockboxes](../examples/cotrained_shard_lockbox/README.md) wrap
  recipient-specific grants and support signed provenance, revocation state,
  and scoped key release. Consumer provenance verification first proves that
  the presented grant is an exact canonical member of the authority-signed
  lockbox; a detached or cross-lockbox grant cannot borrow that authority.
- [Cargo manifests](../tests/test_cargo.py) authorize vectors or context before
  release to an ordinary model. Client keys and manifest AAD bind tenant,
  subject, regime, model, operation, policy version, and the exact partition;
  changing any one of them fails closed. The operation is a finite capability:
  `load`, `retrieve`, or the explicit `load_and_retrieve` union. Each session
  method checks that capability and the live partition-to-Regime binding before
  it calls the storage adapter. Load payloads must match the signed payload
  family, item count, embedding dimensions, finite-number requirements, and
  bounded strict-JSON representation. A declared load must complete exactly
  once before the bus can issue a success receipt. If the signed items declare
  document IDs, the store must return those exact IDs in the same positions;
  replacement IDs fail before completion is receipted.
  Receipt keys and verification bind the same complete expected manifest.
  Retrieval receipts hash the resolved query, request contract, ordered returned
  records, and returned vector material. Untrusted store results are revalidated
  for exact partition, requested kind, dimensions, finite scores/vectors, and
  bounded metadata before release. When a `VectorBridge` is used, the Gate is
  applied in the authenticated source space before projection; public vectors,
  masks, metadata, and provenance identifiers are detached copies.
  Session transitions are serialized, and Cargo loads require an all-or-nothing
  `insert_many` store operation so a partial write cannot produce a misleading
  receipt. The PostgreSQL adapter also serializes complete transactions and
  queries on its shared connection. The default bus signs only completion state
  it can evaluate itself (currently TTL expiry); caller-signaled completion
  kinds fail closed.
  Cross-process replay prevention belongs in a durable serving store.
- [Exact finite-operation gates](../tests/test_operation_gate.py) authenticate a
  complete native transition contract and redeem once. They do not infer the
  semantics of user-defined symbols or grant learned execution authority.
- [Fail-closed defaults](../docs/FAIL_CLOSED_DEFAULTS.md) require verifier-owned
  signer keys, finite default lifetimes, explicit Gate and storage scope,
  exact wire metadata, and exact-host revocation policy. Integrity-only
  inspection is available only through APIs named `*_self_consistency`.
- [Gated RAG](../tests/test_security_hardening.py) controls partition
  ingest/retrieval. It supports only `CachePolicy.NONE` and deliberately has no
  model-training or absorption method because an arbitrary optimizer cannot be
  proven support-restricted. The downstream generator and any audited training
  loop remain separate capability and trust boundaries.
- [`fold_vector`](../src/schemen_gate/_regime0_fold.py) is a lossless row codec,
  not a Gate or storage-confinement mechanism. It applies no mask and grants no
  write authority; callers must enforce those boundaries separately.

Public entry points are grouped in
[`schemen_gate.__init__`](../src/schemen_gate/__init__.py). Executable contract
examples are indexed in [`examples/README.md`](../examples/README.md), with
adversarial coverage in [`tests`](../tests/). The supported 1.x public surface,
schema-version boundary, and deprecation rules are defined in
[`docs/API_STABILITY.md`](../docs/API_STABILITY.md).

## `GateMask` reference

| Constructor | Purpose |
|---|---|
| `GateMask.from_file(path, regime_id=...)` | Load an authority-exported `.npy` mask; `regime_id` must be supplied directly or by its JSON sidecar |
| `GateMask.from_indices(indices, n_dims)` | Build an explicit mask for tests or trusted tooling |
| `GateMask.from_numpy(array)` | Validate and copy an existing binary array |
| `GateMask.from_dict(data)` | Reconstruct a JSON-safe export containing explicit `regime_id` |
| `GateMask.derive(key, regime_id, n_dims, n_regimes)` | Derive one equal-width keyed partition using the NumPy-only core |
| `GateMask.public(n_dims)` | Explicit all-ones public-for-all policy; not a tenant-isolation control |
| `GateMask.full(n_dims)` | Backward-compatible alias for `GateMask.public` |

| Method | Purpose |
|---|---|
| `.apply(hidden)` | Element-wise gate for NumPy, Torch, or compatible tensors |
| `.mask` | Return a detached, read-only owning copy of the binary mask |
| `.to_torch(device, dtype)` | Create an independent Torch tensor |
| `.to_numpy()` | Return a writable copy |
| `np.asarray(mask)` | Standard NumPy array protocol; always detached |
| `np.from_dlpack(mask)` | Standard DLPack interchange; exports detached storage |
| `.to_dict()` | Serialize active indices and metadata |
| `.save(path)` | Save `.npy` plus an optional JSON sidecar |
| `mask_a \| mask_b` | Create an explicit union of same-width supports |

## Security and identity FAQ

### What OS facility handles the chip binding underneath?

Gate does not require one particular OS facility. Its portable credential path
is PKCS#12 plus X.509 verification against a verifier-configured root. The
reference `Pkcs12KeyProvider` deserializes its software key into process memory;
it does not claim TPM, Secure Enclave, CNG, Pluton, Titan, confidential-
computing, or GPU-attestation guarantees. A deployment can supply a native
`KeyProvider` when hardware-backed, non-exportable signing is required.

### Which trusted-compute implementation was targeted and tested?

The tested release target is platform-agnostic PKCS#12/X.509 and software
cryptography across Ed25519, Ed448, ECDSA, and RSA authority keys. Certificate
chains may use those supported issuer-key families and may end at either a
self-signed root or an explicitly pinned non-self-signed CA trust anchor. No
vendor-specific trusted-compute backend is part of the current evidence.
Hardware custody is an integration choice at the classical IT boundary, not a
prerequisite for using the Gate.

### Is Schemen Gate authentication or authorization?

Both. The certificate and signature authenticate the external machine or
issuer. The signed scope authorizes a specific Regime. The resolved Regime then
acts as authenticated execution context for downstream Gates that consume the
same cryptographically bound scope.

### What is the security boundary?

The algebraic Gate and its modeled composition properties are proven under the
assumptions named in [`docs/SECURITY_CLAIMS.md`](../docs/SECURITY_CLAIMS.md). In a
deployment, identity assurance is only as strong as the configured roots,
certificate issuance, private-key custody, rotation, revocation, and bypass
closure. Gate brings AI AuthN/AuthZ to that conventional enterprise PKI
boundary; it does not replace the boundary.

### Can I use a self-signed certificate?

Yes. Pin its fingerprint out of band as an explicit trust decision. This proves
possession and continuity of the pinned key. It does not create an independent
third-party identity assertion, and a bundle cannot nominate its own root as
trusted.

### Which CA does Schemen Gate require?

None. The verifier supplies the exact root fingerprints it trusts, and Gate
verifies against that set. Selecting and operating those roots is the user's
responsibility; Gate's software boundary assumes the selected machine root is
trustworthy.
