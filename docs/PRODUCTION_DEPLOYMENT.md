# Production deployment contract

Schemen Gate 1.0.2 is a production-ready **library boundary**, not a complete
serving platform. A deployment is conforming only when the operator owns every
control below and has tested the denial path at the actual execution boundary.

## Required trust path

```text
independent trust root
  -> certificate validation and external AuthN
  -> signed, scoped, expiring grant
  -> subject/model/operation/Regime AuthZ
  -> Gate before protected execution or release
  -> durable redemption and audit evidence
```

The credential being verified never selects its own trust root. Configure root
fingerprints independently, distribute them through the organization's trusted
configuration channel, and fail closed when a chain, signature, scope, clock,
or required field cannot be verified.

Gate has no built-in CA bundle and imposes no preferred issuer. The operator
may supply fingerprints for any administrator-approved X.509 roots supported by
the configured cryptographic runtime, including a private enterprise CA or an
explicitly pinned self-signed machine root. Gate validates against that set; it
does not decide whether the operator chose, issued, protected, or distributed
those roots wisely. The production claim is conditional on the selected machine
trust root being trustworthy.

## Key custody

- Use `Pkcs12KeyProvider` when an exportable PKCS#12/PFX credential is an
  accepted classical IT boundary. Load its password through the deployment's
  secret facility; never commit, log, echo, or place it in command history.
- The reference provider deserializes its Ed25519, Ed448, ECDSA, or RSA private
  key into process memory. Use a native `KeyProvider` for non-exportable HSM, TPM, Secure
  Enclave, CNG, or cloud-KMS signing. Do not claim hardware custody without
  platform-specific evidence.
- Separate issuer signing keys, Gate verification roots, model/runtime
  credentials, and audit-store credentials. Compromise of one must not silently
  mint all others.
- Define rotation, revocation publication, expiry, recovery, and emergency root
  replacement before launch.

## Revocation contract

- Every trusted verification call must choose `RevocationCheck.ENFORCE`,
  `WARN`, or `SKIP` explicitly. Use `ENFORCE` for production leaf credentials.
- CRL and OCSP results are accepted only after issuer, signature, certificate
  identifier, responder authorization, freshness, status, and applicable CRL
  distribution-point/reason-scope checks.
- A delegated OCSP responder must carry the issuer-signed OCSPSigning purpose
  and a noncritical OCSP No Check extension. Gate otherwise rejects the
  response because responder revocation has not been established.
- Recipient revocation under `ENFORCE` requires a recipient chain authenticated
  to a fingerprint in the signed lockbox's `trusted_recipient_cas`. An
  unpinned caller-supplied chain is evidence to reject, not a trust source.
- Gate performs no certificate-directed network egress. ENFORCE and WARN
  deployments that retrieve CRL or OCSP data must provide a bounded fetcher
  backed by an approved local revocation service or equivalently controlled
  transport. That component owns DNS, redirects, proxies, TLS peer validation,
  destination-address checks, and egress policy.
- Configure the fetcher, byte/time limits, and exact responder-host allowlist in
  one `RevocationPolicy` passed to the trusted verification call. Gate still
  validates endpoint syntax and the returned signed revocation object.
- `WARN` may admit an indeterminate answer but never an authenticated revoked
  certificate. Alert on every warning. `SKIP` is for an explicit offline or
  separately managed trust-anchor policy, not an implicit fallback.

## Authority contract

Every production grant must bind at least:

- issuer and authenticated subject;
- tenant and Regime;
- model identity and immutable version/digest;
- operation and resource scope;
- policy version and purpose context;
- issuance and expiry time;
- exact partition or attachment identity when applicable; and
- replay identifier or finite-use contract when the action is not idempotent.

Never accept a caller-provided `regime_id`, mask, model, corpus, adapter, tool,
or operation as authority. Resolve them from the verified grant.

## Placement and bypass closure

- Apply the Gate before the protected operation, resource release, model
  attachment, shard decryption, or declared activation path.
- Enumerate every alternate route to the same resource: debug endpoints,
  batch jobs, caches, maintenance commands, direct database access, fallback
  models, and administrative tools.
- Deny unknown and partially resolved states. A missing authority service is
  not an allow condition.
- Treat shared attention, residuals, caches, adapters, optimizer state, and live
  host memory as shared unless each is separately governed and tested.
- Keep cross-process replay and exact-use redemption in an atomic, durable
  serving store. The library's in-process protections are not a distributed
  ledger.
- Cargo loads require a store whose `insert_many` implementation is
  all-or-nothing. The bundled PostgreSQL adapter uses one database transaction,
  serializes all work on its shared psycopg connection across complete
  transactions and queries, and requires the server-side pgvector extension.
  The in-memory adapter commits a prepared batch under one lock. A custom store
  without that guarantee is not a conforming Cargo store.
- A custom vector store is untrusted input to the library. It must return one
  ordered list no longer than `top_k`, preserve the exact requested partition
  and optional kind, and supply nonempty unique IDs, bounded strict-JSON
  metadata, finite scores, and finite vectors with the registered dimensions.
  For a load with explicit signed document IDs, the returned IDs must match at
  the same item positions. Gate revalidates those properties and rejects the
  complete result on any mismatch. It never substitutes a synthetic vector for
  malformed storage.
- A bridged vector payload is gated in its authenticated source embedding space
  before the bridge projection. Do not implement projection-before-gating in a
  custom path: the target-space coordinates need not correspond one-for-one to
  the signed source-space support.
- Verify every Cargo receipt against the exact expected manifest. Signature-only
  verification is not a scope decision. The default bus accepts only TTL expiry
  as a completion condition; other completion kinds require a separately
  implemented trusted evidence provider and otherwise fail closed.
- Issue Cargo manifests with the least operation needed: `load` for ingestion,
  `retrieve` for retrieval, and `load_and_retrieve` only when one transaction
  genuinely requires both. Unknown operations and method crossings fail before
  the storage adapter is called. A manifest that declares a load cannot depart
  successfully until the exact signed payload has completed once.
- Call `verify_grant_provenance()` on the exact grant being consumed. It accepts
  a serialized/reconstructed copy only when every canonical field matches a
  grant committed by the authority-signed lockbox; do not treat a valid
  lockbox signature as authority for a separately supplied grant.

## Minimum production tests

Run these against the deployed boundary, not only against unit-test fixtures:

1. correct root, subject, scope, model, operation, Regime, and time succeeds;
2. wrong root and self-nominated root fail;
3. wrong subject, tenant, model digest, operation, Regime, or partition fails;
4. expired, not-yet-valid, revoked, malformed, and tampered grants fail;
5. a grant copied from another signed lockbox and every changed nested grant
   field fail provenance;
6. Cargo `load`/`retrieve` method crossings fail without a store call;
7. replay beyond the allowed use count fails atomically across replicas;
8. authority-service, clock, audit-store, and policy-store failures deny;
9. every known alternate path fails without the same authority decision;
10. rollback restores the previous known-good binary and policy without
   widening authority.

The repository tests cover the library contracts and negative cases. They do
not prove that an operator's network, host, CA, runtime, or bypass inventory is
correct.

## Observability and evidence

Record a non-secret decision receipt containing the request/correlation ID,
subject fingerprint, issuer, grant ID, model digest, operation, Regime, policy
version, decision, reason code, `GateReleaseIdentity` (including the full
source commit), deployment version, and timestamp. Never log
private keys, PKCS#12 passwords, bearer credentials, decrypted shards, or raw
sensitive payloads.

Alert on signature failures, unknown roots, replay, clock skew, repeated scope
mismatch, revocation-store failure, policy rollback, and unexpected bypass-path
traffic. Define retention and access policy for the receipts before production.

## Rollout sequence

1. Verify the source manifest and package checksums.
2. Verify GitHub artifact provenance and require the wheel's
   `current_release_identity()` to match the approved version, repository, and
   full source commit.
3. Install the exact wheel in an isolated image; do not resolve an unpinned
   fallback package.
4. Run local positive and negative examples.
5. Deploy one non-production canary with the same authority and denial path.
6. Exercise every minimum production test above.
7. Promote to a small traffic slice with fail-closed monitoring.
8. Expand only while decision, denial, latency, and error SLOs remain green.
9. Retain an immediate binary and policy rollback path.

## Release verification

From the repository root, with the `dev` and production extras installed:

```bash
python -m pip install -e '.[crypto,lockbox,onnx,rag,spiffe,torch,dev]'
python scripts/bootstrap_build_env.py
python scripts/release_check.py
```

The bootstrap downloads only hash-locked universal build wheels. The checker
then builds offline from a tracked export and checks the tracked-tree manifest,
lint, the full Gate test suite, the research release bundle, the pinned Lean
project, exact wheel/sdist membership and source bytes, and clean-wheel
execution of all three quickstarts: NumPy, PKCS#12, and certificate-to-Gate.
The complete check requires `lake` from an elan/Lean installation. Passing
`--skip-lean` produces only a partial local check and is not complete release
evidence. The checker verifies tracked paper artifacts but does not rebuild
them; rebuilding all four PDFs requires a TeX distribution plus `latexmk` and
`make -C research/cdp paper`. None of these commands publishes, deploys,
pushes, or contacts a production service.

## Production claim

A conforming deployment may say:

> Schemen Gate 1.0.2 provides a tested, fail-closed cryptographic AuthN/AuthZ
> library boundary for explicitly scoped model operations and declared Gate
> placements, conditional on the deployment's CA, key custody, durable replay,
> runtime integrity, and bypass-closure controls.

It must not say that the library alone makes an AI system, host, network, or
model universally secure.
