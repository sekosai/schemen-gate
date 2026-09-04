# Security engineering and review gates

Schemen Gate treats security review as an executable maintenance discipline,
not as a complexity score or a claim that one audit found every defect. This
document defines the controls that apply to the importable package. Research
experiments retain their recorded methods and results; they are not rewritten
merely to satisfy a source-shape metric.

## Trust-boundary review scope

Changes to the following behavior require an independent security review and a
permanent regression test:

- Cargo manifest admission, replay handling, finite operation scope, receipt
  fulfillment, and result binding;
- certificate parsing, path construction, verifier-owned trust anchors, name
  constraints, EKU and Key Usage evaluation;
- CRL and OCSP authentication, freshness, scope, responder authority, and
  revoked-result handling;
- grant membership, recipient binding, delegation, release identity, or
  attestation verification; and
- cryptographic canonicalization, AAD, key derivation, wrapping, or signature
  selection.

The review must identify the exact version and commit, affected invariant,
positive case, denial cases, and executable test evidence. A refactor is not
accepted on complexity reduction alone: behavior tests, type checks, package
complexity checks, and the complete release gate must still pass.

## Decision inventory

The detailed claim mapping lives in
[`CLAIM_TEST_MATRIX.md`](CLAIM_TEST_MATRIX.md). The minimum trust-boundary
decision inventory is:

| Surface | Accept only when | Required denials |
|---|---|---|
| Cargo manifest | identity, Regime, partition, model, policy, operation, payload, time, and replay state match | missing, expired, replayed, cross-partition, wrong-operation, malformed, or mismatched manifests |
| Cargo receipt | receipt authentication and every expected manifest and material-result field match | altered content, embedding, query, manifest, partition, subject, model, policy, or operation |
| Certificate path | every link verifies, the path terminates at a verifier-selected root, validity and CA constraints hold, and supported name/EKU/Key Usage semantics permit the purpose | self-nominated root, issuer/signature mismatch, invalid time, path-length breach, disjoint purpose, violated name constraint, or unsupported path semantics |
| CRL | issuer identity and signature, AKI, distribution scope, reason coverage, freshness, and CRL-signing authority hold | wrong issuer/key, bad signature, stale/future response, unsupported critical semantics, partial coverage, or revoked serial |
| OCSP | request identity, responder authority, signature, purpose, validity, freshness, and GOOD status hold | mismatched request, ambiguous/unauthorized responder, missing delegated-responder contract, bad signature, stale/future response, UNKNOWN, or REVOKED |
| Grant and attestation | exact signed membership, recipient, scope, signer, release, operation, target, status, and time contract match verifier-owned expectations | embedded trust substitution, altered signed fields, wrong recipient/root/release, expired scope, or incomplete execution contract |

`RevocationCheck.ENFORCE` denies when no fresh authenticated revocation answer
is available. `WARN` may return an indeterminate warning, but an authenticated
revoked result is always denied. `SKIP` is an explicit operator policy, never a
fallback selected by a certificate or remote endpoint.

## Enforced source-shape boundary

Every function in `src/schemen_gate` must remain at or below a McCabe complexity
of 20:

```bash
python -m ruff check \
  --select C901 \
  --config 'lint.mccabe.max-complexity=20' \
  src/schemen_gate
```

This check runs in CI and the local release gate. The ceiling is a reviewability
control, not evidence of semantic correctness. A smaller incorrect function is
still incorrect; decision-table tests and the full release verification remain
mandatory.

## Existing executable evidence

- `tests/test_authority_and_cargo_integrity.py` freezes finite Cargo operation,
  exact manifest, exact signed-grant membership, and pre-adapter denial behavior.
- `tests/test_cargo.py` covers AAD, expiry, replay, concurrency, completion, and
  receipt behavior.
- `tests/test_pkcs12.py` covers supported key families and certificate path,
  purpose, name-constraint, and external-root decisions.
- `tests/test_x509_path_and_revocation.py` covers certificate-path, CRL, OCSP,
  Cargo receipt, and completion-evidence regressions.
- `tests/test_security_hardening.py` retains malformed, stale, tampered,
  revoked, ambiguous, and wrong-authority inputs as regression fixtures.
- `scripts/release_check.py` runs lint, the package complexity ceiling, strict
  typing, package and research tests, Lean checks, the tracked-source build,
  distribution verification, isolated installation, and quickstarts.

## Supply-chain and deployment boundary

The publication scanner rejects credentials, machine-local residue, and private
product names across tracked text and extracted PDF content. One reviewed
product-boundary section in the root README is pinned by SHA-256 in
`scripts/public_hygiene.py`. Only its companion-name checks receive that
exception; credential and other hygiene checks still inspect every byte.
Changed prose, duplicate sections, mentions elsewhere, private artifact paths,
and private source remain outside the exception. Updating the pinned digest
requires reviewing the complete replacement section for public disclosure.

The release manifest, history verifier, clean tracked-source build, wheel/sdist
membership verifier, signed commit/tag contract, and GitHub release-attestation
workflow are documented in [`PROVENANCE.md`](PROVENANCE.md) and
[`RELEASE_ATTESTATION.md`](RELEASE_ATTESTATION.md). GitHub-hosted secret
scanning, dependency review, branch protection, environment reviewers, and
CODEOWNERS enforcement depend on separately administered repository settings.
Source checks do not attest to those settings. For consumer verification and
issue reporting, see the [maintenance guide](POST_RELEASE_CHECKLIST.md).

The open-source package does not emit production deployment or incident
telemetry, so repository commit counts must not be presented as DORA deployment
metrics. Operators that need DORA evidence must bind source commits to their
own deployment and incident records.

## Deliberate limits

The current deterministic malformed-input corpus is replayable and release
blocking. It is not a claim of exhaustive parser fuzzing, mutation coverage, or
complete bypass closure. Those controls may add useful evidence later, but
their absence must not be disguised by a generic scanner badge or an unbounded
security claim.
