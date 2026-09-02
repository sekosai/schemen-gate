# Schemen Gate roadmap

This roadmap separates release gates from planned features. An unchecked item
is not shipped behavior and must not appear in launch claims.

## Public-release gates

- [x] The Apache-2.0/CC BY 4.0 path map and adoption-first patent notice are
  recorded in the repository.
- [x] A private staging lineage is maintained with a tracked-tree manifest,
  credential/excluded-material scanning, and reproducible release checks.
- [x] The complete local 1.0.2 release gate passes from a clean candidate:
  library and research tests, lint, typing, Lean proofs, research validation,
  tracked-export wheel and sdist verification, clean-wheel installation, and
  all three quickstarts.
- [x] Independent security findings were converted into permanent regression
  tests and all actionable findings in the reviewed candidate were remediated.
- [ ] Transfer the final reviewed tree into a fresh, single-root public
  repository and verify that no development history, archival refs, reflogs, or
  recoverable predecessor objects crossed the boundary.
- [ ] Publish only the audited `main` lineage; exclude archival refs and
  inherited development history from the public repository.
- [ ] Required CI, branch protection, secret scanning, dependency review,
  private vulnerability reporting, and maintainer two-factor authentication are
  enabled.
- [ ] `schemen-gate` is published from a reviewed tag with provenance, source
  archive, wheel, checksums, and clean-index install tests.
- [ ] Gate quickstarts, package tests, Lean proofs, paper sources, and the
  research release checker pass in GitHub CI from the exact public 1.0.2 tag.
- [ ] Record one independent Modal CPU canary from the release tag.
- [x] The operator execution boundary is explicitly excluded from the library
  claim unless a downstream integration supplies and tests it.
- [x] The public Python API and schema-versioning policy are explicit for the
  1.x line.

## First 30 days

- [ ] Label a small set of bounded `good first issue` and `help wanted` tasks.
- [x] Publish one certificate-to-denial end-to-end tutorial for local Gate use.
- [ ] Publish one operator-owned integration example.
- [ ] Record three independent Modal canary completions from outside the core
  maintainer environment.
- [ ] Invite external technical review of the authority and bypass boundary;
  treat resulting findings as engineering input, not as a prerequisite for the
  library's documented, executable production contract.
- [ ] Merge two substantive external integrations or pull requests.
- [x] Publish the tested X.509/key-profile matrix.
- [ ] Publish the release-tag compatibility matrix for supported Python, NumPy,
  Torch, and Modal versions after final CI completes.

## Next integrations

- public examples for a frozen-backbone FFN lifecycle and a private-adapter
  lifecycle, each with owning and wrong-authority negatives;
- platform-native non-exportable `KeyProvider` adapters, each with explicit
  identity, expiry, revocation, replay, and hardware-evidence contracts;
- deployment recipes whose claims are tied to pinned images, configuration,
  artifacts, and captured evidence; and
- reproducible benchmarks for Gate overhead, capacity scaling, and deployment
  consolidation; and
- a compatibility-preserving split of the internal X.509/revocation,
  lockbox-model, serialization, secret-sharing, and ONNX-attestation modules,
  guarded by the existing public API, golden serialization, and adversarial
  regression suites.

## Success measures

Measure completed user outcomes: clean installs, successful examples, Modal
canaries, independent reproductions, external integrations, substantive issues
and pull requests, and qualified design-partner conversations. Do not use stars
or impressions as proof of adoption, and track any uncorrected claim
overstatement as a release-quality failure.
