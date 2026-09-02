# Changelog

All notable changes to Schemen Gate are documented here.

## 1.0.2 - Production-ready release candidate

- Continue at the next unused semantic version without moving or reusing an
  existing signed tag for different source.
- Preserve the complete 1.0.0 Gate implementation and its bounded security
  claims while reconciling package, workflow, attestation, documentation,
  research-lock, and distribution contracts on 1.0.2.
- Add the public-safe provisional-patent notice and remove private filing
  identifiers, unpublished claim strategy, and contradictory proprietary
  experiment-language from the open-source tree.
- Finalize the Apache-2.0 and CC BY 4.0 path map, repair machine-readable
  citation metadata, and add a tracked public-hygiene gate for credentials,
  local-machine residue, and private patent identifiers.
- Normalize non-semantic developer-machine paths in retained research records,
  document the affected JSON byte digests, and preserve measurements, source
  revisions, dependency versions, and recorded artifact hashes.
- Regenerate the two canonical learned-routing receipts after public-boundary
  hardening, publish their exact artifact digests, and document the independent
  cross-run comparison without representing those local CPU runs as Modal runs.
- Make every shipped research example directly importable, reconcile the Modal
  and protobuf pins, and validate the complete requirements lock.
- Add a certificate-to-grant-to-Regime quickstart with successful enforcement
  and wrong-root/wrong-recipient denial paths.
- Expand the reference PKCS#12 authority provider and verifier across Ed25519,
  Ed448, ECDSA P-256-or-stronger, and RSA-2048-or-stronger signing keys while
  keeping an explicit, fail-closed X.509 profile.
- Add explicit public-for-all Gate construction and detached NumPy/DLPack array
  protocol adapters.
- Add permanent security regression tests and a claim-to-test matrix.
- Decompose the X.509 path, CRL, and OCSP decision paths into typed,
  independently reviewable helpers without changing their fail-closed
  decisions; enforce a McCabe complexity ceiling of 20 across the importable
  package in CI and the release gate.
- Export only exact tracked Gate bytes to Modal, generate the release identity
  from the clean source commit, and verify that commit again in the container.
- Add a machine-readable release contract and a tag-bound PyPI description;
  keep the full paper and research bundle at the matching Git revision.
- Document Cargo Mode and the separate operator-integration integration boundary
  without making runtime guarantees part of this open-source library.
- Require authenticated recipient issuer paths before revocation checks and
  bind SPIFFE SVID validation to recipient roots committed in the signed
  lockbox.
- Require delegated OCSP responders to present the standard noncritical OCSP
  No Check contract in addition to issuer signature, validity, EKU, and Key
  Usage checks.
- Reject ambiguous lockbox YAML, bound PKCS#12 inputs, validate Gate tensor
  shape before multiplication, and reject mutable or coerced authority fields
  in key and operation-gate wire contracts.
- Remove public audit-diary residue, add local documentation-link verification,
  and add standard community and dependency-maintenance files.
- Raise the `cryptography` minimum to 50.0 so the tested timezone-aware X.509,
  CRL, and OCSP behavior stays on one maintained API generation.
- Pin pytest and its child Python processes to the reviewed checkout so a
  different editable installation cannot satisfy source tests.
- Require verifier-owned signer pins for delegation, runtime, and public
  operation attestations; move integrity-only inspection to explicitly named
  `*_self_consistency` functions.
- Give authority-bearing tokens and rights finite default lifetimes, default
  rights to no permissions, and require a conspicuous opt-in for non-expiring
  adapter and operation credentials.
- Require explicit scope for Gate-configured skill dispatch, explicit Cargo
  storage-gating policy in authenticated manifest schema v7, immutable
  partition registration by default, and explicit RAG ingestion policy.
- Reject missing Regime and stream-frame metadata, bind clean source checkouts
  to their exact Git commit, and require exact-host allowlists for
  network-capable revocation policies.
- Rename SPIFFE's lightweight identity check to
  `validate_spiffe_id_syntax` so it cannot be confused with full certificate,
  key, validity, and trust-domain verification.
- Ship documented public-safe evidence exports for fourteen historical run
  records cited by the result manifest, data inventory, and papers; each
  export preserves every measured value and records the original record's
  SHA-256 and every transformed field. Add research release checks that every
  cited result filename resolves to a tracked artifact and that every export
  is indexed with its original digest.
- Bind a source checkout's release identity only from the canonical
  `src/schemen_gate` layout, so a wheel installed inside an unrelated Git
  repository neither inherits that repository's commit nor fails because it is
  dirty.
- Serialize lockboxes without YAML anchors so a hierarchy that shares Python
  objects between levels can be reloaded by the alias-rejecting strict loader.
- Report malformed lockbox document structure as `ValueError`, propagate
  revoked-certificate results through a typed error instead of matching
  exception text, and serve lazy public exports from one audited table that is
  reconciled against `__all__` by a test.
- Add executable evidence for GF(256) Gate-key sharing, lockbox reissue and
  diagnostics, YAML round trips, malformed-document rejection, and the public
  export table; report package coverage in CI.
- Pin `transformers==5.10.1` in the research reproduction environment and the
  Modal launchers (was 5.9.0, affected by GHSA-xrqw-3rrv-vx5w); the recorded
  results are unchanged.
- Add a live-demo section to the README and `docs/DEMO_WALKTHROUGH.md`, which
  maps every request, denial, and evidence field of the public demo to the
  library call behind it.
- Validate `GateRights` authority fields at construction (exact integers,
  booleans, non-empty issuer, string-keyed metadata, validated release) so a
  coerced value such as a boolean regime cannot be signed; the v1 rights byte
  format is unchanged. Document dependency licenses. Repair the last
  terminology-replacement scars in the paper headings and rebuild the PDFs.
- Format the library, tests, scripts, and examples with `ruff format` and
  enforce that boundary in CI and the release gate (the research bundle keeps
  its recorded source form); make release scripts executable; drop the
  proof-of-concept suffix from six test modules; export the four operation-gate
  schema identifiers; read only the Subject Alternative Name in SPIFFE
  identity extraction so malformed extensions fail closed; add executable
  evidence for retrieval-architecture analysis and SPIFFE extraction.
- Make the research reproduction setup (`research/cdp/scripts/setup.sh test`)
  work from a fresh clone by putting the Gate root on the research test path,
  start the README quickstart with `python3 -m venv`, and make
  `release_check.py --skip-lean` skip the Lean build even when Lake is
  installed.
- Install CPU-only torch from the official PyTorch index before the editable
  install in CI so hosted runners are not filled by the CUDA wheel set that
  PyPI torch 2.14 pulls on Linux.

## 1.0.1 - Historical retained release

- Historical signed version retained privately as immutable evidence and not
  reused for different source; its earlier public tag and release were
  withdrawn before the 1.0.2 line was re-rooted. Ongoing releases continue at
  1.0.2.

## 1.0.0 - Initial release line

- First production release candidate.
- Shipped immutable binary masks, cryptographic capability tokens, lockboxes,
  complete-scope Cargo manifests, exact finite-operation gates, gated retrieval,
  and partition-safe vector storage.
- Enforced externally pinned trust anchors, strict canonical encodings, immutable
  authority state, atomic in-process redemption, and fail-closed scope checks.
- Added a self-contained PKCS#12 Ed25519 authority provider with leaf-to-root
  chain packaging, independent root pinning, negative tests, and an executable
  example.
- Documented the external AuthN, boundary AuthZ, and downstream Regime AuthN
  chain, including pinned self-signed operation and the classical IT PKI
  boundary.
- Made hardware custody explicit: the reference PKCS#12 provider loads its key
  into process memory; non-exportable keys require a native `KeyProvider` and
  separate platform evidence.
- Included the complete reviewed CDP research snapshot: full and focused papers,
  LaTeX sources, 21 Lean modules, Modal launchers, offline experiments,
  machine-readable receipts, provenance, and license mapping.
- Added pinned Modal onboarding, a production deployment contract, distribution
  inspection, a tracked-tree SHA-256 manifest, and a single-command release
  verifier.
- Bound Cargo receipts to the complete expected manifest plus canonical query,
  request, ordered record, and vector-output evidence; unverifiable completion
  conditions now fail closed.
- Enforced certificate path-length, signing-leaf Key Usage, supported critical
  extensions, and CRL distribution-point/reason scope.
- Removed certificate-directed network egress from Gate; production revocation
  retrieval now requires an operator-controlled fetcher.
- Replaced optimization-sensitive example and release assertions with explicit
  runtime failures.
- Required consumer grant provenance to match an exact canonical grant committed
  by the authority-signed lockbox; detached mutations and cross-lockbox grants
  now fail closed.
- Restricted Cargo manifests to `load`, `retrieve`, or the explicit
  `load_and_retrieve` union and enforced the method capability before any
  storage-adapter call.
- Bound explicit signed Cargo document IDs to the corresponding accepted store
  results and serialized all operations on a shared PostgreSQL connection.
- Bound provenance-bearing Modal runs to image-baked Gate and launcher digests,
  then recorded remotely measured values instead of caller assertions.
- Detached every public mask/vector/metadata view from authority-bearing state;
  enforced exact live partition, payload-family, dimensionality, finiteness,
  strict-JSON, declared-load, and receipt obligations across Cargo.
- Revalidated hostile vector-store outputs and rejected malformed PostgreSQL
  vector rows instead of synthesizing data; bridged vectors now apply the Gate
  in signed source space before projection.
- Replaced the bundled executable research companion artifact with a
  source-visible fail-closed preflight, pinned remote model revisions to exact
  commits, and linted/tested the complete research launcher surface.
- Removed the misleading fold-and-gate surface; folding is now explicitly a
  lossless codec with no authorization or storage-confinement claim.
- Removed arbitrary model/optimizer training from `GatedRAGAdapter` and made
  non-retrieval cache policies fail closed.
- Added a release-transfer verifier that requires one current-branch ref and
  rejects commits retained by any other ref, recoverable pre-amend objects,
  extra commit history, and reflogs retaining earlier commits.
- Made the Gate software release itself part of every authenticated contract:
  semantic version, canonical GitHub repository, and exact stamped commit are
  now covered by token AAD, rights HMACs, lockbox authority hashes, capability
  signatures, Cargo manifests/receipts, and operation-gate attestations.
- Added a build-time source-commit stamp, distribution inspection, installed
  identity verification, and commit-SHA-pinned GitHub artifact provenance for
  explicitly authorized public version tags.

Historical research receipts retain the package versions recorded when each run
was produced. They are evidence and are not rewritten to match this public
release version.
