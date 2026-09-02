# Complete Schemen Gate open-source release plan

Schemen Gate is the open enforcement layer for **AI PKI**: infrastructure that
carries verifier-trusted identity and signed, scoped authority to an AI
execution boundary. **AI provenance PKI** describes its signed identity and
evidence surfaces; **inference trust infrastructure** describes its runtime
enforcement and deployment surfaces. These labels do not replace classical PKI
or expand the repository's bounded cryptographic and deployment claims.

## Release posture

The release objective is one complete Schemen Gate repository: library, paper,
proofs, examples, and reproducibility material. Repository visibility is an
operational maintainer setting, not a state asserted or changed by this plan.

`sekosai/schemen-gate` is the sole adoption repository. It owns the
dependency-light Gate library and the reviewed CDP research bundle. No external
model-server repository or executable server wheel is required by the release.

The research bundle was imported as a reviewed snapshot without its source
repository history. Public release refs must contain only the audited Gate
lineage. Every installable artifact must match source in that lineage; a
wheel-only or source-unavailable dependency is a release blocker.

The exported project is limited to CDP and Schemen Gate. No private product
name, architecture, strategy, dependency, configuration, or integration hook
may appear in the exported tree, release metadata, documentation, PDFs, issue
templates, or Git history.

## Gates

### 1. Claim and build integrity

- [x] Paper and three focused manuscripts compile from tracked LaTeX.
- [x] Canonical JSON and wheel digests are machine-checked.
- [x] Local contract tests run without private GitHub access.
- [x] All 21 Lean modules have a pinned Lake/Mathlib build.
- [x] Headline theorem dependencies are audited separately from historical
  conditional theorem chains.
- [x] README and claim-boundary language separate algebra, measured state,
  runtime authorization, and unsupported end-to-end claims.
- [ ] Clean GitHub Actions runs pass on the exact Gate and research release
  commit.
- [x] A new user can install Gate and run the local examples without GitHub
  authentication or private services.
- [ ] Record an independent Modal canary from the signed release tag.
- [x] The final PDFs are visually inspected page by page after the last edit.

### 2. Rights and licensing

- [x] Original Python, shell, build code, examples, tests, and Lean proof source
  use Apache-2.0. Authored papers, explanatory prose, figures, and designated
  result records use CC BY 4.0 under `LICENSES.md`.
- [x] Canonical license texts, SPDX identifiers, notices, and the path map are
  included.
- [x] No executable third-party or server wheel is bundled; the research
  preflight is reviewable source and explicitly disclaims production authority.
- [ ] Publish Gate `v1.0.2` from the exact approved source snapshot. Historical
  receipts retain the dependency versions recorded when they ran; they are
  evidence rather than current installation dependencies.
- [x] The final licenses keep the complete Gate stack open
  source. Do not introduce noncommercial or field-of-use restrictions while
  describing the project as open source.
- [x] No model weights or dataset rows are redistributed. Source terms and
  research-only dataset boundaries are recorded in `THIRD_PARTY_NOTICES.md`.
- [x] Include a public-safe patent notice in the adoption README, packaged
  `NOTICE`, and research README. It states that a U.S. provisional was filed
  before release, adds no separate restriction, and points Apache-2.0 material
  to Section 3 while explaining that CC BY 4.0 does not license patent rights.

### 3. Source and history hygiene

- [x] Current-tree checks reject common credential formats, private Git VCS
  dependencies, oversized tracked files, malformed JSON, and broken local
  Markdown links.
- [x] Run dedicated signature and entropy-aware credential scanners over the
  release snapshot; manually resolve detector candidates without publishing
  candidate values.
- [x] Review every tracked path in the release snapshot. Exclude patent working
  files, private corpora, customer material, local config, caches, and generated
  logs.
- [x] Keep the provisional application number, internal docket, unpublished
  claims, filing documents, receipts, and prosecution records out of the public
  tree and release metadata.
- [x] Private launch-planning documents and unrelated workspace material are
  absent from the exported tree.
- [x] Run the private product-name and dependency denylist against the exported
  tree and generated PDFs. Keep the denylist itself outside the public project.
- [x] Bootstrap the Gate lineage from a history-free audited root rather than
  copying inherited development history.
- [ ] Before a visibility change, publish only the audited `main` lineage and
  remove archival refs that retain excluded pre-release history.
- [x] Record and verify the exported tree in `RELEASE_MANIFEST.sha256`.

Suggested snapshot procedure after the release commit is approved:

```bash
git archive --format=tar.gz \
  --prefix=schemen-gate-v1.0.2/ \
  --output=schemen-gate-v1.0.2.tar.gz \
  <approved-release-commit>
shasum -a 256 schemen-gate-v1.0.2.tar.gz
```

Verify that the archive tree matches the approved commit before release. Do not
copy unrelated Git refs or working-directory state. Confirm that the archive
contains no private paths.

### 4. Public GitHub configuration

- [ ] Make `sekosai/schemen-gate` the sole public adoption repository through an
  explicit visibility-change action after every release gate passes.
- [ ] Publish Gate to a public package index from a signed tag, with
  matching wheels, source archives, checksums, and provenance.
- [ ] Gate description: `Open-source identity-bound model execution: binary
  activation, capabilities, lockboxes, Cargo, proofs, and
  reproducible research`.
- [ ] Topics: `machine-learning`, `model-security`, `identity`, `authorization`,
  `formal-verification`, `lean4`, `modal`, `multi-tenant`, and
  `reproducible-research`.
- [ ] Enable issues, discussions, private vulnerability reporting, dependency
  review, secret scanning, and branch protection.
- [ ] Require Gate package, research, and security jobs
  before merging to `main`; require review and dismiss stale approvals.
- [ ] Add a repository social preview and a paper/preprint homepage or DOI.
- [x] Include root and paper `CITATION.cff` files plus community-health files.
- [ ] Confirm citation rendering and detected licenses in
  GitHub's UI.
- [ ] Create a signed, immutable release tag; attach the four PDFs, snapshot
  archive, checksum manifest, and release notes. Never move a published tag.
- [ ] Archive the release with Zenodo or another durable repository and add the
  DOI to `CITATION.cff` and the paper in a follow-up release.

### 5. Release-day verification

From a clean clone of the proposed public Gate repository:

```bash
git clone https://github.com/sekosai/schemen-gate.git

cd schemen-gate
python -m pip install -e '.[crypto,lockbox,onnx,rag,spiffe,torch,dev]'
python -m pytest -q
python scripts/bootstrap_build_env.py
python scripts/build_release.py
python scripts/verify_dist.py
```

Then run the CDP proof, paper, and Modal checks from the research bundle in the
Gate repository. The first public quickstart must not require users to clone the
paper repository separately.

The Modal step is a single scale-to-zero CPU canary. It validates public
onboarding and artifact transport, not a paper result. Do not run a GPU matrix
until one documented experiment smoke artifact has completed and been
inspected.

Record the exact clean-clone commit, package tests, security regressions, PDF
checksums, source archive checksums, GitHub Actions URLs, and Modal canary result
in the release notes.

## Public support promise

The public Gate ecosystem should support installation, local Gate execution,
proof checking, contract tests, examples, and one cheap Modal
onboarding canary without access to private infrastructure. External IdPs,
customer KMS/HSM/Vault systems, production model weights, cloud accounts, and
customer data remain outside the repository; the adapters and interfaces needed
to integrate them must remain public.
