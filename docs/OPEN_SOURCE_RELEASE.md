# Open-source release contract

Schemen Gate is intended to be a complete open-source product, not an open-core
preview. A release is ready only when a new user can inspect the full source,
install it without private infrastructure, run the first example, and reproduce
the published checks.

## Repository and release boundary

- Publish only an audited release lineage. Exclude archival refs, development
  history, and unreachable objects that were not part of the reviewed snapshot.
- Confirm the shipped files match the adopted path-based license policy.
- Keep the public patent notice in the root `README.md`, packaged `NOTICE`, and
  research README: A U.S. provisional patent application was filed before
  release for subject matter related to portions of Schemen Gate; the notice
  adds no separate restriction; Apache-2.0 Section 3 governs patent rights for
  Apache-2.0 material; and CC BY 4.0 does not license patent rights.
- Keep application numbers, internal dockets, unpublished claims, filing
  documents, receipts, and prosecution records in private rights-holder
  custody. None belongs in the public tree, distributions, generated PDFs, or
  release metadata.
- Run a whole-history secret, credential, personal-data, and third-party-source
  audit on the history that will actually be published. A clean current tree
  does not prove clean Git history.
- Confirm `LICENSE`, `NOTICE`, dependency licenses, and the separately licensed
  `research/cdp/` bundle match the material actually shipped.
- Remove every private source, package, issue, documentation, and CI dependency.
- Enable branch protection, required CI, secret scanning, dependency review,
  private vulnerability reporting, and two-factor authentication for release
  maintainers.

## Required before announcing the release

- Publish source archives and wheels from the tagged commit.
- Publish `schemen-gate` on PyPI with provenance and verify installation in a
  clean environment using only public endpoints.
- Verify that every runnable example and research launcher uses only this Gate
  source plus public third-party dependencies; no companion Schemen package is
  a release prerequisite.
- Run the README quickstart verbatim in CI and from an independent clean clone.
- Bootstrap the release builder from `requirements/build.lock`, which pins the
  complete build toolchain by exact version and wheel hash. Build offline from
  the tracked commit export; do not build from the mutable working directory.
- Run the full test matrix, exact wheel/sdist member and source-byte
  verification, metadata verification, clean-wheel examples, and CDP research
  release checker.
- Record the tag, commit, artifact digests, CI run, and verification commands in
  the GitHub release notes.
- Require every wheel and source distribution to contain the `GateReleaseIdentity`
  stamp for the exact tagged GitHub commit. Verify it with
  `python3 scripts/verify_dist.py` and by importing the clean-installed wheel.
- Require `scripts/verify_dist.py` to compare every packaged Gate module with
  `git show HEAD:<path>`, reject extra or missing archive members, validate
  wheel RECORD hashes and installer metadata, and reject dependency-metadata
  drift before provenance is signed.
- Generate the GitHub release-admission attestation for the wheel and source
  distribution, then verify its signature and custom predicate with the exact
  commands in `docs/RELEASE_ATTESTATION.md`.
  Before creating the tag, configure the `public-release` GitHub environment to
  allow only the `main` branch, require a rights-holder reviewer, prevent
  self-review, and disallow administrator bypass. `workflow_run` jobs use the
  default branch as `GITHUB_REF`, so a tag-only environment rule would block
  the attestation. Treat a missing or different protection rule as a release
  blocker. The separate privileged workflow is loaded from the trusted default
  branch, not from a pull request or tagged payload. It accepts only a
  successful CI run for a `v1.0.2` tag in the same repository and only
  when the default branch and tag resolve to the same commit. It executes no
  repository-controlled code and receives write/OIDC authority only after the
  environment approves it. The checked-in workflows do not publish, tag, or
  change repository visibility.
- Create a cryptographically signed `v1.0.2` tag whose target is the same commit
  stamped into the distributions and present at the tip of `main`. Freeze
  `main` from tag creation until the attestation completes; the workflow fails
  closed if their commits differ. The GitHub attestation supplements the signed
  tag; neither substitutes for verifying the artifact digest.
- Verify `RELEASE_MANIFEST.sha256` against the tagged tree with
  `python3 scripts/release_manifest.py --verify`.
- Use `python3 scripts/release_check.py --require-history-free` only when
  constructing a replacement single-root bootstrap repository. Normal releases
  after the public line begins intentionally retain prior public commits and
  tags and therefore must not use that option.
- Transfer from a clean tracked Git export or clone. Never archive a working
  directory containing ignored build caches or nested third-party repositories.

## Claim boundary

The Gate construction and release checks are deterministic relative to their
declared inputs and authority. GPU empirical protocols are reproducible by their
pinned source, assets, dependencies, configuration, seed, and hardware policy;
they are not claimed to be bitwise identical across different GPU models,
drivers, CUDA stacks, or library versions. Neither boundary proves that a policy
is wise, that an operator is trustworthy, or that a deployment has no bypass.
Execution and model isolation claims remain bounded by the lifecycle, threat
model, and tests named in the documentation.
