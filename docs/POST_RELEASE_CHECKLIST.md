# Post-release operating checklist

This list begins only after the reviewed Schemen Gate tree is deliberately made
public. It does not authorize publication, tagging, package upload, deployment,
or announcement.

## Private handoff completed

- [x] Complete local release gate passes from a clean private candidate.
- [x] The permanent security regression suite passes on the candidate.
- [x] Public-facing documentation, X.509 profile, claim-to-test matrix,
  certificate-to-Gate quickstart, research, papers, and release distributions
  are included in the reviewed tree.
- [x] Candidate credential, excluded-material, local-path, and public-hygiene
  sweeps pass without exposing candidate secret values.
- [x] Publication remains a separate explicit authorization gate.

## Immediately after repository publication

- [x] Enable GitHub private vulnerability reporting. This option becomes
  reporter-facing only on a public repository and is independent of Gate
  security modes, public-for-all policy, and Cargo Mode.
- [ ] Verify that **Security → Report a vulnerability** is visible from a
  logged-out or non-maintainer account and that `SECURITY.md` renders from the
  repository Security tab.
- [ ] Enable required CI and branch protection on `main`.
- [ ] Add exact GitHub users or teams to `CODEOWNERS` for Cargo, PKI,
  cryptographic, and release-boundary files; require an independent matching
  approval and prevent self-review or administrator bypass.
- [ ] Enable secret scanning, push protection, supported non-provider-pattern
  scanning, dependency review, and maintainer two-factor-authentication
  controls recorded in `OPEN_SOURCE_RELEASE.md`.
- [ ] Add an independent `public-release` environment reviewer, preserve
  self-review prevention, and disable administrator bypass.
- [ ] Re-run the whole published-history credential and excluded-material scan;
  retain the command, commit, and result without copying candidate secrets into
  the record.
- [ ] Verify public local links, package URLs, paper PDFs, release artifacts,
  signed tag, stamped commit SHA, manifest, and GitHub attestation from a clean
  unauthenticated environment.
- [ ] Run the clean-index wheel installation and all three quickstarts from a
  fresh clone of the public repository.
- [ ] Run and record the Modal CPU AI-PKI canary from the exact release tag.

## Communications and support

- [ ] Confirm that `ryan@sekos.ai` receives a test security report
  and a operator-integration inquiry without exposing either message publicly.
- [ ] Publish the claim-to-test matrix, architecture explanation,
  public-for-all policy, Cargo Mode boundary, and X.509 profile alongside the
  release announcement.
- [ ] Triage reports against a reproducible version/commit first; translate any
  valid security finding into a permanent regression test before release of a
  fix.

## Adoption evidence

- [ ] Publish one operator-owned integration example and the tested Python,
  NumPy, Torch, Modal, and certificate-profile compatibility matrix.
- [ ] Record independent clean installs, canary completions, substantive issues,
  and real integrations. Treat stars and impressions as reach, not proof that a
  Gate protected a production boundary.
- [ ] Invite integration and consulting conversations through the documented
  email without making support a prerequisite for using the open-source Gate.
