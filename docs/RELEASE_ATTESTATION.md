# Release-attestation contract

Schemen Gate 1.0.2 uses a GitHub-signed release-admission attestation. It is a
binding among exact artifact bytes, the reviewed source commit, the release
version and tag, the successful CI run, and a privileged signer workflow loaded
from the protected default branch.

## Why this is a custom predicate

GitHub defines `GITHUB_REF` and `GITHUB_SHA` for a `workflow_run` workflow as
the default branch and its last commit. They do not identify the tag that ran
the upstream CI job. Schemen Gate intentionally retains that default-branch
context because it prevents a candidate tag or pull request from supplying the
workflow that receives OIDC and attestation-write authority.

The attestation therefore has two complementary authenticated layers:

1. A standard GitHub provenance statement and its signing certificate identify
   the repository, default-branch admission workflow, its commit, and the
   GitHub-hosted runner. This statement describes the trusted admitting
   workflow; by itself it does not identify the upstream tag.
2. A second custom predicate identifies Gate 1.0.2, `refs/tags/v1.0.2`, the
   upstream repository and CI workflow identities, CI `head_sha`, run and
   attempt, and immutable artifact name. Both in-toto statements contain the
   action-computed SHA-256 digests of the exact wheel and source archive.

The privileged workflow fails closed unless the default-branch commit equals
the upstream tag commit. It also requires a successful `push`-triggered CI run,
the exact CI workflow path, the same repository as both event and head
repository, public visibility, and a protected `public-release` environment.
It performs no checkout and has no shell, interpreter, or repository-controlled
code step.

## Predicate type and fields

Predicate type:

`https://github.com/sekosai/schemen-gate/attestations/release/v1`

The predicate has exactly these release-relevant fields:

- `schema`: `schemen/gate-release-attestation-v1`
- `package`: `schemen-gate`
- `version`: `1.0.2`
- `source_repository`: `https://github.com/sekosai/schemen-gate`
- `source_repository_id`: the triggering head repository's immutable GitHub ID
- `source_ref`: `refs/tags/v1.0.2`
- `source_commit`: the triggering CI run's full `head_sha`
- `ci_workflow`: `.github/workflows/ci.yml`
- `ci_workflow_id`: the triggering CI workflow's immutable GitHub ID
- `ci_run_id` and `ci_run_attempt`: the triggering GitHub Actions run
- `artifact_name`: `schemen-gate-dist-<source_commit>`

The in-toto subject list, rather than the predicate, carries each artifact
filename and digest.

## Required GitHub environment

After publication is explicitly authorized, but before creating or replacing
the release tag, configure `public-release` to:

- allow only the `main` branch;
- require a rights-holder reviewer;
- prevent self-review; and
- disallow administrator bypass.

The branch rule must be `main`: GitHub matches environment rules against the
privileged `workflow_run` job's `GITHUB_REF`, which is the default branch. The
trusted workflow itself independently requires the triggering upstream
`head_branch` to be exactly `v1.0.2`.

Keep `main` at the release commit from tag creation until attestation finishes.
Any mismatch between `github.sha` and the upstream CI `head_sha` skips the
privileged job.

## Verification

From the exact release checkout, verify a downloaded artifact with:

```bash
EXPECTED_GATE_COMMIT="$(git rev-parse HEAD)"
gh attestation verify dist/schemen_gate-1.0.2-py3-none-any.whl \
  --repo sekosai/schemen-gate \
  --predicate-type https://github.com/sekosai/schemen-gate/attestations/release/v1 \
  --signer-workflow sekosai/schemen-gate/.github/workflows/release-attestation.yml \
  --source-ref refs/heads/main \
  --source-digest "$EXPECTED_GATE_COMMIT" \
  --deny-self-hosted-runners \
  --format json > gate-attestation.json
jq -e --arg commit "$EXPECTED_GATE_COMMIT" '
  any(.[].verificationResult.statement;
    .predicate.schema == "schemen/gate-release-attestation-v1" and
    .predicate.package == "schemen-gate" and
    .predicate.version == "1.0.2" and
    .predicate.source_repository == "https://github.com/sekosai/schemen-gate" and
    (.predicate.source_repository_id | test("^[1-9][0-9]*$")) and
    .predicate.source_ref == "refs/tags/v1.0.2" and
    .predicate.source_commit == $commit and
    .predicate.ci_workflow == ".github/workflows/ci.yml" and
    (.predicate.ci_workflow_id | test("^[1-9][0-9]*$")) and
    (.predicate.ci_run_id | test("^[1-9][0-9]*$")) and
    (.predicate.ci_run_attempt | test("^[1-9][0-9]*$")) and
    .predicate.artifact_name == ("schemen-gate-dist-" + $commit))
' gate-attestation.json
```

Run the same verification for the source archive. The `gh` command verifies
the artifact digest before emitting JSON; `jq -e` then fails unless a verified
statement contains the exact Gate predicate values. Record the tag signature,
release commit, signer commit, CI run, attestation URL, and both artifact
SHA-256 digests in the release notes.
