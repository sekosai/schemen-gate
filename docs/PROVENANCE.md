# Source provenance

Schemen Gate 1.0.0 established the audited release baseline. Version 1.0.2
continues at the next unused semantic version without moving or reusing an
existing signed tag for different source. Generated build directories, package
metadata, bytecode, caches, credentials, model artifacts, private planning, and
unrelated workspace code are excluded.

The release tree includes the Gate library, tests, examples, documentation,
release tooling, and the complete reviewed CDP research snapshot under
`research/cdp/`. The research snapshot records its own source custody in
`research/cdp/SOURCE.json` and uses a path-based Apache-2.0/CC-BY-4.0 license
map. Historical experiment records retain their measurements, dependency
versions, source revisions, and recorded artifact hashes. Non-semantic local
machine paths are normalized to public-safe placeholders; the affected JSON
byte digests and normalization boundary are recorded in
`research/cdp/experiments/results/README.md`.

The release commit must match `RELEASE_MANIFEST.sha256`. Verify it with:

```bash
python3 scripts/release_manifest.py --verify
```

## Release identity and signed build provenance

Authenticated Gate objects carry a `GateReleaseIdentity` with four fields:
`package`, `version`, `source_repository`, and `source_commit`. The release
identity is part of the authenticated bytes, not unsigned logging metadata.
Production verification compares it to a verifier-owned expected identity.

The tracked source contains no claimed value for its own commit SHA because
that is self-referential. After checkout, `scripts/stamp_release.py` receives
the already-existing full GitHub commit SHA and generates an ignored
`_build_identity.py` for packaging. `scripts/verify_dist.py` inspects both the
wheel and source distribution without executing them and requires the stamp to
match the checked-out commit. The clean-wheel release check then imports the
installed package and verifies the same value.

The artifact builder is intentionally split in two. The only networked phase,
`scripts/bootstrap_build_env.py`, first retrieves the exact pip wheel through
the standard library from a fixed PyPI file URL and checks its size and SHA-256
before executing it. That locked pip uses an explicit index, ignores user
configuration, disables dependency discovery, accepts universal wheels only,
and verifies their exact versions and SHA-256 hashes from
`requirements/build.lock`. The artifact phase, `scripts/build_release.py`,
disables package-index access and uses `--no-isolation` with that locked
environment. It builds from a temporary Git index export plus the generated
identity stamp, never from untracked or ignored working-tree files.
`scripts/verify_dist.py` then compares every packaged Gate source byte to the
reviewed `HEAD`, enforces exact archive-member allowlists, verifies dependency
and wheel installer metadata, and validates every wheel RECORD digest. Before
decompression it also enforces compressed-file, member-count, per-member,
aggregate-size, and compression-ratio bounds. Expected members are read in
bounded chunks; a hostile zip or gzip payload cannot be fully materialized
before its release contract is checked.

On an explicitly authorized public `v1.0.2` tag, a separate `workflow_run`
workflow observes successful completion of the entire read-only CI workflow:
the full test matrix, research proof checks, artifact build, and release
contract. GitHub loads this privileged workflow from the trusted default branch,
not from the triggering commit or a pull-request merge. It has no checkout,
interpreter, shell, or repository-controlled code step. Only this workflow
receives OIDC and attestation write permissions; it enters the separately
protected `public-release` environment, downloads the immutable build artifact
from the exact triggering run with digest mismatch configured as an error, and
invokes GitHub's official commit-SHA-pinned attestation action. This creates two
complementary bindings:

1. Gate cryptography signs the release identity inside each Gate contract.
2. GitHub signs the exact artifact digests from the triggering CI run twice:
   standard provenance identifies the trusted default-branch admission
   workflow, while a custom predicate separately binds version `1.0.2`, tag
   `v1.0.2`, repository identity, exact triggering source commit, CI workflow
   and run, and artifact name.

GitHub defines `GITHUB_REF` and `GITHUB_SHA` for `workflow_run` as the default
branch and its last commit, not the triggering tag. The workflow therefore
requires its trusted default-branch SHA to equal the triggering tag's
`head_sha`, and the `public-release` environment must allow `main`, not the tag.
This deliberately makes the signed certificate digest and custom-predicate
source commit agree while retaining the default-branch privilege boundary.
See `docs/RELEASE_ATTESTATION.md` for the exact verifier contract.

Neither binding proves that arbitrary bytes currently executing on an
operator's host are those published artifacts. Execution measurement or native
code-signing enforcement remains a deployment control.

The packaged wheel intentionally contains only the Gate Python library. The
source distribution includes documentation, examples, tests, and release tools
but excludes the research tree. The Git repository is the canonical publication
unit for the papers, proofs, and research receipts.

This file performs no release action. Tags, package uploads, visibility changes,
and announcements remain explicit maintainer operations subject to the checks
in `docs/OPEN_SOURCE_RELEASE.md`.
