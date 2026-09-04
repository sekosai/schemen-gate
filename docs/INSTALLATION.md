# Installation and release verification

[Back to Schemen Gate](../README.md)

Run commands from the repository root. Source installation and release-artifact
verification are separate paths; neither command below publishes a release.

## Install version 1.0.2

### The Gate release is part of the cryptographic contract

Gate protocol schemas and Gate software releases are different version axes.
Every authenticated contract carries a `GateReleaseIdentity` containing the
package name, semantic version, canonical GitHub repository, and source commit.
That identity is covered by the same AEAD tag, HMAC, authority signature, or
public attestation as the rest of the contract. The binding applies to mask and
adapter tokens, `GateRights`, signed lockboxes, capability/attestation tokens,
Cargo manifests and receipts, and exact-operation gates.

[`release-contract.json`](../release-contract.json) is the machine-readable source
for the mechanical package version, tag, and canonical repository fields.
`python scripts/validate_release_contract.py` fails if package metadata,
release tooling, CI, attestation, citation, or the research lock drifts from it.

```python
from schemen_gate import current_release_identity

release = current_release_identity()
print(release.to_dict())
release.require_source_commit()  # fail closed for an unstamped development build
```

An ordinary clean Git checkout binds `source_commit` to its exact `HEAD` and
rejects tracked, staged, or non-ignored untracked source drift. An installed
archive has no Git authority to consult, so release distributions carry the
generated commit stamp. A stale ignored stamp in a checkout is not trusted over
the verified Git commit. The release workflow stamps the already-existing
`github.sha` into the wheel and source distribution, verifies the installed
value, and—on an explicitly authorized public version tag—asks GitHub to sign a
release-admission statement for the artifact digests.
Production verifiers should compare against a verifier-owned expected
`GateReleaseIdentity`, including the full commit SHA.

A commit cannot contain its own final SHA: adding that SHA to a tracked file
would create a different commit. The generated stamp is therefore an ignored
build input, and the signed GitHub artifact attestation is external to the
commit. This is the same reason final artifact checksums and attestations are
published beside a release rather than inside the artifacts they identify.

Before publication, build and install the reviewed local candidate rather than
resolving the project name from an untrusted fallback package index:

```bash
python scripts/stamp_release.py \
  --version 1.0.2 \
  --repository https://github.com/sekosai/schemen-gate \
  --commit "$(git rev-parse HEAD)"
python scripts/bootstrap_build_env.py
python scripts/build_release.py
python scripts/verify_dist.py
shasum -a 256 dist/schemen_gate-1.0.2-py3-none-any.whl
python -m pip install 'dist/schemen_gate-1.0.2-py3-none-any.whl[lockbox]'
```

`build_release.py` rejects tracked changes, staged changes, untracked
non-ignored files, and tracked symlinks. It copies only the reviewed Git index
plus the generated commit stamp into a temporary export and invokes the locked
builder with package-index access disabled. `verify_dist.py` independently
checks exact source bytes, archive paths, dependency metadata, wheel installer
metadata, and RECORD hashes before any artifact can be attested. The complete
release checker performs the build stamp automatically. A release custodian
can verify the generated GitHub attestation after publication. The first
command verifies the artifact digest, GitHub signature, trusted signer
workflow, public runner, and exact default-branch commit. The second checks the
signed custom predicate's version, tag, source commit, and immutable CI artifact
name:

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

See [release attestations](RELEASE_ATTESTATION.md) for the complete trust and verification
contract. A generic provenance check is insufficient because the privileged
`workflow_run` signer executes in GitHub's default-branch context.

After publication is explicitly authorized, install only from the canonical
package index or signed GitHub release and compare its published SHA-256 digest
before deployment. Never substitute an artifact merely because its filename
matches.

The eventual canonical package-index command will be:

```bash
python -m pip install 'schemen-gate[lockbox]==1.0.2'
```

Do not use it until the release page and package-index provenance point to the
same reviewed commit and checksums.

The core wheel depends only on NumPy. Install optional capabilities from a
checked-out source tree when developing:

```bash
python -m pip install -e '.[crypto,lockbox,onnx,rag,spiffe,torch,dev]'
```

The extras are:

- `crypto`: encrypted tokens, key wrapping, capability attestations, lockboxes,
  and operation gates. Deterministic HMAC mask derivation is available in the
  NumPy-only core.
- `torch`: tensor conversion helpers. Schemen Gate does not own your training
  loop or optimizer, and `GatedRAGAdapter` deliberately exposes no model-training
  or absorption method.
- `lockbox`: signed grants, key wrapping, and lockbox persistence.
- `onnx`: model-artifact metadata binding and attestation verification.
- `rag`: gated retrieval and optional vector-store support. `PgVectorStore`
  additionally requires the server-side PostgreSQL pgvector extension; the
  Python extra does not install database extensions.
- `spiffe`: workload identity helpers.

## Optional encoder and bootstrap-history checks

The deterministic release suite does not download an embedding model. After
making the pinned model available, opt into the real-encoder integration:

```bash
SCHEMEN_RUN_REAL_ENCODER=1 python -m pytest -q tests/test_real_encoder_dispatch.py
```

The `--require-history-free` mode is reserved for a replacement single-root
bootstrap snapshot, not ordinary releases after the public lineage begins:

```bash
python scripts/release_check.py --require-history-free
```

That mode requires the current branch to be the only Git ref and rejects extra
reachable commits, predecessor reflogs, and unreachable or dangling objects.
If a bootstrap root must change, rebuild a fresh repository rather than
amending a root and copying its `.git` directory. Published release tags are
immutable; a new release uses a new version and tag.
