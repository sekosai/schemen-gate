# Contributing

Schemen Gate welcomes focused issues and pull requests. For substantial API,
authority, or threat-model changes, open an issue first so the intended boundary
can be reviewed before implementation.

## Development

```bash
python -m pip install -e ".[crypto,lockbox,onnx,rag,spiffe,torch,dev]"
python -m pytest -q
python -m ruff check src tests examples scripts research/cdp/experiments research/cdp/scripts research/cdp/examples
python -m ruff format --check src tests scripts examples
python scripts/validate_release_contract.py
python scripts/check_pypi_readme.py
python scripts/bootstrap_build_env.py
python scripts/build_release.py
python scripts/verify_dist.py
```

The complete release check requires `lake` from an elan/Lean installation.
`python scripts/release_check.py --skip-lean` is useful for a partial local
check, but it is not complete release evidence. Rebuilding the four checked-in
paper PDFs separately requires a TeX distribution plus `latexmk`; run
`make -C research/cdp paper`.

Pytest prepends this checkout's `src/` tree and propagates that path to child
Python processes, so a test run cannot silently use an editable installation
from a different Schemen Gate checkout.

The release bootstrap establishes pip from a fixed file URL, size, and SHA-256
before a package installer contacts an index, then accepts only the explicitly
enumerated hashes in `requirements/build.lock`. The subsequent build is offline
and uses only a Git-tracked export plus the generated commit-identity stamp.
The distribution verifier rejects any unexpected archive member and compares
every packaged Gate source file byte-for-byte with the reviewed commit.

Security-sensitive changes must include adversarial rejection tests. Changes to
authority, key derivation, canonicalization, signatures, replay handling, trust
anchors, or runtime contracts must document the exact old and new boundary.

Keep pull requests narrow, explain the user-visible behavior, and add or update
tests and documentation together. Contributions outside `research/cdp/` are
provided under the root Apache-2.0 license as described in section 5 of that
license. Contributions inside `research/cdp/` use the path-specific license in
[`research/cdp/LICENSES.md`](research/cdp/LICENSES.md). No separate contributor
license agreement is currently required.

Do not commit credentials, generated `build/` or `dist/` trees, `.egg-info`,
model weights, caches, or production receipts.
