# CDP research bundle license map

This snapshot is intentionally multi-licensed. The rule for a path is the
first matching section below. A file not classified here is not approved for a
public release until the license map is updated.

## Third-party exceptions

Third-party files retain their original terms; this project does not relicense
them. No executable third-party package is currently shipped in this bundle.

Any new third-party source, data, model artifact, font, image, or binary must be
listed here and in `THIRD_PARTY_NOTICES.md` before it may ship.

## Apache-2.0

The following executable and operational material is licensed under the Apache
License, Version 2.0. The canonical text is in
[`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).

- `proofs/**/*.lean`
- `experiments/**/*.py`, excluding third-party files listed above
- `examples/**/*.py`
- `scripts/**/*.py`
- `scripts/**/*.sh`
- `scripts/**/*.lean`
- `Makefile`
- `pyproject.toml`
- `lakefile.lean`
- `lake-manifest.json`
- `lean-toolchain`
- `experiments/requirements*.txt`
- `experiments/schemen-library-lock.json`
- `gated-transformer-regime-lanes/*.py`
- `gated-transformer-regime-lanes/.gitignore`
- `.gitignore`

## CC-BY-4.0

The following authored research material is licensed under Creative Commons
Attribution 4.0 International. The canonical text is in
[`LICENSES/CC-BY-4.0.txt`](LICENSES/CC-BY-4.0.txt); the short notice and
authoritative Creative Commons legal-code link are in [`LICENSE`](LICENSE).

- `paper/**`
- `output/pdf/**`
- `experiments/results/**`
- `docs/**`
- `README.md`
- `CHANGELOG.md`
- `CITATION.cff`
- `CODE_OF_CONDUCT.md`
- `CONTRIBUTING.md`
- `SECURITY.md`
- `SOURCE.json`
- `THIRD_PARTY_NOTICES.md`
- `experiments/PLAN.md`
- `experiments/README.md`
- `experiments/orthogonal-superposition-experiment.md`
- `examples/README.md`
- `gated-transformer-regime-lanes/*.md`
- `gated-transformer-regime-lanes/*.json`
- `gated-transformer-regime-lanes/*.png`
- `gated-transformer-regime-lanes/figures/**`
- `gated-transformer-regime-lanes/results/**`

Attribution should identify “Cryptographic Dimension Partitioning,” Ryan
McCormick, the source repository, the CC BY 4.0 license, and whether the
material was modified.

## Packaging-only files

`LICENSE`, `LICENSES.md`, and the legal texts under `LICENSES/` exist to state
or reproduce licensing terms. They are not independently relicensed by this
map.

## Release check

Before publication, enumerate every shipped path, apply these adopted rules in
order, and fail if a path is unclassified or matches conflicting rules. Any
future path or third-party addition must update this map, the canonical texts,
attribution, patent notice when applicable, and third-party exceptions before
it ships.
