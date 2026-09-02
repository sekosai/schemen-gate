# Contributing

Contributions are welcome when they preserve the repository's evidence and
claim boundaries. A passing computation is not automatically a paper result.

## Before opening a pull request

Create a test environment and run the fast release checks:

```bash
./scripts/setup.sh test
python3 scripts/release_check.py
```

For proof changes, also run:

```bash
lake build
./scripts/check_lean_claims.sh
```

For manuscript changes, build all four PDFs with `make paper` and inspect the
rendered pages. Do not commit LaTeX auxiliary files.

## Evidence contract

A pull request that changes an empirical claim must include:

- the exact runner and pinned inputs;
- a timestamped machine-readable result under `experiments/results/`;
- complete denominators and matched controls;
- the acceptance rule and any negative or failed result;
- an update to `experiments/results/README.md` and
  `docs/experiment-data-inventory.md`; and
- paper wording no stronger than the retained evidence.

Cloud work must complete and inspect one persisted canary before a larger
matrix. Do not launch paid Modal work from CI or as part of ordinary review.

## Pull request hygiene

- Keep changes narrow and do not rewrite unrelated result artifacts.
- Never commit credentials, local Modal configuration, model weights, private
  corpora, or generated caches.
- Treat changes to threat models, licensing, public visibility, and canonical
  artifacts as maintainer decisions.
- State which tests ran and which expensive tests did not run.

Unless a contribution says otherwise, it is submitted under the repository's
current path-based license: Apache-2.0 for executable code and Lean proof source,
and CC BY 4.0 for authored papers, prose, figures, and designated research
records. See `LICENSES.md` for the authoritative path map.
