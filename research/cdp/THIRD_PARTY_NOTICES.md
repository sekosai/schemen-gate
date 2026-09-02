# Third-party notices

The repository license does not override third-party terms.

Original Lean files are not third-party material; each carries an explicit
Apache-2.0 notice and refers to `LICENSES/Apache-2.0.txt`. The adopted
path-based license map in `LICENSES.md` applies Apache-2.0 to executable code
and Lean source, and CC BY 4.0 to the designated authored research material.

## Vendored Python wheels

No executable Python wheel is shipped in this research bundle. Reproduction
setup and Modal launchers install Schemen Gate from the enclosing repository,
whose version and source custody are bound in
`experiments/schemen-library-lock.json`. Historical result receipts retain the
dependency records captured by the original runs; they are evidence, not
installable release dependencies.

## Models and datasets

This repository does not redistribute model weights or benchmark datasets.
Model, dataset, and tool licenses are inventoried in `experiments/PLAN.md`.
Users are responsible for obtaining those materials under their own terms.

The two orthogonal-superposition reproduction scripts resolve their inputs at
immutable Hugging Face Git revisions:

- `distilbert-base-uncased` at
  `12040accade4e8a0f71eabdb258fecc2e7e948be` (Apache-2.0 model repository).
- `fancyzhx/ag_news` at
  `eb185aade064a813bc0b7f42de02595523103ca4` (dataset card and loader snapshot;
  the underlying corpus remains subject to the original non-commercial
  research terms described in `experiments/PLAN.md`).

The revisions make the download inputs reproducible; they do not change or
supersede the upstream license terms.
