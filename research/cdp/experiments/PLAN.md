# Experiment Reproduction Plan

This document is the execution policy for the public CDP experiment bundle.
It covers only source and evidence shipped in this repository. Historical
studies whose source is absent are not listed as reproducible work and are not
evidence for the papers.

The detailed construction, command reference, measured results, and claim
boundaries are in [`README.md`](README.md). The result custody map is in
[`results/README.md`](results/README.md).

## Reproduction order

Run the smallest falsifiable check first. Do not spend cloud compute until the
preceding local or smoke tier passes and its artifact has been inspected.

1. Verify the exact library source and experiment lock:

   ```bash
   python3 experiments/library_provenance.py
   ```

2. Run the deterministic, no-network constructions:

   ```bash
   python3 experiments/local_exact_extraction.py
   python3 experiments/capacity_preserving_wide_classifier.py
   python3 experiments/train_authorized_moe.py \
     --synthetic --input-dimensions 32 --hidden-dimensions 16 --epochs 8
   python3 experiments/train_capability_token_moe.py \
     --synthetic --vocabulary-size 128 --maximum-tokens 24 \
     --embedding-dimensions 32 --hidden-dimensions 16 --epochs 8
   ```

3. Run model-backed local canaries before a full local matrix:

   ```bash
   python3 experiments/reproduce_distilbert_classification.py --smoke
   python3 experiments/local_transformer_cotenancy_suite.py --canary
   ```

4. For a Modal launcher, use its documented reduced mode where one exists,
   inspect the returned artifact and acceptance fields, and only then run a
   bounded full matrix. Most current canonical launchers use `--smoke`; Cargo
   authorization is already a small full protocol and has no separate smoke
   flag. Read the static command reference in [`README.md`](README.md); the
   campaign planner deliberately does not import launchers while preparing a
   no-compute plan because their top-level Modal resource declarations may have
   provider-side effects.

For the coordinated, commit-bound re-certification campaign, use
[`../../../docs/MODAL_RECERTIFICATION.md`](../../../docs/MODAL_RECERTIFICATION.md).
It records the exact included and excluded launchers, current price snapshot,
campaign ceiling, provider-budget review, staged approvals, receipt validators,
and external artifact custody. It does not run every `modal_*.py` file
indiscriminately.

Provider prices and account credits change. The dated rates used for a
coordinated campaign live in `modal-recertification.json` and the sealed plan;
check them against current provider pricing before launching remote work. A
command that would allocate cloud compute is never part of the local release
check.

## Acceptance contract

A canonical experiment must:

- resolve `schemen-gate` to the version, clean commit, source-tree digest, and
  experiment lock declared by `schemen-library-lock.json`;
- reject a dirty or mismatched checkout rather than silently relabel it;
- exercise the wrong-authority controls before scored model execution and
  record zero unauthorized model calls;
- state exactly which activation or whole-model surface is governed;
- preserve raw measurements, dependency versions, model and dataset
  revisions, and construction-specific acceptance fields in JSON;
- distinguish exact equality, declared numerical tolerance, statistical
  evidence, and an unmeasured hypothesis; and
- keep smoke, failed, exploratory, and canonical artifacts in their designated
  result tiers.

A smoke run validates installation and control flow. It is not a utility
estimate. A successful construction does not establish confidentiality,
semantic correctness, full-Transformer isolation, or hardware key residency
unless the specific experiment measures that property.

## Shipped evidence families

The public bundle contains runnable source and archived artifacts for:

- exact addressed extraction and capacity-preserving modular classifiers;
- formative and intermediate-FFN DistilBERT classification surfaces;
- strict Transformer FFN cotenancy and private-lane constructions;
- whole-model runtime authorization and Cargo scope binding;
- learned MoE routing inside a fixed authorized expert set;
- capability-token prefix routing;
- service-consolidation and physical-extraction controls;
- orthogonal placement, multiplexing, and fused-benchmark studies; and
- explicitly labeled negative or exploratory Transformer, generation, and
  cache studies.

The archived negative controls are retained because they falsify overbroad
placements and claims. Their presence does not turn them into positive evidence
for the papers.

## Result custody

New runs write timestamped JSON under `experiments/results/`. Canonical results
live at the top level. Protocol smokes, failures, and exploratory results live
under `experiments/results/archive/`. Re-runs create new files; they do not
overwrite prior evidence.

Historical machine paths in retained artifacts are normalized to public-safe
placeholders. Measurements, dependency versions, source commits, and recorded
artifact hashes are preserved. The canonical JSON byte digests affected by
that non-semantic normalization are documented in `results/README.md`.

## Publication and licensing

Executable experiments, launchers, tests, and Lean proof source are licensed
under Apache-2.0. Authored papers, prose, figures, and designated research
records are licensed under CC BY 4.0. The authoritative path map is
[`../LICENSES.md`](../LICENSES.md).

A U.S. provisional patent application was filed before release for subject
matter related to portions of Schemen Gate. No application numbers, internal
dockets, filing documents, receipts, private claim charts, or prosecution
strategy are part of this public bundle. The notice adds no restriction beyond
the repository licenses and does not change the evidentiary status of any
experiment.

## Data and third-party boundaries

Model and dataset revisions used by canonical runners are pinned in source and
recorded in result artifacts. Third-party weights and datasets are downloaded
from their upstream providers and are not redistributed by this repository.
Users remain responsible for the terms of any model, dataset, and cloud service
they choose to run.
