# Reproducibility and custody

This page is the compact entry point for reproducing or auditing the gated
Transformer regime-lane result. It does not replace the frozen protocols and
receipts in the [full evidence archive](EVIDENCE_ARCHIVE.md).

## Evaluated checkpoints

| Model | Immutable revision | Role |
|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | `eaa56b503cc0a8a4d15de1dd8bd2a7e95a716be2` | Small real-model positioning, layerwise, SFT, and concurrency bridge. |
| `Qwen/Qwen3-4B-Instruct-2507` | `cdbee75f17c01a7cc42f958dc650907174af0554` | Principal 4B replication, SFT, decoder, lifecycle, footprint, and throughput studies. |

Principal 4B work used BF16 execution on Modal A100-80GB workers; selected R32
mechanics used H200. Exact Torch, Transformers, source, dependency, app, run,
and artifact identifiers remain in the archived per-study receipts because
those values differ across experimental generations.

## Publication-branch executable core

The curated branch retains two standard-library executable models:

- [`toy_gated_delta_head.py`](toy_gated_delta_head.py), which tests
  authority-first candidate restriction and receipt integrity; and
- [`toy_multi_regime_transformer.py`](toy_multi_regime_transformer.py), which
  tests complete attention lanes, private state, shared read-only operators,
  RoPE controls, explicit joins, and mutation isolation.

Their tests and sealed JSON outcomes remain colocated.

Run them with:

```sh
python3 -m pytest -q research/cdp/gated-transformer-regime-lanes
```

The historical full suite also depended on an unshipped private serving
package and its source-repository experiment harness. That path is not a public
reproduction route and is not required for the 16 toy checks above. The exact
public-integration transformations are recorded in `SOURCE.json`.

The final paper-branch verification before curation was:

```text
240 passed in 5.96s
LANE_PRESERVATION_CUSTODY=PASS
SCALEUP_SOURCE_CUSTODY_AND_COMPLETION=PASS
```

After curation, the publication-branch toy suite and link/claim checks remain
the local executable gate; the two full custody verifiers operate against the
archived complete repository evidence.

## Final experimental denominators

| Evidence | Denominator | Outcome |
|---|---:|---:|
| Final independently trained SFT states | 32 | All retained; one switch-treatment state failed acquisition. |
| Frozen private-mapping comparisons | 49,152 | 0 exact. |
| Wrong-corpus R4 opportunities | 147,456 | 0 foreign answers. |
| Keyed accesses in final SFT confirmation | 7,310,124 | Exact owner map. |
| Fixed-shape foreign intervention cases | 992 | 992 own-lane detector positives. |
| Fixed-shape off-lane comparisons | 2,976 | 2,976 bit exact. |
| Controlled long-decoder target comparisons | 2,056 | 2,056 bit exact. |
| R8 lifecycle target comparisons | 448 | 448 bit exact. |
| R16 lifecycle target comparisons | 896 | 896 bit exact. |
| Final 256-token qualitative responses | 128 | 125 EOS; 128 topical; 0 foreign label substitutions. |

## Evidence rules

1. A file hash establishes byte identity, not correctness.
2. A passing custody verifier does not change a failed hypothesis into a pass.
3. Frozen protocols and failed runs remain in the archive unchanged.
4. Training-state, prompt, and fact denominators are reported separately.
5. Same-corpus order seeds are not treated as independent corpus draws.
6. Cross-shape numerical effects are controlled against ordinary Qwen and are
   not called foreign semantic influence.
7. Zero observed foreign answers is a finite empirical result, not a universal
   security or noninterference theorem.
8. Provider resource costs are configured-rate estimates, not audited invoices.

## Archive custody

The complete evidence is pinned by:

```text
branch: codex/gated-head-evidence-archive
commit: 4e8e0962afc920e15de1731c626384554dd6534c
gated-head subtree: 50de978111b5f5dad7f246fa682fd945e154a22e
```

See [EVIDENCE_ARCHIVE.md](EVIDENCE_ARCHIVE.md) for recovery commands and the
preservation rule.
