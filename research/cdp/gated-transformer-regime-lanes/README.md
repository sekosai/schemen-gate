# Schemen-gated Transformer regime lanes

This is the curated publication surface for the gated Transformer regime-lane
research program.

## Start here

1. [Formal paper](SCHEMEN_GATED_TRANSFORMER_REGIME_LANES_PAPER.md)
2. [All results, failures, and corrections](RESULTS_AND_CORRECTIONS.md)
3. [Claim ledger](CLAIM_LEDGER.md)
4. [Reproducibility and custody](REPRODUCIBILITY.md)
5. [Full evidence archive](EVIDENCE_ARCHIVE.md)

## Result in one paragraph

The bounded architecture result is positive. A Schemen Gate selects a complete
regime-specific attention and decoder-state lane before any private lookup.
Every regime retains full Q/K/V/O attention, all native heads, and native RoPE,
plus its own residual trajectory, KV cache, sequence, positions, masks, output
state, and parameter generation. Embedding, FFN, final-normalization, and
vocabulary weights remain shared and read-only. In the tested fixed-shape Qwen
experiments, foreign content or state did not alter the target lane, every final
wrong-corpus cell produced zero foreign answers, and R8/R16 passed their cached
lifecycle controls. The current switch-oriented SFT curriculum, universal
cross-shape exactness, R32 deployment, and production-security claims remain
withheld.

## Headline evidence

| Measurement | Result |
|---|---:|
| Independently trained final-confirmation states | 32 |
| Frozen private-mapping comparisons | 0/49,152 exact |
| Wrong-corpus opportunities | 0/147,456 foreign answers |
| Final-confirmation keyed accesses | 7,310,124 exact |
| Fixed-shape foreign interventions | 992/992 own-lane detector positives |
| Corresponding off-lane comparisons | 2,976/2,976 bit exact |
| Controlled 256-step decoder targets | 2,056/2,056 bit exact |
| R8 lifecycle targets | 448/448 bit exact |
| R16 lifecycle targets | 896/896 bit exact |
| R8 persistent inventory | 33.866 GiB |
| R16 persistent inventory | 61.999 GiB |
| R2 artifact saving versus two checkpoints | 38.27% |

## What “R” means

`R` counts complete regime alternatives. It does not count or partition the
model's native attention heads. For every decoder block, each regime owns a
complete Q/K/V/O attention module. A request receives one canonical
`(model_id, regime_id, generation)` route before the first bank access.

Mutable lane ownership includes:

- complete private attention parameters;
- selected private normalization leaves during SFT;
- the request's full-width residual trajectory;
- K/V cache and cache length;
- token sequence, valid span, logical positions, and mask;
- emitted tokens, EOS, and stopping state;
- parameter generation; and
- training masters, optimizer moments, gradients, and RNG state when training.

Shared components are immutable operators, not shared mutable activations:

- token embedding;
- frozen FFN/MLP weights;
- final normalization;
- LM/vocabulary projection; and
- tokenizer/vocabulary.

Multiple rows may occupy one batch allocation. Their element ranges and owners
remain distinct, and there is no regime-axis reduction. A join must be an
explicit separately authorized operation.

## Current disposition

- **R8:** production-integration candidate, not production-ready.
- **R16:** retained stress tier.
- **R32:** memory/backend limit evidence only.
- **Lane architecture:** accepted as a controlled research baseline.
- **Switch-oriented SFT curriculum:** not promoted.
- **Automatic SDPA bit-exactness:** not promoted.
- **Naive multi-stream acceleration:** rejected by matched controls.
- **Universal isolation/security:** not claimed.

## Why the directory is small

The original flat directory contained 136 tracked files: frozen protocols,
amendments, results, failures, conversations, receipts, figures, and code. They
are preserved byte-for-byte on `codex/gated-head-evidence-archive` at commit
`4e8e0962afc920e15de1731c626384554dd6534c`.

The publication branch retains the paper, one complete disposition summary, a
concise claim ledger, reproducibility instructions, selected figures, and the
executable toy core. Nothing was discarded; the evidence was moved out of the
reader's critical path.
