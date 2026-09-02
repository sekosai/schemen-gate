# Experiment result manifest

Top-level JSON files are canonical full-run artifacts or durable legacy results
still discussed by the repository. Reduced, failed, superseded, and excluded
studies live under `archive/`.

Historical receipts may name dependencies that are not shipped or required
now. Absolute workstation and worktree paths were normalized to
`<historical-local-root>` and `historical-local-source:` markers before public
distribution; measured values, dependency versions, source commits, and
recorded artifact hashes were preserved. Current runners use the Gate source
and bundled research preflight.

The two canonical 2026-08-31 artifacts below were regenerated after public
release-boundary hardening removed private integration material and local-path
provenance from the candidate. Their measured outcomes were then compared with
the independent 2026-08-21 runs: all numerical and security outcomes match;
the only observed record differences are the scikit-learn version and the
public-safe terminology. These are local CPU experiments, not Modal runs.

## Public-safe evidence exports

Fourteen historical run records named in this manifest, in the data inventory,
and in the papers were produced under a private companion authorization harness
that is not part of this open-source release. Each ships as a documented public
evidence export rather than as the raw run record. Every measured value,
verdict, denominator, seed, dependency version, source commit, and recorded
artifact digest is unchanged. The private harness's package name, wheel path,
checkout path, callback-surface identifier, and product-styled prose references
are replaced by explicit `private-companion-harness` markers. Each export ends
with a `public_evidence_export` block naming the original record, its SHA-256,
the rules applied, and every transformed field path; the raw records are
retained privately. Two exports also drop a private-harness prefix from the
original filename, as listed below. These exports are not Gate-only reruns and
are not Modal recertification results. `scripts/release_check.py` verifies that
every concrete result filename cited in Markdown or TeX resolves to a tracked
artifact and that every export is indexed here with its original digest.

| Export | Original record | Original record SHA-256 |
|---|---|---|
| `capacity_preserving_wide_20260820_215713.json` | `capacity_preserving_wide_20260820_215713.json` | `c29d2bf03a58aa2f2f72758a7237c1542177d550aa61b7a1108e4483c5cb3f2b` |
| `capacity_preserving_wide_20260820_215848.json` | `capacity_preserving_wide_20260820_215848.json` | `291cdb6de597cf77a080a99d5e23868f2fec8548aebf5a57d1a45ca81b0f8688` |
| `capacity_preserving_wide_20260820_225331.json` | `capacity_preserving_wide_20260820_225331.json` | `aa3df79fb23194745a6bdbeaa0eed640bb76daa6e480b9ed1c537c3008017bb3` |
| `cargo_transformer_20260821T063827_829714Z.json` | `cargo_transformer_20260821T063827_829714Z.json` | `a31665448ef69a03f0733b09f73ff04026325422918ed4aaaf18fdcd67f8ea1a` |
| `dense_ffn_cotenancy_20260821T061623_781627Z.json` | `dense_ffn_cotenancy_20260821T061623_781627Z.json` | `12c753d54cb1f98bccb259a48bd6a5d74fa669f7d799e5e5db9b8c4190b5e851` |
| `distilbert_service_consolidation_20260821T135530_707107Z.json` | `distilbert_service_consolidation_20260821T135530_707107Z.json` | `d738806266a43dd67748b1407107923735e94dfef23f22f29ca6fcfa06fbedf7` |
| `generative_intermediate_combined_20260821T071152_101574Z.json` | `generative_intermediate_combined_20260821T071152_101574Z.json` | `add22cc070b223a59b43b0d0c69a4a6b5435a62b66283644af80ab1a08f7ec84` |
| `local_exact_extraction_20260820_215652.json` | `local_exact_extraction_20260820_215652.json` | `0ec391f2b064d5c34d23ee6c4af6c492a42bc869f88753b1a5d660df3b3ca592` |
| `private_transformer_lanes_20260821T062244_931925Z.json` | `private_transformer_lanes_20260821T062244_931925Z.json` | `e3490a9780da50827ab18b48e3cf7f480bcca8b3e2de5b473cbbf72b0464e6d6` |
| `public_gate_adaptation_factorial_20260821T063703_914999Z.json` | `public_gate_adaptation_factorial_20260821T063703_914999Z.json` | `0ad1a233684008fa284bdebf529fe62a491a4feb0bff10bb56234ed276d539f8` |
| `archive/smoke/transformer_cotenancy_local_20260812_100546.json` | `transformer_cotenancy_local_20260812_100546.json` | `2b4419a1f2585eff97eef42477acfaf388b485cddae42123afe9abc48db67781` |
| `transformer_cotenancy_local_20260820_215640.json` | `transformer_cotenancy_local_20260820_215640.json` | `0be48c39a1529397e4561db99ae2814609c9d518f557ca789f1f93095be1dd83` |
| `orthogonal_superposition_20260821T060951_717885Z_reanalysis.json` | same name with a private-harness prefix | `b1f58d1208c0b2d4ba602871ed73b6e04a401a0e64f9338e70fd2d0df8017123` |
| `archive/failed/orthogonal_superposition_20260821T060951_717885Z.json` | same name with a private-harness prefix | `c711fdfec949187a6bff8af04d8ec9d626e173180422896841e455fd6a2900c6` |

## Current classification and cotenancy evidence

- `series1_results_20260803_202919.json`: formative five-seed
  post-encoder DistilBERT classification.
- `archive/superseded/capacity_preserving_wide_20260812_111527.json`:
  historical pre-library `R=128`, `d=10R` synthetic wide-classifier run. The
  later locked-library rerun below is canonical.
- `dense_ffn_cotenancy_*.json`: strict intermediate-FFN capacity and
  confinement matrix.
- `private_transformer_lanes_*.json`: private adapter and expert lanes.
- `dense_ffn_cotenancy_20260821T061623_781627Z.json`: current-library,
  preflight-backed one-seed R=8 strict run; 87.756% owning accuracy and exact
  zero for every unauthorized state delta.
- `private_transformer_lanes_20260821T062244_931925Z.json`: current-library
  one-seed adapter/expert run; 90.66%/90.15% owning and 3.11%/3.28% wrong-key,
  with zero shared and inactive-lane change.
- `public_gate_adaptation_factorial_20260821T063703_914999Z.json`: matched
  one-seed R=8 factorial. Mask awareness contributes +1.331 pp and the full
  pipeline +1.438 pp; all separation checks pass.
- `public_gate_distillation_20260806T193105_795325Z.json`: preliminary
  three-seed all-mask distillation pipeline. The baseline lacks the extra
  public epoch, so this artifact is not an equal-work causal comparison.
- `cargo_transformer_20260821T063827_829714Z.json`: current-library Cargo and
  callback-preflight run; 4/4 exact owning answers and all 23 wrong-scope attempts
  rejected with zero unauthorized model calls.
- `transformer_cotenancy_summary.json`: aggregate of the 2026-08-06 three-seed
  strict-cotenancy and private-lane matrices, generated on 2026-08-12 and
  reported in the papers. The regeneration command below globs every top-level
  `dense_ffn_cotenancy_*.json` and `private_transformer_lanes_*.json`, so it
  now also folds in the 2026-08-21 one-seed corroboration runs and yields
  different means; the tracked file is the paper's aggregate.
- `transformer_cotenancy_local_20260820_215640.json`: historical locked-library
  local matrix; all seven designs pass, with zero unauthorized parameter
  and optimizer-state deltas and zero unauthorized model calls.

## Extraction and deployment

- `local_exact_extraction_20260820_215652.json`: locked-library extraction
  matrix. All 160 comparisons pass the conservative forward-error bound;
  fp32 is bit-identical in this 768-dimensional protocol and fp16 has maximum
  absolute difference 0.015625.
- `distilbert_deployment_20260806_103232.json`: publication-safe deployment
  smoke.
- `distilbert_service_consolidation_20260821T135530_707107Z.json`: full SST-2
  validation benchmark at R=8 through research preflight authority. One frozen backbone
  plus eight zero-initialized private adapters preserves 91.06% utility while
  reducing checkpoint and resident CUDA bytes by about 7.1x. Post-hoc 1/8 FFN
  slicing reaches 67.66%; physical extraction preserves those sliced logits
  within tolerance and raises throughput from 1,585 to 2,872 samples/s. This
  supports service consolidation, not utility-preserving naive FFN slicing.

## Authority-constrained learned routing

- `authorized_learned_moe_20260831T182431_055309Z.json`: clean-source R=8
  learned top-1 MoE result on eight held-out 20 Newsgroups binary tasks. Macro
  accuracy is 81.80% (5,101/6,231 micro-correct). Copying the independently
  trained routers and experts into one preflight-authorized bank changes zero
  logits and zero predictions. Both experts are used in every regime (minimum
  held-out share 42.4%), every router records nonzero parameter movement,
  unauthorized dispatch is zero, and a dedicated training-step audit records
  zero inactive gradients, optimizer state, and parameter change. All malformed
  probes against eight single-Regime preflights make zero model calls.
  SHA-256:
  `8988c2bb2e9dc8ddda08da6b4f75cf043bd05c2d8c234e702211e48fbe4f1c03`.

## Authority-selected capability prefixes and token routing

- `capability_prefix_token_moe_20260831T182453_013039Z.json`: clean-source
  R=8 token-level top-1 routing result on the same held-out 20 Newsgroups task
  family. Research preflight authority selects a private continuous router prefix and the
  corresponding two-expert candidate set; user text is ordinary data and
  cannot select either. Macro accuracy is 77.76% (4,848/6,231 micro-correct),
  across 311,477 non-padding token decisions. Separate and packed execution
  have exactly equal routes, logits, and predictions; global negative-infinity
  masking exactly matches candidate restriction; unauthorized dispatch is
  zero. A user-text spoof containing every `CAPABILITY_REGIME_n` marker also
  causes zero unauthorized dispatch. Every prefix and router moves during
  training, both experts are used in every regime (minimum held-out share
  41.7%), and the active-regime training audit records zero inactive gradient,
  optimizer state, and parameter change. All malformed probes against eight
  singleton preflight authorities make zero model calls. The packed and separate
  parameter bytes are identical, so this proves safe token routing and
  lossless packing, not compression or language-model generation. SHA-256:
  `37e0323a0d4657cf4f06739974476e0aa7ce94daefda494b73d43fc3d986cdc1`.

## Capacity sensitivity under Gate 1.0.1

- `capacity_preserving_wide_20260820_225331.json`: corrected canonical R=128
  synthetic sparse-capacity result. Owning is 256/256, wrong-key is 0/32,512,
  selected execution is exactly equal to dense execution, and all 128 preflight
  authorities pass. The ungated all-regime union is retained only as a
  diagnostic and does not control the verdict because serving always selects
  one authorized regime.
- `capacity_preserving_wide_20260820_215713.json`: the same 2,000-step metrics
  under the superseded verdict. Its `failure` status came from incorrectly
  requiring the ungated all-regime diagnostic to exceed 97.6%, despite perfect
  owning and wrong-key behavior.
- `capacity_preserving_wide_20260820_215848.json`: separately labeled
  3,000-step convergence sensitivity. Owning and wrong-key results are
  unchanged; the all-regime union rises to 252/256. It is not a replacement
  for the matched-budget result.

## Addressed-use orthogonal-placement evidence

- `orthogonal_superposition_20260821T060951_717885Z_reanalysis.json`:
  current-library preflight-backed run. Baseline and every addressed regime have
  identical 91.421% accuracy at R=8 and R=128. Maximum fp32 logit drift is
  $1.06\times10^{-5}$ and changes no prediction. The raw artifact with an
  over-strict auxiliary logit threshold is retained under `archive/failed/`.

- `superposition_sweep_*.json`
- `r1024_test_20260622_103122.json`
- `true_multiplexing_*.json`
- `mux_scaling_20260622_110934.json`

## Causal generation

- `generative_intermediate_combined_20260821T071152_101574Z.json`: pinned,
  current-library one-seed TinyLlama R=8 run at the post-SwiGLU,
  pre-down-projection gate. Exact inactive gradient/parameter confinement and
  preflight rejection checks pass. Gated token loss is 5.442 versus 4.230 for
  the matched ungated control; owning versus wrong-key canary loss is 3.013
  versus 3.452, with 0/16 exact owning generations. This is not utility parity.

Run `python3 experiments/analyze_transformer_cotenancy.py` to regenerate the
strict cotenancy aggregate.

Reduced canaries and canary-discovered orchestration failures are retained in
`archive/smoke/` and `archive/failed/`. They are debugging evidence, not
publication utility estimates.
