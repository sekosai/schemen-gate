# Full evidence archive

The publication branch is intentionally small. The complete research dossier
is preserved on a separate Git branch so protocols, amendments, failed pilots,
raw receipts, dialogue transcripts, and superseded interpretations remain
recoverable without overwhelming the paper surface.

## Immutable local pointer

- Branch: `codex/gated-head-evidence-archive`
- Commit: `4e8e0962afc920e15de1731c626384554dd6534c`
- Repository tree: `5822a03bb937f3176b973bf0cee06d5ded838bd0`
- Gated-head subtree: `50de978111b5f5dad7f246fa682fd945e154a22e`
- Tracked gated-head files: 136
- Intended public branch: `research/gated-transformer-head-evidence`

The intended public branch is a destination, not a claim that publication has
already occurred. Public transfer requires explicit authorization and remote
verification.

## Recovering the archive

The following commands apply only to an authorized checkout of the original
private source repository. The archive branch was deliberately not copied into
this public Schemen Gate repository.

List every preserved path:

```sh
git ls-tree -r --name-only codex/gated-head-evidence-archive -- \
  cdp/research/gated_transformer_head
```

Read one original file without switching branches:

```sh
git show \
  codex/gated-head-evidence-archive:cdp/research/gated_transformer_head/FILE.md
```

Create a temporary worktree for full review:

```sh
git worktree add /tmp/gated-head-evidence \
  codex/gated-head-evidence-archive
```

## Archive contents

The pinned subtree includes:

- toy implementations, tests, and sealed results;
- Qwen 0.5B complete-lane, positioning, SFT, layerwise, and concurrency studies;
- Qwen3-4B R2/R4 replication, duplex, decoder-stress, batching, and profiling;
- unsuccessful grouped and sequence-SFT studies plus their diagnostics;
- repaired SFT, causal influence, topology, and concurrency evidence;
- 24-state and 32-state scale-up/confirmation protocols and results;
- R32 lifecycle and SDPA failure diagnosis;
- R8/R16 production-bound measurements;
- complete conversation appendices and review packets;
- preregistrations, launch/timeout amendments, execution logs, and validation;
- source and claim custody ledgers; and
- machine-readable JSON receipts and figures.

The readable disposition of every study family, including failures and their
corrections, is in [RESULTS_AND_CORRECTIONS.md](RESULTS_AND_CORRECTIONS.md).
The archive remains authoritative for exact historical wording and bytes.

## Preservation rule

Do not rewrite the archive branch to make failed experiments look cleaner.
Corrections belong in new commits on the publication branch and in the summary
ledger. If the archive is eventually removed from ordinary branch listings,
first preserve the commit in a signed tag or an immutable release bundle and
verify that the gated-head subtree ID remains
`50de978111b5f5dad7f246fa682fd945e154a22e`.
