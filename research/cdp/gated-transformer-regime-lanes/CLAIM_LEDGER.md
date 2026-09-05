# Claim ledger

**Decision:** accepted as a controlled research baseline<br>
**Production status:** not deployed; not production-ready<br>
**Primary paper:** [SCHEMEN_GATED_TRANSFORMER_REGIME_LANES_PAPER.md](SCHEMEN_GATED_TRANSFORMER_REGIME_LANES_PAPER.md)

This ledger records the final disposition. Detailed result history and every
correction are in [RESULTS_AND_CORRECTIONS.md](RESULTS_AND_CORRECTIONS.md). Exact
historical documents remain in the [evidence archive](EVIDENCE_ARCHIVE.md).

## Supported within the tested scope

### C1. Gate placement

Schemen authorization completed before the first private attention-bank lookup.
Wrong-model, wrong-width, wrong-regime, empty-authority, unscoped, and stale-
generation probes failed before private access in the evaluated implementations.

**Scope:** pinned Qwen 0.5B and 4B runners and enumerated malformed requests.

### C2. Complete attention, not head partitioning

Each regime owned complete Q/K/V/O attention at every evaluated decoder block,
retaining the checkpoint's native heads, grouped-query structure, and RoPE.

**Scope:** implemented attention banks; no claim that ordinary native heads are
independently partitionable.

### C3. Same-shape matched equivalence

Dormant regime copies did not change the selected lane under the evaluated
same-shape controls. Qwen 0.5B positioning and layerwise captures were bit
exact; Qwen3-4B R1/R2/R4 replication was exact at captured sites; R8/R16
lifecycle targets were exact.

**Scope:** explicitly matched model revision, backend policy, shape, parameters,
and request state.

### C4. Finite off-lane causal separation

Foreign content, attention parameters, residual/FFN rows, and KV state changed
their own lanes while leaving 2,976/2,976 tested other-lane comparisons bit
exact. Controlled foreign-history decoders preserved 2,056/2,056 target
comparisons. Deliberately broken joins and selected-lane mutations were
detector-positive.

**Scope:** enumerated fixed-shape interventions; not arbitrary hostile memory.

### C5. Corpus separation

The final confirmation produced zero foreign answers across all 96 wrong-
corpus R4 cells and 147,456 opportunities. The frozen backbone scored zero on
49,152 private-mapping comparisons.

**Scope:** disjoint random mapping corpora and held-out wording; not natural-
language secrecy in general.

### C6. Independent SFT state

Successful standalone learned checkpoints retained their task performance when
loaded into owning gated lanes. Checkpoint transfers were exact. A resumed B
update matched its physical control and left A/C/D logits, versions, gradients,
and parameters unchanged.

**Scope:** selected private attention/norm leaves and the tested sequential
update protocol; simultaneous SFT and serving remain disabled.

### C7. Shared immutable backbone

Embedding, frozen FFN, final normalization, and LM-head weights were shared
without observed foreign-lane influence. Batched rows remained independently
addressable; no regime-axis semantic reduction was present.

**Scope:** immutable tested operators. A mutable shared operator would be a new
architecture and requires new controls.

### C8. R8/R16 lifecycle

R8 passed 448/448 and R16 passed 896/896 cached target comparisons under tested
foreign histories, cancellation, replacement, row reorder, refresh, and stale-
generation events, with zero off-lane logit delta or argmax flips.

**Scope:** one pinned Qwen revision, one A100 worker, eager policy, and active
batches no larger than eight.

### C9. Footprint and bounded throughput

The R2 composite artifact was 38.27% smaller than a two-checkpoint ceiling. R8
and R16 persistent inventories were 33.866 GiB and 61.999 GiB. R4 row batching
improved 64/256-token prefills but not 1,024-token prefill. The R16 contiguous
selector improved the tested half-roster median by 18.68%.

**Scope:** exact reported processes, layouts, workloads, and hardware.

## Rejected or not promoted

### N1. Native-head or RoPE authority

Rejected. Native attention heads and RoPE do not own the complete residual,
cache, sequence, training, or lifecycle state. RoPE remains unmodified within
the selected attention computation.

### N2. Universal native/R1/R4 bit identity

Rejected. Some batch, padding, and backend changes produced numerical
differences in both gated and ordinary native Qwen. Exact claims require an
explicit same-shape execution contract.

### N3. Current switch-oriented SFT curriculum

Not promoted. It improved R4 exact switching from 36/512 to 438/512 but reduced
lookup from 24,504/24,576 to 23,317/24,576. One state failed acquisition and
only 6/16 treatment states cleared the strict combined gate.

### N4. Naive concurrent eager acceleration

Rejected. Independent CUDA-stream loops were slower than serial execution in
the attention bank, ordinary shared model, and two-full-model controls.

### N5. R32 deployment

Rejected as a current target. Persistent inventory was 126,984,060,928 bytes,
and one automatic-SDPA path was nondeterministic. R32 remains useful limit
evidence.

### N6. Unrestricted conversational quality

Not established. Long-form outputs were generally topical and interpretable,
but retained truncations, factual errors, and same-lane lexical/label-lock
failures. Review was not an independent expert panel.

## Open claims requiring new evidence

- heterogeneous continuous batching with independent EOS and row compaction;
- cancellation and cache teardown under sustained concurrent load;
- grouped private-attention kernels for long-context throughput;
- simultaneous lane-local SFT and inference;
- multi-process or hostile-tenant isolation;
- allocator erasure and cache-memory reuse;
- timing and resource-contention noninterference;
- distributed cache migration and parameter-generation revocation;
- arbitrary explicit joins;
- broad semantic/private-corpus generalization; and
- independent blinded human and domain-expert evaluation.

## Interpretation rule

No zero in this ledger is enlarged into a theorem. A custody PASS establishes
that the expected bytes, runs, and denominators were checked; it does not make
a failed hypothesis pass. A shared vocabulary or immutable representational
operator is not itself foreign learning. Conversely, any new mutable shared
state, alias, reduction, or implicit join falls outside this claim and must be
tested as a new architecture.
