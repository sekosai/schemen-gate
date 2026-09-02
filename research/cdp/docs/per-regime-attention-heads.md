# Archived Hypothesis: Per-Regime Attention Heads

## Status

This note records a rejected intermediate design. It is not an evaluated CDP
embodiment and is not evidence for the paper.

Assigning complete heads to regimes is insufficient to isolate a Transformer.
A gate on the concatenated head outputs can zero the immediate contribution of
an inactive head before the output projection, but it does not close the
surrounding Transformer paths.

## What the head gate does establish

For

$$
\operatorname{MHA}(x)
=\operatorname{Concat}(\operatorname{head}_1,\ldots,
\operatorname{head}_h)W_O,
$$

a block-uniform mask applied to the concatenated head tensor before $W_O$
removes the selected head blocks at that tensor. The same mask zeros the loss
gradient flowing backward through those blocks. With an optimizer that respects
the same support, the corresponding Q/K/V and input-side $W_O$ slices can be
kept inactive for that branch.

That is a local head-output statement only. It is analogous to the local FFN
gate identity $0\cdot x=0$; it is not a complete Transformer isolation result.

## Why head-only partitioning fails

An ordinary Transformer recombines the gated attention output with paths that
are not head-local:

- the shared residual stream carries the block input around attention;
- normalization mixes or rescales the recombined representation;
- the FFN and the next block consume that shared residual state;
- later attention heads can read information already written into it; and
- KV caches, output heads, batching state, and runtime bypasses are not scoped
  by the head mask.

A pretrained model that was not trained for the ablation can also lose utility
because valid attention contributions are replaced with zeros. The archived
head and generative stress tests therefore do not demonstrate multi-regime
attention isolation or a performance-preserving attention gate.

Grouped-query attention makes the unit larger rather than solving the problem:
a K/V group and all associated query heads must remain aligned, but the shared
residual and normalization paths still remain.

## Construction required for attention isolation

Moving attention outside the shared trust boundary requires a complete private
lane per regime. Each lane must include:

1. all attention heads and their Q/K/V/output projections;
2. the FFN and normalization operations for the block;
3. every residual route carrying the lane's representation;
4. any lane-local cache, adapter, optimizer, and output state; and
5. a correctly placed runtime-selected gate before any cross-lane
   recombination.

Every lane and gate must be resolved from the authenticated tenant + sub-id +
capability scope. One missed residual, normalization, cache, or serving bypass
invalidates the claimed lane boundary even though the immediate gated tensor
still contains literal zeros.

This duplicated construction is expensive and has a substantial operator-error
surface. It remains future work outside the current paper. The evaluated CDP
construction leaves attention shared. Its internal coordinate claims are
limited to declared FFN surfaces: the narrower pre-MLP input gate and the
stronger evaluated intermediate-FFN gate. A separate whole-model gate can deny
invocation of an entire Transformer as one atomic resource, but does not
partition that Transformer's attention internals.
