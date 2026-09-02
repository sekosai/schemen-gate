# Experiment Reproduction

This directory is the research reproduction bundle for the CDP paper. Current
reruns install `schemen-gate` from the enclosing, manifest-verified source tree.
The exact Gate version, source commit, tree digest, and the schema of the small
source-visible execution preflight are fail-closed in
`experiments/schemen-library-lock.json` and recorded in new artifacts.

From `research/cdp`, install the enclosing Gate checkout by its resolved
absolute path plus the pinned research dependencies:

```bash
./scripts/setup.sh full
source .venv/bin/activate
python3 experiments/library_provenance.py
```

The Gate source must resolve to the exact clean commit under test; a different
version, commit, or dirty checkout aborts a canonical run. Remote runners copy
that reviewed source and the launcher into the Modal image. The image embeds the
clean build provenance; the remote process hashes the installed Gate package
and copied launcher, rejects caller-supplied provenance drift, and records the
remotely verified values. No executable server wheel, package registry
credential, or private-repository credential is included.

Every completed experiment writes a timestamped JSON artifact under
`experiments/results/`.

For the exact version 1.0.2, multi-launcher, cost-bounded re-certification
workflow, use
[`../../../docs/MODAL_RECERTIFICATION.md`](../../../docs/MODAL_RECERTIFICATION.md).
It is the authoritative campaign inventory; this file remains the construction
and individual-command reference.

Scored model evaluation in current canonical runners first enters a minimal
in-process research preflight. Model id, vector dimensions, and configured
Regime are checked before the architecture-specific callback is invoked. Each
runner probes wrong Regime, model, dimensions, and empty authority and requires
zero unauthorized model calls. This fixture tests callback ordering only; it is
not a server, credential verifier, or production authorization claim.

Canonical full-run artifacts live directly in `experiments/results/`.
Protocol smokes, failed controls, and superseded runs are retained under
`experiments/results/archive/` so they remain auditable without obscuring the
paper's evidence set.

## Gate-surface taxonomy

- **Pre-MLP FFN gate:** masks the FFN input immediately before the
  first projection. It zeros excluded input coordinates and loss
  gradients on aligned first-projection inputs, but does not by itself
  partition the expanded hidden activation or down projection.
- **Intermediate-FFN gate:** masks the expanded activation before the
  down projection. This is the placement used by the paper's strict
  parameter/optimizer confinement experiment.
- **Whole-model gate:** authorizes the complete model as one atomic
  resource before invocation. It does not mask internal coordinates,
  so the authorized model computation is unchanged.

Pre-MLP and intermediate placements are both FFN gates. Whole-model
authorization is an operational runtime boundary and is not the
all-ones special case of an FFN partition.

## Classification experiments

DistilBERT is a Transformer classifier. The limitation stated by the paper is
that these experiments do not partition its attention, normalization, residual
stream, caches, or other undeclared shared state. A shared Transformer encoder
may still feed a governed classifier or FFN surface.

The formative DistilBERT Series 1 code used a post-encoder
classification gate: it multiplied the final 768-dimensional CLS
vector by a regime mask. It did **not** gate DistilBERT's expanded
intermediate FFN activations.

The public reproduction makes that distinction explicit:

```bash
# Reproduce the formative Series 1 surface.
python3 experiments/reproduce_distilbert_classification.py \
  --surface legacy-post-encoder

# Run the paper-aligned gate at every DistilBERT FFN lin2 input.
python3 experiments/reproduce_distilbert_classification.py \
  --surface intermediate-ffn

# Validate installation and control flow on a tiny sample.
python3 experiments/reproduce_distilbert_classification.py --smoke
```

Results from the two surfaces are not interchangeable. Archived
`series1_results_*.json` files describe the legacy post-encoder
surface. A paper claim about intermediate FFN classification requires
new `intermediate-ffn` artifacts.

`reproduce_distilbert_classification.py` is an activation-placement and
utility experiment. Its co-training loop updates the whole encoder, including
shared attention and normalization parameters. It therefore does **not**
establish separated tenant state in a shared Transformer.

The legacy path preserves the formative protocol: all 24 four-class
label permutations are shuffled with NumPy seed 42, masks are derived
from each run seed using the published SHA-256/NumPy construction, and
only four separately trained controls are sampled at each ratio while
the gated model is trained across all `R` regimes. Model and dataset
revisions are pinned in the script and recorded in every artifact.

## Strict shared-cotenancy experiments

The newer cotenancy suite freezes every shared Transformer path that is not
explicitly governed and tests exact parameter and optimizer-state deltas.

```bash
# Complete one persisted reduced cycle and inspect it first.
python3 experiments/local_transformer_cotenancy_suite.py --canary

# CPU: pre-MLP/intermediate FFN, private lanes, broken placements,
# released Cargo scope binding and source-visible research callback preflight.
python3 experiments/local_transformer_cotenancy_suite.py

# GPU: dense FFN slices at R={1,2,4,8,16}.
modal run experiments/modal_dense_ffn_cotenancy.py --smoke
modal run experiments/modal_dense_ffn_cotenancy.py \
  --r-values 1,2,4,8,16 --seeds 42,123,256

# GPU: complete private residual-adapter and expert lanes.
modal run experiments/modal_private_transformer_lanes.py --smoke
modal run experiments/modal_private_transformer_lanes.py \
  --designs adapter,expert --seeds 42,123,256

# GPU: public-adaptation factorial, then strict R=8 tenant training.
# The full runner completes a reduced-data pilot before dispatching full seeds.
modal run experiments/modal_public_gate_adaptation_factorial.py --smoke
modal run experiments/modal_public_gate_adaptation_factorial.py \
  --seeds 42,123,256,512,1024

# Exact addressed-use equivariance, evaluated through execution authority.
modal run experiments/modal_orthogonal_superposition.py \
  --ratios 8,128
```

Full remote matrices dispatch in bounded batches of at most three GPU jobs.
Each command still writes one combined local artifact only after all of its
remote results have returned.

`modal_dense_ffn_cotenancy.py` trains only the active rows of each DistilBERT
FFN up-projection, the matching down-projection columns, and a private
regime-selected classifier. A regime-scoped Adam implementation prevents
momentum from updating inactive slices. Attention, embeddings, normalization,
residual parameters, and parameters outside the aligned FFN surface remain
frozen. Its masks use the publication-safe SHA-256-seed/NumPy balanced
research construction and record that algorithm explicitly; they are not
production key derivation.

`modal_private_transformer_lanes.py` keeps the complete pretrained backbone
frozen. The adapter design adds a small private residual block after every
Transformer layer; the expert design adds a full-width private FFN after every
layer. Both include a private classifier. These designs preserve full shared
attention while ensuring tenant corpus training cannot alter shared weights or
another tenant's trainable lane.

`modal_public_gate_adaptation_factorial.py` tests four initializations at
R=8: no extra public adaptation, an extra ungated hard-label epoch, an extra
all-mask hard-label epoch, and an extra all-mask hard-label-plus-distillation
epoch. The three adapted arms share examples, batch order, dropout seed
schedule, optimizer steps, teacher evaluations, and student forward/backward
counts. The ungated arm repeats each batch R times to match model-pass counts.
Public-adaptation and tenant-stage examples are disjoint. The runner reports
paired contrasts for ordinary extra training, mask-aware training, and
distillation given masks, plus exact frozen, off-partition, optimizer-moment,
and inactive-classifier deltas. The older three-seed distillation artifact is
retained as a preliminary unequal-training-budget comparison. A non-smoke run
first completes one reduced-data pilot and aborts before full dispatch if any
separation assertion fails. The pilot also populates and commits the shared
Hugging Face cache before the full-seed containers are launched.

The exact-zero checks establish model-state confinement only under the declared
runtime boundary. They do not claim that shared attention coordinates are
partitioned. The CPU suite also requires two deliberately invalid constructions
to fail: LayerNorm between a source activation and its gate must create a
cross-mask value, and an ungated residual bypass must carry the excluded source.
It separately checks the pre-MLP surface and verifies that unauthorized
whole-model credentials make zero model calls while the authorized output is
identical to direct execution.

The current one-seed R=8 strict artifact reports 87.756% owning accuracy and
exactly zero frozen, off-partition, optimizer-moment, and inactive-classifier
deltas. The current full public factorial reports +0.075 pp from ordinary
extra training, +1.331 pp from mask awareness, +0.031 pp from distillation
given masks, and +1.438 pp for the full pipeline. Its reduced-data pilot is
negative and is retained under `archive/smoke/`; that pilot is a useful warning
that tiny optimization runs are protocol checks, not utility estimates.

## Capacity-preserving modular classifier

The formative wide-classifier runner holds each regime's allocation fixed at
`q` hidden dimensions and scales total hidden width as `d = q * R`:

```bash
python3 experiments/capacity_preserving_wide_classifier.py
```

The default publication run uses `R=128`, `q=10`, 256 synthetic one-hot
observations, and 256 disjoint answers. It is a deterministic memorization and
scalability check, not a held-out generalization or privacy experiment. The
runner evaluates every owning key, all 32,512 wrong-key/foreign-observation
pairs, the all-regime union, and equivalence between sparse selected-support
execution and an explicit dense binary mask. This is the appropriate sizing
pattern for block-sparse classifiers or expert banks when capacity should grow
with the number of regimes rather than be divided among them.

## Deployment service consolidation

The R=8 DistilBERT SST-2 benchmark compares eight complete services, one frozen
backbone with eight private adapter slots, one shared FFN with authorized
slices, and physically extracted authorized slices:

```bash
modal run experiments/modal_distilbert_service_consolidation.py
```

The full-validation artifact reports 1.071 GB of checkpoint tensors and 1.103
GB resident CUDA state for eight services. The shared-backbone adapter control
retains 91.06% accuracy with 151.6 MB of checkpoint tensors and 156.2 MB
resident CUDA state. The adapters are zero-initialized in this deployment
control; this is not an adapter-training result. Post-hoc 1/8 FFN slicing lowers
accuracy to 67.66%. Physical extraction preserves the sliced result within the
declared fp16 tolerance and raises throughput from 1,585 to 2,872 samples/s.
All scored calls pass through execution authority and unauthorized probes make
zero model calls. The result supports frozen-backbone service consolidation;
it does not support utility-preserving naive narrowing of a dense Transformer.

## Authority-constrained learned MoE

`train_authorized_moe.py` trains a standard learned top-1 router inside each
fixed execution-authorized expert set. The security gate is not learned: execution
selects the regime and its candidate experts before softmax or dispatch. The
semantic router then learns which of two authorized experts should process each
example, using task loss plus a Switch-style load-balancing loss computed only
over that authorized set.

```bash
# Deterministic no-network canary
python3 experiments/train_authorized_moe.py \
  --synthetic --input-dimensions 32 --hidden-dimensions 16 --epochs 8

# Eight real held-out 20 Newsgroups binary tasks
python3 experiments/train_authorized_moe.py --epochs 20
```

The hard acceptance contract requires exact separate-to-packed logits and
predictions, zero unauthorized expert dispatch, zero inactive gradients,
optimizer state, and parameter change, fail-closed execution probes with zero
unauthorized model calls, and detection of the deliberately invalid
pre-softmax zero-logit construction. Every expert must also receive at least
10% of its regime's held-out dispatches, preventing a nominal MoE that has
collapsed to one expert, and every semantic router must record a nonzero
parameter delta. This is learned MoE routing under fixed
authorization, not learned security authority and not shared-attention
partitioning.

The canonical full artifact is
`results/authorized_learned_moe_20260831T182431_055309Z.json`. Across eight
held-out tasks it obtains 81.80% macro accuracy (5,101/6,231 micro-correct).
Packing the trained routers and experts changes zero logits and zero
predictions. Unauthorized dispatch is zero; every router moves during training;
both experts are used in every regime with a minimum held-out share of 42.4%;
and the dedicated active-regime training audit records zero inactive gradients,
optimizer state, and parameter change. Eight separate execution surfaces each
authorize exactly one regime; every cross-regime and malformed probe makes zero
model calls. The packed bank contains the same
router/expert bytes as the separate lanes, so this artifact proves learned
routing and lossless packing, not storage compression. Backbone deduplication
is measured separately by the service-consolidation benchmark above.

## execution-selected capability prefixes and token routing

`train_capability_token_moe.py` moves the learned router to the token level and
adds a private continuous prefix to each regime. The prefix is an internal
model parameter selected only after a singleton research preflight authority
accepts the regime. It is never parsed from request text. User tokens can
influence top-1 semantic routing among the two already-authorized experts, but
cannot select another prefix or expand the candidate expert set.

```bash
# Deterministic no-network spoofing and routing canary
python3 experiments/train_capability_token_moe.py \
  --synthetic --vocabulary-size 128 --maximum-tokens 24 \
  --embedding-dimensions 32 --hidden-dimensions 16 --epochs 8

# Eight real held-out 20 Newsgroups binary tasks
python3 experiments/train_capability_token_moe.py --epochs 20
```

The acceptance contract requires exact separate-to-packed token routes,
logits, and predictions; equivalence between candidate restriction and a
global pre-softmax authorization mask using negative infinity; zero
unauthorized token dispatch; and zero inactive gradients, optimizer state, and
parameter movement. A spoof probe includes literal `CAPABILITY_REGIME_n`
markers for every regime in user text and requires that execution-selected
authority remain unchanged. Every learned prefix and token router must move,
and both experts must receive at least 10% of held-out token dispatches.

This experiment deliberately does not treat a plaintext prompt prefix as a
credential: user-authored text is forgeable. The trusted capability is the
authenticated execution decision that selects an internal prefix and expert set.
The experiment is a token-routing classifier, not an autoregressive language
model, and exact zero dip means copy-and-pack equivalence rather than a claim
that an arbitrary narrowed Transformer preserves utility.

## Cargo and keyed corpus authorization

The current local cotenancy suite exercises released `schemen-gate` Cargo
scope binding and the source-visible research execution preflight. It binds
tenant, subject, regime, model digest, operation, policy version, and exact
partition key, and verifies every mismatched field is rejected before model
execution. Historical receipts retain their original dependency metadata, but
no obsolete duplicate implementation is shipped.

The GPU Cargo runner installs the same reviewed Gate source and research
preflight:

```bash
modal run experiments/modal_cargo_transformer_authorization.py
```

It stores two customer corpora in separate vector-store partitions, derives
operation-specific keys bound to tenant, sub-id, regime, model digest, and
policy version; the partition registration is bound to the same scope. A
successful Cargo dock is required before retrieved context is sent to a normal
shared FLAN-T5 Transformer. Generator invocation then passes through a separate
whole-model gate bound to tenant, sub-id, generation-model digest, and operation.
Random, other-tenant, other-sub-id,
other-operation, other-regime, and same-regime wrong-partition attempts must
be rejected with zero model calls.
Attention is not ablated or coordinate-gated; it sees only context released by
the authorized runtime. The vector store, authorization root key, and runtime
process remain trusted assets. This experiment demonstrates the public
protocol. It does not include a production IdP, policy authority, durable use
ledger, or root-key custody deployment.

The current artifact records 4/4 exact owning answers and rejects all 23
wrong-scope attempts with zero unauthorized generator calls.

## Causal generation

`modal_generative_intermediate.py` is the current causal-LM gate-placement
runner. It uses TinyLlama 1.1B with a gate on the expanded SwiGLU product at
the input of every `down_proj`, exactly the theorem-aligned intermediate FFN
surface. It uses pinned model/data revisions, the reviewed Gate source and
source-visible research preflight, matched initialization, packed non-padded
token blocks, and an exact
inactive gradient/parameter probe.

```bash
modal run experiments/modal_generative_intermediate.py --seeds 0 --smoke
modal run experiments/modal_generative_intermediate.py --seeds 11 --r 8
```

The one-seed R=8 run passes confinement and preflight authorization, but does not
show utility parity: gated mean token loss is 5.442 versus 4.230 ungated.
Owning canary loss is lower than wrong-key loss (3.013 versus 3.452), but no
canary is generated exactly. This is evidence for the placement and state
boundary, not a successful generative utility claim.

## Deployment measurements

The DistilBERT storage, live accelerator-memory, and routed-throughput
protocol is independently runnable without retraining:

```bash
python3 experiments/reproduce_distilbert_deployment.py
python3 experiments/reproduce_distilbert_deployment.py --smoke
```

These measurements describe the formative post-encoder architecture.
Hardware and software versions are recorded in each output artifact.

## Other included experiments

- `local_exact_extraction.py`: algebraically equivalent full-gated versus
  sliced projection with an explicit finite-precision forward-error bound.
- `orthogonal_superposition_sweep.py`: permutation conjugation and
  component ablation.
- `true_multiplexing_test.py`: concurrent keyed-basis placement.
- Other `modal_*.py` files are archived exploratory and follow-up studies.
  They are outside the narrowed paper evidence unless the paper says
  otherwise; in particular, older generative runners are not substitutes for
  the migrated intermediate-FFN runner above.

## Research masks

New canonical local experiments derive partitions with the locked
`schemen_gate.GateMask`. `research_masks.py` remains only for archived studies
that must reproduce the publication-safe predecessor construction; it is not
an authorization runtime.
