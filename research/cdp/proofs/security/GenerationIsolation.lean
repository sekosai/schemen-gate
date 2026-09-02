/-
Copyright (c) 2026 Ryan R. All rights reserved.
Released under Apache 2.0; see LICENSES/Apache-2.0.txt.
Authors: Ryan R
-/
import SerialGateSecurity

/-!
# Generation Isolation — Formal Verification (Tier 1.5, Generation)

Machine-checked Lean formalization of the Non-Execution Independence
Theorem and its corollaries for the frozen-backbone / per-regime
adapter architecture ("Schemen-Adapter"), as specified in
`docs/generation-isolation-handoff.md` §3-5.

This module builds directly on `SerialGateSecurity.lean`
(`Vec`, `Adapter`, `indicator`, `hmul` / `⊙`, `serial_output_confined`).
Everything below is added on top: no existing theorem is weakened
or replaced.

## Informal statement

In the frozen-backbone serial-gate architecture, at inference with
active regime set `S`, the forward pass never evaluates `adapter_R`
for any `R ∉ S`. Therefore the forward pass output is a pure
function of `(input, θ_backbone, {adapter_R}_{R ∈ S})`. This
propagates through the LM head, any logit-function sampler, and
full autoregressive decoding: the entire generated token sequence
is pointwise identical for any choice of inactive-regime adapters.

## What is proven (module contents)

### Block-level (§ 2 below)
- `adapterInjection_congr_on_active`: adapter-sum depends only on
  active regimes.
- `blockForward_independent_of_inactive`: one transformer block's
  output depends only on active-regime adapters.

### Stack-level (§ 3 below)
- `layerStack_independent_of_inactive`: any finite composition of
  frozen blocks inherits the block-level independence by induction
  on the list of blocks. (Handoff §5.3 theorem 2.)

### Readout-level (§ 4 below)
- `logits_independent_of_inactive`: the LM head, modeled as an
  arbitrary function `Vec n → Logits V`, preserves independence.
- `function_of_logits_independent`: ANY function of the logits
  (greedy argmax, top-k selection, temperature-scaled sampling
  under a fixed random seed, or any deterministic post-processor)
  inherits the independence.

### Autoregressive-level (§ 5 below)
- `autoregressive_independent_of_inactive`: the full autoregressive
  decoding loop, for any number of generation steps `T`, produces
  bit-identical token sequences under any two adapter families that
  agree on the active set.
- `generation_invariant_under_inactive_replacement`: replacing any
  inactive regime's adapter by ANY alternative (zero, random,
  adversarial) leaves the output identical. This is the formal
  "the adversary cannot even distinguish the deployed model from
  a model with their adapter wiped" statement.

### Steganographic failure, generation analogue (§ 6 below)
- `generation_steganographic_failure`: if two adapter families
  agree on the masked regime set `A`, the generation output under
  mask `A` is identical. Structural / algebraic counterpart to the
  statistical "gen_A is independent of D_B" claim in
  `docs/generation-isolation-handoff.md` §4.
- `gen_independent_of_inactive_data`: explicit data-parameterized
  version — if each adapter is the output of a training function
  applied to per-regime training data, the generation output under
  `A` is a function of data indexed by `A` only. Formalizes the
  handoff §2.2 primary claim that `gen_A(x)` does not depend on
  `{D_R}_{R ∉ A}`.

### Link to partition confinement (§ 7 below)
- `adapterInjection_summand_confined`: when a regime's mask is a
  partition indicator, its summand in the adapter-injection term
  is zero outside the partition. Direct invocation of
  `SerialGateSecurity.serial_output_confined`.

## What is NOT proven here (and why)

Per handoff §5.4, these are deliberately out of scope:

- **Backbone cleanliness** (the backbone was trained without any
  regime's data). Deployment / protocol obligation, not a
  model-math invariant; must be attested by the lockbox.
- **Training-pipeline hygiene** (adapter_A was actually trained on
  D_A only, with no cross-contamination). Operational concern.
- **Opacity under open-vocabulary generation** (the 2^(n/R) mask-
  search argument does not apply; reduces to AES-256-GCM / PRF on
  the lockbox). See `docs/citizen-rights.md` Tier 1.5.
- **Valid-LM hypothesis** (the backbone produces well-formed text).
  Property of the public pretrained LM, not a Schemen invariant.
- **Memorization within a regime**. Orthogonal; see handoff §7.2.
- **Side channels** (timing, memory, cache). execution / sidecar layer.
- **KV-cache hygiene.** `autoregressive` in §5 recomputes the full
  forward pass at every step — it models cache-free inference.
  Real deployments use KV caches; a cache contaminated by a
  previous request under a different active set would violate the
  theorem's assumption that each step is a fresh forward pass.
  Per-request cache isolation is a runtime obligation (sidecar),
  not a model-math invariant.

## Build integration

Registered in `proofs/lakefile.lean` as `lean_lib «GenerationIsolation»`,
together with `lean_lib «SerialGateSecurity»`. `lake build` from the
`proofs/` directory compiles both alongside the rest of the corpus.
-/

set_option autoImplicit false

noncomputable section

namespace Schemen.Generation

open Schemen.SerialGate


-- ════════════════════════════════════════════════════════════════
-- §1. DEFINITIONS (Handoff §2.1 & §5.2)
--
-- A frozen transformer block carries only backbone parameters
-- (attention, FFN, two LayerNorms), all of which are opaque
-- functions that pre-date any regime. A regime bundles an
-- opaque adapter function with a coordinate-mask. An adapter
-- family is an indexed collection of regimes.
--
-- The active regime set `S : Finset (Fin K)` is a Finset so
-- that the adapter-injection sum is finite. K (total number
-- of registered regimes) is a parameter.
-- ════════════════════════════════════════════════════════════════

/-- A regime: an opaque adapter function together with its mask.
    The adapter reads the full hidden state; its output is
    multiplied pointwise by the mask. Nothing is assumed about
    the mask being a 0/1 indicator — the independence theorems
    hold for any mask vector. When the mask is an `indicator`,
    `serial_output_confined` additionally guarantees partition
    confinement of the active contribution. -/
structure RegimeAdapter (n : ℕ) where
  f    : Adapter n
  mask : Vec n

/-- A family of K regimes. -/
def AdapterFamily (n K : ℕ) := Fin K → RegimeAdapter n

/-- The frozen backbone of a single transformer block.
    No field depends on any regime or adapter. -/
structure FrozenBlock (n : ℕ) where
  /-- Attention sublayer. Opaque; depends only on backbone params. -/
  attn : Vec n → Vec n
  /-- Feed-forward sublayer. Opaque; depends only on backbone params. -/
  ffn  : Vec n → Vec n
  /-- Pre-attention LayerNorm. -/
  ln1  : Vec n → Vec n
  /-- Pre-FFN LayerNorm. -/
  ln2  : Vec n → Vec n

/-- The adapter-injection term at a layer, indexed over the active
    set `S`. Note the sum iterates over `S` only — inactive regimes'
    adapters (those at indices `R ∉ S`) are never evaluated. This
    is the single structural fact the whole chain of theorems
    depends on. -/
def adapterInjection {n K : ℕ}
    (S : Finset (Fin K)) (fam : AdapterFamily n K) (h : Vec n) : Vec n :=
  S.sum (fun R => (fam R).f h ⊙ (fam R).mask)

/-- Post-attention residual state `h + Attn(LN₁(h))`. Factored so
    proofs about the block's output do not have to unfold a `let`
    chain. Backbone-only (no dependence on any adapter). -/
def postAttn {n : ℕ} (blk : FrozenBlock n) (h : Vec n) : Vec n :=
  h + blk.attn (blk.ln1 h)

/-- One transformer block's forward pass under active set `S`:
    ```
    h₁   = h + Attn(LN₁(h))                                    -- postAttn
    h₂   = h₁ + FFN(LN₂(h₁)) + Σ_{R ∈ S} (mask_R ⊙ f_R(LN₂(h₁)))
    ```
    Matches the equations in `generation-isolation-handoff.md` §2.1. -/
def blockForward {n K : ℕ}
    (blk : FrozenBlock n)
    (S : Finset (Fin K)) (fam : AdapterFamily n K)
    (h : Vec n) : Vec n :=
  postAttn blk h
    + blk.ffn (blk.ln2 (postAttn blk h))
    + adapterInjection S fam (blk.ln2 (postAttn blk h))


-- ════════════════════════════════════════════════════════════════
-- §2. BLOCK-LEVEL INDEPENDENCE (Handoff §5.3 item 1)
--
-- If two adapter families agree on the active set S, the
-- adapter-injection sum is pointwise equal (Finset.sum_congr),
-- and hence the whole block output is equal — everything else
-- in the block is backbone-only.
-- ════════════════════════════════════════════════════════════════

/-- **Lemma (Adapter Injection Congruent On Active).**
    The adapter-injection sum is determined by the restriction of
    the adapter family to the active set `S`. -/
theorem adapterInjection_congr_on_active {n K : ℕ}
    (S : Finset (Fin K))
    (fam fam' : AdapterFamily n K)
    (h : Vec n)
    (h_agree : ∀ R ∈ S, fam R = fam' R) :
    adapterInjection S fam h = adapterInjection S fam' h := by
  unfold adapterInjection
  apply Finset.sum_congr rfl
  intro R hR
  rw [h_agree R hR]

/-- **Theorem 1 — block level (Handoff §3.2, single-block case).**

    The one-block forward pass under active set `S` is identical
    for any two adapter families that agree on `S`. In particular,
    `{f_R}_{R ∉ S}` can be changed to anything — zero, random,
    adversarial — without altering the output.

    Proof: `h`, `attn`, `ffn`, `ln1`, `ln2` depend only on the
    backbone. The adapter-injection term is a `Finset.sum`
    indexed by `S`, so by `Finset.sum_congr` it depends only on
    `fam` restricted to `S`. -/
theorem blockForward_independent_of_inactive {n K : ℕ}
    (blk : FrozenBlock n)
    (S : Finset (Fin K))
    (fam fam' : AdapterFamily n K)
    (h : Vec n)
    (h_agree : ∀ R ∈ S, fam R = fam' R) :
    blockForward blk S fam h = blockForward blk S fam' h := by
  unfold blockForward
  rw [adapterInjection_congr_on_active S fam fam' _ h_agree]


-- ════════════════════════════════════════════════════════════════
-- §3. STACK-LEVEL INDEPENDENCE (Handoff §5.3 item 2)
--
-- The full transformer forward pass is a fold of `blockForward`
-- over a list of blocks. We induct on the list: at each step the
-- previous hidden state is equal (by induction), and the block
-- step preserves equality (by §2).
-- ════════════════════════════════════════════════════════════════

/-- Sequential application of a list of frozen blocks, left-to-right.
    Represents the full transformer stack between embedding and the
    LM head. -/
def layerStack {n K : ℕ}
    (S : Finset (Fin K)) (fam : AdapterFamily n K)
    (blocks : List (FrozenBlock n))
    (h : Vec n) : Vec n :=
  match blocks with
  | []           => h
  | blk :: rest  => layerStack S fam rest (blockForward blk S fam h)

/-- **Theorem 1 — Non-Execution Independence of the Forward Pass.**
    (Handoff §3.2 Theorem 1, formalized through the full layer
    stack.)

    For any finite list of frozen blocks, the final hidden state
    after the full transformer stack depends only on adapters in
    the active set `S`. Two adapter families that agree on `S`
    produce identical stack outputs, bit-for-bit, for every input.

    Proof: induction on `blocks`. Base: identity. Step: use
    `blockForward_independent_of_inactive` at the head block,
    then apply the IH to the tail. -/
theorem layerStack_independent_of_inactive {n K : ℕ}
    (S : Finset (Fin K))
    (fam fam' : AdapterFamily n K)
    (blocks : List (FrozenBlock n))
    (h : Vec n)
    (h_agree : ∀ R ∈ S, fam R = fam' R) :
    layerStack S fam blocks h = layerStack S fam' blocks h := by
  induction blocks generalizing h with
  | nil => rfl
  | cons blk rest ih =>
      have heq : blockForward blk S fam h = blockForward blk S fam' h :=
        blockForward_independent_of_inactive blk S fam fam' h h_agree
      calc layerStack S fam (blk :: rest) h
          = layerStack S fam rest (blockForward blk S fam h) := rfl
        _ = layerStack S fam rest (blockForward blk S fam' h) := by rw [heq]
        _ = layerStack S fam' rest (blockForward blk S fam' h) := ih _
        _ = layerStack S fam' (blk :: rest) h := rfl


-- ════════════════════════════════════════════════════════════════
-- §4. READOUT / LOGIT / SAMPLER INDEPENDENCE
--
-- The LM head is modeled as an arbitrary function `Vec n → Logits V`.
-- (In practice a linear projection W_LM plus LN, but the proofs
-- are indifferent to the specific form — only compositionality
-- matters.)
--
-- The sampler is modeled as an arbitrary function `Logits V → α`
-- for any output type α. This covers greedy argmax, top-k argmax
-- under a fixed tie-breaker, top-p selection under a fixed random
-- seed, or any deterministic post-processing of logits. For
-- stochastic samplers, treat the random seed as an extra function
-- argument; the theorem's universal quantification over α means
-- e.g. `α := Seed → Fin V` also works.
-- ════════════════════════════════════════════════════════════════

/-- Vocabulary-indexed logits. -/
abbrev Logits (V : ℕ) := Fin V → ℝ

/-- **LM-Head Independence (Handoff §5.3 item 3).**
    Applying an arbitrary readout `lm_head` to the stack output
    preserves independence of inactive adapters. -/
theorem logits_independent_of_inactive {n K V : ℕ}
    (blocks : List (FrozenBlock n))
    (lm_head : Vec n → Logits V)
    (S : Finset (Fin K))
    (fam fam' : AdapterFamily n K)
    (h : Vec n)
    (h_agree : ∀ R ∈ S, fam R = fam' R) :
    lm_head (layerStack S fam blocks h) =
    lm_head (layerStack S fam' blocks h) := by
  rw [layerStack_independent_of_inactive S fam fam' blocks h h_agree]

/-- **Sampler Independence (Handoff §5.3 item 4, deterministic form).**
    Any function of the logits — greedy, top-k, any deterministic
    or fixed-seed stochastic sampler — is independent of inactive
    adapters. The theorem is universally quantified over the
    output type `α`, so it covers not only token outputs but also
    sequence outputs, probability scores, or any other
    logit-derived quantity. -/
theorem function_of_logits_independent {n K V : ℕ} {α : Type}
    (blocks : List (FrozenBlock n))
    (lm_head : Vec n → Logits V)
    (sampler : Logits V → α)
    (S : Finset (Fin K))
    (fam fam' : AdapterFamily n K)
    (h : Vec n)
    (h_agree : ∀ R ∈ S, fam R = fam' R) :
    sampler (lm_head (layerStack S fam blocks h)) =
    sampler (lm_head (layerStack S fam' blocks h)) := by
  rw [logits_independent_of_inactive blocks lm_head S fam fam' h h_agree]


-- ════════════════════════════════════════════════════════════════
-- §5. AUTOREGRESSIVE GENERATION INDEPENDENCE
--
-- Full autoregressive decoding: given a prompt and a step count T,
-- iterate T times { embed the prefix, run the stack, apply LM head,
-- sample, append }. We prove by induction on T that the generated
-- sequence is identical under any two adapter families agreeing on
-- the active set.
--
-- This is the empirical analog of the differential-equivalence
-- test in handoff §6.3: bitwise identity, not "close," not "low
-- divergence," for every prompt and every T.
-- ════════════════════════════════════════════════════════════════

/-- A single autoregressive step: embed current sequence, run stack,
    apply LM head, sample next token, append. All non-adapter
    components (`embed_seq`, `blocks`, `lm_head`, `sampler`) are
    opaque — they represent the fixed backbone plus a deterministic
    sampler. For a stochastic sampler, treat the random seed as a
    closed-over value; the theorem still holds under the same seed. -/
def arStep {n K V : ℕ}
    (blocks : List (FrozenBlock n))
    (embed_seq : List (Fin V) → Vec n)
    (lm_head : Vec n → Logits V)
    (sampler : Logits V → Fin V)
    (S : Finset (Fin K)) (fam : AdapterFamily n K)
    (seq : List (Fin V)) : List (Fin V) :=
  seq ++ [sampler (lm_head (layerStack S fam blocks (embed_seq seq)))]

/-- Autoregressive generation for `T` steps starting from `prompt`. -/
def autoregressive {n K V : ℕ}
    (blocks : List (FrozenBlock n))
    (embed_seq : List (Fin V) → Vec n)
    (lm_head : Vec n → Logits V)
    (sampler : Logits V → Fin V)
    (S : Finset (Fin K)) (fam : AdapterFamily n K)
    (prompt : List (Fin V)) : ℕ → List (Fin V)
  | 0       => prompt
  | T' + 1  =>
      arStep blocks embed_seq lm_head sampler S fam
        (autoregressive blocks embed_seq lm_head sampler S fam prompt T')

/-- **arStep Independence.** One autoregressive step is a
    function of the logits, hence independent of inactive adapters. -/
theorem arStep_congr {n K V : ℕ}
    (blocks : List (FrozenBlock n))
    (embed_seq : List (Fin V) → Vec n)
    (lm_head : Vec n → Logits V)
    (sampler : Logits V → Fin V)
    (S : Finset (Fin K))
    (fam fam' : AdapterFamily n K)
    (seq : List (Fin V))
    (h_agree : ∀ R ∈ S, fam R = fam' R) :
    arStep blocks embed_seq lm_head sampler S fam seq =
    arStep blocks embed_seq lm_head sampler S fam' seq := by
  unfold arStep
  rw [layerStack_independent_of_inactive S fam fam' blocks _ h_agree]

/-- **Theorem (Autoregressive Generation Independence).**
    (Handoff §5.3 item 4 / Corollary 1 of §3.5.)

    For any list of frozen blocks, any embedder, any LM head, any
    deterministic sampler, any prompt, and any number of generation
    steps `T`: two adapter families that agree on the active set
    produce bit-identical token sequences.

    Proof: induction on `T`. Base (T = 0) returns the prompt,
    trivially equal. Step: apply the IH to make the prefix equal,
    then apply `arStep_congr` to make the next token equal. -/
theorem autoregressive_independent_of_inactive {n K V : ℕ}
    (blocks : List (FrozenBlock n))
    (embed_seq : List (Fin V) → Vec n)
    (lm_head : Vec n → Logits V)
    (sampler : Logits V → Fin V)
    (S : Finset (Fin K))
    (fam fam' : AdapterFamily n K)
    (prompt : List (Fin V))
    (h_agree : ∀ R ∈ S, fam R = fam' R) :
    ∀ T : ℕ,
      autoregressive blocks embed_seq lm_head sampler S fam prompt T =
      autoregressive blocks embed_seq lm_head sampler S fam' prompt T := by
  intro T
  induction T with
  | zero => rfl
  | succ T' ih =>
      show arStep blocks embed_seq lm_head sampler S fam
             (autoregressive blocks embed_seq lm_head sampler S fam prompt T')
         = arStep blocks embed_seq lm_head sampler S fam'
             (autoregressive blocks embed_seq lm_head sampler S fam' prompt T')
      rw [ih]
      exact arStep_congr blocks embed_seq lm_head sampler S fam fam' _ h_agree


-- ════════════════════════════════════════════════════════════════
-- §6. REPLACEMENT COROLLARY AND STEGANOGRAPHIC FAILURE (GEN.)
--
-- The corollaries in handoff §3.5 / §4: an adversary's inactive
-- adapters can be replaced by anything (zero, random, a red-team's
-- deliberate leak attempt) and the output under the active set
-- does not move. This is the sharpest form of the algebraic
-- "generated text reveals nothing about inactive regimes" claim.
-- ════════════════════════════════════════════════════════════════

/-- **Corollary (Inactive Adapter Replacement).**
    Build a new family `fam'` from `fam` by replacing every
    inactive regime (`R ∉ S`) with an arbitrary `replacement R`.
    The autoregressive output is unchanged.

    Formal version of: an adversary observing samples from
    `gen_S(·)` cannot distinguish the deployed model from a model
    in which every inactive `adapter_R` has been replaced by an
    arbitrary function — zero, random, or deliberately adversarial. -/
theorem generation_invariant_under_inactive_replacement
    {n K V : ℕ}
    [DecidableEq (Fin K)]
    (blocks : List (FrozenBlock n))
    (embed_seq : List (Fin V) → Vec n)
    (lm_head : Vec n → Logits V)
    (sampler : Logits V → Fin V)
    (S : Finset (Fin K))
    (fam : AdapterFamily n K)
    (replacement : Fin K → RegimeAdapter n)
    (prompt : List (Fin V)) (T : ℕ) :
    autoregressive blocks embed_seq lm_head sampler S fam prompt T =
    autoregressive blocks embed_seq lm_head sampler S
      (fun R => if R ∈ S then fam R else replacement R) prompt T := by
  apply autoregressive_independent_of_inactive
    blocks embed_seq lm_head sampler S fam
    (fun R => if R ∈ S then fam R else replacement R)
    prompt
  intro R hR
  simp [hR]

/-- **Theorem (Generation Steganographic Failure — Structural).**
    (Handoff §4, structural / algebraic counterpart.)

    Fix a masked regime set `A`. If two adapter families `fam`
    and `fam'` agree on `A`, the autoregressive generation output
    under mask `A` is bit-identical for any prompt and any number
    of steps.

    Interpretation: if the backbone was pretrained without access
    to any regime's data (a protocol obligation, not a theorem),
    and each `adapter_R` for `R ∈ A` was trained only on `D_R`
    (ditto), then the deployed generation output is a function of
    `(prompt, backbone, {adapter_R}_{R ∈ A})` — which in turn is
    a function of `(prompt, backbone, {D_R}_{R ∈ A})`. No argument
    of this function references `D_B` for any `B ∉ A`. Hence
    (structurally/algebraically) the output reveals nothing about
    any other regime's training corpus.

    This is the generation analogue of V3's `regime_output_locality`
    at the forward-pass level, extended through the LM head and
    the autoregressive loop. -/
theorem generation_steganographic_failure {n K V : ℕ}
    (blocks : List (FrozenBlock n))
    (embed_seq : List (Fin V) → Vec n)
    (lm_head : Vec n → Logits V)
    (sampler : Logits V → Fin V)
    (A : Finset (Fin K))
    (fam fam' : AdapterFamily n K)
    (h_agree_on_A : ∀ R ∈ A, fam R = fam' R)
    (prompt : List (Fin V)) (T : ℕ) :
    autoregressive blocks embed_seq lm_head sampler A fam prompt T =
    autoregressive blocks embed_seq lm_head sampler A fam' prompt T :=
  autoregressive_independent_of_inactive
    blocks embed_seq lm_head sampler A fam fam' prompt h_agree_on_A T

/-- **Theorem (Generation Independent of Inactive-Regime Training Data).**
    (Handoff §2.2 primary claim, explicit data-parameterized form.)

    Parameterize each regime's adapter by a hypothetical training
    function `train R : Data R → RegimeAdapter n`. Two data
    assignments that agree on the active set `A` produce identical
    generation output.

    Since the disagreement on `Aᶜ` can be arbitrary — different
    corpora, zero data, noise, or adversarially chosen data — the
    output under `A` is (functionally) independent of the training
    data of any regime outside `A`.

    This is the explicit formalization of the handoff's primary
    claim: `gen_A(x)` does not depend on `{D_R}_{R ∉ A}` in any
    way. It follows immediately from
    `autoregressive_independent_of_inactive`.

    The theorem is per-regime polymorphic in the training-data
    type (`Data : Fin K → Type`) so different regimes can have
    different data types. The `train` function is opaque and
    can embed any optimizer, hyperparameters, or stateful training
    pipeline, provided it is deterministic in its own regime's
    data. -/
theorem gen_independent_of_inactive_data {n K V : ℕ}
    {Data : Fin K → Type}
    (blocks : List (FrozenBlock n))
    (embed_seq : List (Fin V) → Vec n)
    (lm_head : Vec n → Logits V)
    (sampler : Logits V → Fin V)
    (A : Finset (Fin K))
    (train : (R : Fin K) → Data R → RegimeAdapter n)
    (data₁ data₂ : (R : Fin K) → Data R)
    (h_agree_on_A : ∀ R ∈ A, data₁ R = data₂ R)
    (prompt : List (Fin V)) (T : ℕ) :
    autoregressive blocks embed_seq lm_head sampler A
      (fun R => train R (data₁ R)) prompt T =
    autoregressive blocks embed_seq lm_head sampler A
      (fun R => train R (data₂ R)) prompt T := by
  apply autoregressive_independent_of_inactive
  intro R hR
  rw [h_agree_on_A R hR]


-- ════════════════════════════════════════════════════════════════
-- §7. LINK TO PARTITION CONFINEMENT
--
-- When a regime's mask is a partition indicator — the SOP
-- assumption under which `SerialGateSecurity` gives its output
-- confinement guarantees — each active regime's contribution is
-- ALSO confined to its partition coordinates at the point of
-- injection. The §2-§6 theorems above do NOT need this: they hold
-- for any mask vector. The stronger combined statement below
-- connects the GENERATION isolation theorems of this module to
-- the PARTITION confinement theorems of `SerialGateSecurity` at
-- the hand-off point: the adapter-injection term.
-- ════════════════════════════════════════════════════════════════

/-- **Theorem (Adapter-Injection Summand Confinement).**
    When regime `R`'s mask is the indicator of its partition
    `P_R`, the `R`-th summand of the adapter-injection term is
    zero at every coordinate outside `P_R`. Direct invocation of
    `serial_output_confined` restated at the injection level. -/
theorem adapterInjection_summand_confined {n K : ℕ}
    (fam : AdapterFamily n K)
    (h : Vec n)
    (R : Fin K) (P_R : Finset (Fin n))
    (h_mask : (fam R).mask = indicator P_R)
    (j : Fin n) (hj : j ∉ P_R) :
    ((fam R).f h ⊙ (fam R).mask) j = 0 := by
  rw [h_mask]
  exact serial_output_confined (fam R).f h P_R j hj

end Schemen.Generation
