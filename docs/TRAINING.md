# Training and model boundaries

[Back to Schemen Gate](../README.md)

## Training is a lifecycle choice

The Binary Activation papers report several training protocols. They answer
different questions and should not be collapsed into one generic “gated
training” recipe.

### 1. Gate-aware post-encoder co-training

Use this when all regimes may update one shared encoder and the goal is useful
support-constrained task learning. Apply each sample's gate during training so
the optimizer can learn features that survive that support.

```python
import torch

from schemen_gate import GateMask

n_dims = 768
n_regimes = 8
masks = [GateMask.from_file(f"masks/regime_{r}.npy") for r in range(n_regimes)]

# Build once per device, not inside the hot path.
gate_bank = torch.stack(
    [mask.to_torch(device="cuda", dtype=torch.float32) for mask in masks]
)

# hidden: [batch, 768]; regime_ids: [batch], authority-resolved integers.
hidden = encoder(input_ids).last_hidden_state[:, 0]
gated = hidden * gate_bank[regime_ids]
logits = classifier(gated)
loss = torch.nn.functional.cross_entropy(logits, labels)
loss.backward()
optimizer.step()
```

This is co-training: all regimes can influence the shared encoder parameters.
It measures whether the model can learn under the support constraint; it is not
evidence that the shared encoder contains separated tenant-private state.

Mixed-regime minibatches are valid because each sample receives its own mask.
An intentionally authorized union (`mask_a | mask_b`) removes the boundary
between those included supports for that invocation.

### 2. Strict FFN tenant training over a frozen public backbone

Use this when the shared Transformer is accepted as public/frozen and private
trainable state must be confined to aligned FFN slices. Place the gate on the
expanded FFN activation after the element-wise nonlinearity and immediately
before the down projection:

```python
def gated_ffn(x, up_projection, down_projection, activation, gate_mask):
    expanded = activation(up_projection(x))
    gated = gate_mask.apply(expanded)
    return down_projection(gated)
```

That multiply gives an exact local forward zero and zero loss gradient at the
inactive activation coordinates. Exact parameter-state confinement additionally
requires all of the following:

1. Freeze attention, embeddings, normalization, residual/shared parameters,
   caches, and every other path outside the declared FFN surface.
2. Align the active support with the first projection's output rows, hidden
   bias entries, and the down projection's input columns in PyTorch storage.
3. Restrict the complete optimizer update—including momentum, Adam moments,
   weight decay, and post-step transforms—to the same active slices, or restore
   inactive weights and state after every step.
4. Close ungated residual, adapter, cache, and alternate-serving bypasses.
5. Evaluate the owning key and every wrong key on the same trained model, and
   audit frozen parameters, inactive slices, optimizer state, and inactive
   classifiers for exact change.

`GateMask.apply` supplies the activation gate. It does not currently install
model hooks, freeze a backbone, or provide a masked optimizer wrapper. A normal
optimizer with decoupled weight decay can move an inactive parameter even when
its loss gradient is zero.

### 3. Public mask-aware adaptation, then frozen tenant training

A shared backbone may first be adapted on public data with all intended masks,
then frozen before tenant-stage training. The paper reports preliminary
DistilBERT evidence that this can recover utility, but the retained comparison
is not a causal estimate: extra training, distillation, and mask awareness are
confounded. Treat this as an experimental initialization protocol, not a
proven default.

Never mix tenant-private data into the public adaptation stage if the resulting
shared weights are supposed to remain public.

### 4. Frozen backbone plus private adapters or experts

Use identity-selected private lanes when dividing one fixed FFN into `R`
narrow slices costs too much utility. Freeze the shared backbone, allocate a
separate adapter, classifier, or expert to each regime, and let the trusted
authority select which attachment may load.

This restores per-regime trainable capacity at a linear storage cost. It
controls attachment reachability and inactive-lane state; it does not make the
shared attention, residual stream, caches, or logs private.

The runnable fixture at
[`examples/cotrained_shard_lockbox`](../examples/cotrained_shard_lockbox/README.md)
shows gate-aware MLP co-training followed by encrypted per-regime shard release.
It is a fixture-scale protocol check, not Transformer or production evidence.

## Dimension and capacity rules

The current equal-width derivation API requires:

```text
n_dims % n_regimes == 0
```

Therefore `n_dims >= n_regimes`, and every regime receives
`n_dims / n_regimes` coordinates. Increasing `n_regimes` while holding
`n_dims` fixed reduces per-regime capacity.

If each regime needs a fixed private width `q`, design the governed layer as:

```text
n_dims = q * n_regimes
```

This preserves per-regime width but increases parameters or storage linearly.
Whole-model authorization and private attachments avoid dividing an existing
backbone, but they protect different surfaces.

## Model-training and Transformer boundaries

- Freezing a backbone prevents tenant-stage parameter updates; it does not make
  its existing representations private.
- Post-hoc masking of ordinary pretrained attention heads is a destructive
  ablation, not demonstrated Binary Activation isolation.
- The strict result governs intermediate FFN activations and aligned trainable
  state. Shared attention remains shared.
- Complete attention lanes would need separate Q/K/V, output projection,
  normalization, FFN, residual, and cache paths trained or distilled as lanes
  from the outset. That result is open.
- MoE experts are a more natural future authorization surface, but masking
  pre-softmax router logits with zero is insufficient; unauthorized experts
  must be excluded from selection and dispatch.
