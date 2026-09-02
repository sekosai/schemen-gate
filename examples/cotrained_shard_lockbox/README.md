# Co-trained Gated MLP + Lockbox — Executable Fixture

This fixture co-trains a small MLP with the regime gate present from the first
step, packages aligned per-regime weight shards under AES-256-GCM keys, and
releases only the shards named by an Ed25519-signed grant. It is useful for
checking lifecycle and custody mechanics without pretending that a toy teacher
task is Transformer or production evidence.

| Boundary | Epistemic basis | Executable evidence in this fixture |
|---|---|---|
| Gate-aware task learning | Observed fixture measurement | Correct-mask accuracy is much greater than wrong-mask accuracy |
| Local gradient/update support | Proven for the modeled aligned-update contract (`weight_update_confined`, `end_to_end_isolation`); observed for this plain-SGD fixture | The Gate zeros inactive loss-gradient paths; this fixture intentionally has no momentum or weight decay |
| R4: released (shard-decrypted) inference is exact | Observed implementation behavior | Released-model logits are compared *exactly* with the reference |
| R5: non-keyed reads zero | Proven Gate algebra plus observed fixture behavior | Gated activations are exactly zero with nothing released |
| R1: AAD/grant substitution rejected | Observed behavior under the standard Ed25519 signature assumption | An AAD-tampered grant fails Ed25519 verification |
| R2: release scope does not widen | Observed implementation behavior | Only the regimes named in the verified grant are decrypted |
| Custody: unreleased shards are ciphertext | Observed behavior under standard AES-GCM and Ed25519 assumptions | Wrong-key decryption raises `InvalidTag` (AES-256-GCM) |

The co-training result is observed, not a theorem that arbitrary knowledge is
private. The output bias is shared, and this fixture has no Transformer
attention, residual, normalization, cache, or serving path. For the paper's
strict frozen-backbone protocol and optimizer requirements, see the root
[training guide](../../README.md#training-is-a-lifecycle-choice).

The bundled Lean modules prove the Gate algebra and a modeled aligned-update
contract. They do not contain an implementation-refinement theorem for this
lockbox fixture. The R1/R2/R4/R5 labels above name executable checks in this
example, not Lean theorem names.

AES-256-GCM and Ed25519 are real constructions. The teachers, data, and
single-process authority are fixtures. Failure of any assertion exits nonzero.

## Run

```bash
# From the repository root:
python examples/cotrained_shard_lockbox/demo.py
pytest -q tests/test_cotrained_shard_lockbox.py
```
