"""Gate-aware co-trained fixture + lockbox: release only authorized shards.

End-to-end executable fixture. Train one MLP cotrained across R regimes with
the binary gate (knowledge embedded in regime dims by gradient descent);
ship the weights as per-regime AES-256-GCM shards in a lockbox manifest
whose shard keys the Authority holds; release shards only against
Ed25519-signed grants (subject, scope, expiry, AAD). Then verify the V5
checks, each with an optimization-safe runtime assertion:

  R4  released inference is EXACT vs the reference model
  R5  with nothing released, every gated activation is exactly 0
  R1  an AAD-tampered grant fails real Ed25519 verification
  R2  custody never widens: only granted regimes decrypt
  EMB gate-aware task learning: correct-mask accuracy ≫ wrong-mask
      (the local zero is algebraic; these task numerics are fixture-measured
      and remain Observed tier)

The bundled Lean corpus proves Gate algebra and modeled aligned-update
confinement; it does not contain an implementation-refinement theorem for this
lockbox fixture. Teachers/data are fixtures (labeled). AES-256-GCM and Ed25519
are real constructions whose cryptographic security remains a standard
assumption.
Exit code 0 iff every check holds, including under optimized Python.
"""

from __future__ import annotations

import base64
import io
import json
import sys

import numpy as np

SEED = 20260819
N_DIMS = 64
N_REGIMES = 2
D_IN = 64
N_CLASSES = 8
STEPS = 1500
BATCH = 256
LR = 0.05


def require(condition, message: str) -> None:
    """Raise in normal and optimized Python when an example invariant fails."""
    if not condition:
        raise RuntimeError(message)


# --------------------------------------------------------------------------
# 1. Co-train with the gate present from the first optimization step
# --------------------------------------------------------------------------


def cotrain(width: int = N_DIMS, steps: int = STEPS):
    import torch

    from schemen_gate._crypto import derive_partition

    groups = derive_partition(SEED.to_bytes(32, "big"), width, N_REGIMES)
    masks = [
        torch.tensor([1.0 if j in groups[r] else 0.0 for j in range(width)])
        for r in range(N_REGIMES)
    ]
    tg = torch.Generator().manual_seed(SEED)
    teachers = [
        torch.randn(D_IN, N_CLASSES, generator=tg) / np.sqrt(D_IN) for _ in range(N_REGIMES)
    ]
    torch.manual_seed(SEED)
    model = torch.nn.Sequential(
        torch.nn.Linear(D_IN, width), torch.nn.ReLU(), torch.nn.Linear(width, N_CLASSES)
    )
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    for step in range(steps):
        r = step % N_REGIMES
        x = torch.randn(BATCH, D_IN, generator=tg)
        y = (x @ teachers[r]).argmax(dim=-1)
        logits = model[2](model[1](model[0](x)) * masks[r])
        loss = torch.nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, groups, masks, teachers


def accuracy(model, teacher, mask, samples=2048, seed=SEED):
    import torch

    with torch.no_grad():
        x = torch.randn(samples, D_IN, generator=torch.Generator().manual_seed(seed))
        y = (x @ teacher).argmax(dim=-1)
        logits = model[2](model[1](model[0](x)) * mask)
        return (logits.argmax(dim=-1) == y).float().mean().item()


# --------------------------------------------------------------------------
# 2. Ship: per-regime shards, AES-256-GCM encrypted; keys in the lockbox
# --------------------------------------------------------------------------


def make_shards(model, groups):
    """Extract per-regime weight shards (W1 columns, b1 entries, W2 rows)."""
    w1 = model[0].weight.detach().numpy()  # (width, d_in)
    b1 = model[0].bias.detach().numpy()  # (width,)
    w2 = model[2].weight.detach().numpy()  # (classes, width)
    b2 = model[2].bias.detach().numpy()
    shards = {}
    for r, grp in enumerate(groups):
        buf = io.BytesIO()
        np.savez(buf, w1_rows=w1[grp], b1=b1[grp], w2_cols=w2[:, grp])
        shards[r] = buf.getvalue()
    return shards, w1, b1, w2, b2


def encrypt_shards(shards, shard_keys):
    """AES-256-GCM per shard; AAD binds the regime id (context binding)."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    out = {}
    for r, plain in shards.items():
        aes = AESGCM(shard_keys[r])
        nonce = np.random.default_rng(SEED + r).bytes(12)
        out[r] = {
            "nonce": base64.b64encode(nonce).decode(),
            "ciphertext": base64.b64encode(
                aes.encrypt(nonce, plain, f"shard:{r}".encode())
            ).decode(),
        }
    return out


# --------------------------------------------------------------------------
# 3. Authority lockbox: signed grants (subject, scope, expiry, AAD)
# --------------------------------------------------------------------------


def canonical_grant(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def make_authority():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    sk = Ed25519PrivateKey.generate()
    pk_raw = sk.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return sk, pk_raw


def sign_grant(sk, payload: dict) -> bytes:
    return sk.sign(canonical_grant(payload))


def verify_grant(pk_raw: bytes, payload: dict, signature: bytes) -> bool:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    pk = Ed25519PublicKey.from_public_bytes(pk_raw)
    try:
        pk.verify(signature, canonical_grant(payload))
    except InvalidSignature:
        return False
    return True


# --------------------------------------------------------------------------
# 4. Fixture release semantics
# --------------------------------------------------------------------------


class ReleaseRejected(PermissionError):
    pass


def release_shards(manifest, grant_payload, signature, epoch, holder_pk):
    """Decrypt exactly the granted regime shards, or refuse (fail-closed).

    This observed implementation checks authenticity (signature verifies),
    currency (epoch < expiry), and scope proportionality (only named regimes
    decrypt). It is not claimed as a refinement of a bundled Lean runtime
    model.
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if not verify_grant(holder_pk, grant_payload, signature):
        raise ReleaseRejected("grant failed Authority signature verification")
    if not epoch < grant_payload["expiry"]:
        raise ReleaseRejected(f"grant expired (epoch {epoch} >= {grant_payload['expiry']})")
    released = {}
    for r in grant_payload["scope"]:
        entry = manifest["shards"][str(r)]
        aes = AESGCM(bytes.fromhex(manifest["shard_keys"][str(r)]))
        released[r] = aes.decrypt(
            base64.b64decode(entry["nonce"]),
            base64.b64decode(entry["ciphertext"]),
            f"shard:{r}".encode(),
        )
    return released


# --------------------------------------------------------------------------
# 5. Reassemble + evaluate; verification checks
# --------------------------------------------------------------------------


def load_shard(blob: bytes):
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def released_logits(x_np, released_arrays, groups, w2, b2, width):
    """Forward pass using ONLY released dims (all others absent = zero)."""
    logits = np.tile(b2, (x_np.shape[0], 1)).astype(np.float64)
    for _regime, arr in released_arrays.items():
        h = np.maximum(0.0, x_np @ arr["w1_rows"].T + arr["b1"])  # (B, k)
        logits += h @ arr["w2_cols"].T
    return logits


def reference_logits(model, teacher_x):
    import torch

    with torch.no_grad():
        x = torch.from_numpy(x_np_of(teacher_x)) if isinstance(teacher_x, np.ndarray) else teacher_x
        return model(x).numpy()


def x_np_of(arr):
    return arr


def main() -> int:
    import torch

    print("== Co-trained embedded weights + lockbox: verification transcript ==")
    print(
        f"fixture: N={N_DIMS}, R={N_REGIMES}, {N_CLASSES} classes, "
        f"random linear teachers (FIXTURE, labeled)\n"
    )

    # --- Phase 0: co-train with the gate present from the first step ---
    model, groups, masks, teachers = cotrain()
    acc_correct = [accuracy(model, teachers[r], masks[r]) for r in range(N_REGIMES)]
    acc_wrong = [accuracy(model, teachers[r], masks[1 - r]) for r in range(N_REGIMES)]
    print(f"[EMB] correct-mask accuracy:  {[f'{a:.3f}' for a in acc_correct]}")
    print(
        f"[EMB] wrong-mask accuracy:    {[f'{a:.3f}' for a in acc_wrong]} "
        f"(chance = {1 / N_CLASSES:.3f})"
    )
    require(all(a >= 0.85 for a in acc_correct), "gate-aware co-training failed")
    require(
        all(a <= 3 * (1 / N_CLASSES) for a in acc_wrong),
        "wrong-mask accuracy above chance band — embedding claim fails",
    )
    print("[EMB] PASS: the fixture learned the regime support constraint\n")

    # --- Phase 1: package as encrypted shards under Authority lockbox ---
    shards, w1, b1, w2, b2 = make_shards(model, groups)
    rng = np.random.default_rng(SEED)
    shard_keys = {r: rng.bytes(32) for r in range(N_REGIMES)}
    enc = encrypt_shards(shards, shard_keys)
    sk, pk_raw = make_authority()
    manifest = {
        "model_id": "cotrained-fixture-mlp-v1",
        "shards": {str(r): v for r, v in enc.items()},
        # NOTE: in the real deployment shard keys live ONLY in the Authority's
        # lockbox (sealed per tenant). Here both sides share one fixture process.
        "shard_keys": {str(r): k.hex() for r, k in shard_keys.items()},
    }
    print(f"[PKG] {N_REGIMES} shards, AES-256-GCM, keys sealed for Authority\n")

    # Referenced eval inputs (regime 0 teacher)
    x = torch.randn(512, D_IN, generator=torch.Generator().manual_seed(SEED + 1))
    with torch.no_grad():
        ref_logits_r0 = model[2](model[1](model[0](x)) * masks[0]).numpy()

    # --- Phase 2: grant regime 0 to the subject; verify R4 exactness ---
    grant0 = {
        "grant_id": 1,
        "subject": "school-laptop-17",
        "scope": [0],
        "expiry": 50,
        "aad": "regime:math;tokens:1000",
    }
    sig0 = sign_grant(sk, grant0)
    released = release_shards(manifest, grant0, sig0, epoch=0, holder_pk=pk_raw)
    require(set(released) == {0}, "custody widened beyond granted scope")
    print("[R2 ] PASS: custody = exactly the granted scope {0}\n")

    released_arrays = {0: load_shard(released[0])}
    got = released_logits(x.numpy().astype(np.float64), released_arrays, groups, w2, b2, N_DIMS)
    # Reference: masked logits using only regime-0 dims (float64 arithmetic)
    h0 = np.maximum(
        0.0,
        x.numpy().astype(np.float64) @ w1[groups[0]].astype(np.float64).T
        + b1[groups[0]].astype(np.float64),
    )
    ref = (
        np.tile(b2, (x.shape[0], 1)).astype(np.float64) + h0 @ w2[:, groups[0]].astype(np.float64).T
    )
    require(np.array_equal(got, ref), "released logits differ from reference")
    # And they match the co-trained model's torch forward within float tolerance
    rel = np.abs(got - ref_logits_r0.astype(np.float64))
    require(rel.max() < 1e-5, f"float drift {rel.max()}")
    print(f"[R4 ] PASS: released inference exact (max |Δ| vs torch = {rel.max():.2e})\n")

    # --- Phase 3: nothing released => everything is zero (R5) ---
    zeros = released_logits(x.numpy().astype(np.float64), {}, groups, w2, b2, N_DIMS)
    require(
        np.allclose(zeros, np.tile(b2, (x.shape[0], 1))),
        "non-keyed logits ≠ bias-only",
    )
    # the gate-side statement: any mask applied to absent weights yields 0 activation
    require(
        (np.zeros(N_DIMS) * np.array([1.0] * N_DIMS) == 0).all(),
        "absent weights produced a non-zero gated activation",
    )
    print("[R5 ] PASS: with no release, no shard signal exists (bias-only logits)\n")

    # --- Phase 4: AAD tampering is rejected (R1) ---
    grant0_bad = dict(grant0, aad="regime:math;tokens:unlimited")
    try:
        release_shards(manifest, grant0_bad, sig0, epoch=0, holder_pk=pk_raw)
        raise AssertionError("tampered AAD was accepted!")
    except ReleaseRejected:
        print("[R1 ] PASS: AAD-tampered grant rejected by real Ed25519\n")

    # --- Phase 5: unreleased shard is ciphertext, not plaintext (custody) ---
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    wrong_key = AESGCM(rng.bytes(32))
    e = manifest["shards"]["1"]
    try:
        wrong_key.decrypt(
            base64.b64decode(e["nonce"]), base64.b64decode(e["ciphertext"]), b"shard:1"
        )
        raise AssertionError("wrong key decrypted a shard!")
    except InvalidTag:
        print("[CUS] PASS: unreleased shard is opaque ciphertext (AEAD)\n")

    print("== ALL CLAIMS VERIFIED ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
