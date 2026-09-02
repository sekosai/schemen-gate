"""Tests for GateMask — the full-gate mode interface.

Covers mask derivation, application, composition, serialization,
and cross-compatibility with the main Schemen SDK.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from schemen_gate import GateMask


@pytest.fixture
def key():
    return b"\x42" * 32


class TestDerivation:
    def test_derive_is_available_without_cryptography_import(self):
        script = """
import builtins
real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == 'cryptography' or name.startswith('cryptography.'):
        raise AssertionError('GateMask.derive imported the optional cryptography package')
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from schemen_gate import GateMask
mask = GateMask.derive(bytes([0x42]) * 32, regime_id=0, n_dims=8, n_regimes=2)
assert mask.active_dims == 4
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            check=False,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_derive_produces_binary(self, key):
        mask = GateMask.derive(key, regime_id=0, n_dims=64, n_regimes=4)
        unique = np.unique(mask.mask)
        assert set(unique) == {0.0, 1.0}

    def test_derive_correct_active_count(self, key):
        mask = GateMask.derive(key, regime_id=0, n_dims=64, n_regimes=4)
        assert mask.active_dims == 16

    def test_derive_deterministic(self, key):
        m1 = GateMask.derive(key, regime_id=0, n_dims=64, n_regimes=4)
        m2 = GateMask.derive(key, regime_id=0, n_dims=64, n_regimes=4)
        np.testing.assert_array_equal(m1.mask, m2.mask)

    def test_different_keys_different_masks(self):
        m1 = GateMask.derive(b"\x01" * 32, regime_id=0, n_dims=64, n_regimes=2)
        m2 = GateMask.derive(b"\x02" * 32, regime_id=0, n_dims=64, n_regimes=2)
        assert not np.array_equal(m1.mask, m2.mask)

    def test_regimes_are_disjoint(self, key):
        masks = [GateMask.derive(key, r, 64, 4) for r in range(4)]
        for i in range(4):
            for j in range(i + 1, 4):
                assert np.all(masks[i].mask * masks[j].mask == 0.0)

    def test_regimes_cover_all_dims(self, key):
        masks = [GateMask.derive(key, r, 64, 4) for r in range(4)]
        total = sum(m.mask for m in masks)
        np.testing.assert_array_equal(total, np.ones(64))


class TestApplication:
    def test_apply_numpy(self, key):
        mask = GateMask.derive(key, regime_id=0, n_dims=16, n_regimes=2)
        h = np.ones(16) * 3.0
        gated = mask.apply(h)
        assert gated.shape == (16,)
        assert np.sum(gated == 3.0) == 8
        assert np.sum(gated == 0.0) == 8

    def test_apply_preserves_active(self, key):
        mask = GateMask.derive(key, regime_id=0, n_dims=16, n_regimes=2)
        h = np.arange(16, dtype=np.float64)
        gated = mask.apply(h)
        active = np.where(mask.mask == 1.0)[0]
        np.testing.assert_array_equal(gated[active], h[active])

    def test_apply_zeros_inactive(self, key):
        mask = GateMask.derive(key, regime_id=0, n_dims=16, n_regimes=2)
        h = np.ones(16) * 99.0
        gated = mask.apply(h)
        inactive = np.where(mask.mask == 0.0)[0]
        np.testing.assert_array_equal(gated[inactive], 0.0)

    def test_apply_supports_batches_on_the_final_axis(self, key):
        mask = GateMask.derive(key, regime_id=0, n_dims=16, n_regimes=2)
        hidden = np.ones((3, 16), dtype=np.float64)

        gated = mask.apply(hidden)

        assert gated.shape == (3, 16)
        np.testing.assert_array_equal(gated[0], mask.to_numpy())

    @pytest.mark.parametrize("shape", [(), (15,), (16, 1)])
    def test_apply_rejects_an_incompatible_final_dimension(self, key, shape):
        mask = GateMask.derive(key, regime_id=0, n_dims=16, n_regimes=2)
        hidden = np.ones(shape, dtype=np.float64)

        with pytest.raises(ValueError, match="final dimension"):
            mask.apply(hidden)

    @pytest.mark.parametrize("hidden", [1.0, [1.0] * 16])
    def test_apply_rejects_shapeless_inputs(self, key, hidden):
        mask = GateMask.derive(key, regime_id=0, n_dims=16, n_regimes=2)

        with pytest.raises(ValueError, match="explicit non-scalar shape"):
            mask.apply(hidden)


class TestComposition:
    def test_compose_or(self, key):
        m0 = GateMask.derive(key, 0, 16, 4)
        m1 = GateMask.derive(key, 1, 16, 4)
        composed = m0 | m1
        assert composed.active_dims == 8

    def test_compose_all_regimes_gives_full(self, key):
        masks = [GateMask.derive(key, r, 16, 4) for r in range(4)]
        composed = masks[0] | masks[1] | masks[2] | masks[3]
        np.testing.assert_array_equal(composed.mask, np.ones(16))

    def test_compose_mismatch_raises(self, key):
        m16 = GateMask.derive(key, 0, 16, 2)
        m32 = GateMask.derive(key, 0, 32, 2)
        with pytest.raises(ValueError, match="dimensionality"):
            _ = m16 | m32


class TestSerialization:
    def test_rejects_non_binary_mask(self):
        with pytest.raises(ValueError, match="binary"):
            GateMask.from_numpy(np.array([0.0, 0.5, 1.0]))

    def test_input_and_property_cannot_mutate_authority_mask(self):
        source = np.array([1.0, 0.0, 1.0, 0.0])
        mask = GateMask.from_numpy(source)
        source[:] = 1.0
        np.testing.assert_array_equal(mask.to_numpy(), [1.0, 0.0, 1.0, 0.0])
        with pytest.raises(ValueError):
            mask.mask[:] = 1.0

    def test_negative_index_is_rejected(self):
        with pytest.raises(ValueError, match="indices"):
            GateMask.from_indices([-1], n_dims=4)

    @pytest.mark.parametrize(
        "document",
        [
            {"n_dims": 4, "active_indices": [-1]},
            {"n_dims": 4, "active_indices": [True]},
            {"n_dims": 4, "active_indices": [4]},
            {"n_dims": True, "active_indices": [0]},
            {"n_dims": 4, "active_indices": [0], "regime_id": True},
        ],
    )
    def test_from_dict_rejects_noncanonical_authority_fields(self, document):
        with pytest.raises(ValueError):
            GateMask.from_dict(document)

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("not_finite", float("nan")),
            ("tuple_value", ("not", "json")),
            ("object_value", object()),
            (7, "non-string-key"),
        ],
        ids=["nan", "tuple", "object", "non-string-key"],
    )
    def test_from_dict_rejects_non_json_metadata(self, key, value):
        document = {
            "n_dims": 4,
            "active_indices": [0],
            "regime_id": 0,
            key: value,
        }

        with pytest.raises(ValueError, match="Gate mask metadata"):
            GateMask.from_dict(document)

    def test_from_dict_rejects_cyclic_metadata(self):
        cycle = []
        cycle.append(cycle)

        with pytest.raises(ValueError, match="cycle"):
            GateMask.from_dict(
                {
                    "n_dims": 4,
                    "active_indices": [0],
                    "regime_id": 0,
                    "cycle": cycle,
                }
            )

    def test_from_dict_rejects_oversized_metadata(self):
        with pytest.raises(ValueError, match="1 MiB safety limit"):
            GateMask.from_dict(
                {
                    "n_dims": 4,
                    "active_indices": [0],
                    "regime_id": 0,
                    "blob": "x" * (1024 * 1024),
                }
            )

    def test_save_load_npy(self, key, tmp_path):
        mask = GateMask.derive(key, regime_id=0, n_dims=64, n_regimes=4)
        path = tmp_path / "mask.npy"
        mask.save(path)

        loaded = GateMask.from_file(path, regime_id=0)
        np.testing.assert_array_equal(loaded.mask, mask.mask)

    def test_from_file_requires_regime_without_sidecar(self, tmp_path):
        path = tmp_path / "mask.npy"
        np.save(path, np.ones(4, dtype=np.float64))
        with pytest.raises(ValueError, match="regime_id is required"):
            GateMask.from_file(path)

    def test_from_file_rejects_oversized_metadata_sidecar(self, tmp_path):
        path = tmp_path / "mask.npy"
        np.save(path, np.ones(4, dtype=np.float64))
        path.with_suffix(".json").write_bytes(b" " * (16 * 1024 * 1024 + 1))

        with pytest.raises(ValueError, match="16 MiB safety limit"):
            GateMask.from_file(path)

    def test_save_rejects_oversized_sidecar_before_writing_mask(self, tmp_path, monkeypatch):
        import schemen_gate._mask as mask_module

        gate = GateMask.from_indices(range(16), n_dims=16)
        path = tmp_path / "mask.npy"
        monkeypatch.setattr(mask_module, "_MAX_MASK_SIDECAR_BYTES", 32)

        with pytest.raises(ValueError, match="16 MiB safety limit"):
            gate.save(path)

        assert not path.exists()
        assert not path.with_suffix(".json").exists()

    def test_from_file_rejects_duplicate_metadata_keys(self, tmp_path):
        path = tmp_path / "mask.npy"
        np.save(path, np.ones(4, dtype=np.float64))
        path.with_suffix(".json").write_text(
            '{"regime_id": 1, "regime_id": 2}',
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="valid strict JSON"):
            GateMask.from_file(path)

    def test_from_file_rejects_non_standard_json_numbers(self, tmp_path):
        path = tmp_path / "mask.npy"
        np.save(path, np.ones(4, dtype=np.float64))
        path.with_suffix(".json").write_text(
            '{"n_dims": NaN}',
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="valid strict JSON"):
            GateMask.from_file(path)

    def test_to_dict_from_dict(self, key):
        mask = GateMask.derive(key, regime_id=2, n_dims=64, n_regimes=4)
        d = mask.to_dict()
        restored = GateMask.from_dict(d)
        np.testing.assert_array_equal(restored.mask, mask.mask)
        assert restored.regime_id == 2

    def test_from_indices(self):
        mask = GateMask.from_indices([0, 3, 7], n_dims=16, regime_id=5)
        assert mask.active_dims == 3
        assert mask.regime_id == 5
        assert mask.mask[0] == 1.0
        assert mask.mask[3] == 1.0
        assert mask.mask[7] == 1.0
        assert mask.mask[1] == 0.0

    def test_full_mask(self):
        mask = GateMask.full(32)
        assert mask.active_dims == 32
        np.testing.assert_array_equal(mask.mask, np.ones(32))
        assert mask.metadata["access_policy"] == "public-for-all"

    def test_public_for_all_is_an_explicit_all_ones_policy(self):
        mask = GateMask.public(8)

        assert mask.regime_id == -1
        assert mask.active_dims == 8
        assert mask.metadata == {
            "access_policy": "public-for-all",
            "full_access": True,
        }
        np.testing.assert_array_equal(mask.apply(np.arange(8)), np.arange(8))


class TestStandardArrayProtocols:
    def test_numpy_array_protocol_returns_detached_storage(self):
        mask = GateMask.from_indices([0, 2], n_dims=4)
        exported = np.asarray(mask)

        assert mask.shape == (4,)
        assert mask.dtype == np.dtype(np.float64)
        exported[:] = 0.0
        np.testing.assert_array_equal(mask.to_numpy(), [1.0, 0.0, 1.0, 0.0])

    def test_numpy_copy_false_cannot_alias_authority_storage(self):
        mask = GateMask.from_indices([0], n_dims=2)

        with pytest.raises(ValueError, match="copy=False"):
            mask.__array__(copy=False)

    def test_dlpack_protocol_returns_detached_storage(self):
        if not hasattr(np, "from_dlpack"):
            pytest.skip("NumPy does not expose DLPack in this environment")
        mask = GateMask.from_indices([1, 3], n_dims=4)
        exported = np.from_dlpack(mask)

        np.testing.assert_array_equal(exported, [0.0, 1.0, 0.0, 1.0])
        np.testing.assert_array_equal(mask.to_numpy(), [0.0, 1.0, 0.0, 1.0])


class TestProtocolVectors:
    """Lock exact partition output without an unavailable companion package."""

    @pytest.mark.parametrize(
        ("key", "n_dims", "n_regimes", "expected"),
        [
            (
                bytes(32),
                8,
                2,
                ((4, 6, 5, 1), (7, 2, 3, 0)),
            ),
            (
                bytes([0x42]) * 32,
                16,
                4,
                ((12, 1, 3, 2), (9, 11, 5, 0), (7, 15, 8, 14), (10, 6, 4, 13)),
            ),
            (
                bytes(range(32)),
                12,
                3,
                ((3, 4, 6, 8), (9, 1, 11, 0), (5, 2, 7, 10)),
            ),
        ],
    )
    def test_partition_protocol_vectors(
        self,
        key: bytes,
        n_dims: int,
        n_regimes: int,
        expected: tuple[tuple[int, ...], ...],
    ) -> None:
        from schemen_gate._crypto import derive_partition

        assert derive_partition(key, n_dims, n_regimes) == [
            list(partition) for partition in expected
        ]
