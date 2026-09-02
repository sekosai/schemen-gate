"""GateMask — the single object a trainer needs to apply a Schemen gate.

Design constraints:
- No dependency on the ``schemen`` package.
- Only ``numpy`` is required.
- Optional ``torch`` / ``cryptography`` imports are deferred.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

_MAX_MASK_METADATA_BYTES = 1024 * 1024
_MAX_MASK_METADATA_DEPTH = 64
_MAX_MASK_METADATA_ITEMS = 10_000
_MAX_MASK_SIDECAR_BYTES = 16 * 1024 * 1024


def _validate_metadata_value(
    value: object,
    *,
    path: str = "$",
    depth: int = 0,
    active: set[int] | None = None,
    item_count: list[int] | None = None,
) -> None:
    """Require a bounded, exact JSON data model for mask metadata."""

    if depth > _MAX_MASK_METADATA_DEPTH:
        raise ValueError("Gate mask metadata exceeds the maximum nesting depth")
    count = [0] if item_count is None else item_count
    count[0] += 1
    if count[0] > _MAX_MASK_METADATA_ITEMS:
        raise ValueError("Gate mask metadata exceeds the maximum item count")

    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not np.isfinite(value):
            raise ValueError(f"Gate mask metadata number at {path} must be finite")
        return

    if type(value) not in {dict, list}:
        raise ValueError(f"Gate mask metadata at {path} must use exact JSON value types")

    seen = set() if active is None else active
    identity = id(value)
    if identity in seen:
        raise ValueError(f"Gate mask metadata at {path} contains a cycle")
    seen.add(identity)
    try:
        if isinstance(value, list):
            for index, item in enumerate(value):
                _validate_metadata_value(
                    item,
                    path=f"{path}[{index}]",
                    depth=depth + 1,
                    active=seen,
                    item_count=count,
                )
            return

        if not isinstance(value, dict):
            raise AssertionError("validated metadata container must be a dict or list")
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"Gate mask metadata key at {path} must be a string")
            _validate_metadata_value(
                item,
                path=f"{path}.{key}",
                depth=depth + 1,
                active=seen,
                item_count=count,
            )
    finally:
        seen.remove(identity)


def _validated_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return an isolated metadata value after strict JSON and size checks."""

    _validate_metadata_value(metadata)
    encoded = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    if len(encoded) > _MAX_MASK_METADATA_BYTES:
        raise ValueError("Gate mask metadata exceeds the 1 MiB safety limit")
    return copy.deepcopy(metadata)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build one JSON object while refusing ambiguous duplicate members."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_non_json_constant(value: str) -> object:
    """Reject Python's non-standard NaN and Infinity JSON extensions."""

    raise ValueError(f"non-JSON numeric constant {value!r}")


@dataclass(frozen=True, repr=False)
class GateMask:
    """A binary gate mask for a single regime.

    The mask is a 1-D float64 array of length ``n_dims`` with values in
    {0.0, 1.0}.  Applying it to a hidden activation vector via
    element-wise multiplication confines the signal to the active
    partition.

    Construction options (no crypto)::

        GateMask.from_file("masks/regime_0.npy")
        GateMask.from_indices([3, 7, 12], n_dims=16)
        GateMask.from_dict(saved_dict)

    Deterministic keyed construction (NumPy-only core)::

        GateMask.derive(key_bytes, regime_id=0, n_dims=768, n_regimes=4)
    """

    _mask: np.ndarray
    _regime_id: int
    _metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        """Freeze a validated binary mask.

        ``frozen=True`` only prevents rebinding dataclass attributes; it does
        not make a numpy buffer immutable.  Gate masks are authority-bearing
        data, so reject non-binary inputs and retain a private read-only copy.
        """
        arr = np.asarray(self._mask, dtype=np.float64)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("Gate mask must be a non-empty 1-D array")
        if not np.all(np.isfinite(arr)) or not np.all((arr == 0.0) | (arr == 1.0)):
            raise ValueError("Gate mask values must be finite and binary (0.0 or 1.0)")
        if isinstance(self._regime_id, bool) or not isinstance(self._regime_id, int):
            raise ValueError("regime_id must be an integer")
        if type(self._metadata) is not dict:
            raise ValueError("metadata must be a dictionary")

        private = np.array(arr, dtype=np.float64, copy=True)
        private.setflags(write=False)
        metadata = _validated_metadata(
            {
                key: value
                for key, value in self._metadata.items()
                if key not in {"n_dims", "regime_id", "active_indices"}
            }
        )
        object.__setattr__(self, "_mask", private)
        object.__setattr__(self, "_metadata", metadata)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mask(self) -> np.ndarray:
        """Return an isolated, read-only copy of the binary mask.

        A read-only numpy *view* is not an authority boundary: callers can
        reach its owning array through ``.base`` and make that owner writable.
        Returning an owning copy keeps all caller mutations detached from the
        private mask used by :meth:`apply`.
        """
        public = self._mask.copy()
        public.setflags(write=False)
        return public

    @property
    def n_dims(self) -> int:
        return len(self._mask)

    @property
    def regime_id(self) -> int:
        return self._regime_id

    @property
    def active_dims(self) -> int:
        """Number of dimensions set to 1.0."""
        return int(self._mask.sum())

    @property
    def shape(self) -> tuple[int]:
        """Array-protocol shape without exposing the authority buffer."""

        return (self.n_dims,)

    @property
    def dtype(self) -> np.dtype[Any]:
        """Array-protocol dtype of the canonical in-memory mask."""

        return self._mask.dtype

    @property
    def metadata(self) -> Dict[str, Any]:
        """Bounded JSON metadata attached at export time."""
        return copy.deepcopy(self._metadata)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: Union[str, Path], regime_id: int | None = None) -> GateMask:
        """Load a mask from a ``.npy`` file exported by the Schemen authority.

        If a companion ``.json`` sidecar exists (same stem), its contents
        are loaded as metadata.
        """
        path = Path(path)
        arr = np.load(path, allow_pickle=False).astype(np.float64)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1-D mask, got shape {arr.shape}")

        meta: Dict[str, Any] = {}
        sidecar = path.with_suffix(".json")
        if sidecar.exists():
            with open(sidecar, "rb") as f:
                encoded = f.read(_MAX_MASK_SIDECAR_BYTES + 1)
            if len(encoded) > _MAX_MASK_SIDECAR_BYTES:
                raise ValueError("Gate mask metadata sidecar exceeds the 16 MiB safety limit")
            try:
                loaded = json.loads(
                    encoded,
                    object_pairs_hook=_reject_duplicate_json_keys,
                    parse_constant=_reject_non_json_constant,
                )
            except (UnicodeDecodeError, ValueError, RecursionError) as exc:
                raise ValueError("Gate mask metadata sidecar is not valid strict JSON") from exc
            if not isinstance(loaded, dict):
                raise ValueError("Gate mask metadata sidecar must contain a JSON object")
            meta = loaded
            if "regime_id" in meta:
                regime_id = meta["regime_id"]

        if regime_id is None:
            raise ValueError("regime_id is required when the mask sidecar does not supply it")

        return cls(_mask=arr, _regime_id=regime_id, _metadata=meta)

    @classmethod
    def from_indices(
        cls,
        indices: Sequence[int],
        n_dims: int,
        regime_id: int = 0,
    ) -> GateMask:
        """Build a mask from active dimension indices."""
        if isinstance(n_dims, bool) or not isinstance(n_dims, int) or n_dims <= 0:
            raise ValueError("n_dims must be a positive integer")
        try:
            canonical_indices = list(indices)
        except TypeError as exc:
            raise ValueError("indices must be a sequence of integers") from exc
        if any(
            isinstance(index, bool)
            or not isinstance(index, (int, np.integer))
            or index < 0
            or index >= n_dims
            for index in canonical_indices
        ):
            raise ValueError(f"indices must be integers in [0, {n_dims})")
        m = np.zeros(n_dims, dtype=np.float64)
        m[canonical_indices] = 1.0
        return cls(_mask=m, _regime_id=regime_id, _metadata={})

    @classmethod
    def from_numpy(cls, arr: np.ndarray, regime_id: int = 0) -> GateMask:
        """Wrap an existing numpy array as a GateMask."""
        arr = np.asarray(arr, dtype=np.float64)
        return cls(_mask=arr, _regime_id=regime_id, _metadata={})

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> GateMask:
        """Reconstruct from a dict produced by :meth:`to_dict`."""
        if not isinstance(d, dict):
            raise ValueError("Gate mask document must be a dictionary")
        if not {"active_indices", "n_dims", "regime_id"}.issubset(d):
            raise ValueError("Gate mask document requires active_indices, n_dims, and regime_id")
        indices = d["active_indices"]
        if not isinstance(indices, list):
            raise ValueError("active_indices must be a list of integers")
        n_dims = d["n_dims"]
        regime_id = d["regime_id"]
        meta = {k: v for k, v in d.items() if k not in {"active_indices", "n_dims", "regime_id"}}
        validated = cls.from_indices(indices, n_dims=n_dims, regime_id=regime_id)
        return cls(_mask=validated._mask, _regime_id=regime_id, _metadata=meta)

    @classmethod
    def derive(
        cls,
        key: bytes,
        regime_id: int,
        n_dims: int,
        n_regimes: int = 2,
    ) -> GateMask:
        """Derive a mask from raw key bytes using the Schemen partition algorithm.

        Uses HMAC-SHA256 from the Python standard library.
        """
        from schemen_gate._crypto import derive_gate_mask_raw

        arr = derive_gate_mask_raw(key, regime_id, n_dims, n_regimes)
        return cls(
            _mask=arr,
            _regime_id=regime_id,
            _metadata={"n_regimes": n_regimes},
        )

    @classmethod
    def full(cls, n_dims: int) -> GateMask:
        """Backward-compatible alias for :meth:`public`."""

        return cls.public(n_dims)

    @classmethod
    def public(cls, n_dims: int) -> GateMask:
        """All-ones public-for-all policy.

        This is an explicit authorization result, not a missing Gate or a
        cryptographic isolation control. Every coordinate is intentionally
        available to every caller governed by the surrounding public policy.
        """

        if isinstance(n_dims, bool) or not isinstance(n_dims, int) or n_dims <= 0:
            raise ValueError("n_dims must be a positive integer")
        return cls(
            _mask=np.ones(n_dims, dtype=np.float64),
            _regime_id=-1,
            _metadata={
                "access_policy": "public-for-all",
                "full_access": True,
            },
        )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def apply(self, hidden: Any) -> Any:
        """Element-wise gate: ``hidden * mask``.

        Works with numpy arrays, torch tensors, or shaped array objects
        supporting ``__mul__``. Scalar and shapeless inputs are rejected.
        For torch tensors the mask is converted on use.
        """
        shape = getattr(hidden, "shape", None)
        if shape is None:
            raise ValueError("Gate input must expose an explicit non-scalar shape")
        try:
            input_shape = tuple(shape)
        except TypeError as exc:
            raise ValueError("Gate input shape must be a finite sequence") from exc
        if not input_shape or input_shape[-1] != self.n_dims:
            raise ValueError(
                "Gate input's final dimension must equal "
                f"n_dims={self.n_dims}; got shape {input_shape}"
            )

        try:
            import torch

            if isinstance(hidden, torch.Tensor):
                t = self.to_torch(device=hidden.device, dtype=hidden.dtype)
                return hidden * t
        except ImportError:
            pass
        return hidden * self._mask

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------

    def to_torch(
        self,
        device: Optional[Any] = None,
        dtype: Optional[Any] = None,
    ) -> Any:
        """Convert to a ``torch.Tensor``.

        Parameters
        ----------
        device : torch device or string, optional
        dtype : torch dtype, optional (defaults to float32)
        """
        import torch

        if dtype is None:
            dtype = torch.float32
        # Copy so a tensor write can never mutate the authority mask buffer.
        t = torch.tensor(self._mask, dtype=dtype)
        if device is not None:
            t = t.to(device)
        return t

    def to_numpy(self) -> np.ndarray:
        """Return a copy of the mask as a numpy array."""
        return self._mask.copy()

    def __array__(
        self,
        dtype: np.dtype[Any] | None = None,
        copy: bool | None = None,
    ) -> np.ndarray:
        """Implement NumPy's array-conversion protocol using a detached copy.

        ``copy=False`` cannot be honored without exposing authority-bearing
        storage, so it fails explicitly instead of returning an alias.
        """

        if copy is False:
            raise ValueError("GateMask cannot expose its private buffer with copy=False")
        return np.array(self._mask, dtype=dtype, copy=True)

    def __dlpack__(self, stream: Any = None) -> Any:
        """Export a detached copy through the standard DLPack protocol."""

        detached = self.to_numpy()
        return detached.__dlpack__(stream=stream)

    def __dlpack_device__(self) -> tuple[int, int]:
        """Return the standard DLPack CPU device descriptor."""

        return self._mask.__dlpack_device__()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        d: Dict[str, Any] = {
            "n_dims": self.n_dims,
            "regime_id": self._regime_id,
            "active_indices": [int(i) for i in np.where(self._mask == 1.0)[0]],
        }
        d.update(self.metadata)
        return d

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path], *, with_metadata: bool = True) -> None:
        """Save the mask to a ``.npy`` file with optional JSON sidecar."""
        path = Path(path)
        encoded: bytes | None = None
        if with_metadata:
            payload = self.to_dict()
            encoded = json.dumps(
                payload,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            if len(encoded) > _MAX_MASK_SIDECAR_BYTES:
                raise ValueError("Gate mask metadata sidecar exceeds the 16 MiB safety limit")
        np.save(path, self._mask)
        if encoded is not None:
            sidecar = path.with_suffix(".json")
            with open(sidecar, "wb") as f:
                f.write(encoded)

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def __or__(self, other: GateMask) -> GateMask:
        """Compose two disjoint masks via element-wise OR (addition)."""
        if self.n_dims != other.n_dims:
            raise ValueError("Cannot compose masks of different dimensionality")
        combined = np.clip(self._mask + other._mask, 0.0, 1.0)
        return GateMask(
            _mask=combined,
            _regime_id=-1,
            _metadata={"composed_from": [self._regime_id, other._regime_id]},
        )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"GateMask(regime={self._regime_id}, dims={self.n_dims}, active={self.active_dims})"
