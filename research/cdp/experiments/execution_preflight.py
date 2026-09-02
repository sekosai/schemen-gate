"""Minimal fail-closed execution preflight for research experiments.

This module is deliberately not a web server, authentication service, or
production authorization implementation. It makes one falsifiable experiment
property explicit: a mismatched model, dimension, or Regime is rejected before
the model callback runs.

Production callers should use Schemen Gate's signed authority and Cargo
surfaces at their actual execution boundary. This fixture exists only so the
paper runners do not need to bundle an unrelated server distribution.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class ExecutionPreflightError(RuntimeError):
    """Raised before model execution when a research frame is out of scope."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class ExecutionFrame:
    """Exact research execution scope checked before a callback."""

    frame_id: str
    model_id: str
    dimensions: int
    regime_id: int


@dataclass(frozen=True)
class GateExecutionGrant:
    """A successful preflight decision for one model evaluation callback."""

    frame_id: str
    model_id: str
    dimensions: int
    regime_id: int
    receipt_sha256: str


class GateExecutionPreflight:
    """Research-only callback preflight with no network or server surface."""

    def __init__(
        self,
        *,
        model_id: str,
        dimensions: int,
        authorized_regime_ids: list[int],
    ) -> None:
        if not isinstance(model_id, str) or not model_id or "\x00" in model_id:
            raise ValueError("model_id must be a non-empty string without NUL")
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        if (
            not isinstance(authorized_regime_ids, list)
            or not authorized_regime_ids
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in authorized_regime_ids
            )
        ):
            raise ValueError("authorized regime ids must be non-negative integers")
        if len(set(authorized_regime_ids)) != len(authorized_regime_ids):
            raise ValueError("authorized regime ids must be unique")
        self.model_id = model_id
        self.dimensions = dimensions
        self.authorized_regime_ids = tuple(sorted(authorized_regime_ids))
        self._sequence = 0
        self._receipts: list[dict[str, Any]] = []
        self.model_calls = 0
        self._state_lock = threading.RLock()

    def _frame(
        self,
        *,
        regime_id: int,
        model_id: str | None = None,
        dimensions: int | None = None,
    ) -> ExecutionFrame:
        with self._state_lock:
            self._sequence += 1
            return ExecutionFrame(
                frame_id=f"evaluation-{self._sequence}",
                model_id=self.model_id if model_id is None else model_id,
                dimensions=self.dimensions if dimensions is None else dimensions,
                regime_id=regime_id,
            )

    @staticmethod
    def _validate(
        frame: ExecutionFrame,
        *,
        configured_model: str,
        configured_dimensions: int,
        authorized_regime_ids: tuple[int, ...],
    ) -> None:
        if not authorized_regime_ids:
            raise ExecutionPreflightError(
                "no Regime authority is configured",
                status_code=403,
            )
        if frame.model_id != configured_model:
            raise ExecutionPreflightError("model scope mismatch", status_code=403)
        if frame.dimensions != configured_dimensions:
            raise ExecutionPreflightError("dimension scope mismatch", status_code=422)
        if frame.regime_id not in authorized_regime_ids:
            raise ExecutionPreflightError("Regime scope mismatch", status_code=403)

    def authorize(
        self,
        regime_id: int,
        *,
        model_id: str | None = None,
        dimensions: int | None = None,
    ) -> GateExecutionGrant:
        frame = self._frame(
            regime_id=regime_id,
            model_id=model_id,
            dimensions=dimensions,
        )
        self._validate(
            frame,
            configured_model=self.model_id,
            configured_dimensions=self.dimensions,
            authorized_regime_ids=self.authorized_regime_ids,
        )
        receipt_body = {
            "dimensions": frame.dimensions,
            "frame_id": frame.frame_id,
            "model_id": frame.model_id,
            "regime_id": frame.regime_id,
            "schema": "schemen-gate/research-execution-preflight-v1",
        }
        receipt_sha256 = hashlib.sha256(
            json.dumps(
                receipt_body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest()
        with self._state_lock:
            self._receipts.append(
                {**receipt_body, "receipt_sha256": receipt_sha256}
            )
        return GateExecutionGrant(
            frame_id=frame.frame_id,
            model_id=frame.model_id,
            dimensions=frame.dimensions,
            regime_id=frame.regime_id,
            receipt_sha256=receipt_sha256,
        )

    def invoke(self, regime_id: int, callback: Callable[[], T]) -> T:
        """Run the callback only after exact research-scope validation."""
        self.authorize(regime_id)
        with self._state_lock:
            self.model_calls += 1
        return callback()

    def rejection_probe(self) -> dict[str, Any]:
        """Prove common malformed scopes fail before model invocation."""
        attempts = {
            "other_regime": lambda: self.authorize(
                max(self.authorized_regime_ids) + 1
            ),
            "other_model": lambda: self.authorize(
                self.authorized_regime_ids[0],
                model_id=self.model_id + "-other",
            ),
            "other_dimensions": lambda: self.authorize(
                self.authorized_regime_ids[0],
                dimensions=self.dimensions + 1,
            ),
            "empty_authority": lambda: self._validate(
                self._frame(regime_id=self.authorized_regime_ids[0]),
                configured_model=self.model_id,
                configured_dimensions=self.dimensions,
                authorized_regime_ids=(),
            ),
        }
        rows: dict[str, Any] = {}
        for name, attempt in attempts.items():
            with self._state_lock:
                before = self.model_calls
                try:
                    attempt()
                    rejected = False
                    status_code = None
                except ExecutionPreflightError as exc:
                    rejected = exc.status_code in {403, 422}
                    status_code = exc.status_code
                calls = self.model_calls - before
            rows[name] = {
                "rejected": rejected,
                "status_code": status_code,
                "model_calls": calls,
            }
        return {
            "attempts": rows,
            "all_rejected": all(row["rejected"] for row in rows.values()),
            "unauthorized_model_calls": sum(
                row["model_calls"] for row in rows.values()
            ),
        }

    def evidence(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "surface": "schemen-gate/research-execution-preflight-v1",
                "model_id": self.model_id,
                "dimensions": self.dimensions,
                "authorized_regime_ids": list(self.authorized_regime_ids),
                "authorized_model_calls": self.model_calls,
                "authorization_receipts": copy.deepcopy(self._receipts),
                "production_authority_claim": False,
            }
