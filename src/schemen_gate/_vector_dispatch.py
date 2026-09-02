"""Vector Dispatch -- semantic skill/tool routing via embedding fingerprints.

Register skills (tools, actions, capabilities) by embedding their intent
descriptions.  Dispatch incoming queries to the correct skill by cosine
similarity in embedding space.  Optionally gate each skill into a disjoint
regime subspace for algebraic (not statistical) dispatch boundaries.

The dispatch is deterministic: same encoder, same input = same vector = same
skill.  No keyword matching, no regex, no NLU pipeline.  Just geometry.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from schemen_gate._mask import GateMask

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class SkillFingerprint:
    """A skill's vector identity in embedding space."""

    skill_id: str
    description: str
    vector: np.ndarray
    regime_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return int(self.vector.shape[-1])

    def fingerprint_hash(self) -> str:
        return hashlib.sha256(
            np.ascontiguousarray(self.vector, dtype=np.float32).tobytes()
        ).hexdigest()


@dataclass
class DispatchResult:
    """Result of dispatching a query to a skill."""

    skill_id: str
    score: float
    margin: float
    runner_up_id: Optional[str] = None
    runner_up_score: float = 0.0
    gated: bool = False


# ---------------------------------------------------------------------------
# SkillRegistry
# ---------------------------------------------------------------------------


class SkillRegistry:
    """Semantic registry that maps intent vectors to callable skills.

    Registration: embed a skill description, store the fingerprint.
    Dispatch: embed a query, find the nearest skill by cosine similarity.
    Gating (optional): assign each skill to a regime, project into disjoint
    subspaces, dispatch becomes algebraically unambiguous.
    """

    def __init__(
        self,
        embed_fn: Callable[[str], np.ndarray],
        *,
        gate_key: Optional[bytes] = None,
        n_dims: Optional[int] = None,
        n_regimes: Optional[int] = None,
    ) -> None:
        gate_values = (gate_key, n_dims, n_regimes)
        if any(value is not None for value in gate_values) and not all(
            value is not None for value in gate_values
        ):
            raise ValueError("gate_key, n_dims, and n_regimes must be provided together")
        if gate_key is not None and (not isinstance(gate_key, bytes) or len(gate_key) != 32):
            raise ValueError("gate_key must be exactly 32 bytes")
        for name, value in (("n_dims", n_dims), ("n_regimes", n_regimes)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        self._embed_fn = embed_fn
        self._skills: Dict[str, SkillFingerprint] = {}
        self._gate_key = gate_key
        self._n_dims = n_dims
        self._n_regimes = n_regimes
        self._masks: Dict[int, GateMask] = {}

    @property
    def gating_enabled(self) -> bool:
        return self._gate_key is not None

    @property
    def skill_count(self) -> int:
        return len(self._skills)

    def _get_mask(self, regime_id: int) -> GateMask:
        if regime_id not in self._masks:
            if self._gate_key is None or self._n_dims is None or self._n_regimes is None:
                raise ValueError("gating is not fully configured")
            self._masks[regime_id] = GateMask.derive(
                self._gate_key,
                regime_id,
                self._n_dims,
                self._n_regimes,
            )
        return self._masks[regime_id]

    def register(
        self,
        skill_id: str,
        description: str,
        *,
        regime_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SkillFingerprint:
        """Embed a skill description and register its fingerprint."""
        if self.gating_enabled and regime_id is None:
            raise ValueError("gated registries require an explicit regime_id")
        if not self.gating_enabled and regime_id is not None:
            raise ValueError("regime_id requires a gate-configured registry")
        vec = self._embed_fn(description)
        vec = np.asarray(vec, dtype=np.float64).ravel()
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec = vec / norm

        if regime_id is not None and self.gating_enabled:
            mask = self._get_mask(regime_id)
            vec = vec * mask.to_numpy()
            re_norm = np.linalg.norm(vec)
            if re_norm > 1e-12:
                vec = vec / re_norm

        fp = SkillFingerprint(
            skill_id=skill_id,
            description=description,
            vector=vec,
            regime_id=regime_id,
            metadata=metadata or {},
        )
        self._skills[skill_id] = fp
        return fp

    def dispatch(
        self,
        query: str,
        *,
        gate_regime: Optional[int] = None,
    ) -> DispatchResult:
        """Find the best-matching skill for a query.

        If gate_regime is provided and gating is enabled, the query vector
        is projected into that regime's subspace before comparison.
        """
        if self.gating_enabled and gate_regime is None:
            raise ValueError("gated registries require an explicit gate_regime")
        if not self.gating_enabled and gate_regime is not None:
            raise ValueError("gate_regime requires a gate-configured registry")
        if not self._skills:
            raise ValueError("No skills registered")

        q_vec = self._embed_fn(query)
        q_vec = np.asarray(q_vec, dtype=np.float64).ravel()
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 1e-12:
            q_vec = q_vec / q_norm

        gated = False
        if gate_regime is not None and self.gating_enabled:
            mask = self._get_mask(gate_regime)
            q_vec = q_vec * mask.to_numpy()
            re_norm = np.linalg.norm(q_vec)
            if re_norm > 1e-12:
                q_vec = q_vec / re_norm
            gated = True

        scores: List[Tuple[str, float]] = []
        for skill_id, fp in self._skills.items():
            score = float(np.dot(q_vec, fp.vector))
            scores.append((skill_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        best_id, best_score = scores[0]
        if len(scores) > 1:
            runner_id, runner_score = scores[1]
            margin = best_score - runner_score
        else:
            runner_id, runner_score = None, 0.0
            margin = best_score

        return DispatchResult(
            skill_id=best_id,
            score=best_score,
            margin=margin,
            runner_up_id=runner_id,
            runner_up_score=runner_score,
            gated=gated,
        )

    def dispatch_top_k(
        self,
        query: str,
        k: int = 3,
        *,
        gate_regime: Optional[int] = None,
    ) -> List[DispatchResult]:
        """Return the top-k matching skills."""
        if self.gating_enabled and gate_regime is None:
            raise ValueError("gated registries require an explicit gate_regime")
        if not self.gating_enabled and gate_regime is not None:
            raise ValueError("gate_regime requires a gate-configured registry")
        if not self._skills:
            raise ValueError("No skills registered")

        q_vec = self._embed_fn(query)
        q_vec = np.asarray(q_vec, dtype=np.float64).ravel()
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 1e-12:
            q_vec = q_vec / q_norm

        gated = False
        if gate_regime is not None:
            mask = self._get_mask(gate_regime)
            q_vec = q_vec * mask.to_numpy()
            re_norm = np.linalg.norm(q_vec)
            if re_norm > 1e-12:
                q_vec = q_vec / re_norm
            gated = True

        scores: List[Tuple[str, float]] = []
        for skill_id, fp in self._skills.items():
            score = float(np.dot(q_vec, fp.vector))
            scores.append((skill_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, (sid, sc) in enumerate(scores[:k]):
            runner_id = scores[idx + 1][0] if idx + 1 < len(scores) else None
            runner_score = scores[idx + 1][1] if idx + 1 < len(scores) else 0.0
            results.append(
                DispatchResult(
                    skill_id=sid,
                    score=sc,
                    margin=sc - runner_score if runner_id else sc,
                    runner_up_id=runner_id,
                    runner_up_score=runner_score,
                    gated=gated,
                )
            )
        return results

    def pairwise_similarities(self) -> np.ndarray:
        """Cosine similarity matrix between all registered skill vectors."""
        ids = sorted(self._skills.keys())
        n = len(ids)
        mat = np.zeros((n, n), dtype=np.float64)
        vecs = [self._skills[sid].vector for sid in ids]
        for i in range(n):
            for j in range(n):
                mat[i, j] = float(np.dot(vecs[i], vecs[j]))
        return mat

    def skill_ids(self) -> List[str]:
        return sorted(self._skills.keys())

    def get_fingerprint(self, skill_id: str) -> SkillFingerprint:
        return self._skills[skill_id]
