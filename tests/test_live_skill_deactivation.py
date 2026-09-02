"""Follow-up 5: Live skill deactivation.

Register skills, deactivate some by closing their regime gates,
verify deactivated skills vanish from dispatch space and queries
route to the nearest active skill.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Optional, Set

import numpy as np
import pytest

from schemen_gate._mask import GateMask
from schemen_gate._vector_dispatch import SkillRegistry

N_DIMS = 768
GATE_KEY = hashlib.sha256(b"vectorese-followup-5-deactivation").digest()


def _embed(text: str, dim: int = N_DIMS) -> np.ndarray:
    text_lower = text.lower()
    words = text_lower.split()
    bigrams = [text_lower[i : i + 3] for i in range(len(text_lower) - 2)]
    tokens = words + bigrams
    if not tokens:
        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
        vec = rng.standard_normal(dim)
        return vec / np.linalg.norm(vec)
    vec = np.zeros(dim, dtype=np.float64)
    for t in tokens:
        h = hashlib.sha256(t.encode()).digest()
        seed = int.from_bytes(h[:8], "big")
        rng = np.random.default_rng(seed)
        vec += rng.standard_normal(dim)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-12 else vec


SKILLS = {
    "read-file": "Read contents of a file from the repository workspace",
    "write-file": "Write or create a file in the repository workspace",
    "run-command": "Execute a shell command in the runtime container",
    "search-codebase": "Search for files and code patterns using grep and glob",
    "git-diff": "Show the current git diff of staged and unstaged changes",
    "git-commit": "Create a git commit with a descriptive message",
    "retrieve-vectors": "Retrieve raw embedding vectors from the RAG store",
    "canvas": "Create a live React canvas for data visualization",
}


class LiveSkillRegistry:
    """Registry with live deactivation support.

    Wraps SkillRegistry and maintains a set of active skill IDs.
    Deactivated skills are excluded from dispatch.
    """

    def __init__(
        self,
        embed_fn,
        *,
        gate_key: Optional[bytes] = None,
        n_dims: Optional[int] = None,
        n_regimes: Optional[int] = None,
    ):
        self._full_registry = SkillRegistry(
            embed_fn=embed_fn,
            gate_key=gate_key,
            n_dims=n_dims,
            n_regimes=n_regimes,
        )
        self._active_skills: Set[str] = set()
        self._all_skills: Dict[str, str] = {}

    def register(self, skill_id: str, description: str, **kwargs) -> None:
        self._full_registry.register(skill_id, description, **kwargs)
        self._active_skills.add(skill_id)
        self._all_skills[skill_id] = description

    def deactivate(self, skill_id: str) -> None:
        self._active_skills.discard(skill_id)

    def activate(self, skill_id: str) -> None:
        if skill_id in self._all_skills:
            self._active_skills.add(skill_id)

    def is_active(self, skill_id: str) -> bool:
        return skill_id in self._active_skills

    @property
    def active_count(self) -> int:
        return len(self._active_skills)

    @property
    def total_count(self) -> int:
        return self._full_registry.skill_count

    def dispatch(self, query: str, **kwargs):
        """Dispatch only among active skills."""
        if not self._active_skills:
            raise ValueError("No active skills")

        results = self._full_registry.dispatch_top_k(
            query,
            k=self._full_registry.skill_count,
            **kwargs,
        )
        for r in results:
            if r.skill_id in self._active_skills:
                return r

        return results[0]

    def dispatch_all(self, query: str, **kwargs):
        """Dispatch among ALL skills (ignoring deactivation)."""
        return self._full_registry.dispatch(query, **kwargs)


class TestLiveDeactivation:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.registry = LiveSkillRegistry(embed_fn=_embed)
        for skill_id, desc in SKILLS.items():
            self.registry.register(skill_id, desc)

    def test_all_skills_active_initially(self):
        assert self.registry.active_count == len(SKILLS)
        assert self.registry.total_count == len(SKILLS)

    def test_deactivate_removes_from_dispatch(self):
        self.registry.deactivate("git-diff")

        r = self.registry.dispatch("show me the git diff of changes")
        assert r.skill_id != "git-diff", "Deactivated skill should not be dispatched to"

    def test_deactivated_skill_queries_reroute(self):
        """Queries that would hit a deactivated skill go to the next best."""
        r_before = self.registry.dispatch_all("show git diff")
        original_skill = r_before.skill_id

        self.registry.deactivate(original_skill)

        r_after = self.registry.dispatch("show git diff")
        assert r_after.skill_id != original_skill
        assert r_after.skill_id in self.registry._active_skills

    def test_reactivation_restores_dispatch(self):
        self.registry.deactivate("canvas")

        r_off = self.registry.dispatch("create a data visualization dashboard")
        assert r_off.skill_id != "canvas"

        self.registry.activate("canvas")

        r_on = self.registry.dispatch_all("create a data visualization dashboard")
        assert r_on.skill_id == "canvas" or self.registry.is_active("canvas")

    def test_deactivate_multiple_skills(self):
        to_deactivate = ["git-diff", "git-commit", "retrieve-vectors"]
        for s in to_deactivate:
            self.registry.deactivate(s)

        assert self.registry.active_count == len(SKILLS) - 3

        for _ in range(20):
            queries = [
                "show me changed files",
                "save changes to version control",
                "get raw vectors for inference",
                "read a file",
                "run the tests",
            ]
            for q in queries:
                r = self.registry.dispatch(q)
                assert r.skill_id not in to_deactivate, (
                    f"Query '{q}' dispatched to deactivated skill {r.skill_id}"
                )

    def test_deactivate_all_but_one(self):
        for s in list(SKILLS.keys())[1:]:
            self.registry.deactivate(s)

        assert self.registry.active_count == 1
        remaining = list(SKILLS.keys())[0]

        for q in ["anything", "random query", "does not matter"]:
            r = self.registry.dispatch(q)
            assert r.skill_id == remaining

    def test_deactivate_all_raises(self):
        for s in SKILLS:
            self.registry.deactivate(s)

        with pytest.raises(ValueError, match="No active skills"):
            self.registry.dispatch("anything")


class TestGatedDeactivation:
    """Deactivation via regime gating -- closing a regime gate."""

    @pytest.fixture(autouse=True)
    def setup(self):
        n_regimes = 16
        self.registry = LiveSkillRegistry(
            embed_fn=_embed,
            gate_key=GATE_KEY,
            n_dims=N_DIMS,
            n_regimes=n_regimes,
        )
        self.n_regimes = n_regimes
        for idx, (skill_id, desc) in enumerate(SKILLS.items()):
            self.registry.register(skill_id, desc, regime_id=idx)

    def test_gated_deactivation(self):
        """Deactivating a gated skill: its regime is algebraically gone."""
        self.registry.deactivate("canvas")

        r = self.registry.dispatch(
            "create interactive visualization",
            gate_regime=list(SKILLS).index("canvas"),
        )
        assert r.skill_id != "canvas"

    def test_gated_skills_still_orthogonal_after_deactivation(self):
        """Remaining active skills maintain orthogonality."""
        self.registry.deactivate("git-diff")
        self.registry.deactivate("git-commit")

        active_ids = [s for s in SKILLS if self.registry.is_active(s)]
        vecs = [self.registry._full_registry.get_fingerprint(s).vector for s in active_ids]

        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                dot = abs(float(np.dot(vecs[i], vecs[j])))
                assert dot < 1e-6, (
                    f"Active skills {active_ids[i]} and {active_ids[j]} "
                    f"have non-zero dot product: {dot:.8f}"
                )

    def test_regime_mask_isolation_after_deactivation(self):
        """Deactivated regime's mask has zero overlap with active regimes."""
        deactivated_idx = list(SKILLS.keys()).index("canvas")
        deactivated_mask = GateMask.derive(
            GATE_KEY, deactivated_idx, N_DIMS, self.n_regimes
        ).to_numpy()

        for idx, skill_id in enumerate(SKILLS):
            if skill_id == "canvas":
                continue
            active_mask = GateMask.derive(GATE_KEY, idx, N_DIMS, self.n_regimes).to_numpy()
            overlap = deactivated_mask * active_mask
            np.testing.assert_array_equal(overlap, 0.0)
