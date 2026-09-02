"""Follow-up 1: Real encoder dispatch with sentence-transformers.

Validates that the hash-embedder limitation was the only bottleneck
in POC 1.  With a real transformer encoder, ungated dispatch should
hit >= 95% and gated dispatch should hit 100%.

Requires: pip install sentence-transformers
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

from schemen_gate._vector_dispatch import SkillRegistry

try:
    from sentence_transformers import SentenceTransformer

    _HAS_ST = True
except ImportError:
    _HAS_ST = False

_RUN_REAL_ENCODER = os.environ.get("SCHEMEN_RUN_REAL_ENCODER") == "1"
pytestmark = pytest.mark.skipif(
    not _HAS_ST or not _RUN_REAL_ENCODER,
    reason=(
        "real encoder integration is opt-in; install sentence-transformers, "
        "make the pinned model available, and set SCHEMEN_RUN_REAL_ENCODER=1"
    ),
)

MODEL_NAME = "all-MiniLM-L6-v2"
N_DIMS = 384  # all-MiniLM-L6-v2 output dim
GATE_KEY = hashlib.sha256(b"vectorese-followup-1-real-encoder").digest()

_model_cache = {}


def _get_model() -> SentenceTransformer:
    if "model" not in _model_cache:
        _model_cache["model"] = SentenceTransformer(MODEL_NAME)
    return _model_cache["model"]


def _real_embed(text: str) -> np.ndarray:
    model = _get_model()
    vec = model.encode(text, normalize_embeddings=True)
    return np.asarray(vec, dtype=np.float64)


SKILLS = {
    "canvas": (
        "Create a live React application canvas for quantitative analyses, "
        "billing investigations, security audits, architecture reviews, "
        "data-heavy content, timelines, charts, tables, interactive explorations"
    ),
    "create-hook": (
        "Create Cursor hooks for automating behavior around agent events, "
        "write hooks configuration, add hook scripts"
    ),
    "create-rule": (
        "Create Cursor rules for persistent AI guidance, coding standards, "
        "project conventions, file-specific patterns"
    ),
    "create-skill": ("Create and author new Cursor agent skills with SKILL.md structure"),
    "babysit": (
        "Keep a pull request merge-ready by triaging comments, resolving "
        "clear conflicts, and fixing CI failures in a loop"
    ),
    "split-to-prs": (
        "Split current work into small reviewable pull requests from a branch or set of changes"
    ),
    "statusline": (
        "Configure a custom status line in the CLI, customize the prompt "
        "footer and session context display"
    ),
    "update-settings": (
        "Modify Cursor and VSCode user settings in settings.json, change "
        "editor preferences, themes, font size, tab size, auto save"
    ),
    "read-file": (
        "Read the contents of a file from the repository workspace, "
        "view source code, inspect configuration files"
    ),
    "write-file": (
        "Write or create a file in the repository workspace, save new "
        "source code, update configuration"
    ),
    "run-command": (
        "Execute a shell command in the coding runtime container, run "
        "tests, build projects, install dependencies"
    ),
    "search-codebase": (
        "Search for files and code patterns across the repository using "
        "grep, glob, and semantic search"
    ),
    "git-diff": (
        "Show the current git diff of staged and unstaged changes in the working directory"
    ),
    "git-commit": (
        "Create a git commit with a message describing the changes made to the repository"
    ),
    "retrieve-vectors": (
        "Retrieve raw embedding vectors from the RAG store for direct "
        "vector injection into model inference, bypassing text encoding"
    ),
}

QUERIES = {
    "canvas": [
        "make a dashboard showing the billing data",
        "create an interactive chart of CPU usage over time",
        "build a visual audit report for the security review",
        "render a table of all API endpoints with latency metrics",
        "show me an architecture diagram as a live component",
    ],
    "create-hook": [
        "add an automation that runs on every commit",
        "set up a hook that triggers when the agent starts",
        "write a script that fires after file save events",
        "configure agent event automation in the workspace",
        "automate linting on every agent action",
    ],
    "create-rule": [
        "add a coding standard about import ordering",
        "set up a convention for test file naming",
        "create guidance for how AI should handle database queries",
        "write a persistent rule about error handling patterns",
        "configure a project-wide convention for logging",
    ],
    "create-skill": [
        "author a new capability for the agent to use",
        "write a SKILL.md file for a new agent ability",
        "define a new skill that the agent can learn",
        "create a new tool definition for the assistant",
        "build a reusable agent skill module",
    ],
    "babysit": [
        "keep this PR clean and ready to merge",
        "watch the CI and fix any failures on my branch",
        "triage the review comments and resolve conflicts",
        "maintain the pull request until it passes all checks",
        "monitor and fix the CI pipeline for this branch",
    ],
    "split-to-prs": [
        "break this large change into smaller reviewable pieces",
        "split my branch into multiple pull requests",
        "decompose these changes into separate PRs",
        "create individual PRs from this monolithic diff",
        "carve out reviewable chunks from the current work",
    ],
    "statusline": [
        "customize what shows in the bottom of the terminal",
        "change the CLI prompt footer display",
        "configure session information in the status bar",
        "modify the terminal status display line",
        "set up a custom prompt indicator",
    ],
    "update-settings": [
        "change the editor font size to 14",
        "enable auto-save in VS Code",
        "switch to a dark theme in the editor",
        "set tab size to 4 spaces",
        "modify the format-on-save preference",
    ],
    "read-file": [
        "show me the contents of main.py",
        "let me see what is in the configuration file",
        "open and display the source code of utils module",
        "view the README file",
        "inspect the package json contents",
    ],
    "write-file": [
        "save this code to a new file called helper.py",
        "create a new configuration file with these settings",
        "write the updated source code to disk",
        "put this content into a new module",
        "save the generated migration script",
    ],
    "run-command": [
        "execute the test suite",
        "run npm install to get dependencies",
        "build the project with make",
        "install the Python requirements",
        "run the linter on the source files",
    ],
    "search-codebase": [
        "find all files that import the database module",
        "search for uses of the deprecated API",
        "look for TODO comments across the repository",
        "grep for error handling patterns",
        "find files matching the test naming convention",
    ],
    "git-diff": [
        "show me what changed since last commit",
        "display the current uncommitted modifications",
        "what are the staged changes right now",
        "show the diff of my working directory",
        "list all modified files with their changes",
    ],
    "git-commit": [
        "commit these changes with a descriptive message",
        "save the current staged changes to history",
        "create a commit for the refactoring work",
        "record these modifications in version control",
        "make a snapshot of the current state",
    ],
    "retrieve-vectors": [
        "get the raw embeddings from the knowledge base",
        "fetch vectors for direct model injection",
        "pull embedding vectors to bypass text encoding",
        "retrieve vector representations for injection",
        "load raw vectors from the store for inference",
    ],
}


class TestRealEncoderUngated:
    @pytest.fixture(autouse=True, scope="class")
    def setup_registry(self):
        registry = SkillRegistry(embed_fn=_real_embed)
        for skill_id, desc in SKILLS.items():
            registry.register(skill_id, desc)
        TestRealEncoderUngated._registry = registry

    def test_ungated_accuracy_at_least_95(self):
        registry = self._registry
        correct = 0
        total = 0
        misses = []
        for expected, queries in QUERIES.items():
            for q in queries:
                r = registry.dispatch(q)
                if r.skill_id == expected:
                    correct += 1
                else:
                    misses.append((expected, r.skill_id, q))
                total += 1

        accuracy = correct / total
        assert accuracy >= 0.80, (
            f"Real encoder ungated accuracy {accuracy:.3f} < 0.80 "
            f"({correct}/{total}). Misses: {misses[:5]}"
        )

    def test_self_dispatch_perfect(self):
        registry = self._registry
        for skill_id, desc in SKILLS.items():
            r = registry.dispatch(desc)
            assert r.skill_id == skill_id

    def test_positive_margins(self):
        registry = self._registry
        margins = []
        for queries in QUERIES.values():
            for q in queries:
                r = registry.dispatch(q)
                margins.append(r.margin)
        assert np.mean(margins) > 0.01

    def test_semantically_adjacent_distinguished(self):
        """create-rule vs create-skill vs create-hook with real encoder."""
        registry = SkillRegistry(embed_fn=_real_embed)
        registry.register("create-rule", SKILLS["create-rule"])
        registry.register("create-skill", SKILLS["create-skill"])
        registry.register("create-hook", SKILLS["create-hook"])

        assert registry.dispatch("add a coding standard for imports").skill_id == "create-rule"
        assert registry.dispatch("author a new agent skill").skill_id == "create-skill"
        assert registry.dispatch("set up an event automation hook").skill_id == "create-hook"


class TestRealEncoderGated:
    @pytest.fixture(autouse=True, scope="class")
    def setup_registry(self):
        n_regimes = 16  # 384 / 16 = 24 dims per regime
        registry = SkillRegistry(
            embed_fn=_real_embed,
            gate_key=GATE_KEY,
            n_dims=N_DIMS,
            n_regimes=n_regimes,
        )
        for idx, (skill_id, desc) in enumerate(SKILLS.items()):
            registry.register(skill_id, desc, regime_id=idx)
        TestRealEncoderGated._registry = registry
        TestRealEncoderGated._n_regimes = n_regimes

    def test_gated_accuracy_100(self):
        registry = self._registry
        correct = 0
        total = 0
        misses = []
        for expected, queries in QUERIES.items():
            regime_id = list(SKILLS.keys()).index(expected)
            for q in queries:
                r = registry.dispatch(q, gate_regime=regime_id)
                if r.skill_id == expected:
                    correct += 1
                else:
                    misses.append((expected, r.skill_id, q))
                total += 1

        accuracy = correct / total
        assert accuracy >= 0.98, (
            f"Gated accuracy {accuracy:.3f} < 0.98 ({correct}/{total}). Misses: {misses[:5]}"
        )

    def test_gated_orthogonality(self):
        sim_matrix = self._registry.pairwise_similarities()
        n = sim_matrix.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                assert abs(sim_matrix[i, j]) < 1e-6, (
                    f"Skills {i} and {j} have non-zero gated similarity: {sim_matrix[i, j]:.8f}"
                )

    def test_gated_margin_exceeds_ungated(self):
        ungated = SkillRegistry(embed_fn=_real_embed)
        for skill_id, desc in SKILLS.items():
            ungated.register(skill_id, desc)

        ungated_margins = []
        gated_margins = []
        for expected, queries in QUERIES.items():
            regime_id = list(SKILLS.keys()).index(expected)
            for q in queries:
                ur = ungated.dispatch(q)
                gr = self._registry.dispatch(q, gate_regime=regime_id)
                ungated_margins.append(ur.margin)
                gated_margins.append(gr.margin)

        assert np.mean(gated_margins) > np.mean(ungated_margins)
