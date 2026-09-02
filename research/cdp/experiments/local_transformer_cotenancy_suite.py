"""Local separation suite for Transformer-friendly CDP designs.

The suite evaluates distinct boundaries:

1. a pre-MLP FFN-input gate and its first-projection gradient surface;
2. dense intermediate-FFN partitions with regime-scoped Adam moments;
3. complete gated residual adapters on a frozen shared Transformer;
4. complete hard-routed private experts on a frozen shared Transformer;
5. deliberately broken LayerNorm and bypass placements;
6. Cargo corpus authorization; and
7. whole-model atomic authorization before model execution.

It is intentionally small enough to run on CPU.  Utility numbers are protocol
smokes, while exact off-regime parameter/optimizer deltas and authorization
rejections are pass/fail separation checks.

Run from this repository:

    python3 experiments/local_transformer_cotenancy_suite.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from execution_preflight import GateExecutionPreflight
from library_provenance import collect_library_provenance

from schemen_gate import (
    BusTerminal,
    CargoAuthenticationError,
    CargoItem,
    EmbeddingSpec,
    GatedRAGAdapter,
    GateMask,
    InMemoryVectorStore,
    PartitionMap,
    PartitionMode,
    compute_payload_hash,
    create_manifest,
    derive_cargo_access_key,
)


@dataclass(frozen=True)
class Config:
    seed: int = 42
    regimes: int = 4
    model_dim: int = 32
    ffn_dim: int = 64
    classes: int = 4
    sequence_length: int = 6
    train_examples: int = 384
    test_examples: int = 256
    steps: int = 240
    learning_rate: float = 0.03


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def source_provenance() -> dict:
    root = Path(__file__).resolve().parent.parent

    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *arguments],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def make_frozen_transformer(config: Config) -> nn.Module:
    layer = nn.TransformerEncoderLayer(
        d_model=config.model_dim,
        nhead=4,
        dim_feedforward=config.ffn_dim,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    backbone = nn.TransformerEncoder(layer, num_layers=2)
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    return backbone


def gate_placement_negative_control(config: Config) -> dict:
    """Require known-bad placements to violate the intended boundary."""

    source = torch.zeros(config.ffn_dim)
    source[0] = 3.0
    target_mask = torch.zeros(config.ffn_dim)
    target_mask[config.ffn_dim // config.regimes] = 1.0

    correctly_gated = source * target_mask
    normalized_before_gate = F.layer_norm(
        source,
        (config.ffn_dim,),
    ) * target_mask
    ungated_residual_bypass = correctly_gated + source

    correct_max = float(correctly_gated.abs().max())
    layernorm_leak = float(normalized_before_gate.abs().max())
    bypass_leak = float(ungated_residual_bypass.abs().max())
    return {
        "design": "deliberately_broken_gate_placement_controls",
        "correctly_placed_cross_mask_max": correct_max,
        "layernorm_between_source_and_gate_max": layernorm_leak,
        "ungated_residual_bypass_max": bypass_leak,
        "negative_controls_triggered": (
            layernorm_leak > 0.0 and bypass_leak > 0.0
        ),
        "separation_pass": (
            correct_max == 0.0
            and layernorm_leak > 0.0
            and bypass_leak > 0.0
        ),
    }


def pre_mlp_ffn_gate_test(config: Config) -> dict:
    """Check the narrower parameter surface of an FFN-input gate."""

    gate_key = hashlib.sha256(b"cdp-local-pre-mlp-suite").digest()
    mask = GateMask.derive(
        gate_key,
        regime_id=0,
        n_dims=config.model_dim,
        n_regimes=config.regimes,
    ).to_torch(dtype=torch.float32)
    active = mask.bool()

    generator = torch.Generator().manual_seed(config.seed + 1000)
    inputs = torch.randn(
        12,
        config.model_dim,
        generator=generator,
        requires_grad=True,
    )
    first_projection = torch.randn(
        config.model_dim,
        config.ffn_dim,
        generator=generator,
        requires_grad=True,
    )
    bias = torch.randn(
        config.ffn_dim,
        generator=generator,
        requires_grad=True,
    )

    gated = F.gelu((inputs * mask) @ first_projection + bias)
    extracted = F.gelu(
        inputs[:, active] @ first_projection[active, :] + bias
    )
    maximum_extraction_difference = float(
        (gated - extracted).detach().abs().max()
    )

    gated.square().sum().backward()
    maximum_inactive_input_gradient = float(
        inputs.grad[:, ~active].abs().max()
    )
    maximum_inactive_w1_row_gradient = float(
        first_projection.grad[~active, :].abs().max()
    )
    maximum_active_w1_row_gradient = float(
        first_projection.grad[active, :].abs().max()
    )

    tolerance = 1e-5
    return {
        "design": "pre_mlp_ffn_input_gate",
        "declared_surface": (
            "FFN input coordinates and aligned rows of the first projection; "
            "the hidden activation, down projection, and upstream mixed paths "
            "are not partitioned by this placement alone"
        ),
        "dimensions_per_regime": config.model_dim // config.regimes,
        "maximum_full_vs_extracted_difference": (
            maximum_extraction_difference
        ),
        "finite_precision_tolerance": tolerance,
        "maximum_inactive_input_gradient": maximum_inactive_input_gradient,
        "maximum_inactive_w1_row_gradient": (
            maximum_inactive_w1_row_gradient
        ),
        "maximum_active_w1_row_gradient": maximum_active_w1_row_gradient,
        "separation_pass": (
            maximum_extraction_difference <= tolerance
            and maximum_inactive_input_gradient == 0.0
            and maximum_inactive_w1_row_gradient == 0.0
            and maximum_active_w1_row_gradient > 0.0
        ),
    }


@torch.no_grad()
def make_features(
    backbone: nn.Module,
    config: Config,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    generator = torch.Generator().manual_seed(config.seed)
    teachers = [
        torch.randn(
            config.model_dim,
            config.classes,
            generator=generator,
        )
        for _ in range(config.regimes)
    ]
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []
    for regime in range(config.regimes):
        train_tokens = torch.randn(
            config.train_examples,
            config.sequence_length,
            config.model_dim,
            generator=generator,
        )
        test_tokens = torch.randn(
            config.test_examples,
            config.sequence_length,
            config.model_dim,
            generator=generator,
        )
        train = backbone(train_tokens).mean(dim=1)
        test = backbone(test_tokens).mean(dim=1)
        train_features.append(train)
        test_features.append(test)
        train_labels.append((train @ teachers[regime]).argmax(dim=-1))
        test_labels.append((test @ teachers[regime]).argmax(dim=-1))
    return train_features, train_labels, test_features, test_labels


class DensePartitionedFFN(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.up = nn.Linear(config.model_dim, config.ffn_dim)
        self.down = nn.Linear(config.ffn_dim, config.model_dim, bias=False)
        self.classes = config.classes

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.up(features)) * mask
        return (features + self.down(hidden))[:, : self.classes]


class ResidualAdapter(nn.Module):
    def __init__(self, model_dim: int, bottleneck: int) -> None:
        super().__init__()
        self.up = nn.Linear(model_dim, bottleneck)
        self.down = nn.Linear(bottleneck, model_dim, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(features)))


class GatedAdapterBank(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.adapters = nn.ModuleList(
            [
                ResidualAdapter(config.model_dim, config.model_dim)
                for _ in range(config.regimes)
            ]
        )
        self.classes = config.classes

    def forward(self, features: torch.Tensor, regime: int) -> torch.Tensor:
        return (features + self.adapters[regime](features))[:, : self.classes]


class PrivateExpert(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(config.model_dim, config.ffn_dim),
            nn.GELU(),
            nn.Linear(config.ffn_dim, config.model_dim, bias=False),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.ffn(features)


class PrivateExpertBank(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [PrivateExpert(config) for _ in range(config.regimes)]
        )
        self.classes = config.classes

    def forward(self, features: torch.Tensor, regime: int) -> torch.Tensor:
        return self.experts[regime](features)[:, : self.classes]


class RegimeScopedAdam:
    """Adam that updates only explicitly authorized tensor entries."""

    def __init__(
        self,
        parameters: Iterable[tuple[nn.Parameter, torch.Tensor]],
        *,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.entries = list(parameters)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.step_count = 0
        self.moments = {
            id(parameter): (
                torch.zeros_like(parameter),
                torch.zeros_like(parameter),
            )
            for parameter, _ in self.entries
        }

    @torch.no_grad()
    def step(self) -> None:
        self.step_count += 1
        correction1 = 1 - self.beta1**self.step_count
        correction2 = 1 - self.beta2**self.step_count
        for parameter, authorized in self.entries:
            if parameter.grad is None:
                continue
            first, second = self.moments[id(parameter)]
            gradient = parameter.grad
            first[authorized] = (
                self.beta1 * first[authorized]
                + (1 - self.beta1) * gradient[authorized]
            )
            second[authorized] = (
                self.beta2 * second[authorized]
                + (1 - self.beta2) * gradient[authorized].square()
            )
            estimate = first[authorized] / correction1
            variance = second[authorized] / correction2
            parameter[authorized] -= (
                self.learning_rate
                * estimate
                / (variance.sqrt() + self.epsilon)
            )

    def zero_grad(self) -> None:
        for parameter, _ in self.entries:
            parameter.grad = None


def dense_authorizations(
    model: DensePartitionedFFN,
    active: torch.Tensor,
) -> list[tuple[nn.Parameter, torch.Tensor]]:
    rows = active[:, None].expand_as(model.up.weight)
    columns = active[None, :].expand_as(model.down.weight)
    return [
        (model.up.weight, rows),
        (model.up.bias, active),
        (model.down.weight, columns),
    ]


def clone_parameters(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }


def max_parameter_delta(
    before: dict[str, torch.Tensor],
    module: nn.Module,
) -> float:
    return max(
        float((parameter.detach() - before[name]).abs().max())
        for name, parameter in module.named_parameters()
    )


@torch.no_grad()
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == labels).float().mean())


def train_dense(
    config: Config,
    train_features: list[torch.Tensor],
    train_labels: list[torch.Tensor],
    test_features: list[torch.Tensor],
    test_labels: list[torch.Tensor],
) -> dict:
    model = DensePartitionedFFN(config)
    gate_key = hashlib.sha256(b"cdp-local-dense-suite").digest()
    masks = [
        GateMask.derive(
            gate_key,
            regime_id=regime,
            n_dims=config.ffn_dim,
            n_regimes=config.regimes,
        ).to_torch(dtype=torch.float32)
        for regime in range(config.regimes)
    ]
    rows = []
    maximum_unauthorized_delta = 0.0
    maximum_unauthorized_moment = 0.0

    for regime in range(config.regimes):
        active = masks[regime].bool()
        authorized = dense_authorizations(model, active)
        optimizer = RegimeScopedAdam(
            authorized,
            learning_rate=config.learning_rate,
        )
        before = clone_parameters(model)
        for _ in range(config.steps):
            optimizer.zero_grad()
            logits = model(train_features[regime], masks[regime])
            F.cross_entropy(logits, train_labels[regime]).backward()
            optimizer.step()

        for name, parameter in model.named_parameters():
            if name == "up.weight":
                allowed = active[:, None].expand_as(parameter)
            elif name == "up.bias":
                allowed = active
            else:
                allowed = active[None, :].expand_as(parameter)
            unauthorized_delta = (parameter.detach() - before[name])[~allowed]
            maximum_unauthorized_delta = max(
                maximum_unauthorized_delta,
                float(unauthorized_delta.abs().max()),
            )

        for parameter, allowed in authorized:
            first, second = optimizer.moments[id(parameter)]
            if bool((~allowed).any()):
                maximum_unauthorized_moment = max(
                    maximum_unauthorized_moment,
                    float(first[~allowed].abs().max()),
                    float(second[~allowed].abs().max()),
                )

        owning = accuracy(
            model(test_features[regime], masks[regime]),
            test_labels[regime],
        )
        wrong = [
            accuracy(model(test_features[regime], masks[other]), test_labels[regime])
            for other in range(config.regimes)
            if other != regime
        ]
        rows.append(
            {
                "regime": regime,
                "owning_accuracy": owning,
                "wrong_key_mean_accuracy": float(np.mean(wrong)),
            }
        )

    return {
        "design": "dense_intermediate_ffn_partition",
        "shared_paths": "frozen Transformer backbone; fixed output coordinates",
        "dimensions_per_regime_per_layer": config.ffn_dim // config.regimes,
        "maximum_unauthorized_parameter_delta": maximum_unauthorized_delta,
        "maximum_unauthorized_optimizer_moment": maximum_unauthorized_moment,
        "mean_owning_accuracy": float(
            np.mean([row["owning_accuracy"] for row in rows])
        ),
        "mean_wrong_key_accuracy": float(
            np.mean([row["wrong_key_mean_accuracy"] for row in rows])
        ),
        "per_regime": rows,
        "separation_pass": (
            maximum_unauthorized_delta == 0.0
            and maximum_unauthorized_moment == 0.0
        ),
    }


def train_private_modules(
    model: GatedAdapterBank | PrivateExpertBank,
    modules: nn.ModuleList,
    config: Config,
    train_features: list[torch.Tensor],
    train_labels: list[torch.Tensor],
    test_features: list[torch.Tensor],
    test_labels: list[torch.Tensor],
    *,
    design: str,
) -> dict:
    rows = []
    maximum_inactive_delta = 0.0
    for regime in range(config.regimes):
        inactive_before = {
            other: clone_parameters(modules[other])
            for other in range(config.regimes)
            if other != regime
        }
        optimizer = torch.optim.Adam(
            modules[regime].parameters(),
            lr=config.learning_rate,
        )
        for _ in range(config.steps):
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_features[regime], regime)
            F.cross_entropy(logits, train_labels[regime]).backward()
            optimizer.step()

        for other, before in inactive_before.items():
            maximum_inactive_delta = max(
                maximum_inactive_delta,
                max_parameter_delta(before, modules[other]),
            )

        owning = accuracy(
            model(test_features[regime], regime),
            test_labels[regime],
        )
        wrong = [
            accuracy(model(test_features[regime], other), test_labels[regime])
            for other in range(config.regimes)
            if other != regime
        ]
        rows.append(
            {
                "regime": regime,
                "owning_accuracy": owning,
                "wrong_key_mean_accuracy": float(np.mean(wrong)),
            }
        )

    return {
        "design": design,
        "shared_paths": "frozen Transformer backbone; fixed output coordinates",
        "maximum_inactive_module_delta": maximum_inactive_delta,
        "mean_owning_accuracy": float(
            np.mean([row["owning_accuracy"] for row in rows])
        ),
        "mean_wrong_key_accuracy": float(
            np.mean([row["wrong_key_mean_accuracy"] for row in rows])
        ),
        "per_regime": rows,
        "separation_pass": maximum_inactive_delta == 0.0,
    }


def cargo_authorization_test(config: Config) -> dict:
    gate_key = hashlib.sha256(b"cdp-cargo-authorization-suite").digest()
    dimensions = 64
    embedding_spec = EmbeddingSpec(
        model_id="deterministic-suite-embedder",
        dimensions=dimensions,
        vocabulary_hash=hashlib.sha256(b"suite-vocabulary").hexdigest(),
        pooling="mean",
    )

    def embed(text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode()).digest()
        raw = np.frombuffer(digest * 2, dtype=np.uint8).astype(np.float64)
        return raw / (np.linalg.norm(raw) + 1e-12)

    store = InMemoryVectorStore()
    partition_map = PartitionMap(gate_key, dimensions, config.regimes)
    adapter = GatedRAGAdapter(store, partition_map, embed_fn=embed)
    terminal = BusTerminal(adapter, gate_key, embedding_spec)

    model_digest = "deterministic-suite-model-v1"
    for regime in range(2):
        partition = f"tenant-{regime}"
        partition_map.register(
            partition,
            regime_id=regime,
            mode=PartitionMode.READ_WRITE,
        )
        terminal.register_bus(partition, regime)
    alias_partition = "tenant-zero-same-regime-alias"
    partition_map.register(
        alias_partition,
        regime_id=0,
        mode=PartitionMode.IMMUTABLE,
    )
    terminal.register_bus(alias_partition, 0)

    items = [
        CargoItem(
            content="Tenant zero launch code is ALPHA-714.",
            embedding=embed("Tenant zero launch code is ALPHA-714."),
            doc_id="tenant-zero-secret",
        )
    ]
    manifest = create_manifest(
        tenant_id="customer-zero",
        regime_id=0,
        embedding_spec=embedding_spec,
        payload_hash=compute_payload_hash(items),
        payload_kind="rag_documents",
        item_count=1,
        subject_id="claims-team",
        model_digest=model_digest,
        operation="load",
        policy_version="research-v1",
        partition_key="tenant-0",
        gate_embeddings_at_rest=False,
    )
    bus = terminal.get_bus("tenant-0")
    load_key = derive_cargo_access_key(
        gate_key,
        manifest.regime_id,
        manifest.tenant_id,
        subject_id=manifest.subject_id,
        model_digest=manifest.model_digest,
        operation=manifest.operation,
        policy_version=manifest.policy_version,
        partition_key=manifest.partition_key,
    )
    session = bus.dock(manifest, gate_key_secret=load_key)
    session.load_cargo(items)
    load_receipt = session.depart()

    read_manifest = create_manifest(
        tenant_id=manifest.tenant_id,
        regime_id=manifest.regime_id,
        embedding_spec=manifest.embedding_spec,
        payload_hash=compute_payload_hash([]),
        payload_kind=manifest.payload_kind,
        item_count=0,
        subject_id=manifest.subject_id,
        model_digest=manifest.model_digest,
        operation="retrieve",
        policy_version=manifest.policy_version,
        partition_key=manifest.partition_key,
        gate_embeddings_at_rest=False,
    )
    access_key = derive_cargo_access_key(
        gate_key,
        read_manifest.regime_id,
        read_manifest.tenant_id,
        subject_id=read_manifest.subject_id,
        model_digest=read_manifest.model_digest,
        operation=read_manifest.operation,
        policy_version=read_manifest.policy_version,
        partition_key=read_manifest.partition_key,
    )
    session = bus.dock(read_manifest, gate_key_secret=access_key)
    result = session.unload_cargo("launch code", top_k=1)
    retrieval_receipt = session.depart()

    wrong_random_rejected = False
    try:
        bus.dock(read_manifest, gate_key_secret=os.urandom(32))
    except CargoAuthenticationError:
        wrong_random_rejected = True

    wrong_operation_rejected = False
    try:
        bus.dock(read_manifest, gate_key_secret=load_key)
    except CargoAuthenticationError:
        wrong_operation_rejected = True

    wrong_tenant_rejected = False
    wrong_tenant_key = derive_cargo_access_key(
        gate_key,
        read_manifest.regime_id,
        "customer-one",
        subject_id=read_manifest.subject_id,
        model_digest=read_manifest.model_digest,
        operation=read_manifest.operation,
        policy_version=read_manifest.policy_version,
        partition_key=read_manifest.partition_key,
    )
    try:
        bus.dock(read_manifest, gate_key_secret=wrong_tenant_key)
    except CargoAuthenticationError:
        wrong_tenant_rejected = True

    wrong_sub_id_rejected = False
    wrong_subject_key = derive_cargo_access_key(
        gate_key,
        read_manifest.regime_id,
        read_manifest.tenant_id,
        subject_id="unauthorized-subject",
        model_digest=read_manifest.model_digest,
        operation=read_manifest.operation,
        policy_version=read_manifest.policy_version,
        partition_key=read_manifest.partition_key,
    )
    try:
        bus.dock(read_manifest, gate_key_secret=wrong_subject_key)
    except CargoAuthenticationError:
        wrong_sub_id_rejected = True

    wrong_model_rejected = False
    wrong_model_key = derive_cargo_access_key(
        gate_key,
        read_manifest.regime_id,
        read_manifest.tenant_id,
        subject_id=read_manifest.subject_id,
        model_digest="other-model-v1",
        operation=read_manifest.operation,
        policy_version=read_manifest.policy_version,
        partition_key=read_manifest.partition_key,
    )
    try:
        bus.dock(read_manifest, gate_key_secret=wrong_model_key)
    except CargoAuthenticationError:
        wrong_model_rejected = True

    wrong_policy_rejected = False
    wrong_policy_key = derive_cargo_access_key(
        gate_key,
        read_manifest.regime_id,
        read_manifest.tenant_id,
        subject_id=read_manifest.subject_id,
        model_digest=read_manifest.model_digest,
        operation=read_manifest.operation,
        policy_version="research-v2",
        partition_key=read_manifest.partition_key,
    )
    try:
        bus.dock(read_manifest, gate_key_secret=wrong_policy_key)
    except CargoAuthenticationError:
        wrong_policy_rejected = True

    wrong_regime_rejected = False
    wrong_regime_key = derive_cargo_access_key(
        gate_key,
        1,
        read_manifest.tenant_id,
        subject_id=read_manifest.subject_id,
        model_digest=read_manifest.model_digest,
        operation=read_manifest.operation,
        policy_version=read_manifest.policy_version,
        partition_key=read_manifest.partition_key,
    )
    try:
        bus.dock(read_manifest, gate_key_secret=wrong_regime_key)
    except CargoAuthenticationError:
        wrong_regime_rejected = True

    wrong_same_regime_partition_rejected = False
    try:
        terminal.get_bus(alias_partition).dock(
            read_manifest,
            gate_key_secret=access_key,
        )
    except CargoAuthenticationError:
        wrong_same_regime_partition_rejected = True

    model_calls = 0
    invoke_manifest = create_manifest(
        tenant_id=read_manifest.tenant_id,
        regime_id=read_manifest.regime_id,
        embedding_spec=read_manifest.embedding_spec,
        payload_hash=read_manifest.payload_hash,
        payload_kind=read_manifest.payload_kind,
        item_count=0,
        subject_id=read_manifest.subject_id,
        model_digest=read_manifest.model_digest,
        operation=read_manifest.operation,
        policy_version=read_manifest.policy_version,
        partition_key=read_manifest.partition_key,
        gate_embeddings_at_rest=False,
    )

    def invoke_after_authorization(key: bytes) -> None:
        nonlocal model_calls
        authorized = bus.dock(invoke_manifest, gate_key_secret=key)
        model_calls += 1
        authorized.depart()

    calls_before_wrong_key = model_calls
    try:
        invoke_after_authorization(os.urandom(32))
    except CargoAuthenticationError:
        pass
    wrong_key_model_calls = model_calls - calls_before_wrong_key
    invoke_after_authorization(access_key)

    checks = {
        "owning_retrieval_matches": (
            len(result.docs) == 1
            and result.docs[0].doc_id == "tenant-zero-secret"
        ),
        "random_key_rejected": wrong_random_rejected,
        "other_operation_key_rejected": wrong_operation_rejected,
        "other_tenant_key_rejected": wrong_tenant_rejected,
        "other_sub_id_key_rejected": wrong_sub_id_rejected,
        "other_model_key_rejected": wrong_model_rejected,
        "other_policy_key_rejected": wrong_policy_rejected,
        "other_regime_key_rejected": wrong_regime_rejected,
        "other_partition_same_regime_rejected": (
            wrong_same_regime_partition_rejected
        ),
        "receipt_verifies": (
            bus.verify_receipt(load_receipt, expected_manifest=manifest)
            and bus.verify_receipt(
                retrieval_receipt,
                expected_manifest=read_manifest,
            )
        ),
        "wrong_key_model_calls": wrong_key_model_calls,
        "owning_key_model_calls": model_calls,
    }
    return {
        "design": "cargo_corpus_gate_authorization",
        **checks,
        "separation_pass": (
            checks["owning_retrieval_matches"]
            and checks["random_key_rejected"]
            and checks["other_operation_key_rejected"]
            and checks["other_tenant_key_rejected"]
            and checks["other_sub_id_key_rejected"]
            and checks["other_model_key_rejected"]
            and checks["other_policy_key_rejected"]
            and checks["other_regime_key_rejected"]
            and checks["other_partition_same_regime_rejected"]
            and checks["receipt_verifies"]
            and checks["wrong_key_model_calls"] == 0
            and checks["owning_key_model_calls"] == 1
        ),
    }


def runtime_inference_authorization_test() -> dict:
    """Exercise the research preflight immediately before a model callback."""

    model_id = "deterministic-runtime-embedder"
    dimensions = 4
    values = [1.0, -2.0, 3.5, 0.25]
    surface = GateExecutionPreflight(
        model_id=model_id,
        dimensions=dimensions,
        authorized_regime_ids=[0],
    )
    probe = surface.rejection_probe()
    rejections = probe["attempts"]
    owning_output = surface.invoke(0, lambda: list(values))
    maximum_authorized_output_difference = float(
        np.max(np.abs(np.asarray(owning_output) - np.asarray(values)))
    )
    return {
        "design": "research_execution_preflight",
        "declared_surface": (
            "configured model + dimensions + regime authorization immediately before "
            "the experiment callback; this is not a production serving surface"
        ),
        "unauthorized_attempts": rejections,
        "unauthorized_model_calls": probe["unauthorized_model_calls"],
        "owning_model_calls": surface.model_calls,
        "maximum_authorized_output_difference": (
            maximum_authorized_output_difference
        ),
        "separation_pass": (
            all(row["rejected"] for row in rejections.values())
            and all(row["model_calls"] == 0 for row in rejections.values())
            and surface.model_calls == 1
            and maximum_authorized_output_difference == 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canary",
        action="store_true",
        help="Run a reduced persisted protocol cycle before the full CPU suite.",
    )
    args = parser.parse_args()
    config = (
        Config(train_examples=64, test_examples=64, steps=24)
        if args.canary
        else Config()
    )
    seed_everything(config.seed)
    started = time.perf_counter()
    backbone = make_frozen_transformer(config)
    train_features, train_labels, test_features, test_labels = make_features(
        backbone,
        config,
    )

    pre_mlp = pre_mlp_ffn_gate_test(config)
    dense = train_dense(
        config,
        train_features,
        train_labels,
        test_features,
        test_labels,
    )
    adapters_model = GatedAdapterBank(config)
    adapters = train_private_modules(
        adapters_model,
        adapters_model.adapters,
        config,
        train_features,
        train_labels,
        test_features,
        test_labels,
        design="complete_gated_residual_adapters",
    )
    experts_model = PrivateExpertBank(config)
    experts = train_private_modules(
        experts_model,
        experts_model.experts,
        config,
        train_features,
        train_labels,
        test_features,
        test_labels,
        design="hard_routed_private_experts",
    )
    placement = gate_placement_negative_control(config)
    cargo = cargo_authorization_test(config)
    whole_model = runtime_inference_authorization_test()

    results = [
        pre_mlp,
        dense,
        adapters,
        experts,
        placement,
        cargo,
        whole_model,
    ]
    artifact = {
        "schema_version": 2,
        "experiment": "local_transformer_cotenancy_suite",
        "run_mode": "canary" if args.canary else "full",
        "status": (
            "pass"
            if all(result["separation_pass"] for result in results)
            else "separation_failure"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "libraries": collect_library_provenance(),
        },
        "config": config.__dict__,
        "source": source_provenance(),
        "results": results,
    }
    result_root = Path(__file__).resolve().parent / "results"
    output_dir = result_root / "archive" / "smoke" if args.canary else result_root
    output = output_dir / (
        f"transformer_cotenancy_{'canary' if args.canary else 'local'}_"
        f"{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))
    print(f"Results saved to {output}")
    if artifact["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
