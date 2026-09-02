"""Token-level MoE routing with a trusted, execution-selected soft prefix.

The capability prefix is an internal learned vector selected only after
execution authorizes a regime. It is never parsed from user text. User tokens may
condition semantic routing among authorized experts, but cannot select the
prefix or expand the authorized expert set.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TokenRouteTrace:
    """Selections for non-padding tokens only."""

    local_experts: torch.Tensor
    global_experts: torch.Tensor
    probabilities: torch.Tensor


class TokenExpert(nn.Module):
    def __init__(self, dimensions: int, hidden_dimensions: int) -> None:
        super().__init__()
        self.up = nn.Linear(dimensions, hidden_dimensions)
        self.down = nn.Linear(hidden_dimensions, dimensions)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(inputs)))


class CapabilityTokenLane(nn.Module):
    """One private token router, soft prefix, expert set, and classifier."""

    def __init__(
        self,
        *,
        vocabulary_size: int,
        embedding_dimensions: int,
        hidden_dimensions: int,
        classes: int,
        experts: int,
    ) -> None:
        super().__init__()
        if min(
            vocabulary_size,
            embedding_dimensions,
            hidden_dimensions,
            classes,
            experts,
        ) <= 0:
            raise ValueError("token MoE dimensions must be positive")
        self.vocabulary_size = vocabulary_size
        self.embedding_dimensions = embedding_dimensions
        self.hidden_dimensions = hidden_dimensions
        self.classes = classes
        self.expert_count = experts
        self.token_embedding = nn.Embedding(
            vocabulary_size,
            embedding_dimensions,
            padding_idx=0,
        )
        self.capability_prefix = nn.Parameter(
            torch.empty(embedding_dimensions)
        )
        nn.init.normal_(
            self.capability_prefix,
            mean=0.0,
            std=embedding_dimensions**-0.5,
        )
        self.router = nn.Linear(2 * embedding_dimensions, experts)
        self.experts = nn.ModuleList(
            [TokenExpert(embedding_dimensions, hidden_dimensions) for _ in range(experts)]
        )
        self.classifier = nn.Linear(embedding_dimensions, classes)

    def router_logits(
        self,
        token_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        prefix = self.capability_prefix.view(1, 1, -1).expand(
            token_embeddings.shape[0],
            token_embeddings.shape[1],
            -1,
        )
        router_inputs = torch.cat((token_embeddings, prefix), dim=-1)
        return self.router(router_inputs)

    def route(
        self,
        token_embeddings: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = F.softmax(self.router_logits(token_embeddings), dim=-1)
        selected = probabilities.argmax(dim=-1)
        return probabilities, selected.masked_fill(~token_mask, 0)

    def forward(
        self,
        token_ids: torch.Tensor,
        *,
        expert_offset: int = 0,
    ) -> tuple[torch.Tensor, TokenRouteTrace]:
        if token_ids.ndim != 2:
            raise ValueError("token ids must have shape [batch, sequence]")
        if token_ids.dtype != torch.long:
            raise ValueError("token ids must use torch.long")
        if bool(((token_ids < 0) | (token_ids >= self.vocabulary_size)).any()):
            raise ValueError("token ids must be inside the configured vocabulary")
        token_mask = token_ids != 0
        if bool((token_mask.sum(dim=1) == 0).any()):
            raise ValueError("every example must contain at least one user token")
        embedded = self.token_embedding(token_ids)
        probabilities, selected = self.route(embedded, token_mask)
        expert_output = torch.zeros_like(embedded)
        for expert_index, expert in enumerate(self.experts):
            positions = torch.logical_and(selected == expert_index, token_mask)
            if bool(positions.any()):
                transformed = expert(embedded[positions])
                weights = probabilities[..., expert_index][positions].unsqueeze(-1)
                expert_output[positions] = transformed * weights
        combined = (embedded + expert_output) * token_mask.unsqueeze(-1)
        pooled = combined.sum(dim=1) / token_mask.sum(dim=1, keepdim=True)
        valid_selected = selected[token_mask]
        return self.classifier(pooled), TokenRouteTrace(
            local_experts=valid_selected,
            global_experts=valid_selected + expert_offset,
            probabilities=probabilities[token_mask],
        )

    def load_balance_loss(self, trace: TokenRouteTrace) -> torch.Tensor:
        assignment = F.one_hot(
            trace.local_experts,
            num_classes=self.expert_count,
        ).to(trace.probabilities.dtype)
        dispatch_fraction = assignment.mean(dim=0).detach()
        probability_fraction = trace.probabilities.mean(dim=0)
        return self.expert_count * torch.sum(
            dispatch_fraction * probability_fraction
        )


class CapabilityTokenBank(nn.Module):
    """Packed bank whose trusted regime selects an internal soft prefix."""

    def __init__(self, lanes: Iterable[CapabilityTokenLane]) -> None:
        super().__init__()
        copied = [copy.deepcopy(lane) for lane in lanes]
        if not copied:
            raise ValueError("at least one capability token lane is required")
        first = copied[0]
        for lane in copied[1:]:
            if (
                lane.vocabulary_size != first.vocabulary_size
                or lane.embedding_dimensions != first.embedding_dimensions
                or lane.hidden_dimensions != first.hidden_dimensions
                or lane.classes != first.classes
                or lane.expert_count != first.expert_count
            ):
                raise ValueError("all capability token lanes must match")
        self.lanes = nn.ModuleList(copied)
        self.regimes = len(copied)
        self.experts_per_regime = first.expert_count

    @property
    def total_experts(self) -> int:
        return self.regimes * self.experts_per_regime

    def allowed_experts(self, regime: int) -> range:
        if isinstance(regime, bool) or not isinstance(regime, int):
            raise PermissionError("regime authority must be an integer identifier")
        if regime < 0 or regime >= self.regimes:
            raise PermissionError(f"regime {regime} has no token-routing authority")
        start = regime * self.experts_per_regime
        return range(start, start + self.experts_per_regime)

    def forward(
        self,
        token_ids: torch.Tensor,
        regime: int,
    ) -> tuple[torch.Tensor, TokenRouteTrace]:
        allowed = self.allowed_experts(regime)
        return self.lanes[regime](token_ids, expert_offset=allowed.start)

    def dense_masked_route(
        self,
        token_ids: torch.Tensor,
        regime: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Global reference with authorization applied before softmax/top-1."""

        allowed = self.allowed_experts(regime)
        lane = self.lanes[regime]
        token_mask = token_ids != 0
        embedded = lane.token_embedding(token_ids)
        local_logits = lane.router_logits(embedded)
        global_logits = embedded.new_zeros(
            (*local_logits.shape[:-1], self.total_experts)
        )
        global_logits[..., allowed.start : allowed.stop] = local_logits
        authorization_mask = torch.zeros(
            self.total_experts,
            dtype=torch.bool,
            device=token_ids.device,
        )
        authorization_mask[allowed.start : allowed.stop] = True
        restricted = global_logits.masked_fill(
            ~authorization_mask.view(1, 1, -1),
            -torch.inf,
        )
        probabilities = F.softmax(restricted, dim=-1)
        return probabilities[token_mask], probabilities.argmax(dim=-1)[token_mask]


def snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }


def maximum_parameter_delta(
    before: dict[str, torch.Tensor],
    module: nn.Module,
) -> float:
    if not before:
        return 0.0
    return max(
        float((parameter.detach() - before[name]).abs().max())
        for name, parameter in module.named_parameters()
    )
