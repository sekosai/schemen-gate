"""Authority-constrained learned mixture-of-experts primitives.

The outer gate is not learned: execution authority selects one regime and its
expert set.  Inside that set, a conventional top-1 router is trained from task
loss plus a Switch-style load-balancing auxiliary loss.  Candidate restriction
therefore precedes softmax and dispatch; unauthorized experts are never valid
router candidates.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RouteTrace:
    """Observed local and global expert selections for one forward pass."""

    local_experts: torch.Tensor
    global_experts: torch.Tensor
    probabilities: torch.Tensor


class Expert(nn.Module):
    """Small feed-forward classifier expert."""

    def __init__(self, input_dimensions: int, hidden_dimensions: int, classes: int):
        super().__init__()
        self.up = nn.Linear(input_dimensions, hidden_dimensions)
        self.down = nn.Linear(hidden_dimensions, classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(inputs)))


class LearnedTop1MoE(nn.Module):
    """Trainable token-choice top-1 MoE for one authorized regime."""

    def __init__(
        self,
        *,
        input_dimensions: int,
        hidden_dimensions: int,
        classes: int,
        experts: int,
    ) -> None:
        super().__init__()
        if min(input_dimensions, hidden_dimensions, classes, experts) <= 0:
            raise ValueError("MoE dimensions and expert count must be positive")
        self.input_dimensions = input_dimensions
        self.classes = classes
        self.expert_count = experts
        self.router = nn.Linear(input_dimensions, experts)
        self.experts = nn.ModuleList(
            [
                Expert(input_dimensions, hidden_dimensions, classes)
                for _ in range(experts)
            ]
        )

    def route(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = F.softmax(self.router(inputs), dim=-1)
        selected = probabilities.argmax(dim=-1)
        return probabilities, selected

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        expert_offset: int = 0,
    ) -> tuple[torch.Tensor, RouteTrace]:
        probabilities, selected = self.route(inputs)
        outputs = inputs.new_zeros((inputs.shape[0], self.classes))
        for expert_index, expert in enumerate(self.experts):
            rows = selected == expert_index
            if bool(rows.any()):
                routed = expert(inputs[rows])
                gate_weight = probabilities[rows, expert_index].unsqueeze(-1)
                outputs[rows] = routed * gate_weight
        return outputs, RouteTrace(
            local_experts=selected,
            global_experts=selected + expert_offset,
            probabilities=probabilities,
        )

    def load_balance_loss(self, trace: RouteTrace) -> torch.Tensor:
        """Switch-style auxiliary loss over this authorized expert set only."""

        assignment = F.one_hot(
            trace.local_experts,
            num_classes=self.expert_count,
        ).to(trace.probabilities.dtype)
        dispatch_fraction = assignment.mean(dim=0).detach()
        probability_fraction = trace.probabilities.mean(dim=0)
        return self.expert_count * torch.sum(
            dispatch_fraction * probability_fraction
        )


class AuthorizedExpertBank(nn.Module):
    """Packed regime bank with a learned router inside each authorized set."""

    def __init__(
        self,
        *,
        regimes: int,
        input_dimensions: int,
        hidden_dimensions: int,
        classes: int,
        experts_per_regime: int,
    ) -> None:
        super().__init__()
        if regimes <= 0:
            raise ValueError("regimes must be positive")
        self.regimes = regimes
        self.experts_per_regime = experts_per_regime
        self.lanes = nn.ModuleList(
            [
                LearnedTop1MoE(
                    input_dimensions=input_dimensions,
                    hidden_dimensions=hidden_dimensions,
                    classes=classes,
                    experts=experts_per_regime,
                )
                for _ in range(regimes)
            ]
        )

    @property
    def total_experts(self) -> int:
        return self.regimes * self.experts_per_regime

    def allowed_experts(self, regime: int) -> range:
        if regime < 0 or regime >= self.regimes:
            raise PermissionError(f"regime {regime} has no expert authority")
        start = regime * self.experts_per_regime
        return range(start, start + self.experts_per_regime)

    def forward(
        self,
        inputs: torch.Tensor,
        regime: int,
    ) -> tuple[torch.Tensor, RouteTrace]:
        allowed = self.allowed_experts(regime)
        return self.lanes[regime](inputs, expert_offset=allowed.start)

    def dense_masked_route(
        self,
        inputs: torch.Tensor,
        regime: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference global-router view with -infinity before softmax/top-1."""

        allowed = self.allowed_experts(regime)
        all_logits = torch.cat([lane.router(inputs) for lane in self.lanes], dim=-1)
        mask = torch.zeros(
            self.total_experts,
            dtype=torch.bool,
            device=inputs.device,
        )
        mask[allowed.start : allowed.stop] = True
        restricted_logits = all_logits.masked_fill(~mask.unsqueeze(0), -torch.inf)
        probabilities = F.softmax(restricted_logits, dim=-1)
        return probabilities, probabilities.argmax(dim=-1)

    @classmethod
    def pack(cls, lanes: Iterable[LearnedTop1MoE]) -> "AuthorizedExpertBank":
        copied = [copy.deepcopy(lane) for lane in lanes]
        if not copied:
            raise ValueError("at least one trained lane is required")
        first = copied[0]
        first_expert = first.experts[0]
        bank = cls(
            regimes=len(copied),
            input_dimensions=first.input_dimensions,
            hidden_dimensions=first_expert.up.out_features,
            classes=first.classes,
            experts_per_regime=first.expert_count,
        )
        for lane in copied:
            if (
                lane.input_dimensions != first.input_dimensions
                or lane.classes != first.classes
                or lane.expert_count != first.expert_count
                or lane.experts[0].up.out_features
                != first_expert.up.out_features
            ):
                raise ValueError("all packed lanes must have the same architecture")
        bank.lanes = nn.ModuleList(copied)
        return bank


def unsafe_zero_logit_route(
    logits: torch.Tensor,
    allowed: torch.Tensor,
) -> torch.Tensor:
    """Deliberately invalid pre-softmax masking used as a negative control."""

    return (logits * allowed.to(logits.dtype)).argmax(dim=-1)


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


def snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }
