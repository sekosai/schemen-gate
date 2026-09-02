from __future__ import annotations

import torch
from local_transformer_cotenancy_suite import RegimeScopedAdam

from schemen_gate import GateMask


def test_gate_zeros_inactive_activation_and_aligned_parameter_gradients_exactly() -> None:
    gate = GateMask.from_indices([0, 2, 5], n_dims=6, regime_id=7)
    active = gate.to_torch(dtype=torch.bool)
    inputs = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
        requires_grad=True,
    )
    aligned_projection = torch.nn.Parameter(
        torch.arange(1.0, 19.0).reshape(6, 3)
    )

    gated = gate.apply(inputs)
    loss = (gated @ aligned_projection).square().sum()
    loss.backward()

    assert torch.count_nonzero(gated[:, ~active]) == 0
    assert inputs.grad is not None
    assert aligned_projection.grad is not None
    assert torch.count_nonzero(inputs.grad[:, ~active]) == 0
    assert torch.count_nonzero(aligned_projection.grad[~active]) == 0
    assert torch.count_nonzero(inputs.grad[:, active]) > 0
    assert torch.count_nonzero(aligned_projection.grad[active]) > 0


def test_regime_scoped_adam_confines_parameters_and_first_second_moments_exactly() -> None:
    parameter = torch.nn.Parameter(torch.arange(1.0, 7.0))
    active = torch.tensor([True, False, True, False, False, True])
    before = parameter.detach().clone()
    optimizer = RegimeScopedAdam(
        [(parameter, active)],
        learning_rate=0.01,
    )

    for scale in (1.0, 3.0):
        # Deliberately supply nonzero gradients in every coordinate. The
        # optimizer authorization mask, not an accidental zero gradient, must
        # protect the inactive parameter and state entries.
        parameter.grad = scale * torch.arange(1.0, 7.0)
        optimizer.step()

        first, second = optimizer.moments[id(parameter)]
        assert torch.equal(parameter.detach()[~active], before[~active])
        assert torch.count_nonzero(first[~active]) == 0
        assert torch.count_nonzero(second[~active]) == 0

    first, second = optimizer.moments[id(parameter)]
    assert torch.count_nonzero(parameter.detach()[active] - before[active]) > 0
    assert torch.count_nonzero(first[active]) == active.count_nonzero()
    assert torch.count_nonzero(second[active]) == active.count_nonzero()
