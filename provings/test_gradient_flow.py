from __future__ import annotations

import torch

from epiresist.balances import LossWeights, total_loss


def test_all_parameters_receive_gradient(model, batch):
    output = model(batch)
    loss, _ = total_loss(
        output.regression,
        batch.target,
        output.logits,
        batch.label,
        output.physics,
        output.attention_mass,
        output.trajectory,
        LossWeights(),
    )
    loss.backward()
    missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert missing == []


def test_gradient_flows_through_ode_solver(model, batch):
    output = model(batch)
    output.physics.backward()
    grads = [p.grad for n, p in model.named_parameters() if "parameter_head" in n]
    assert any(g is not None and torch.any(g != 0) for g in grads)
