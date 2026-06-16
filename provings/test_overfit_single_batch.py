from __future__ import annotations

import torch

from epiresist.apparatus import seed_all
from epiresist.balances import LossWeights, total_loss


def test_single_batch_overfits(model, batch):
    seed_all(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    weights = LossWeights()
    first = None
    last = None
    for step in range(40):
        output = model(batch)
        loss, _ = total_loss(
            output.regression,
            batch.target,
            output.logits,
            batch.label,
            output.physics,
            output.attention_mass,
            output.trajectory,
            weights,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 0:
            first = float(loss.detach())
        last = float(loss.detach())
    assert first is not None and last is not None
    assert last < 0.5 * first
