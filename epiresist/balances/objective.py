from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossWeights:
    physics: float = 0.1
    sparse: float = 0.01
    temporal: float = 0.05


def prediction_loss(
    regression: torch.Tensor, target: torch.Tensor, logits: torch.Tensor, label: torch.Tensor
) -> torch.Tensor:
    return F.mse_loss(regression, target) + F.binary_cross_entropy_with_logits(logits, label)


def sparsity_loss(attention_mass: torch.Tensor) -> torch.Tensor:
    return attention_mass.abs().mean()


def temporal_smoothness(trajectory: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    if trajectory.shape[0] < 2:
        return trajectory.new_zeros(())
    jumps = trajectory[1:] - trajectory[:-1]
    magnitude = jumps.pow(2).sum(dim=-1)
    mask = (magnitude.sqrt() > threshold).to(magnitude.dtype)
    return (magnitude * mask).mean()


def total_loss(
    regression: torch.Tensor,
    target: torch.Tensor,
    logits: torch.Tensor,
    label: torch.Tensor,
    physics: torch.Tensor,
    attention_mass: torch.Tensor,
    trajectory: torch.Tensor,
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred = prediction_loss(regression, target, logits, label)
    sparse = sparsity_loss(attention_mass)
    temporal = temporal_smoothness(trajectory)
    total = pred + weights.physics * physics + weights.sparse * sparse + weights.temporal * temporal
    components = {
        "prediction": float(pred.detach()),
        "physics": float(physics.detach()),
        "sparse": float(sparse.detach()),
        "temporal": float(temporal.detach()),
        "total": float(total.detach()),
    }
    return total, components
