from __future__ import annotations

import torch
from torch import nn


class DoseResponseEncoder(nn.Module):
    def __init__(
        self,
        n_replicates: int,
        d_model: int,
        filters: tuple[int, int, int] = (64, 128, 256),
        kernels: tuple[int, int, int] = (3, 5, 7),
    ) -> None:
        super().__init__()
        channels = [n_replicates, *filters]
        layers: list[nn.Module] = []
        for i, k in enumerate(kernels):
            layers.append(nn.Conv1d(channels[i], channels[i + 1], kernel_size=k, padding=k // 2))
            layers.append(nn.ReLU())
        self.convolutions = nn.Sequential(*layers)
        self.project = nn.Linear(filters[-1], d_model)

    def forward(self, drug: torch.Tensor) -> torch.Tensor:
        features = self.convolutions(drug)
        pooled = features.mean(dim=-1)
        return self.project(pooled)
