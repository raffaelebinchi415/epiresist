from __future__ import annotations

import torch
from torch import nn


class CrossModalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, modalities: torch.Tensor) -> torch.Tensor:
        n_modalities = modalities.shape[1]
        attended = []
        for m in range(n_modalities):
            query = modalities[:, m : m + 1]
            others = torch.cat([modalities[:, :m], modalities[:, m + 1 :]], dim=1)
            out, _ = self.attention(query, others, others)
            attended.append(self.norm(out + query))
        return torch.cat(attended, dim=1)
