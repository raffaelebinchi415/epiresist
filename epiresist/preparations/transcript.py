from __future__ import annotations

import math

import torch
from torch import nn


class TranscriptEncoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_gene_sets: int,
        d_model: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.n_gene_sets = n_gene_sets
        self.set_size = math.ceil(n_genes / n_gene_sets)
        self.padded = self.set_size * n_gene_sets

        self.gene_embed = nn.Linear(1, d_model)

        intra = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 2, dropout=dropout, batch_first=True
        )
        self.intra_set = nn.TransformerEncoder(intra, num_layers=1)

        inter = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 2, dropout=dropout, batch_first=True
        )
        self.inter_set = nn.TransformerEncoder(inter, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)
        self.enzyme_head = nn.Linear(d_model, 3)

    def forward(self, expr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = expr.shape[0]
        if expr.shape[1] < self.padded:
            pad = self.padded - expr.shape[1]
            expr = torch.nn.functional.pad(expr, (0, pad))
        grouped = expr.reshape(batch, self.n_gene_sets, self.set_size, 1)
        tokens = self.gene_embed(grouped)
        tokens = tokens.reshape(batch * self.n_gene_sets, self.set_size, -1)
        intra = self.intra_set(tokens)
        set_repr = intra.mean(dim=1).reshape(batch, self.n_gene_sets, -1)
        inter = self.inter_set(set_repr)
        pooled = self.norm(inter.mean(dim=1))
        enzymes = torch.nn.functional.softplus(self.enzyme_head(pooled))
        return pooled, enzymes
