from __future__ import annotations

import torch


def log2_quantile(expr: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    logged = torch.log2(expr.clamp_min(0.0) + eps)
    ranks = logged.argsort(dim=-1).argsort(dim=-1).to(logged.dtype)
    denom = max(logged.shape[-1] - 1, 1)
    quantiles = ranks / denom
    reference = logged.mean(dim=0, keepdim=True).sort(dim=-1).values
    index = (quantiles * denom).round().long().clamp(0, denom)
    gathered = reference.expand(expr.shape[0], -1).gather(-1, index)
    return gathered


def beta_values(
    intensity_methylated: torch.Tensor, intensity_unmethylated: torch.Tensor, offset: float = 100.0
) -> torch.Tensor:
    m = intensity_methylated.clamp_min(0.0)
    u = intensity_unmethylated.clamp_min(0.0)
    return m / (m + u + offset)


def minmax_doseresponse(drug: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    flat = drug.reshape(drug.shape[0], -1)
    lo = flat.min(dim=-1, keepdim=True).values
    hi = flat.max(dim=-1, keepdim=True).values
    scaled = (flat - lo) / (hi - lo + eps)
    return scaled.reshape(drug.shape)
