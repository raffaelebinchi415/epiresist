from __future__ import annotations

import torch


def dnmt_activity(
    dnmt3a: torch.Tensor, dnmt3b: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return torch.sigmoid(weights[:, 0] * dnmt3a + weights[:, 1] * dnmt3b + bias)


def tet_activity(
    tet2: torch.Tensor, concentration: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return torch.sigmoid(weights[:, 0] * tet2 + weights[:, 1] * concentration + bias)


def epigenetic_rate(
    methylation: torch.Tensor,
    concentration: torch.Tensor,
    k_methyl: torch.Tensor,
    k_demethyl: torch.Tensor,
    f_weights: torch.Tensor,
    f_bias: torch.Tensor,
    g_weights: torch.Tensor,
    g_bias: torch.Tensor,
    dnmt3a: torch.Tensor,
    dnmt3b: torch.Tensor,
    tet2: torch.Tensor,
) -> torch.Tensor:
    f = dnmt_activity(dnmt3a, dnmt3b, f_weights, f_bias)
    g = tet_activity(tet2, concentration, g_weights, g_bias)
    return k_methyl * (1.0 - methylation) * f - k_demethyl * methylation * g
