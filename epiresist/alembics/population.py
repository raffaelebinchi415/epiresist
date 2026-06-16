from __future__ import annotations

import torch


def population_rate(
    sensitive: torch.Tensor,
    resistant: torch.Tensor,
    effect: torch.Tensor,
    r_s: torch.Tensor,
    r_r: torch.Tensor,
    carrying: torch.Tensor,
    delta_s_max: torch.Tensor,
    delta_r_max: torch.Tensor,
    alpha: torch.Tensor,
    mu: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    total = sensitive + resistant
    occupancy = 1.0 - total / (carrying + 1e-6)
    delta_s = delta_s_max * effect
    delta_r = delta_r_max * effect * alpha
    d_sensitive = r_s * sensitive * occupancy - delta_s * sensitive - mu * sensitive
    d_resistant = r_r * resistant * occupancy - delta_r * resistant + mu * sensitive
    return d_sensitive, d_resistant
