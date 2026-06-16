from __future__ import annotations

import torch


def concentration_rate(
    concentration: torch.Tensor, dose: torch.Tensor, k_e: torch.Tensor, k_a: torch.Tensor
) -> torch.Tensor:
    return -k_e * concentration + k_a * dose


def effect_rate(
    concentration: torch.Tensor,
    effect: torch.Tensor,
    k_eff: torch.Tensor,
    ec50: torch.Tensor,
    hill: torch.Tensor,
    k_off: torch.Tensor,
) -> torch.Tensor:
    conc = concentration.clamp_min(0.0)
    hill_term = conc.pow(hill) / (ec50.pow(hill) + conc.pow(hill) + 1e-9)
    return k_eff * hill_term - k_off * effect
