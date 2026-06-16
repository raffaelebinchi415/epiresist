from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from epiresist.alembics.epigenetic import epigenetic_rate
from epiresist.alembics.pharmacology import concentration_rate, effect_rate
from epiresist.alembics.population import population_rate

STATE_M = 0
STATE_C = 1
STATE_E = 2
STATE_S = 3
STATE_R = 4
STATE_DIM = 5


@dataclass
class OdeParameters:
    k_methyl: torch.Tensor
    k_demethyl: torch.Tensor
    f_weights: torch.Tensor
    f_bias: torch.Tensor
    g_weights: torch.Tensor
    g_bias: torch.Tensor
    k_e: torch.Tensor
    k_a: torch.Tensor
    k_eff: torch.Tensor
    ec50: torch.Tensor
    hill: torch.Tensor
    k_off: torch.Tensor
    r_s: torch.Tensor
    r_r: torch.Tensor
    carrying: torch.Tensor
    delta_s_max: torch.Tensor
    delta_r_max: torch.Tensor
    alpha: torch.Tensor
    mu: torch.Tensor
    coupling_gain: torch.Tensor
    dnmt3a: torch.Tensor
    dnmt3b: torch.Tensor
    tet2: torch.Tensor
    dose: torch.Tensor


@dataclass(frozen=True)
class OdeSwitches:
    enable_epigenetic: bool = True
    enable_pharmacology: bool = True
    enable_population: bool = True
    use_coupling: bool = True


class CoupledSystem(nn.Module):
    def __init__(
        self, params: OdeParameters, switches: OdeSwitches, baseline: torch.Tensor
    ) -> None:
        super().__init__()
        self.params = params
        self.switches = switches
        self.baseline = baseline

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        p = self.params
        s = self.switches
        methyl = state[:, STATE_M]
        conc = state[:, STATE_C]
        effect = state[:, STATE_E]
        sensitive = state[:, STATE_S]
        resistant = state[:, STATE_R]

        drug_for_methyl = conc if s.use_coupling else torch.zeros_like(conc)
        effect_for_pop = effect if s.use_coupling else self.baseline[:, STATE_E]

        d_methyl = epigenetic_rate(
            methyl,
            drug_for_methyl,
            p.k_methyl,
            p.k_demethyl,
            p.f_weights,
            p.f_bias,
            p.g_weights,
            p.g_bias,
            p.dnmt3a,
            p.dnmt3b,
            p.tet2,
        )
        if s.use_coupling:
            fraction = resistant / (sensitive + resistant + 1e-6)
            d_methyl = d_methyl + p.coupling_gain * fraction

        d_conc = concentration_rate(conc, p.dose, p.k_e, p.k_a)
        d_effect = effect_rate(conc, effect, p.k_eff, p.ec50, p.hill, p.k_off)
        d_sensitive, d_resistant = population_rate(
            sensitive,
            resistant,
            effect_for_pop,
            p.r_s,
            p.r_r,
            p.carrying,
            p.delta_s_max,
            p.delta_r_max,
            p.alpha,
            p.mu,
        )

        if not s.enable_epigenetic:
            d_methyl = torch.zeros_like(d_methyl)
        if not s.enable_pharmacology:
            d_conc = torch.zeros_like(d_conc)
            d_effect = torch.zeros_like(d_effect)
        if not s.enable_population:
            d_sensitive = torch.zeros_like(d_sensitive)
            d_resistant = torch.zeros_like(d_resistant)

        return torch.stack([d_methyl, d_conc, d_effect, d_sensitive, d_resistant], dim=-1)
