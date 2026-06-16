from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import msgspec
import torch
import yaml
from torch import nn

from epiresist.alembics import (
    STATE_E,
    STATE_M,
    STATE_R,
    STATE_S,
    CoupledSystem,
    OdeParameters,
    OdeSwitches,
    integrate,
    physics_residual,
)
from epiresist.compounding import CrossModalAttention, PhysicsGate
from epiresist.materia import CohortBatch
from epiresist.preparations import DoseResponseEncoder, MethylationEncoder, TranscriptEncoder

N_ODE_RAW = 22


class MateriaCfg(msgspec.Struct, frozen=True):
    name: str = "beat_aml"
    n_samples: int = 512
    n_genes: int = 2048
    n_cpg: int = 1024
    n_concentrations: int = 7
    n_replicates: int = 3
    n_gene_sets: int = 32
    seed: int = 42
    manifest: str | None = None


class PreparationCfg(msgspec.Struct, frozen=True):
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    gat_heads: int = 4
    gat_layers: int = 2
    n_graph_genes: int = 128
    graph_degree: int = 3
    cnn_filters: tuple[int, int, int] = (64, 128, 256)
    cnn_kernels: tuple[int, int, int] = (3, 5, 7)
    dropout: float = 0.1
    active_modalities: tuple[bool, bool, bool] = (True, True, True)


class AlembicCfg(msgspec.Struct, frozen=True):
    solver: str = "dopri5"
    rtol: float = 1e-5
    atol: float = 1e-7
    integration_steps: int = 10
    t_final: float = 1.0
    adjoint: bool = True
    enable_epigenetic: bool = True
    enable_pharmacology: bool = True
    enable_population: bool = True
    use_coupling: bool = True


class BalanceCfg(msgspec.Struct, frozen=True):
    lambda_physics: float = 0.1
    lambda_sparse: float = 0.01
    lambda_temporal: float = 0.05


class RegimenCfg(msgspec.Struct, frozen=True):
    lr: float = 5e-4
    weight_decay: float = 1e-5
    epochs: int = 200
    patience: int = 20
    batch_size: int = 32
    grad_clip: float = 1.0
    amp: bool = False
    ema_decay: float = 0.999
    seed: int = 42
    folds: int = 5
    device: str = "auto"


class AssayCfg(msgspec.Struct, frozen=True):
    top_k: int = 10
    bootstrap_iters: int = 1000
    n_perm: int = 200


class FormularyConfig(msgspec.Struct, frozen=True):
    name: str = "main"
    enable_mechanism: bool = True
    materia: MateriaCfg = msgspec.field(default_factory=MateriaCfg)
    preparation: PreparationCfg = msgspec.field(default_factory=PreparationCfg)
    alembic: AlembicCfg = msgspec.field(default_factory=AlembicCfg)
    balance: BalanceCfg = msgspec.field(default_factory=BalanceCfg)
    regimen: RegimenCfg = msgspec.field(default_factory=RegimenCfg)
    assay: AssayCfg = msgspec.field(default_factory=AssayCfg)


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve(path: Path, seen: set[Path]) -> dict:
    if path in seen:
        raise ValueError(f"circular inherit chain at {path.name}")
    seen.add(path)
    raw = yaml.safe_load(path.read_text()) or {}
    parents = raw.pop("inherit", [])
    if isinstance(parents, str):
        parents = [parents]
    resolved: dict = {}
    for parent in parents:
        resolved = _deep_merge(resolved, _resolve(path.parent / parent, seen))
    return _deep_merge(resolved, raw)


def load_formula(name_or_path: str, recipes_dir: str | Path = "recipes/formula") -> FormularyConfig:
    candidate = Path(name_or_path)
    if not candidate.exists():
        candidate = Path(recipes_dir) / f"{name_or_path}.yaml"
    merged = _resolve(candidate, set())
    return msgspec.convert(merged, FormularyConfig)


@dataclass
class ModelOutput:
    regression: torch.Tensor
    logits: torch.Tensor
    trajectory: torch.Tensor
    terminal_state: torch.Tensor
    attention_mass: torch.Tensor
    physics: torch.Tensor
    ode_params: OdeParameters


class EpiResistPINN(nn.Module):
    def __init__(self, config: FormularyConfig) -> None:
        super().__init__()
        self.config = config
        prep = config.preparation
        d = prep.d_model
        self.transcript = TranscriptEncoder(
            config.materia.n_genes,
            config.materia.n_gene_sets,
            d,
            prep.n_heads,
            prep.n_layers,
            prep.dropout,
        )
        self.methylation = MethylationEncoder(
            config.materia.n_cpg,
            prep.n_graph_genes,
            d,
            prep.gat_heads,
            prep.gat_layers,
            prep.graph_degree,
            config.materia.seed,
        )
        self.doseresponse = DoseResponseEncoder(
            config.materia.n_replicates,
            d,
            prep.cnn_filters,
            prep.cnn_kernels,
        )
        self.crossmodal = CrossModalAttention(d, prep.n_heads, prep.dropout)
        self.gate = PhysicsGate(d, n_modalities=3, n_state=3)
        self.parameter_head = nn.Linear(d, N_ODE_RAW)
        self.initial_head = nn.Linear(d, 5)
        self.regression_head = nn.Linear(d, 1)
        self.resistance_head = nn.Linear(d, 1)
        nn.init.zeros_(self.parameter_head.bias)

    def _fuse_initial(self, batch: CohortBatch) -> tuple[torch.Tensor, torch.Tensor, OdeParameters]:
        h_expr, enzymes = self.transcript(batch.expr)
        h_meth = self.methylation(batch.methyl)
        h_drug = self.doseresponse(batch.drug)
        flags = self.config.preparation.active_modalities
        h_expr = h_expr * float(flags[0])
        enzymes = enzymes * float(flags[0])
        h_meth = h_meth * float(flags[1])
        h_drug = h_drug * float(flags[2])
        modalities = torch.stack([h_expr, h_meth, h_drug], dim=1)
        attended = self.crossmodal(modalities)
        zero_state = self.gate.zero_state(batch.expr.shape[0], batch.expr.device)
        fused, _ = self.gate(attended, zero_state)
        params = self._build_parameters(fused, enzymes, batch.dose)
        return attended, fused, params

    def _build_parameters(
        self, fused: torch.Tensor, enzymes: torch.Tensor, dose: torch.Tensor
    ) -> OdeParameters:
        raw = self.parameter_head(fused)
        sp = torch.nn.functional.softplus
        return OdeParameters(
            k_methyl=sp(raw[:, 0]),
            k_demethyl=sp(raw[:, 1]),
            f_weights=raw[:, 2:4],
            f_bias=raw[:, 4],
            g_weights=raw[:, 5:7],
            g_bias=raw[:, 7],
            k_e=sp(raw[:, 8]),
            k_a=sp(raw[:, 9]),
            k_eff=sp(raw[:, 10]),
            ec50=sp(raw[:, 11]) + 1e-3,
            hill=sp(raw[:, 12]) + 1.0,
            k_off=sp(raw[:, 13]),
            r_s=sp(raw[:, 14]),
            r_r=sp(raw[:, 15]),
            carrying=sp(raw[:, 16]) + 1.0,
            delta_s_max=sp(raw[:, 17]),
            delta_r_max=sp(raw[:, 18]),
            alpha=torch.sigmoid(raw[:, 19]),
            mu=sp(raw[:, 20]),
            coupling_gain=0.1 * torch.tanh(raw[:, 21]),
            dnmt3a=enzymes[:, 0],
            dnmt3b=enzymes[:, 1],
            tet2=enzymes[:, 2],
            dose=dose,
        )

    def ode_parameters(self, batch: CohortBatch) -> OdeParameters:
        _, _, params = self._fuse_initial(batch)
        return params

    def _physics_active(self) -> bool:
        a = self.config.alembic
        return a.enable_epigenetic or a.enable_pharmacology or a.enable_population

    def forward(self, batch: CohortBatch) -> ModelOutput:
        attended, fused, params = self._fuse_initial(batch)
        raw_init = self.initial_head(fused)
        sp = torch.nn.functional.softplus
        y0 = torch.stack(
            [
                torch.sigmoid(raw_init[:, 0]),
                sp(raw_init[:, 1]),
                sp(raw_init[:, 2]),
                sp(raw_init[:, 3]) + 0.5,
                sp(raw_init[:, 4]) + 0.1,
            ],
            dim=-1,
        )

        a = self.config.alembic
        if self._physics_active():
            switches = OdeSwitches(
                a.enable_epigenetic, a.enable_pharmacology, a.enable_population, a.use_coupling
            )
            system = CoupledSystem(params, switches, y0)
            times = torch.linspace(0.0, a.t_final, a.integration_steps, device=y0.device)
            trajectory = integrate(system, y0, times, a.solver, a.rtol, a.atol, a.adjoint)
            terminal = trajectory[-1]
            physics = physics_residual(system, trajectory, times)
            fraction = terminal[:, STATE_R] / (terminal[:, STATE_S] + terminal[:, STATE_R] + 1e-6)
            state = torch.stack([terminal[:, STATE_M], terminal[:, STATE_E], fraction], dim=-1)
        else:
            trajectory = y0.unsqueeze(0)
            terminal = y0
            physics = y0.new_zeros(())
            state = self.gate.zero_state(batch.expr.shape[0], y0.device)

        z, attention_mass = self.gate(attended, state)
        regression = self.regression_head(z).squeeze(-1)
        logits = self.resistance_head(z).squeeze(-1)
        return ModelOutput(
            regression=regression,
            logits=logits,
            trajectory=trajectory,
            terminal_state=terminal,
            attention_mass=attention_mass,
            physics=physics,
            ode_params=params,
        )

    def export_forward(
        self, expr: torch.Tensor, methyl: torch.Tensor, drug: torch.Tensor, dose: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        blank = torch.zeros(expr.shape[0], device=expr.device)
        batch = CohortBatch(expr, methyl, drug, dose, blank, blank)
        attended, fused, _ = self._fuse_initial(batch)
        state = self.gate.zero_state(expr.shape[0], expr.device)
        z, _ = self.gate(attended, state)
        return self.regression_head(z).squeeze(-1), self.resistance_head(z).squeeze(-1)


class EncoderPathway(nn.Module):
    def __init__(self, model: EpiResistPINN) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, expr: torch.Tensor, methyl: torch.Tensor, drug: torch.Tensor, dose: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.export_forward(expr, methyl, drug, dose)


def build_model(config: FormularyConfig) -> EpiResistPINN:
    return EpiResistPINN(config)
