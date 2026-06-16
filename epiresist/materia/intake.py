from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CohortSpec:
    name: str
    n_samples: int
    n_genes: int
    n_cpg: int
    n_concentrations: int
    n_replicates: int
    n_gene_sets: int
    seed: int


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _draw_sample(spec: CohortSpec, index: int) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(spec.seed * 1_000_003 + index)
    rho = float(rng.uniform(-2.0, 2.0))

    driver = np.array([rho, rho, -rho], dtype=np.float64)
    expr = rng.normal(6.0, 1.0, size=spec.n_genes)
    expr[:3] = 6.0 + 1.5 * driver
    expr = np.maximum(expr, 0.0)

    methyl_logit = rng.normal(0.0, 0.6, size=spec.n_cpg) + 0.8 * rho
    methyl = _sigmoid(methyl_logit)

    log_ec50 = 0.6 * rho + rng.normal(0.0, 0.05)
    ec50 = float(np.exp(log_ec50))
    concentrations = np.logspace(-3, 1, spec.n_concentrations)
    hill = concentrations**2 / (ec50**2 + concentrations**2)
    viability = 1.0 - hill
    drug = viability[None, :] + rng.normal(
        0.0, 0.02, size=(spec.n_replicates, spec.n_concentrations)
    )
    drug = np.clip(drug, 0.0, 1.5)

    dose = float(rng.uniform(0.5, 1.5))
    target = log_ec50
    label = 1.0 if rho > 0.0 else 0.0

    return {
        "expr": torch.tensor(expr, dtype=torch.float32),
        "methyl": torch.tensor(methyl, dtype=torch.float32),
        "drug": torch.tensor(drug, dtype=torch.float32),
        "dose": torch.tensor(dose, dtype=torch.float32),
        "target": torch.tensor(target, dtype=torch.float32),
        "label": torch.tensor(label, dtype=torch.float32),
    }


class SyntheticCohort(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, spec: CohortSpec) -> None:
        self.spec = spec
        self._cache: dict[int, dict[str, torch.Tensor]] = {}

    def __len__(self) -> int:
        return self.spec.n_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index not in self._cache:
            self._cache[index] = _draw_sample(self.spec, index)
        return self._cache[index]


def _load_manifest(spec: CohortSpec, root: Path) -> SyntheticCohort:
    expr = np.load(root / "expression.npy")
    if expr.shape[1] != spec.n_genes:
        raise ValueError("manifest gene count does not match cohort spec")
    cohort = SyntheticCohort(spec)
    methyl = np.load(root / "methylation.npy")
    drug = np.load(root / "doseresponse.npy")
    target = np.load(root / "target.npy")
    label = np.load(root / "label.npy")
    dose = np.load(root / "dose.npy")
    for i in range(min(spec.n_samples, expr.shape[0])):
        cohort._cache[i] = {
            "expr": torch.tensor(expr[i], dtype=torch.float32),
            "methyl": torch.tensor(methyl[i], dtype=torch.float32),
            "drug": torch.tensor(drug[i], dtype=torch.float32),
            "dose": torch.tensor(dose[i], dtype=torch.float32),
            "target": torch.tensor(target[i], dtype=torch.float32),
            "label": torch.tensor(label[i], dtype=torch.float32),
        }
    return cohort


def load_cohort(spec: CohortSpec, manifest: str | None = None) -> SyntheticCohort:
    if manifest is not None:
        root = Path(manifest)
        if root.exists():
            return _load_manifest(spec, root)
    return SyntheticCohort(spec)
