from __future__ import annotations

from typing import Protocol

import numpy as np
import torch

from epiresist.alembics import OdeParameters
from epiresist.materia import CohortBatch


class ParameterModel(Protocol):
    def ode_parameters(self, batch: CohortBatch) -> OdeParameters: ...


def parameter_attribution(model: ParameterModel, batch: CohortBatch, name: str) -> np.ndarray:
    expr = batch.expr.detach().clone().requires_grad_(True)
    probed = CohortBatch(
        expr=expr,
        methyl=batch.methyl,
        drug=batch.drug,
        dose=batch.dose,
        target=batch.target,
        label=batch.label,
    )
    params = model.ode_parameters(probed)
    value = getattr(params, name)
    grad = torch.autograd.grad(value.sum(), expr, create_graph=False)[0]
    return grad.abs().mean(dim=0).detach().cpu().numpy()


def resistance_signature(attribution: np.ndarray, top_k: int = 20) -> np.ndarray:
    order = np.argsort(-attribution)
    return order[: min(top_k, attribution.shape[0])]
