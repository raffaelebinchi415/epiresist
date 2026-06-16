from __future__ import annotations

import pytest
import torch

from epiresist.formulary import FormularyConfig, build_model, load_formula
from epiresist.materia import CohortSpec, SyntheticCohort, collate


@pytest.mark.parametrize(
    "name",
    [
        "ablation_no_ode1",
        "ablation_no_ode2",
        "ablation_no_ode3",
        "ablation_no_coupling",
        "supplement_solver_rk4",
    ],
)
def test_alembic_variants_run(name):
    base = load_formula(name)
    smoke = load_formula("_smoke")
    config = FormularyConfig(
        name=base.name,
        enable_mechanism=False,
        materia=smoke.materia,
        preparation=smoke.preparation,
        alembic=base.alembic,
        balance=smoke.balance,
        regimen=smoke.regimen,
        assay=smoke.assay,
    )
    cohort = SyntheticCohort(
        CohortSpec(
            config.materia.name,
            config.materia.n_samples,
            config.materia.n_genes,
            config.materia.n_cpg,
            config.materia.n_concentrations,
            config.materia.n_replicates,
            config.materia.n_gene_sets,
            config.materia.seed,
        )
    )
    batch = collate([cohort[i] for i in range(6)])
    output = build_model(config)(batch)
    assert torch.isfinite(output.regression).all()


def test_modality_drop_zeroes_contribution():
    smoke = load_formula("_smoke")
    dropped = FormularyConfig(
        name="drop",
        enable_mechanism=False,
        materia=smoke.materia,
        preparation=smoke.preparation.__class__(
            **{
                **{f: getattr(smoke.preparation, f) for f in smoke.preparation.__struct_fields__},
                "active_modalities": (False, True, True),
            }
        ),
        alembic=smoke.alembic,
        balance=smoke.balance,
        regimen=smoke.regimen,
        assay=smoke.assay,
    )
    cohort = SyntheticCohort(
        CohortSpec(
            smoke.materia.name,
            smoke.materia.n_samples,
            smoke.materia.n_genes,
            smoke.materia.n_cpg,
            smoke.materia.n_concentrations,
            smoke.materia.n_replicates,
            smoke.materia.n_gene_sets,
            smoke.materia.seed,
        )
    )
    batch = collate([cohort[i] for i in range(6)])
    output = build_model(dropped)(batch)
    assert torch.isfinite(output.regression).all()
