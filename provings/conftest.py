from __future__ import annotations

import pytest

from epiresist.formulary import FormularyConfig, build_model, load_formula
from epiresist.materia import CohortSpec, SyntheticCohort, collate


def _cohort_spec(config: FormularyConfig) -> CohortSpec:
    m = config.materia
    return CohortSpec(
        m.name,
        m.n_samples,
        m.n_genes,
        m.n_cpg,
        m.n_concentrations,
        m.n_replicates,
        m.n_gene_sets,
        m.seed,
    )


@pytest.fixture()
def config() -> FormularyConfig:
    return load_formula("_smoke")


@pytest.fixture()
def cohort(config: FormularyConfig) -> SyntheticCohort:
    return SyntheticCohort(_cohort_spec(config))


@pytest.fixture()
def batch(cohort: SyntheticCohort):
    return collate([cohort[i] for i in range(8)])


@pytest.fixture()
def model(config: FormularyConfig):
    return build_model(config)
