from __future__ import annotations

import torch

from epiresist.apparatus import seed_all
from epiresist.formulary import build_model, load_formula
from epiresist.materia import CohortSpec, SyntheticCohort, collate


def _spec(config):
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


def test_cohort_sampling_is_deterministic():
    config = load_formula("_smoke")
    a = SyntheticCohort(_spec(config))[3]
    b = SyntheticCohort(_spec(config))[3]
    assert torch.equal(a["expr"], b["expr"])
    assert torch.equal(a["target"], b["target"])


def test_seeded_model_init_is_reproducible():
    config = load_formula("_smoke")
    cohort = SyntheticCohort(_spec(config))
    batch = collate([cohort[i] for i in range(8)])
    seed_all(123)
    first = build_model(config)(batch).regression
    seed_all(123)
    second = build_model(config)(batch).regression
    assert torch.allclose(first, second, atol=1e-6)
