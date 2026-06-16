from __future__ import annotations

import numpy as np

from epiresist.assay import aggregate_enrichment, enrichment_score, normalized_enrichment_score


def test_top_ranked_gene_set_has_positive_enrichment():
    scores = np.linspace(1.0, 0.0, 100)
    gene_set = np.arange(0, 10)
    assert enrichment_score(scores, gene_set) > 0.5


def test_bottom_ranked_gene_set_has_negative_enrichment():
    scores = np.linspace(1.0, 0.0, 100)
    gene_set = np.arange(90, 100)
    assert enrichment_score(scores, gene_set) < 0.0


def test_normalized_enrichment_scales_with_signal():
    scores = np.linspace(1.0, 0.0, 100)
    strong = normalized_enrichment_score(scores, np.arange(0, 10), n_perm=100, seed=0)
    rng = np.random.default_rng(0)
    diffuse = normalized_enrichment_score(
        scores, rng.choice(100, 10, replace=False), n_perm=100, seed=0
    )
    assert abs(strong) >= abs(diffuse)


def test_aggregate_is_mean_of_top_k():
    values = [3.0, 1.0, 2.0, 0.5]
    assert abs(aggregate_enrichment(values, top_k=2) - 2.5) < 1e-9
