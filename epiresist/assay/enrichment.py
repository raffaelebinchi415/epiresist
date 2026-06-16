from __future__ import annotations

import numpy as np


def enrichment_score(ranked_scores: np.ndarray, gene_set: np.ndarray, weight: float = 1.0) -> float:
    order = np.argsort(-ranked_scores)
    ordered = ranked_scores[order]
    membership = np.isin(order, gene_set)
    hit_weights = np.abs(ordered) ** weight
    hit_norm = hit_weights[membership].sum()
    if hit_norm == 0.0:
        return 0.0
    n = ranked_scores.shape[0]
    n_miss = n - int(membership.sum())
    running = 0.0
    extreme = 0.0
    miss_step = 1.0 / max(n_miss, 1)
    for i in range(n):
        if membership[i]:
            running += hit_weights[i] / hit_norm
        else:
            running -= miss_step
        if abs(running) > abs(extreme):
            extreme = running
    return float(extreme)


def normalized_enrichment_score(
    ranked_scores: np.ndarray,
    gene_set: np.ndarray,
    n_perm: int = 200,
    seed: int = 0,
    weight: float = 1.0,
) -> float:
    observed = enrichment_score(ranked_scores, gene_set, weight)
    rng = np.random.default_rng(seed)
    n = ranked_scores.shape[0]
    size = gene_set.shape[0]
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        permuted = np.asarray(rng.choice(n, size=size, replace=False))
        null[i] = enrichment_score(ranked_scores, permuted, weight)
    scale = np.abs(null).mean()
    if scale == 0.0:
        return 0.0
    return float(observed / scale)


def gsea_prerank(
    ranked_scores: np.ndarray,
    gene_sets: list[np.ndarray],
    n_perm: int = 200,
    seed: int = 0,
    weight: float = 1.0,
) -> list[float]:
    return [
        normalized_enrichment_score(ranked_scores, gs, n_perm, seed + i, weight)
        for i, gs in enumerate(gene_sets)
    ]


def aggregate_enrichment(nes_values: list[float], top_k: int = 10) -> float:
    if not nes_values:
        return 0.0
    ordered = sorted((abs(v) for v in nes_values), reverse=True)
    selected = ordered[: min(top_k, len(ordered))]
    return float(sum(selected) / len(selected))
