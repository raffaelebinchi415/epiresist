from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import stats


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    estimates = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        estimates[i] = statistic(sample)
    point = float(statistic(values))
    lower = float(np.quantile(estimates, alpha / 2.0))
    upper = float(np.quantile(estimates, 1.0 - alpha / 2.0))
    return point, lower, upper


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = x.shape[0]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n, dtype=np.float64)
    out[order] = ranks
    return out


def _auc_components(
    positive: np.ndarray, negative: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    m = positive.shape[0]
    n = negative.shape[0]
    tx = _midrank(positive)
    ty = _midrank(negative)
    txy = _midrank(np.concatenate([positive, negative]))
    auc = (txy[:m].sum() - m * (m + 1) / 2.0) / (m * n)
    v10 = (txy[:m] - tx) / n
    v01 = 1.0 - (txy[m:] - ty) / m
    return float(auc), v10, v01


def delong_test(
    labels: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray
) -> tuple[float, float, float]:
    mask = labels > 0.5
    auc_a, a10, a01 = _auc_components(scores_a[mask], scores_a[~mask])
    auc_b, b10, b01 = _auc_components(scores_b[mask], scores_b[~mask])
    m = int(mask.sum())
    n = int((~mask).sum())
    s10 = np.cov(np.stack([a10, b10]))
    s01 = np.cov(np.stack([a01, b01]))
    cov = s10 / m + s01 / n
    var = cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]
    if var <= 0.0:
        return auc_a, auc_b, 1.0
    z = (auc_a - auc_b) / np.sqrt(var)
    p = float(2.0 * stats.norm.sf(abs(z)))
    return auc_a, auc_b, p


def welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    result = stats.ttest_ind(a, b, equal_var=False)
    return float(result.statistic), float(result.df), float(result.pvalue)
