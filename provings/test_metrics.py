from __future__ import annotations

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score as sk_f1
from sklearn.metrics import roc_auc_score

from epiresist.assay import auroc, f1_score, pearson, rmse, spearman


def test_pearson_matches_scipy():
    rng = np.random.default_rng(1)
    a = rng.normal(size=128)
    b = 0.6 * a + rng.normal(size=128) * 0.4
    ta, tb = torch.tensor(a), torch.tensor(b)
    assert abs(pearson(ta, tb) - pearsonr(a, b)[0]) < 1e-5


def test_spearman_matches_scipy():
    rng = np.random.default_rng(2)
    a = rng.normal(size=128)
    b = np.exp(a) + rng.normal(size=128) * 0.1
    ta, tb = torch.tensor(a), torch.tensor(b)
    assert abs(spearman(ta, tb) - spearmanr(a, b)[0]) < 1e-4


def test_rmse_value():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 2.0, 5.0])
    assert abs(rmse(a, b) - (4.0 / 3.0) ** 0.5) < 1e-6


def test_auroc_matches_sklearn():
    rng = np.random.default_rng(3)
    scores = rng.uniform(size=256)
    labels = (rng.uniform(size=256) > 0.4).astype(float)
    mine = auroc(torch.tensor(scores), torch.tensor(labels))
    assert abs(mine - roc_auc_score(labels, scores)) < 1e-6


def test_f1_matches_sklearn():
    rng = np.random.default_rng(4)
    scores = rng.uniform(size=256)
    labels = (rng.uniform(size=256) > 0.5).astype(float)
    mine = f1_score(torch.tensor(scores), torch.tensor(labels))
    reference = sk_f1(labels, (scores > 0.5).astype(float))
    assert abs(mine - reference) < 1e-6


def test_auroc_degenerate_single_class():
    scores = torch.tensor([0.1, 0.2, 0.3])
    labels = torch.tensor([1.0, 1.0, 1.0])
    assert auroc(scores, labels) == 0.5
