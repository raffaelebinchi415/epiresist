from __future__ import annotations

import torch


def _rank(values: torch.Tensor) -> torch.Tensor:
    order = values.argsort()
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(1, values.numel() + 1, dtype=values.dtype, device=values.device)
    return ranks


def pearson(prediction: torch.Tensor, target: torch.Tensor) -> float:
    p = prediction - prediction.mean()
    t = target - target.mean()
    denom = p.norm() * t.norm()
    if float(denom) == 0.0:
        return 0.0
    return float((p * t).sum() / denom)


def spearman(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return pearson(_rank(prediction), _rank(target))


def rmse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((prediction - target) ** 2)))


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    positive = scores[labels > 0.5]
    negative = scores[labels <= 0.5]
    if positive.numel() == 0 or negative.numel() == 0:
        return 0.5
    ranks = _rank(scores)
    positive_ranks = ranks[labels > 0.5].sum()
    n_pos = positive.numel()
    n_neg = negative.numel()
    auc = (positive_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def f1_score(scores: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> float:
    predicted = (scores > threshold).to(labels.dtype)
    true_positive = float((predicted * labels).sum())
    predicted_positive = float(predicted.sum())
    actual_positive = float(labels.sum())
    if predicted_positive == 0.0 or actual_positive == 0.0:
        return 0.0
    precision = true_positive / predicted_positive
    recall = true_positive / actual_positive
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)
