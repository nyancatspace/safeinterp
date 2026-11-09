"""Small statistics for contrasts between two groups of examples."""
from __future__ import annotations

import torch


def effect_size(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Cohen's d of each column between ``y`` and ``~y``."""
    a, b = X[y], X[~y]
    pooled = ((a.var(0, unbiased=False) + b.var(0, unbiased=False)) / 2).sqrt()
    return (a.mean(0) - b.mean(0)) / (pooled + 1e-6)


def _folds(y: torch.Tensor, k: int) -> list[torch.Tensor]:
    """Stratified fold ids (deterministic interleave within each class)."""
    fold = torch.empty(len(y), dtype=torch.long)
    for cls in (True, False):
        idx = (y == cls).nonzero().flatten()
        fold[idx] = torch.arange(len(idx)) % k
    return [fold == i for i in range(k)]


def cv_accuracy(X: torch.Tensor, y: torch.Tensor, single_feature: bool, k: int = 5) -> float:
    """Cross-validated accuracy of a 1-D classifier: projection on the training
    difference of means, or the single feature with the largest |Cohen's d|.
    Threshold at the midpoint of the two training class means."""
    k = max(2, min(k, int(y.sum()), int((~y).sum())))
    correct = 0
    for test in _folds(y, k):
        Xtr, ytr = X[~test], y[~test]
        if single_feature:
            d = effect_size(Xtr, ytr)
            f = int(d.abs().argmax())
            w = torch.zeros(X.shape[1])
            w[f] = d[f].sign()
        else:
            w = Xtr[ytr].mean(0) - Xtr[~ytr].mean(0)
        s = Xtr @ w
        thr = (s[ytr].mean() + s[~ytr].mean()) / 2
        correct += int(((X[test] @ w > thr) == y[test]).sum())
    return correct / len(y)
