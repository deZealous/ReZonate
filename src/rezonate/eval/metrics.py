"""Ranking metrics with graded relevance (grade 0-3 -> gain 2^g - 1)."""

from __future__ import annotations

import numpy as np


def dcg(gains: np.ndarray, k: int) -> float:
    g = gains[:k]
    return float(np.sum((2.0 ** g - 1) / np.log2(np.arange(2, len(g) + 2))))


def ndcg_at_k(scores: np.ndarray, grades: np.ndarray, k: int = 10, rng=None) -> float:
    """NDCG@k of ranking ``grades`` by ``scores``. Ties broken randomly (not by index,
    which would silently favour whatever order the data came in)."""
    rng = rng or np.random.default_rng(0)
    order = np.lexsort((rng.random(len(scores)), -np.asarray(scores)))
    ideal = dcg(np.sort(grades)[::-1], k)
    return dcg(np.asarray(grades)[order], k) / ideal if ideal > 0 else np.nan


def precision_at_k(scores, relevant: np.ndarray, k: int = 10) -> float:
    order = np.argsort(-np.asarray(scores), kind="stable")[:k]
    return float(np.mean(relevant[order]))


def mean_ci(x, level: float = 0.95, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile-bootstrap CI over folds/repeats."""
    x = np.asarray([v for v in x if np.isfinite(v)])
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(1)
    a = (1 - level) / 2
    return float(x.mean()), float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))
