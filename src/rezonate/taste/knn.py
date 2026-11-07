"""Nearest-neighbour taste baseline with explicit multi-modal structure.

score(x) = mean of the top-k similarities to liked tracks (weighted by grade)
           - beta * mean of the top-k similarities to disliked tracks

Because the score is *local* (top-k, not a centroid), a listener with several unrelated
taste clusters is handled naturally: a candidate only has to sit near one cluster.
Averaging all liked tracks into one taste vector would land between clusters, in music
you might not like at all.

``modes()`` clusters the liked tracks (k-means, k picked by silhouette) for explanations
("near your post-punk cluster"), per-mode diversity in a batch, and coverage-based
exploration.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def topk_mean(S: np.ndarray, w: np.ndarray, k: int, exclude_self: bool = False) -> np.ndarray:
    """Row-wise weighted mean of the k largest similarities. S: (n, m), w: (m,)."""
    if S.shape[1] == 0:
        return np.zeros(S.shape[0])
    S = S.copy()
    if exclude_self:
        np.fill_diagonal(S, -np.inf)
    k = min(k, S.shape[1] - int(exclude_self))
    if k <= 0:
        return np.zeros(S.shape[0])
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    top = np.take_along_axis(S, idx, axis=1)
    ww = np.where(np.isfinite(top), w[idx], 0.0)  # masked (-inf) entries = excluded self
    top = np.where(np.isfinite(top), top, 0.0)
    return (top * ww).sum(1) / (ww.sum(1) + 1e-9)


class KNNTaste:
    name = "knn"

    def __init__(self, k: int = 10, beta: float = 0.5):
        self.k, self.beta = k, beta

    def fit(self, X: np.ndarray, grades: np.ndarray, weights: np.ndarray | None = None):
        w = np.ones(len(grades)) if weights is None else np.asarray(weights, float)
        pos, neg = grades >= 2, grades == 0
        self.P, self.wp = X[pos], w[pos] * (grades[pos] - 1.0)  # love counts double vs. like
        self.N, self.wn = X[neg], w[neg]
        return self

    def features(self, X: np.ndarray) -> np.ndarray:
        """(s_pos, s_neg) per row, against the fitted (training) labels."""
        sp = topk_mean(X @ self.P.T, self.wp, self.k)
        sn = topk_mean(X @ self.N.T, self.wn, self.k) if len(self.N) else np.zeros(len(X))
        return np.stack([sp, sn], axis=1)

    def score(self, X: np.ndarray) -> np.ndarray:
        f = self.features(X)
        return f[:, 0] - self.beta * f[:, 1]

    def modes(self, max_k: int = 8, min_size: int = 3, seed: int = 0) -> np.ndarray:
        """Centroids of liked-track clusters (L2-normalised)."""
        P = self.P
        if len(P) < 2 * min_size:
            c = P.mean(0, keepdims=True)
            return c / np.linalg.norm(c, axis=1, keepdims=True)
        best, best_s = None, -1.0
        for k in range(2, min(max_k, len(P) // min_size) + 1):
            km = KMeans(k, n_init=5, random_state=seed).fit(P)
            if np.bincount(km.labels_).min() < min_size:
                continue
            s = silhouette_score(P, km.labels_, metric="cosine")
            if s > best_s:
                best, best_s = km, s
        if best is None:
            c = P.mean(0, keepdims=True)
        else:
            c = best.cluster_centers_
        return c / np.linalg.norm(c, axis=1, keepdims=True)
