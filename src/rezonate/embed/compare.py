"""Compare embedding models on a labelled sample.

What we measure, and why it matters for a *taste* model:
  * linear probe (5-fold macro-F1 on top-level genre): is coarse style linearly decodable?
  * kNN P@10 (same top-level genre among 10 nearest neighbours): the nearest-neighbour
    taste baseline lives or dies on local neighbourhood quality.
  * sub-genre Jaccard@10 (mean Jaccard of FMA ``genres_all`` sets between a query and its
    neighbours): finer-grained similarity, closer to "sounds like" than top-level genre.
  * anisotropy (mean pairwise cosine): high values compress the cosine range and make
    similarity thresholds brittle; we center + L2-normalise to counter it.
Genre is only a proxy for taste. These numbers rank embeddings for *style structure*,
not for what any one listener likes; the taste evaluation (eval/) is the real test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def prep(X: np.ndarray) -> np.ndarray:
    """Center then L2-normalise (standard fix for anisotropic SSL embeddings)."""
    X = X - X.mean(axis=0, keepdims=True)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def anisotropy(X: np.ndarray, n: int = 2000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    Z = X[idx] / (np.linalg.norm(X[idx], axis=1, keepdims=True) + 1e-9)
    S = Z @ Z.T
    return float(S[np.triu_indices(len(Z), 1)].mean())


def knn_metrics(X: np.ndarray, y: np.ndarray, sets: list[set] | None, k: int = 10) -> dict:
    Z = prep(X)
    S = Z @ Z.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argsort(-S, axis=1)[:, :k]
    p_at_k = float((y[nn] == y[:, None]).mean())
    # LOO kNN classification accuracy (majority vote)
    votes = [pd.Series(y[row]).value_counts().idxmax() for row in nn]
    out = {"knn_p@10": p_at_k, "knn_acc": float(np.mean(np.array(votes) == y))}
    if sets is not None:
        def jac(a, b):
            return len(a & b) / len(a | b) if a | b else 0.0
        out["subgenre_jaccard@10"] = float(np.mean([np.mean([jac(sets[i], sets[j]) for j in nn[i]])
                                                    for i in range(len(X))]))
    return out


def probe(X: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    pred = cross_val_predict(clf, X, y, cv=cv)
    return {"probe_macro_f1": float(f1_score(y, pred, average="macro")),
            "probe_acc": float(np.mean(pred == y))}


def evaluate(X: np.ndarray, y: np.ndarray, sets: list[set] | None = None) -> dict:
    return {**probe(X, y), **knn_metrics(X, y, sets), "anisotropy_raw": anisotropy(X),
            "dim": int(X.shape[1])}


def chance_levels(y: np.ndarray, sets: list[set] | None = None) -> dict:
    """Reference values for a random embedding, so the table has a floor."""
    rng = np.random.default_rng(0)
    return {"name": "random (chance)", **knn_metrics(rng.normal(size=(len(y), 64)), y, sets),
            "probe_macro_f1": 1 / len(set(y)), "probe_acc": float(pd.Series(y).value_counts(
                normalize=True).max())}


def mert_layer_views(X: np.ndarray) -> dict[str, np.ndarray]:
    """Candidate MERT views from (N, 13, 768): each layer, and the mean of all layers."""
    views = {f"mert_L{i}": X[:, i] for i in range(X.shape[1])}
    views["mert_mean_all"] = X.mean(axis=1)
    return views
