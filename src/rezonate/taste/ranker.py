"""Learned ranker: a small supervised model over audio, kNN-similarity and context features.

Why pointwise and not LambdaMART? Learning-to-rank losses compare items *within a
query*. A personal recommender has a single query (you), and only a few hundred labels,
so a pairwise/listwise objective adds variance without new information. A regularised
pointwise classifier with graded sample weights is the robust choice at this scale.
Swap in ``kind="gbt"`` once you have well over 1,000 labels.

Leakage guard: kNN features for *training* rows are computed leaving that row out (a
track must not be its own nearest liked neighbour), and held-out tracks are scored
against training labels only.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rezonate.taste.knn import KNNTaste, topk_mean


class LearnedRanker:
    name = "ranker"

    def __init__(self, pca_dim: int = 32, kind: str = "logreg", k: int = 10, seed: int = 0):
        self.pca_dim, self.kind, self.k, self.seed = pca_dim, kind, k, seed

    def _knn_feats(self, X, train: bool):
        if train:
            # leave-one-out for training rows that are themselves positives/negatives
            Spos = X @ self.Xtr[self.pos].T
            Spos[self.pos_rows, np.arange(len(self.pos_rows))] = -np.inf
            sp = topk_mean(Spos, self.wp, self.k)
            Sneg = X @ self.Xtr[self.neg].T
            Sneg[self.neg_rows, np.arange(len(self.neg_rows))] = -np.inf
            sn = topk_mean(Sneg, self.wn, self.k) if self.neg.any() else np.zeros(len(X))
        else:
            return self.knn.features(X)
        return np.stack([sp, sn], axis=1)

    def fit(self, X, grades, weights=None, C=None, X_all=None):
        w = np.ones(len(grades)) if weights is None else np.asarray(weights, float)
        self.Xtr = X
        self.pos, self.neg = grades >= 2, grades == 0
        self.pos_rows, self.neg_rows = np.where(self.pos)[0], np.where(self.neg)[0]
        self.wp, self.wn = w[self.pos] * (grades[self.pos] - 1.0), w[self.neg]
        self.knn = KNNTaste(self.k).fit(X, grades, w)
        self.pca = PCA(min(self.pca_dim, X.shape[1]), random_state=self.seed).fit(
            X if X_all is None else X_all)
        F = self._features(X, C, train=True)
        y = (grades >= 2).astype(int)
        sw = w * np.where(grades == 3, 1.5, 1.0) * np.where(grades == 1, 0.5, 1.0)
        if self.kind == "gbt":
            self.model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                        max_iter=200, l2_regularization=1.0,
                                                        random_state=self.seed)
        else:
            self.model = make_pipeline(StandardScaler(), LogisticRegressionCV(
                Cs=[0.01, 0.03, 0.1, 0.3, 1.0], cv=min(5, max(2, int(min(y.sum(), (1 - y).sum())))),
                max_iter=3000, scoring="roc_auc", l1_ratios=(0.0,), use_legacy_attributes=False))
        self.model.fit(F, y, **({"logisticregressioncv__sample_weight": sw}
                                if self.kind != "gbt" else {"sample_weight": sw}))
        return self

    def _features(self, X, C, train: bool):
        parts = [self.pca.transform(X), self._knn_feats(X, train)]
        if C is not None:
            parts.append(C)
        return np.concatenate(parts, axis=1)

    def score(self, X, C=None):
        return self.model.predict_proba(self._features(X, C, train=False))[:, 1]
