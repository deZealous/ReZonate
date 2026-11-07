"""Bayesian preference model that updates after every rating.

Model:   p(like | x) = sigmoid(w^T phi(x)),   w ~ N(0, diag(1/lambda))
Basis:   phi(x) = [1, PCA_d(x), RBF(x, landmarks), context(x)]
         RBF bumps centred on k-means landmarks of the *whole* catalogue let a linear-in-w
         model express several separate taste regions (a cheap stand-in for a GP
         classifier). The PCA part captures broad "direction of taste" trends.
Fit:     Laplace approximation. Newton/IRLS to the MAP, covariance = inverse Hessian.
         Soft targets (meh = 0.35, like = 0.8, love = 1.0) use the graded labels.
Online:  ``update`` adds a rating and re-runs a few warm-started Newton steps; with ~100
         basis functions this takes milliseconds, so the rating CLI refits after every key.
Prior:   lambda is picked by the Laplace-approximate evidence (empirical Bayes) on refit.
Uncertainty gives exploration for free: Thompson sampling or UCB on the logit.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

SOFT = {0: 0.0, 1: 0.35, 2: 0.8, 3: 1.0}


def _sqdist(X, L):
    """Pairwise squared distances without materialising an (N, L, D) tensor."""
    return np.maximum((X ** 2).sum(1)[:, None] + (L ** 2).sum(1)[None] - 2 * X @ L.T, 0.0)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class Basis:
    def __init__(self, pca_dim: int = 16, n_landmarks: int = 64, seed: int = 0):
        self.pca_dim, self.n_landmarks, self.seed = pca_dim, n_landmarks, seed

    def fit(self, X_all: np.ndarray, C_all: np.ndarray | None = None) -> Basis:
        """Unsupervised: fit on all tracks (rated + candidates)."""
        self.pca = PCA(min(self.pca_dim, X_all.shape[1], len(X_all) - 1),
                       random_state=self.seed).fit(X_all)
        self.pca_scale = np.sqrt(self.pca.explained_variance_[0]) + 1e-9
        k = min(self.n_landmarks, len(X_all))
        self.L = KMeans(k, n_init=3, random_state=self.seed).fit(X_all).cluster_centers_
        d2 = _sqdist(X_all, self.L)
        self.gamma = 1.0 / (np.median(np.sort(d2, axis=1)[:, :5]) + 1e-9)  # local bandwidth
        self.c_mu = C_all.mean(0) if C_all is not None else None
        self.c_sd = C_all.std(0) + 1e-6 if C_all is not None else None
        return self

    def __call__(self, X: np.ndarray, C: np.ndarray | None = None) -> np.ndarray:
        parts = [np.ones((len(X), 1)), self.pca.transform(X) / self.pca_scale]
        d2 = _sqdist(X, self.L)
        parts.append(np.exp(-self.gamma * d2))
        if C is not None and self.c_mu is not None:
            parts.append((C - self.c_mu) / self.c_sd)
        return np.concatenate(parts, axis=1)


class BayesTaste:
    name = "bayes"

    def __init__(self, basis: Basis, lambdas=(0.3, 1.0, 3.0, 10.0), newton_iters: int = 25):
        self.basis = basis
        self.lambdas = lambdas
        self.iters = newton_iters
        self.Phi = np.zeros((0, 0))
        self.lam = lambdas[len(lambdas) // 2]

    # ---- core Laplace fit -------------------------------------------------------------
    def _map(self, Phi, t, w, lam, m0=None):
        d = Phi.shape[1]
        prior = np.full(d, lam)
        prior[0] = 1e-2  # nearly flat prior on the bias
        m = np.zeros(d) if m0 is None else m0.copy()
        for _ in range(self.iters):
            p = _sigmoid(Phi @ m)
            g = Phi.T @ (w * (p - t)) + prior * m
            H = (Phi * (w * p * (1 - p))[:, None]).T @ Phi + np.diag(prior)
            step = np.linalg.solve(H, g)
            m -= step
            if np.abs(step).max() < 1e-6:
                break
        p = _sigmoid(Phi @ m)
        H = (Phi * (w * p * (1 - p))[:, None]).T @ Phi + np.diag(prior)
        return m, H, prior

    def _log_evidence(self, Phi, t, w, lam):
        m, H, prior = self._map(Phi, t, w, lam)
        p = np.clip(_sigmoid(Phi @ m), 1e-9, 1 - 1e-9)
        ll = np.sum(w * (t * np.log(p) + (1 - t) * np.log(1 - p)))
        lp = -0.5 * np.sum(prior * m ** 2) + 0.5 * np.sum(np.log(prior))
        return ll + lp - 0.5 * np.linalg.slogdet(H)[1]

    def fit(self, X, grades, weights=None, C=None, select_lambda: bool = True):
        self.Phi = self.basis(X, C)
        self.t = np.array([SOFT[int(g)] for g in grades])
        self.w = np.ones(len(grades)) if weights is None else np.asarray(weights, float)
        if select_lambda and len(grades) >= 10:
            self.lam = max(
                self.lambdas, key=lambda lam: self._log_evidence(self.Phi, self.t, self.w, lam))
        self.m, H, _ = self._map(self.Phi, self.t, self.w, self.lam)
        self.S = np.linalg.inv(H)
        return self

    def update(self, x, grade, weight=1.0, c=None):
        """Online update with one new rating (warm-started; lambda kept fixed)."""
        phi = self.basis(x[None], None if c is None else c[None])
        self.Phi = np.vstack([self.Phi, phi])
        self.t = np.append(self.t, SOFT[int(grade)])
        self.w = np.append(self.w, weight)
        self.m, H, _ = self._map(self.Phi, self.t, self.w, self.lam, m0=self.m)
        self.S = np.linalg.inv(H)
        return self

    # ---- prediction ----------------------------------------------------------------
    def logit_moments(self, X, C=None):
        Phi = self.basis(X, C)
        mu = Phi @ self.m
        var = np.einsum("ij,jk,ik->i", Phi, self.S, Phi)
        return mu, np.maximum(var, 1e-12)

    def score(self, X, C=None):
        """Posterior-predictive P(like) via the probit approximation."""
        mu, var = self.logit_moments(X, C)
        return _sigmoid(mu / np.sqrt(1 + np.pi * var / 8))

    def ucb(self, X, C=None, kappa: float = 1.0):
        mu, var = self.logit_moments(X, C)
        return mu + kappa * np.sqrt(var)

    def thompson(self, X, C=None, rng=None):
        rng = rng or np.random.default_rng()
        w = rng.multivariate_normal(self.m, self.S, method="cholesky")
        return self.basis(X, C) @ w
