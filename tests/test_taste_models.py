import numpy as np
from sklearn.metrics import roc_auc_score

from rezonate.eval.metrics import ndcg_at_k, precision_at_k
from rezonate.taste.bayes import Basis, BayesTaste
from rezonate.taste.knn import KNNTaste, topk_mean
from rezonate.taste.ranker import LearnedRanker


def separates(scores, g):
    """Liked (2/3) vs disliked separation. Grade 2 vs 3 is random in the fixture, so NDCG
    can't reach 1; we check top-20 purity and AUC instead."""
    return precision_at_k(scores, g >= 2, 20) == 1.0 and roc_auc_score(g >= 2, scores) > 0.95


def split(n, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    return p[: int(0.7 * n)], p[int(0.7 * n):]


def test_topk_mean_masks_excluded_self():
    S = np.array([[1.0, 0.5], [0.2, 1.0]])
    out = topk_mean(S, np.ones(2), k=2, exclude_self=True)
    assert np.allclose(out, [0.5, 0.2])


def test_knn_beats_centroid_on_multimodal_taste(two_mode_taste):
    X, c, g = two_mode_taste
    tr, te = split(len(X))
    knn = KNNTaste().fit(X[tr], g[tr])
    centroid = X[tr][g[tr] >= 2].mean(0)
    n_knn = ndcg_at_k(knn.score(X[te]), g[te], 20)
    n_cen = ndcg_at_k(X[te] @ centroid, g[te], 20)
    assert separates(knn.score(X[te]), g[te])
    assert n_knn >= n_cen


def test_knn_modes_finds_two_clusters(two_mode_taste):
    X, c, g = two_mode_taste
    modes = KNNTaste().fit(X, g).modes()
    assert len(modes) == 2


def test_bayes_learns_and_updates_online(two_mode_taste):
    X, c, g = two_mode_taste
    tr, te = split(len(X))
    basis = Basis(pca_dim=8, n_landmarks=16).fit(X)
    m = BayesTaste(basis).fit(X[tr], g[tr])
    assert separates(m.score(X[te]), g[te])
    # online update: a new 'love' raises that track's predicted probability
    x = X[te[0]]
    before = m.score(x[None])[0]
    m.update(x, 3)
    m.update(x, 3)
    assert m.score(x[None])[0] > before
    # uncertainty shrinks where we have data
    mu, var_seen = m.logit_moments(X[tr[:20]])
    far = -X[tr[:20]]  # antipodal points: far from all landmarks with data
    _, var_far = m.logit_moments(far / np.linalg.norm(far, axis=1, keepdims=True))
    assert var_seen.mean() < var_far.mean()
    assert m.thompson(X[:5], rng=np.random.default_rng(0)).shape == (5,)


def test_ranker_with_context(two_mode_taste):
    X, c, g = two_mode_taste
    tr, te = split(len(X))
    C = np.random.default_rng(1).normal(size=(len(X), 3))
    m = LearnedRanker(pca_dim=8).fit(X[tr], g[tr], C=C[tr], X_all=X)
    assert separates(m.score(X[te], C[te]), g[te])
