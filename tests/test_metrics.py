import numpy as np

from rezonate.eval.metrics import mean_ci, ndcg_at_k, precision_at_k


def test_ndcg_perfect_and_reversed():
    g = np.array([3, 2, 1, 0, 0])
    assert ndcg_at_k(np.array([5, 4, 3, 2, 1]), g, k=5) == 1.0
    assert ndcg_at_k(np.array([1, 2, 3, 4, 5]), g, k=5) < 0.6


def test_ndcg_no_relevant_is_nan():
    assert np.isnan(ndcg_at_k(np.arange(4), np.zeros(4, int), k=3))


def test_ndcg_ties_are_not_index_ordered():
    # All-equal scores: expected NDCG should be well below 1 regardless of input order.
    g = np.array([3] + [0] * 19)
    vals = [ndcg_at_k(np.zeros(20), g, k=10, rng=np.random.default_rng(s)) for s in range(200)]
    assert 0.1 < np.mean(vals) < 0.4  # analytic expectation ~0.21


def test_precision_and_ci():
    assert precision_at_k(np.array([3, 2, 1]), np.array([True, False, True]), k=2) == 0.5
    mu, lo, hi = mean_ci([0.5, 0.6, 0.7, 0.8])
    assert lo <= mu <= hi
