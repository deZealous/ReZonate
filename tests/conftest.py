import numpy as np
import pytest


@pytest.fixture
def two_mode_taste():
    """Synthetic catalogue: 3 clusters on the sphere; the listener loves clusters 0 and 2
    and dislikes cluster 1. Their *mean* points between 0 and 2, toward neither."""
    rng = np.random.default_rng(0)
    d = 32
    centers = rng.normal(size=(3, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    X, c = [], []
    for k in range(3):
        pts = centers[k] + 0.25 * rng.normal(size=(120, d)) / np.sqrt(d) * 4
        X.append(pts)
        c += [k] * 120
    X = np.vstack(X)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    c = np.array(c)
    grades = np.where(c == 1, 0, 3)
    grades[(c != 1) & (rng.random(len(c)) < 0.3)] = 2
    return X, c, grades
