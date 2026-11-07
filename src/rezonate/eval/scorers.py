"""Scorers shared by the offline evaluation script and ``rezonate evaluate``.

Each scorer has the signature (setup, train_idx, test_idx, rng) -> scores for test_idx,
where indices point into ``setup.lab_rows`` / ``setup.grades``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rezonate.eval import protocol
from rezonate.taste.bayes import Basis, BayesTaste
from rezonate.taste.knn import KNNTaste
from rezonate.taste.ranker import LearnedRanker

CTX_SETS = {"": (), "+tags": ("tags",), "+era": ("era",), "+lyrics": ("lyrics",),
            "+pop": ("pop",), "+tags+era+lyrics": ("tags", "era", "lyrics")}


def make_scorers(setup: protocol.Setup, views: list[str], ctx_view: str = "clap",
                 ctx_sets: dict | None = None) -> dict:
    ctx_sets_all = CTX_SETS if ctx_sets is None else ctx_sets
    def lab(rows):
        return setup.lab_rows[rows]

    def popularity(s, tr, te, rng):
        # setup.popularity should already be comparable across sources (see pop_percentile)
        return np.nan_to_num(s.popularity[lab(te)], nan=-1.0)

    def random(s, tr, te, rng):
        return rng.random(len(te))

    scorers = {"popularity": popularity, "random": random}
    basis_cache: dict = {}

    for v in views:
        X = setup.views[v]

        def knn(s, tr, te, rng, X=X):
            m = KNNTaste().fit(X[lab(tr)], s.grades[tr], s.weights[tr])
            return m.score(X[lab(te)])
        scorers[f"knn | {v}"] = knn

        ctx_sets = ctx_sets_all if v == ctx_view else {"": ()}
        for suffix, groups in ctx_sets.items():
            C = protocol.context_cols(setup, groups)

            def ranker(s, tr, te, rng, X=X, C=C):
                m = LearnedRanker().fit(X[lab(tr)], s.grades[tr], s.weights[tr],
                                        None if C is None else C[lab(tr)], X_all=X)
                return m.score(X[lab(te)], None if C is None else C[lab(te)])
            scorers[f"ranker | {v}{suffix}"] = ranker

            key = (v, suffix)
            if key not in basis_cache:
                basis_cache[key] = Basis().fit(X, C)

            def bayes(s, tr, te, rng, X=X, C=C, key=key):
                m = BayesTaste(basis_cache[key]).fit(X[lab(tr)], s.grades[tr], s.weights[tr],
                                                     None if C is None else C[lab(tr)])
                return m.score(X[lab(te)], None if C is None else C[lab(te)])
            scorers[f"bayes | {v}{suffix}"] = bayes
    return scorers




def pop_percentile(popularity, source) -> np.ndarray:
    """Popularity as a within-source percentile, so FMA listens, Jamendo listens and
    Last.fm listeners can be compared on one scale. Missing values stay NaN."""
    s = pd.Series(np.asarray(popularity, float))
    return s.groupby(pd.Series(np.asarray(source))).rank(pct=True).values
