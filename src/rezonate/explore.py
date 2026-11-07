"""Compose a recommendation batch: exploit slots + a reserved share of exploration slots.

Exploit (1 - eps of the batch): highest posterior-mean score, diversified with MMR
  (maximal marginal relevance) so one taste mode can't take every slot.

Explore (eps of the batch, default 25 %) uses three filters, in order:
  1. Unfamiliar. The candidate's max cosine similarity to anything you've rated or played
     must be in the bottom ``novelty_q`` quantile of the pool. That rules out
     "more of the same" picks.
  2. Under-covered region. The catalogue is clustered (k-means). Clusters are visited in
     order of fewest ratings per track, so whole regions you've never heard get sampled
     before already-known ones get refined.
  3. Not hopeless. Within a cluster we take the best Thompson-sampled score from the
     Bayesian model (a posterior draw). Tracks the model is uncertain about can win;
     tracks it is confident you'd dislike rarely do. This is what makes exploration
     informative and not just random.
  Optionally ``niche_bias`` breaks near-ties toward less popular tracks.

Every pick is tagged with context='explore' or 'exploit' and logged with your rating, so
the explore hit rate can be measured and ``eps`` tuned from data rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class Pick:
    row: int
    context: str   # exploit | explore
    reason: str


def mmr(scores: np.ndarray, X: np.ndarray, n: int, lam: float = 0.75,
        exclude: set[int] | None = None) -> list[int]:
    exclude = exclude or set()
    cand = [i for i in np.argsort(-scores)[: max(10 * n, 200)] if i not in exclude]
    chosen: list[int] = []
    fin = np.isfinite(scores)
    lo, hi = scores[fin].min(), scores[fin].max()
    s = np.where(fin, (scores - lo) / (hi - lo + 1e-9), -np.inf)
    while cand and len(chosen) < n:
        if chosen:
            red = (X[cand] @ X[chosen].T).max(1)
        else:
            red = np.zeros(len(cand))
        j = int(np.argmax(lam * s[cand] - (1 - lam) * red))
        chosen.append(cand.pop(j))
    return chosen


def compose_batch(X: np.ndarray, mean_score: np.ndarray, sample_score: np.ndarray,
                  known_rows: np.ndarray, candidate_mask: np.ndarray, n: int = 20,
                  eps: float = 0.25, novelty_q: float = 0.3, n_clusters: int = 24,
                  popularity: np.ndarray | None = None, niche_bias: float = 0.0,
                  seed: int = 0) -> list[Pick]:
    """X: (N, D) normalised embeddings for all tracks; scores: (N,); masks over N."""
    rng = np.random.default_rng(seed)
    n_explore = int(round(eps * n))
    cand = np.where(candidate_mask)[0]
    exploit_scores = np.full(len(X), -np.inf)
    exploit_scores[cand] = mean_score[cand]
    picks: list[Pick] = []

    # --- explore -----------------------------------------------------------------
    if n_explore and len(known_rows):
        sim_known = (X[cand] @ X[known_rows].T).max(1)
        thresh = np.quantile(sim_known, novelty_q)
        novel = cand[sim_known <= thresh]
        km = KMeans(min(n_clusters, len(X)), n_init=3, random_state=seed).fit(X)
        lab = km.labels_
        coverage = np.bincount(lab[known_rows], minlength=km.n_clusters) / (
            np.bincount(lab, minlength=km.n_clusters) + 1)
        order = np.argsort(coverage + 1e-6 * rng.random(km.n_clusters))
        ts = sample_score.copy()
        if popularity is not None and niche_bias:
            pr = np.argsort(np.argsort(np.nan_to_num(popularity))) / max(len(X) - 1, 1)
            ts = ts - niche_bias * pr * np.std(ts[cand])
        used: set[int] = set()
        for c in np.tile(order, 3):  # cycle through clusters until slots are filled
            if len(picks) >= n_explore:
                break
            pool = [i for i in novel if lab[i] == c and i not in used]
            if not pool:
                continue
            best = max(pool, key=lambda i: ts[i])
            used.add(best)
            picks.append(Pick(int(best), "explore",
                              f"unfamiliar region (cluster {c}, {coverage[c]:.0%} rated)"))
    # --- exploit -----------------------------------------------------------------
    taken = {p.row for p in picks}
    for r in mmr(exploit_scores, X, n - len(picks), exclude=taken):
        if np.isfinite(exploit_scores[r]):
            picks.append(Pick(int(r), "exploit", f"score {mean_score[r]:.2f}"))
    rng.shuffle(picks)  # don't reveal which slots are exploration (avoids rating bias)
    return picks
