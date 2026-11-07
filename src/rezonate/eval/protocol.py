"""Offline evaluation: repeated stratified K-fold over *rated* tracks.

For each fold we train on the other folds' labels and rank only the held-out rated tracks
(their true grades are known), then compute NDCG@k with graded gains. Unrated tracks are
never treated as negatives: "not rated" is not "disliked".

Caveats that will matter on your real data (flagged, not solved):
  * Selection bias. The songs you rate are not a random sample. Apple loves are songs you
    already found, and CLI ratings are songs the system chose to show you. Held-out NDCG
    then rewards models that mimic the selection process. Ratings logged with
    context='explore' are closer to random and are reported separately once there are enough.
  * Small samples. With ~150-300 labels a fold holds 30-60 tracks, so per-fold NDCG is
    noisy. We report bootstrap CIs over folds x repeats; overlapping CIs mean "no evidence
    of a difference", not "equal".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from rezonate.eval.metrics import mean_ci, ndcg_at_k, precision_at_k


@dataclass
class Fold:
    train: np.ndarray  # row indices into the labelled arrays
    test: np.ndarray


@dataclass
class Setup:
    """Everything a scorer needs. Arrays are aligned with ``ids`` (all tracks)."""
    ids: list[str]
    views: dict[str, np.ndarray]        # embedding view name -> (N, D)
    C: np.ndarray | None                # context features (N, F)
    C_slices: dict[str, slice]
    popularity: np.ndarray              # (N,) raw counts (for the baseline + niche metric)
    lab_rows: np.ndarray                # row index into ids for each labelled track
    grades: np.ndarray
    weights: np.ndarray


# scorer(setup, train_rows, test_rows, rng) -> scores for test_rows
Scorer = Callable[[Setup, np.ndarray, np.ndarray, np.random.Generator], np.ndarray]


def context_cols(setup: Setup, groups: tuple[str, ...]):
    if not groups or setup.C is None:
        return None
    return np.hstack([setup.C[:, setup.C_slices[g]] for g in groups])


def run(setup: Setup, scorers: dict[str, Scorer], k: int = 10, n_splits: int = 5,
        n_repeats: int = 3, seed: int = 0, log=print) -> pd.DataFrame:
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    pop_pct = pd.Series(setup.popularity).rank(pct=True).values
    rows = []
    for f, (tr, te) in enumerate(rskf.split(setup.lab_rows, setup.grades)):
        rng = np.random.default_rng(seed + f)
        g_te = setup.grades[te]
        for name, scorer in scorers.items():
            s = scorer(setup, tr, te, rng)
            top = np.argsort(-s, kind="stable")[:k]
            rows.append({
                "model": name, "fold": f,
                f"ndcg@{k}": ndcg_at_k(s, g_te, k, rng),
                "ndcg@5": ndcg_at_k(s, g_te, 5, rng),
                f"p@{k}(like+)": precision_at_k(s, g_te >= 2, k),
                f"top{k}_pop_pct": float(pop_pct[setup.lab_rows[te][top]].mean()),
            })
        if log and (f + 1) % n_splits == 0:
            log(f"  repeat {(f + 1) // n_splits}/{n_repeats} done")
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in df.columns if c not in ("model", "fold")]
    out = []
    for m, g in df.groupby("model", sort=False):
        r = {"model": m}
        for c in metrics:
            mu, lo, hi = mean_ci(g[c].values)
            r[c] = mu
            r[f"{c}_ci"] = f"[{lo:.3f}, {hi:.3f}]"
        out.append(r)
    return pd.DataFrame(out).sort_values(metrics[0], ascending=False).reset_index(drop=True)
