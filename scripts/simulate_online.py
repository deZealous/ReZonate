"""Step 6 evidence: does reserving part of each batch for exploration pay off?

Online loop on the simulated FMA listener. It starts from a *narrow* history: 30 rated
tracks drawn only from 2 of the persona's 7 taste modes, mimicking a mainstream listening
history. Each round fits the Bayesian model, composes a 20-track batch with exploration
share eps, "rates" it with the persona's grades, and repeats.

Reported per eps (mean over seeds):
  * hit rate: share of recommended tracks the listener likes (grade >= 2)
  * modes found: how many of the 7 taste modes have at least one recommended like
  * catalogue NDCG@50: how well the final model ranks ALL still-unrated tracks (only
    measurable because the listener is simulated)
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from rezonate.config import REPO_ROOT, load_config
from rezonate.embed.views import load_view
from rezonate.eval.metrics import ndcg_at_k
from rezonate.eval.simulate import PERSONA_MODES, grades_from_utility, utility
from rezonate.explore import compose_batch
from rezonate.ingest import fma
from rezonate.taste.bayes import Basis, BayesTaste


def mode_of(meta: pd.DataFrame) -> np.ndarray:
    out = np.full(len(meta), -1)
    for k, gs in enumerate(PERSONA_MODES.values()):
        hit = meta.genres_all.map(lambda s, gs=gs: bool(set(s) & set(gs))).values
        out[(out == -1) & hit] = k
    return out


def run_one(X, grades, modes, basis, eps, seed, rounds=10, n=20, start_modes=(0, 1)):
    rng = np.random.default_rng(seed)
    start_pool = np.where(np.isin(modes, start_modes) | (modes == -1) & (rng.random(len(X)) < 0.1))[0]
    known = list(rng.choice(start_pool, 30, replace=False))
    recs, ctxs, first_round = [], [], {}
    for r in range(rounds):
        kr = np.array(known)
        m = BayesTaste(basis).fit(X[kr], grades[kr])
        mask = np.ones(len(X), bool)
        mask[kr] = False
        picks = compose_batch(X, m.score(X), m.thompson(X, rng=rng), kr, mask, n=n, eps=eps,
                              seed=seed * 100 + r)
        rows = [p.row for p in picks]
        for p in picks:
            md = modes[p.row]
            if grades[p.row] >= 2 and md not in start_modes and md != -1:
                first_round.setdefault(md, r + 1)
        recs += rows
        ctxs += [p.context for p in picks]
        known += rows
    kr = np.array(known)
    m = BayesTaste(basis).fit(X[kr], grades[kr])
    unrated = np.setdiff1d(np.arange(len(X)), kr)
    recs, ctxs = np.array(recs), np.array(ctxs)
    liked = recs[grades[recs] >= 2]
    n_new = len(PERSONA_MODES) - len(start_modes)
    # rounds until a new mode is first liked; never found within the horizon counts as rounds+1
    disc = [first_round.get(m, rounds + 1) for m in range(len(PERSONA_MODES)) if m not in start_modes]
    ex = ctxs == "explore"
    return {"eps": eps, "seed": seed, "hit_rate": float(np.mean(grades[recs] >= 2)),
            "explore_hit_rate": float(np.mean(grades[recs[ex]] >= 2)) if ex.any() else np.nan,
            "modes_found": len(set(modes[liked]) - {-1}),
            "new_modes_by_round3": sum(v <= 3 for v in first_round.values()) / n_new,
            "rounds_to_new_mode": float(np.mean(disc)),
            "catalogue_ndcg@50": ndcg_at_k(m.score(X[unrated]), grades[unrated], 50)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config()
    ids, X = load_view(cfg.embeddings_dir, "clap")
    meta = fma.load_tracks(cfg.data_dir / "fma_meta").loc[[int(t.split(":")[1]) for t in ids]]
    modes = mode_of(meta)
    basis = Basis().fit(X)
    res = []
    for seed in range(args.seeds):
        grades = grades_from_utility(utility(meta, seed=seed)).values
        for eps in (0.0, 0.25, 0.5):
            res.append(run_one(X, grades, modes, basis, eps, seed))
            print(res[-1], flush=True)
    df = pd.DataFrame(res)
    metrics = ["hit_rate", "explore_hit_rate", "modes_found", "new_modes_by_round3",
               "rounds_to_new_mode", "catalogue_ndcg@50"]
    summ = df.groupby("eps")[metrics].agg(["mean", "std"])
    summ.columns = [f"{a} ({b})" for a, b in summ.columns]
    md = ["# Exploration: online simulation (SIMULATED listener, CLAP view)", "",
          "Start: 30 ratings drawn only from 2 of 7 taste modes. 10 rounds x 20 recommendations.",
          f"Mean (std) over {args.seeds} seeds.", "",
          summ.reset_index().to_markdown(index=False, floatfmt=".3f"), "",
          "* `hit_rate`: share of recommended tracks the listener likes. Exploration costs some "
          "of this in the short run.",
          f"* `modes_found`: taste modes (of {len(PERSONA_MODES)}) with at least one "
          "recommended like. Includes the 2 starting modes.",
          "* `explore_hit_rate`: like rate of the exploration slots alone.",
          "* `new_modes_by_round3`: share of the 5 non-starting modes liked within 3 batches.",
          "* `rounds_to_new_mode`: mean batch index at which a non-starting mode is first liked "
          "(11 = not found within 10 batches). Lower = faster discovery.",
          "* `catalogue_ndcg@50`: final model's ranking of every still-unrated track.", ""]
    (REPO_ROOT / "reports" / "exploration_simulated.md").write_text("\n".join(md))
    df.to_csv(REPO_ROOT / "reports" / "exploration_simulated.csv", index=False)
    print("\n".join(md))


if __name__ == "__main__":
    main()
