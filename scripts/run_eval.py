"""Steps 4, 5, 7: offline evaluation on the SIMULATED FMA listener (see eval/simulate.py).

  * taste models (kNN, learned ranker, Bayesian) x embeddings (CLAP, MERT, concat)
  * popularity and random baselines
  * context-feature ablation (tags / era / lyrics / popularity) on the ranker and Bayes model
Metric: NDCG@10 with graded gains on held-out rated tracks; 5-fold x 2 repeats x 3 persona
seeds; bootstrap 95 % CIs over all folds.

Usage:  uv run python scripts/run_eval.py [--n-labels 300]
On your real data, use `rezonate evaluate` (same protocol, your labels).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from rezonate import db
from rezonate.config import REPO_ROOT, load_config
from rezonate.embed.views import load_view
from rezonate.eval import protocol
from rezonate.eval.scorers import make_scorers
from rezonate.eval.simulate import PERSONA_MODES, grades_from_utility, utility
from rezonate.features import context
from rezonate.ingest import fma


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-labels", type=int, default=300)
    ap.add_argument("--personas", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()

    cfg = load_config()
    con = db.connect(cfg.db_path)
    sel = json.loads((cfg.data_dir / "selected_views.json").read_text())
    view_names = ["clap", sel["mert"], sel["concat"]]
    ids, _ = load_view(cfg.embeddings_dir, sel["concat"])
    views = {v: load_view(cfg.embeddings_dir, v, ids)[1] for v in view_names}
    meta = fma.load_tracks(cfg.data_dir / "fma_meta").loc[[int(t.split(":")[1]) for t in ids]]
    C, names, slices = context.build(con, ids, groups=("tags", "era", "lyrics", "pop"))
    pop = meta["listens"].values.astype(float)
    cov = {g: float((np.abs(C[:, s]).sum(1) > 0).mean()) for g, s in slices.items()}
    lyr_cov = float(1 - C[:, names.index("lyrics_missing")].mean())
    print("context coverage:", cov, "lyrics:", lyr_cov)

    all_runs = []
    for p in range(args.personas):
        g_all = grades_from_utility(utility(meta, seed=p))
        rng = np.random.default_rng(100 + p)
        lab_rows = np.sort(rng.choice(len(ids), size=min(args.n_labels, len(ids)), replace=False))
        setup = protocol.Setup(ids=ids, views=views, C=C, C_slices=slices, popularity=pop,
                               lab_rows=lab_rows, grades=g_all.values[lab_rows],
                               weights=np.ones(len(lab_rows)))
        print(f"persona {p}: grade counts {np.bincount(setup.grades, minlength=4).tolist()}")
        df = protocol.run(setup, make_scorers(setup, view_names), n_repeats=args.repeats, seed=p)
        df["persona"] = p
        all_runs.append(df)
    runs = pd.concat(all_runs, ignore_index=True)
    out = REPO_ROOT / "reports"
    runs.to_csv(out / "eval_simulated_folds.csv", index=False)
    summ = protocol.summarise(runs.drop(columns=["persona"]))
    summ.to_csv(out / "eval_simulated_summary.csv", index=False)

    def table(mask):
        d = summ[mask(summ.model)]
        cols = ["model", "ndcg@10", "ndcg@10_ci", "ndcg@5", "p@10(like+)", "top10_pop_pct"]
        return d[cols].to_markdown(index=False, floatfmt=".3f")

    def main_mask(m):
        return ~m.str.contains(r"\+(tags|era|lyrics|pop)", regex=True)
    def abl_mask(m):
        return m.str.contains(r"\| clap", regex=True) & ~m.str.startswith("knn")
    n_folds = runs.groupby("model").size().iloc[0]
    md = [
        "# Offline evaluation on a SIMULATED listener (FMA)",
        "",
        "> These numbers validate the pipeline. They are **not** evidence about your taste: the",
        "> listener is simulated from FMA genre labels (see `src/rezonate/eval/simulate.py`).",
        "> Re-run `rezonate evaluate` on your own ratings before trusting any conclusion.",
        "",
        f"* catalogue: {len(ids)} FMA tracks; each persona rates {args.n_labels} random tracks",
        f"* persona taste modes: {', '.join(PERSONA_MODES)}",
        f"* protocol: 5-fold x {args.repeats} repeats x {args.personas} personas = {n_folds} "
        "folds per model; NDCG@10 on held-out rated tracks (gain 2^grade - 1); 95 % bootstrap CI",
        "* `top10_pop_pct`: mean popularity percentile of each model's top 10 "
        "(lower = more niche picks)",
        f"* context coverage on this sample: tags {cov['tags']:.0%}, era {cov['era']:.0%}, "
        f"lyrics {lyr_cov:.1%} (LRCLIB rarely knows FMA artists)",
        "",
        "## Taste models vs baselines (audio only)",
        "",
        table(main_mask),
        "",
        "## Context-feature ablation (CLAP audio + context)",
        "",
        table(abl_mask),
        "",
    ]
    (out / "eval_simulated.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
