"""Step 3: compare CLAP vs MERT (per layer) on the FMA comparison sample.

Writes reports/embedding_comparison.md and .csv.

Layer selection is done on a random half (A) of the sample and the final head-to-head is
reported on the other half (B), so MERT isn't flattered by picking its best of 14 views on
the same data it's scored on.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from rezonate.config import REPO_ROOT, load_config
from rezonate.embed import compare, store
from rezonate.ingest import fma

SELECT_BY = "subgenre_jaccard@10"  # closest proxy to "sounds like"


def main() -> None:
    cfg = load_config()
    comp_ids = (cfg.data_dir / "comparison_ids.txt").read_text().split()
    meta = fma.load_tracks(cfg.data_dir / "fma_meta")
    clap_ids, clap = store.load(store.path_for(cfg.embeddings_dir, "clap"))
    mert_ids, mert = store.load(store.path_for(cfg.embeddings_dir, "mert"))
    ci, mi = {t: i for i, t in enumerate(clap_ids)}, {t: i for i, t in enumerate(mert_ids)}
    ids = [t for t in comp_ids if t in ci and t in mi]
    fids = [int(t.split(":")[1]) for t in ids]
    y = meta.loc[fids, "genre_top"].values
    sets = [set(g) for g in meta.loc[fids, "genres_all"]]
    views = {"clap": clap[[ci[t] for t in ids]]}
    views |= compare.mert_layer_views(mert[[mi[t] for t in ids]])

    rows = [compare.chance_levels(y, sets)]
    for name, X in views.items():
        rows.append({"name": name, **compare.evaluate(X, y, sets)})
        print(rows[-1], flush=True)
    full = pd.DataFrame(rows)

    # Held-out head-to-head.
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(ids))
    A, B = perm[: len(ids) // 2], perm[len(ids) // 2:]
    sweep = []
    for name, X in views.items():
        if name.startswith("mert"):
            sweep.append((compare.knn_metrics(X[A], y[A], [sets[i] for i in A])[SELECT_BY], name))
    best_mert = max(sweep)[1]
    layer = best_mert.replace("mert_", "").replace("mean_all", "mean")
    concat = compare.prep(views["clap"]) / np.sqrt(2)
    concat = np.hstack([concat, compare.prep(views[best_mert]) / np.sqrt(2)])
    h2h = []
    for name, X in (("clap", views["clap"]), (best_mert, views[best_mert]),
                    (f"clap+{best_mert}", concat)):
        h2h.append({"name": name, **compare.evaluate(X[B], y[B], [sets[i] for i in B])})
    h2h = pd.DataFrame(h2h)

    stats_p = cfg.data_dir / "embed_stats.json"
    stats = json.loads(stats_p.read_text()) if stats_p.exists() else {}
    speed = {m: s.get("sec_per_track_model") for m, s in stats.items()}

    out = REPO_ROOT / "reports"
    out.mkdir(exist_ok=True)
    full.to_csv(out / "embedding_comparison.csv", index=False)
    cols = ["name", "probe_macro_f1", "knn_p@10", "knn_acc", "subgenre_jaccard@10",
            "anisotropy_raw", "dim"]
    md = [
        "# Embedding comparison: CLAP vs MERT on FMA",
        "",
        f"Sample: {len(ids)} FMA clips (30 s), {len(set(y))} top-level genres, stratified "
        f"(~{len(ids) // len(set(y))}/genre). Same three 10 s windows per clip for both models, "
        "and the per-track vector is the mean over windows. Metrics are computed on "
        "centered + L2-normalised vectors.",
        "",
        "## Full sample (all views)",
        "",
        full[cols].to_markdown(index=False, floatfmt=".3f"),
        "",
        f"## Held-out head-to-head (MERT layer chosen on half A by {SELECT_BY}, scored on half B, "
        f"n={len(B)})",
        "",
        h2h[cols].to_markdown(index=False, floatfmt=".3f"),
        "",
        f"Selected MERT view: **{best_mert}** (use view name `mert:{layer}`).",
        "",
        "## Cost (CPU, 4 threads, this dev container; model forward only)",
        "",
        *(f"* {m}: {s:.2f} s/track (3 x 10 s windows)" for m, s in speed.items() if s),
        "",
    ]
    (out / "embedding_comparison.md").write_text("\n".join(md))
    (cfg.data_dir / "selected_views.json").write_text(json.dumps(
        {"mert": f"mert:{layer}", "concat": f"clap+mert:{layer}"}))
    print("\n".join(md))


if __name__ == "__main__":
    main()
