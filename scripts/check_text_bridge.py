"""Does the CLAP text bridge work? (Cold start for loved songs that have no audio.)

Setup on the simulated FMA listener: take 30 loved tracks, hide their audio, and describe
each one with text. Then rank the rest of the catalogue by similarity (kNN, k=5) to:
  * text anchors from FMA's free-text tags (often artist or place names: noisy)
  * text anchors from genre words (stands in for Last.fm/MusicBrainz tags; OPTIMISTIC,
    because the simulated listener is defined by these same genre labels)
  * the real audio of the 30 loves (oracle upper reference)
Each text variant is run in the raw CLAP space and with modality-gap centering.
Writes reports/text_bridge_check.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rezonate import db, service
from rezonate.config import REPO_ROOT, load_config
from rezonate.embed import store
from rezonate.embed.models import ClapEmbedder
from rezonate.eval.metrics import ndcg_at_k
from rezonate.eval.simulate import grades_from_utility, utility
from rezonate.ingest import fma
from rezonate.taste.knn import KNNTaste


def main(seeds: int = 8, n_loves: int = 30) -> None:
    cfg = load_config()
    con = db.connect(cfg.db_path)
    cat = service.load_catalog(cfg, con, "clap")
    raw_ids, raw = store.load(store.path_for(cfg.embeddings_dir, "clap"))
    ri = {t: i for i, t in enumerate(raw_ids)}
    A_raw = raw[[ri[t] for t in cat.ids]]
    A_raw = A_raw / np.linalg.norm(A_raw, axis=1, keepdims=True)
    meta = fma.load_tracks(cfg.data_dir / "fma_meta").loc[[int(t.split(":")[1]) for t in cat.ids]]
    tags = pd.read_sql_query("SELECT * FROM tags ORDER BY weight DESC", con)
    fma_txt = tags.groupby("track_id").tag.apply(lambda s: ", ".join(s.head(6)))
    genre_txt = meta.genres_all.map(lambda gs: ", ".join(gs[:4]))
    clap = ClapEmbedder()
    knn = lambda Z: KNNTaste(k=5).fit(Z, np.full(len(Z), 3))  # noqa: E731
    rows = []
    for seed in range(seeds):
        g = grades_from_utility(utility(meta, seed=seed)).values
        loves = np.random.default_rng(seed).choice(np.where(g == 3)[0], n_loves, replace=False)
        rest = np.setdiff1d(np.arange(len(cat.ids)), loves)
        r = {"seed": seed}
        for name, texts in (("FMA free tags", [fma_txt.get(cat.ids[i], "") for i in loves]),
                            ("genre words", [genre_txt.iloc[i] for i in loves])):
            T = clap.embed_text([f"{t} music" for t in texts])
            r[f"{name} (raw)"] = ndcg_at_k(knn(T).score(A_raw[rest]), g[rest], 50)
            Tc = T - T.mean(0)
            Tc /= np.linalg.norm(Tc, axis=1, keepdims=True)
            r[f"{name} (centered)"] = ndcg_at_k(knn(Tc).score(cat.X[rest]), g[rest], 50)
        r["real audio (oracle)"] = ndcg_at_k(knn(cat.X[loves]).score(cat.X[rest]), g[rest], 50)
        r["random"] = float(np.mean([ndcg_at_k(np.random.default_rng(i).random(len(rest)),
                                               g[rest], 50) for i in range(10)]))
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows).drop(columns="seed")
    summ = pd.DataFrame({"NDCG@50 mean": df.mean(), "std": df.std()})
    md = ["# CLAP text bridge check (SIMULATED listener)", "",
          f"{n_loves} loved tracks with audio hidden, {seeds} seeds; catalogue ranked by kNN "
          "similarity to the anchors.", "",
          summ.to_markdown(floatfmt=".3f"), "",
          "Reading: with noisy free tags (artist/place names) the bridge is at chance. With "
          "genre-style descriptions it is close to using the real audio. That condition is "
          "optimistic, since the simulated listener is defined by genre labels, but it supports "
          "using the bridge with Last.fm/MusicBrainz tags only, which is what "
          "`service.text_anchors` now does. Centering makes little difference.", ""]
    (REPO_ROOT / "reports" / "text_bridge_check.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
