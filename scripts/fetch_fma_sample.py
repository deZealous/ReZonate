"""Build the FMA sample used for the embedding comparison and the offline demo.

Two overlapping parts:
  * comparison set: N clips per top-level genre (genre labels = ground truth for probes)
  * taste pool:     extra clips from sub-genres near the listener's stated taste plus a
                    random background, so a taste model has positives *and* negatives.

Usage:  uv run python scripts/fetch_fma_sample.py --per-genre 40 --per-subgenre 50 --background 300
"""

from __future__ import annotations

import argparse

import numpy as np

from rezonate import db
from rezonate.config import load_config
from rezonate.ingest import fma

COMPARISON_GENRES = ["Rock", "Experimental", "Electronic", "Hip-Hop", "Folk", "Pop",
                     "Instrumental", "International", "Classical", "Jazz", "Soul-RnB"]
TASTE_SUBGENRES = ["Post-Punk", "Indie-Rock", "Experimental Pop", "Soul-RnB", "Reggae - Dub",
                   "Free-Jazz", "Synth Pop", "Shoegaze", "Alternative Hip-Hop", "Loud-Rock"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-genre", type=int, default=40)
    ap.add_argument("--per-subgenre", type=int, default=50)
    ap.add_argument("--background", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    meta = fma.load_tracks(fma.fetch_metadata(cfg.data_dir / "fma_meta"))
    rng = np.random.default_rng(args.seed)
    # Only tracks of a sane length and with a top-level genre (keeps ground truth clean).
    meta = meta[(meta.duration_s >= 60) & meta.genre_top.notna()]

    comp = fma.stratified_sample(meta, args.per_genre, COMPARISON_GENRES, seed=args.seed)
    ids = set(comp.index)
    for sg in TASTE_SUBGENRES:
        pool = meta[meta.genres_all.map(lambda gs, sg=sg: sg in gs)].index.difference(list(ids))
        ids |= set(rng.choice(pool, size=min(args.per_subgenre, len(pool)), replace=False))
    rest = meta.index.difference(list(ids))
    ids |= set(rng.choice(rest, size=args.background, replace=False))
    ids = sorted(int(i) for i in ids)
    print(f"selected {len(ids)} tracks; downloading clips (HTTP range reads from fma_large.zip)")

    paths = fma.download_clips(ids, cfg.audio_dir / "fma")
    print(f"downloaded {len(paths)} / {len(ids)}")

    con = db.connect(cfg.db_path)
    db.upsert_tracks(con, fma.to_tracks_frame(meta, paths))
    db.upsert_tags(con, fma.tag_rows(meta, list(paths)))
    comp_ids = [f"fma:{t}" for t in comp.index if t in paths]
    (cfg.data_dir / "comparison_ids.txt").write_text("\n".join(comp_ids))
    print(f"comparison set: {len(comp_ids)} clips -> {cfg.data_dir / 'comparison_ids.txt'}")


if __name__ == "__main__":
    main()
