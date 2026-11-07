"""Embed every track in the DB that has audio, with one or more models (resumable).

Usage:  uv run python scripts/embed_all.py --models clap mert
"""

from __future__ import annotations

import argparse
import json

from rezonate import db
from rezonate.config import load_config
from rezonate.embed import store
from rezonate.embed.models import get_embedder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["clap", "mert"])
    ap.add_argument("--batch-tracks", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config()
    con = db.connect(cfg.db_path)
    items = con.execute("SELECT track_id, audio_path FROM tracks WHERE audio_path IS NOT NULL "
                        "ORDER BY track_id").fetchall()
    stats = {}
    for name in args.models:
        model = get_embedder(name)
        stats[name] = store.embed_tracks(model, items, store.path_for(cfg.embeddings_dir, name),
                                         batch_tracks=args.batch_tracks)
        print(json.dumps({k: v for k, v in stats[name].items() if k != "failed"}),
              f"failed={len(stats[name]['failed'])}", flush=True)
        del model
    (cfg.data_dir / "embed_stats.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
