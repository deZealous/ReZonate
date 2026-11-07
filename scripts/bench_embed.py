"""Clean (uncontended) embedding cost per track: decode + model forward, 3 x 10 s windows.

Usage: uv run python scripts/bench_embed.py --n 16
Writes data/embed_bench.json (read by compare_embeddings.py).
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from rezonate import db
from rezonate.config import load_config
from rezonate.embed.audio import load_audio, windows
from rezonate.embed.models import device, get_embedder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()
    cfg = load_config()
    con = db.connect(cfg.db_path)
    paths = [r[0] for r in con.execute(
        "SELECT audio_path FROM tracks WHERE audio_path IS NOT NULL LIMIT ?", (args.n,))]
    out = {"device": device(), "threads": torch.get_num_threads(), "n": len(paths)}
    for name in ("clap", "mert"):
        m = get_embedder(name)
        wins = [windows(load_audio(paths[0], m.sr), m.sr)]  # warm-up
        m.embed(wins[0])
        t0 = time.perf_counter()
        batches = [windows(load_audio(p, m.sr), m.sr) for p in paths]
        t1 = time.perf_counter()
        for i in range(0, len(batches), 4):
            m.embed([w for b in batches[i:i + 4] for w in b])
        t2 = time.perf_counter()
        out[name] = {"decode_s_per_track": (t1 - t0) / len(paths),
                     "model_s_per_track": (t2 - t1) / len(paths)}
        print(name, out[name], flush=True)
        del m
    (cfg.data_dir / "embed_bench.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
