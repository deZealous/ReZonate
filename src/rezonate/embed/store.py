"""Parquet embedding store: one file per model, one row per track, vector flattened.

Shape of the per-track array (e.g. (13, 768) for MERT layers) is kept in the Parquet
schema metadata, so ``load`` returns it unflattened. Writes are append-and-dedupe so an
overnight run can be resumed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from rezonate.embed.audio import load_audio, windows
from rezonate.embed.models import Embedder


def path_for(emb_dir: Path, name: str) -> Path:
    return Path(emb_dir) / f"{name}.parquet"


def save(path: Path, ids: list[str], arr: np.ndarray, extra_meta: dict | None = None) -> None:
    shape = list(arr.shape[1:])
    if path.exists():
        old_ids, old = load(path)
        keep = [i for i, t in enumerate(old_ids) if t not in set(ids)]
        ids = [old_ids[i] for i in keep] + list(ids)
        arr = np.concatenate([old[keep].reshape(len(keep), *shape), arr]) if keep else arr
    flat = arr.reshape(len(ids), -1).astype(np.float32)
    table = pa.table({
        "track_id": pa.array(ids, pa.string()),
        "emb": pa.FixedSizeListArray.from_arrays(pa.array(flat.ravel()), flat.shape[1]),
    })
    meta = {"shape": json.dumps(shape), **{k: json.dumps(v) for k, v in (extra_meta or {}).items()}}
    table = table.replace_schema_metadata(meta)
    pq.write_table(table, path)


def load(path: Path) -> tuple[list[str], np.ndarray]:
    t = pq.read_table(path)
    shape = json.loads(t.schema.metadata[b"shape"])
    ids = t.column("track_id").to_pylist()
    flat = t.column("emb").combine_chunks().flatten().to_numpy(zero_copy_only=False)
    return ids, flat.reshape(len(ids), *shape)


def embed_tracks(model: Embedder, items: list[tuple[str, str]], out: Path, batch_tracks: int = 4,
                 win_s: float = 10.0, n_windows: int = 3, save_every: int = 50,
                 log=print) -> dict:
    """Embed (track_id, audio_path) pairs; per-track vector = mean over windows.

    Skips tracks already in ``out``. Returns timing stats (seconds per track, model only).
    """
    done = set(load(out)[0]) if out.exists() else set()
    todo = [(t, p) for t, p in items if t not in done]
    buf_ids, buf_vecs, failed = [], [], []
    model_time, decode_time, n = 0.0, 0.0, 0
    for i in range(0, len(todo), batch_tracks):
        chunk = todo[i:i + batch_tracks]
        wins, owners = [], []
        t0 = time.perf_counter()
        for tid, p in chunk:
            try:
                ws = windows(load_audio(p, model.sr), model.sr, win_s, n_windows)
            except Exception as e:  # noqa: BLE001 - corrupt file: record and move on
                failed.append((tid, str(e)[:200]))
                continue
            wins += ws
            owners += [tid] * len(ws)
        decode_time += time.perf_counter() - t0
        if not wins:
            continue
        t0 = time.perf_counter()
        z = model.embed(wins)
        model_time += time.perf_counter() - t0
        owners_arr = np.array(owners)
        for tid in dict.fromkeys(owners):
            buf_ids.append(tid)
            buf_vecs.append(z[owners_arr == tid].mean(axis=0))
        n += len(set(owners))
        if len(buf_ids) >= save_every or i + batch_tracks >= len(todo):
            save(out, buf_ids, np.stack(buf_vecs), {"model": model.name, "win_s": win_s,
                                                     "n_windows": n_windows})
            log(f"[{model.name}] {len(done) + n}/{len(items)} embedded "
                f"({model_time / max(n, 1):.2f}s/track model, {decode_time / max(n, 1):.2f}s decode)")
            buf_ids, buf_vecs = [], []
    return {"model": model.name, "n": n, "sec_per_track_model": model_time / max(n, 1),
            "sec_per_track_decode": decode_time / max(n, 1), "failed": failed}
