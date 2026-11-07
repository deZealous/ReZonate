"""Free Music Archive (FMA) candidate pool.

Dataset: Defferrard et al., "FMA: A Dataset For Music Analysis" (ISMIR 2017),
https://github.com/mdeff/fma . Metadata is CC BY 4.0; each audio clip carries the
artist's own (mostly Creative Commons) licence, stored in ``tracks.license``.

Archives are hosted at os.unil.cloud.switch.ch/fma/. ``fma_large.zip`` holds 30 s
clips for all 106,574 tracks (~93 GB). Rather than download it, we read the zip's
central directory and the members we want through HTTP Range requests.
"""

from __future__ import annotations

import ast
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from rezonate.enrich.http import HTTPRangeFile

BASE_URL = "https://os.unil.cloud.switch.ch/fma"
META_URL = f"{BASE_URL}/fma_metadata.zip"


def fetch_metadata(dest: Path) -> Path:
    """Extract tracks.csv and genres.csv from fma_metadata.zip (range requests, ~tens of MB)."""
    dest.mkdir(parents=True, exist_ok=True)
    needed = ["fma_metadata/tracks.csv", "fma_metadata/genres.csv"]
    if all((dest / Path(n).name).exists() for n in needed):
        return dest
    with zipfile.ZipFile(HTTPRangeFile(META_URL, block_size=4 << 20)) as zf:
        for name in needed:
            (dest / Path(name).name).write_bytes(zf.read(name))
    return dest


def load_tracks(meta_dir: Path) -> pd.DataFrame:
    """Flatten FMA's two-level-header tracks.csv into the columns we use."""
    raw = pd.read_csv(meta_dir / "tracks.csv", index_col=0, header=[0, 1], low_memory=False)
    genres = pd.read_csv(meta_dir / "genres.csv", index_col=0)
    gname = genres["title"].to_dict()

    def parse_list(s):
        try:
            return ast.literal_eval(s) if isinstance(s, str) else []
        except (ValueError, SyntaxError):
            return []

    df = pd.DataFrame(index=raw.index)
    df["title"] = raw[("track", "title")]
    df["artist"] = raw[("artist", "name")]
    df["album"] = raw[("album", "title")]
    df["duration_s"] = raw[("track", "duration")]
    df["listens"] = raw[("track", "listens")]
    df["favorites"] = raw[("track", "favorites")]
    df["interest"] = raw[("track", "interest")]
    df["license"] = raw[("track", "license")]
    df["genre_top"] = raw[("track", "genre_top")]
    df["genres_all"] = raw[("track", "genres_all")].map(parse_list).map(
        lambda ids: [gname.get(i, str(i)) for i in ids])
    df["tags"] = raw[("track", "tags")].map(parse_list)
    df["album_tags"] = raw[("album", "tags")].map(parse_list)
    df["artist_tags"] = raw[("artist", "tags")].map(parse_list)
    # Release year: prefer the album release date, fall back to the upload date.
    rel = pd.to_datetime(raw[("album", "date_released")], errors="coerce")
    up = pd.to_datetime(raw[("track", "date_created")], errors="coerce")
    df["year"] = rel.dt.year.fillna(up.dt.year).astype("Int64")
    df["subset"] = raw[("set", "subset")]
    return df


def member_name(track_id: int, archive: str = "fma_large") -> str:
    tid = f"{track_id:06d}"
    return f"{archive}/{tid[:3]}/{tid}.mp3"


def download_clips(track_ids: list[int], out_dir: Path, archive: str = "fma_large") -> dict[int, Path]:
    """Fetch selected 30 s clips from a remote FMA zip. Returns {track_id: path} for successes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    got: dict[int, Path] = {}
    todo = []
    for t in track_ids:
        p = out_dir / f"{t:06d}.mp3"
        if p.exists() and p.stat().st_size > 0:
            got[t] = p
        else:
            todo.append(t)
    if not todo:
        return got
    rf = HTTPRangeFile(f"{BASE_URL}/{archive}.zip", block_size=1 << 20, max_blocks=8)
    with zipfile.ZipFile(rf) as zf:
        names = set(zf.namelist())
        for t in todo:
            name = member_name(t, archive)
            if name not in names:
                continue
            data = zf.read(name)  # FMA mp3s are 'stored', so this is a single contiguous range
            if len(data) < 10_000:  # a handful of FMA clips are known to be empty/truncated
                continue
            p = out_dir / f"{t:06d}.mp3"
            p.write_bytes(data)
            got[t] = p
    return got


def stratified_sample(df: pd.DataFrame, n_per_genre: int, genres: list[str] | None = None,
                      seed: int = 0, col: str = "genre_top") -> pd.DataFrame:
    """Sample tracks per top-level genre (only tracks that have one)."""
    rng = np.random.default_rng(seed)
    d = df[df[col].notna()]
    if genres:
        d = d[d[col].isin(genres)]
    parts = []
    for _, g in d.groupby(col):
        k = min(n_per_genre, len(g))
        parts.append(g.loc[rng.choice(g.index, size=k, replace=False)])
    return pd.concat(parts)


def to_tracks_frame(df: pd.DataFrame, paths: dict[int, Path]) -> pd.DataFrame:
    """Map FMA rows into the ``tracks`` table schema."""
    d = df.loc[[t for t in df.index if t in paths]]
    return pd.DataFrame({
        "track_id": [f"fma:{t}" for t in d.index],
        "source": "fma",
        "title": d["title"].values,
        "artist": d["artist"].values,
        "album": d["album"].values,
        "year": d["year"].values,
        "duration_s": d["duration_s"].values,
        "audio_path": [str(paths[t]) for t in d.index],
        "popularity": d["listens"].values,
        "license": d["license"].values,
        "genre_hint": d["genre_top"].values,
    })


def tag_rows(df: pd.DataFrame, track_ids: list[int]) -> list[tuple[str, str, float, str]]:
    """User-applied FMA tags (track > album > artist weighting). Genre labels are excluded
    on purpose: they are the ground truth for the embedding comparison."""
    rows = []
    for t in track_ids:
        if t not in df.index:
            continue
        r = df.loc[t]
        for col, w in (("tags", 1.0), ("album_tags", 0.6), ("artist_tags", 0.4)):
            for tag in r[col]:
                tag = str(tag).strip().lower()
                if tag:
                    rows.append((f"fma:{t}", tag, w, f"fma_{col}"))
    # Collapse duplicates keeping max weight.
    best: dict[tuple[str, str], float] = {}
    for tid, tag, w, _ in rows:
        best[(tid, tag)] = max(w, best.get((tid, tag), 0.0))
    return [(tid, tag, w, "fma") for (tid, tag), w in best.items()]


def read_zip_member(url: str, name: str) -> bytes:
    with zipfile.ZipFile(HTTPRangeFile(url)) as zf:
        return zf.read(name)


__all__ = [
    "download_clips",
    "fetch_metadata",
    "load_tracks",
    "read_zip_member",
    "stratified_sample",
    "tag_rows",
    "to_tracks_frame",
]
