"""Context (non-audio) features, built for an aligned list of track IDs.

Groups (each can be switched on/off for the ablation):
  tags    TF-IDF over tags (FMA user tags, Last.fm top tags, MusicBrainz tags, Jamendo tags)
          -> TruncatedSVD(16), plus a has-tags flag. Tags are crowd-sourced and often name
          the genre: they're useful, but they're also the closest thing to a popularity
          signal (niche tracks have fewer tags), so missingness is an explicit feature.
  era     release decade one-hot + z-scored year + missing flag.
  lyrics  sentence-transformers/all-MiniLM-L6-v2 on plain lyrics -> PCA(16), plus
          instrumental and missing flags. Lyrics are a secondary signal for you (you said
          sound matters most), so the default weight in the final ranker is learned, not fixed.
  pop     log(1 + popularity) z-scored within each source. Off by default for discovery,
          because it pulls rankings toward the mainstream; kept for the ablation.

All transforms are unsupervised and fit on the full catalogue (rated + candidates).
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer

GROUPS = ("tags", "era", "lyrics", "pop")
DECADES = [(0, 1979), (1980, 1989), (1990, 1999), (2000, 2009), (2010, 2014), (2015, 2019),
           (2020, 9999)]
LYRICS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LYRICS_REV = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def tag_features(con, ids, dim=16, min_df=3, seed=0):
    tags = pd.read_sql_query("SELECT track_id, tag, MAX(weight) AS w FROM tags GROUP BY track_id, tag",
                             con)
    tags = tags[tags.track_id.isin(set(ids))]
    vc = tags.tag.value_counts()
    tags = tags[tags.tag.isin(vc[vc >= min_df].index)]
    row = {t: i for i, t in enumerate(ids)}
    vocab = {t: j for j, t in enumerate(sorted(tags.tag.unique()))}
    M = np.zeros((len(ids), max(len(vocab), 1)))
    for tid, tag, w in tags.itertuples(index=False, name=None):
        M[row[tid], vocab[tag]] = max(M[row[tid], vocab[tag]], w)
    has = (M.sum(1) > 0).astype(float)[:, None]
    if len(vocab) <= dim:
        Z = M
    else:
        T = TfidfTransformer(sublinear_tf=True).fit_transform(M)
        Z = TruncatedSVD(dim, random_state=seed).fit_transform(T)
    names = [f"tag_svd{i}" for i in range(Z.shape[1])] + ["has_tags"]
    return np.hstack([Z, has]), names


def era_features(years: pd.Series):
    y = pd.to_numeric(years, errors="coerce")
    oh = np.stack([((y >= lo) & (y <= hi)).astype(float).values for lo, hi in DECADES], axis=1)
    z = ((y - y.mean()) / (y.std() + 1e-9)).fillna(0).values[:, None]
    miss = y.isna().astype(float).values[:, None]
    names = [f"era_{lo}_{hi}" for lo, hi in DECADES] + ["year_z", "year_missing"]
    return np.hstack([oh, z, miss]), names


def lyrics_features(con, ids, dim=16, seed=0, encoder=None):
    lyr = pd.read_sql_query("SELECT track_id, plain, instrumental FROM lyrics", con).set_index(
        "track_id").reindex(ids)
    has = lyr.plain.fillna("").str.len().gt(40).values
    instr = lyr.instrumental.fillna(0).astype(float).values[:, None]
    Z = np.zeros((len(ids), dim))
    if has.sum() >= max(dim + 1, 5):
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(LYRICS_MODEL, revision=LYRICS_REV, device="cpu")
        texts = lyr.plain[has].str.slice(0, 2000).tolist()
        E = encoder.encode(texts, batch_size=32, normalize_embeddings=True)
        Z[has] = PCA(min(dim, len(texts) - 1), random_state=seed).fit_transform(E)[:, :dim] \
            if E.shape[1] > dim else E
    miss = (~has & (instr[:, 0] == 0)).astype(float)[:, None]
    names = [f"lyr_pca{i}" for i in range(dim)] + ["instrumental", "lyrics_missing"]
    return np.hstack([Z, instr, miss]), names


def pop_features(tracks: pd.DataFrame):
    lp = np.log1p(pd.to_numeric(tracks.popularity, errors="coerce"))
    z = lp.groupby(tracks.source).transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
    return np.stack([z.fillna(0).values, lp.isna().astype(float).values], axis=1), \
        ["pop_z", "pop_missing"]


def build(con: sqlite3.Connection, ids: list[str], groups=("tags", "era", "lyrics"),
          lyrics_encoder=None) -> tuple[np.ndarray, list[str], dict[str, slice]]:
    """Return (C, names, slices) where slices[group] indexes that group's columns."""
    tracks = pd.read_sql_query("SELECT * FROM tracks", con).set_index("track_id").reindex(ids)
    blocks, names, slices, col = [], [], {}, 0
    for g in groups:
        if g == "tags":
            B, n = tag_features(con, ids)
        elif g == "era":
            B, n = era_features(tracks.year)
        elif g == "lyrics":
            B, n = lyrics_features(con, ids, encoder=lyrics_encoder)
        elif g == "pop":
            B, n = pop_features(tracks.reset_index())
        else:
            raise ValueError(g)
        blocks.append(B)
        names += n
        slices[g] = slice(col, col + B.shape[1])
        col += B.shape[1]
    C = np.hstack(blocks) if blocks else np.zeros((len(ids), 0))
    return C, names, slices
