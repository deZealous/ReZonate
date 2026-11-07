"""Glue between storage, embeddings, labels and models (used by the CLI and notebook)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from rezonate import db
from rezonate.config import Config
from rezonate.embed.views import load_view
from rezonate.explore import Pick, compose_batch
from rezonate.labels import fuse_labels, implicit_grades
from rezonate.taste.bayes import Basis, BayesTaste


@dataclass
class Catalog:
    ids: list[str]
    X: np.ndarray
    tracks: pd.DataFrame  # indexed by track_id, aligned with ids
    row: dict[str, int]


def load_catalog(cfg: Config, con: sqlite3.Connection, view: str | None = None) -> Catalog:
    ids, X = load_view(cfg.embeddings_dir, view or cfg.embedding_model)
    tracks = db.read(con, "tracks").set_index("track_id").reindex(ids)
    return Catalog(ids, X, tracks, {t: i for i, t in enumerate(ids)})


def current_labels(con: sqlite3.Connection) -> pd.DataFrame:
    plays = db.read(con, "plays")
    fb = db.latest_feedback(con)
    likes = fb[fb.source.str.startswith("apple")]
    explicit = fb[~fb.source.str.startswith("apple") & (fb.source != "simulated")]
    imp = implicit_grades(plays) if len(plays) else None
    labels = fuse_labels(imp, likes, explicit)
    # Transfer labels from library entries (no audio) to linked local audio files.
    links = pd.read_sql_query("SELECT track_id, link_id FROM tracks WHERE link_id IS NOT NULL", con)
    if len(links) and len(labels):
        m = labels.merge(links, left_on="track_id", right_on="link_id", suffixes=("_lib", ""))
        if len(m):
            moved = m.drop(columns=["track_id_lib", "link_id"])
            labels = pd.concat([labels, moved]).drop_duplicates("track_id", keep="last")
    return labels.reset_index(drop=True)


def text_anchors(con: sqlite3.Connection, labels: pd.DataFrame, cat: Catalog,
                 clap_text_fn, min_grade: int = 2, max_tags: int = 6) -> tuple[list[str], np.ndarray]:
    """EXPERIMENTAL cold-start bridge for loved songs with no audio (DRM Apple Music).

    We describe each loved library song by its tags ("indie pop, dark pop, electropop") and
    embed that text with CLAP's text tower, which shares the audio embedding space.
    CLAP has a known *modality gap* (text and audio vectors occupy offset regions), so we
    remove each modality's mean before normalising. We haven't validated this against
    real loves; treat it as a prior that your first ratings should quickly override.
    Only valid with the "clap" view.
    """
    need = labels[(labels.grade >= min_grade) & ~labels.track_id.isin(cat.row)]
    if need.empty:
        return [], np.zeros((0, cat.X.shape[1]))
    tags = pd.read_sql_query("SELECT track_id, tag, weight FROM tags", con)
    tags = tags[tags.track_id.isin(need.track_id)].sort_values("weight", ascending=False)
    texts = tags.groupby("track_id").tag.apply(lambda s: ", ".join(s.head(max_tags)))
    if texts.empty:
        return [], np.zeros((0, cat.X.shape[1]))
    T = clap_text_fn([f"{t} music" for t in texts.values])
    T = T - T.mean(0, keepdims=True) if len(T) > 3 else T
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
    return list(texts.index), T


def seed_selection(cat: Catalog, n: int = 150, known: set[str] | None = None,
                   seed: int = 0) -> list[int]:
    """Diverse first batch for cold start: one track nearest each of n k-means centroids.

    Coverage-first because, with no ratings, every region is equally uncertain. Rating a
    spread of styles once teaches the model more than rating 150 near-duplicates.
    """
    known = known or set()
    cand = np.array([i for i, t in enumerate(cat.ids) if t not in known])
    k = min(n, len(cand))
    km = KMeans(k, n_init=3, random_state=seed).fit(cat.X[cand])
    d = km.transform(cat.X[cand])  # distances to each centroid
    picks = []
    for c in range(k):
        order = np.argsort(d[:, c])
        for j in order:
            if cand[j] not in picks:
                picks.append(int(cand[j]))
                break
    return picks


def fit_bayes(cat: Catalog, labels: pd.DataFrame, extra_X: np.ndarray | None = None,
              extra_grades: np.ndarray | None = None, extra_w: float = 0.5) -> BayesTaste:
    lab = labels[labels.track_id.isin(cat.row)]
    rows = lab.track_id.map(cat.row).values.astype(int)
    X, g, w = cat.X[rows], lab.grade.values, lab.weight.values
    if extra_X is not None and len(extra_X):
        X = np.vstack([X, extra_X])
        g = np.concatenate([g, extra_grades])
        w = np.concatenate([w, np.full(len(extra_X), extra_w)])
    basis = Basis().fit(cat.X)
    return BayesTaste(basis).fit(X, g, w)


def recommend(cat: Catalog, model: BayesTaste, labels: pd.DataFrame, n: int = 20,
              eps: float = 0.25, known_extra: set[str] | None = None, seed: int = 0,
              niche_bias: float = 0.0) -> list[Pick]:
    known = set(labels.track_id) | (known_extra or set())
    known_rows = np.array([cat.row[t] for t in known if t in cat.row], dtype=int)
    mask = np.ones(len(cat.ids), bool)
    mask[known_rows] = False
    mean = model.score(cat.X)
    ts = model.thompson(cat.X, rng=np.random.default_rng(seed))
    return compose_batch(cat.X, mean, ts, known_rows, mask, n=n, eps=eps,
                         popularity=cat.tracks.popularity.values.astype(float),
                         niche_bias=niche_bias, seed=seed)
