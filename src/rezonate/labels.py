"""Turn plays, loves and explicit ratings into one graded label per track.

Grades (used directly as NDCG gains): 0 dislike, 1 meh, 2 like, 3 love.

Priority: explicit CLI/notebook rating > Apple Loved/Disliked > implicit plays.

Implicit rules (deliberately conservative — implicit data is noisy, e.g. a skip on
shuffle at 2 a.m. is weak evidence):
  * a play is a SKIP if the end reason is a skip/manual-change AND fewer than 50 % (or
    30 s) of the track was heard, or if any play shorter than 30 s ended early;
  * a play is COMPLETE if it ended naturally or >= 80 % was heard;
  * repeat listening across *distinct days* counts more than a single binge.
Tracks with too little evidence get no label rather than a guessed one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rezonate.ingest.apple_music import NATURAL_END, SKIP_REASONS


def classify_plays(plays: pd.DataFrame) -> pd.DataFrame:
    p = plays.copy()
    played = pd.to_numeric(p.get("played_ms"), errors="coerce")
    media = pd.to_numeric(p.get("media_ms"), errors="coerce")
    frac = (played / media).where(media > 0)
    reason = p.get("end_reason", pd.Series(index=p.index, dtype=object)).astype(str).str.upper()
    early = (frac < 0.5) | (played < 30_000)
    skip_reason = reason.isin(SKIP_REASONS)
    p["is_skip"] = (skip_reason & (early | frac.isna())) | ((played < 30_000) & ~reason.isin(NATURAL_END))
    p["is_complete"] = ~p["is_skip"] & (reason.isin(NATURAL_END) | (frac >= 0.8))
    p["day"] = pd.to_datetime(p.get("ts"), errors="coerce", utc=True, format="mixed").dt.date
    return p


def implicit_grades(plays: pd.DataFrame, prior_a: float = 1.0, prior_b: float = 1.0,
                    min_events: int = 2) -> pd.DataFrame:
    """Per track: completion rate with a Beta(a, b) prior, distinct days, and a grade or NaN."""
    p = classify_plays(plays)
    g = p.groupby("track_id").agg(
        n_complete=("is_complete", "sum"), n_skip=("is_skip", "sum"),
        n_events=("is_skip", "size"), n_days=("day", "nunique"))
    g["completion"] = (g.n_complete + prior_a) / (g.n_complete + g.n_skip + prior_a + prior_b)
    grade = pd.Series(np.nan, index=g.index)
    enough = g.n_events >= min_events
    grade[enough & (g.completion <= 0.3) & (g.n_skip >= 2)] = 0
    grade[enough & (g.completion > 0.3) & (g.completion < 0.6)] = 1
    grade[enough & (g.completion >= 0.6) & (g.n_complete >= 2)] = 1
    grade[enough & (g.completion >= 0.7) & (g.n_complete >= 3) & (g.n_days >= 2)] = 2
    g["grade"] = grade
    # Confidence in [0, 1): saturates with evidence; used as a sample weight in learners.
    g["confidence"] = 1 - np.exp(-(g.n_complete + g.n_skip) / 4)
    return g.reset_index()


def fuse_labels(implicit: pd.DataFrame | None, likes: pd.DataFrame | None,
                explicit: pd.DataFrame | None) -> pd.DataFrame:
    """Combine sources; returns track_id, grade, weight, label_source."""
    parts = []
    if implicit is not None and len(implicit):
        d = implicit.dropna(subset=["grade"])
        parts.append(pd.DataFrame({"track_id": d.track_id, "grade": d.grade,
                                   "weight": 0.3 + 0.4 * d.confidence, "label_source": "implicit",
                                   "prio": 0}))
    if likes is not None and len(likes):
        parts.append(pd.DataFrame({"track_id": likes.track_id, "grade": likes.grade,
                                   "weight": 0.9, "label_source": "apple_pref", "prio": 1}))
    if explicit is not None and len(explicit):
        parts.append(pd.DataFrame({"track_id": explicit.track_id, "grade": explicit.grade,
                                   "weight": 1.0, "label_source": explicit.source, "prio": 2}))
    if not parts:
        return pd.DataFrame(columns=["track_id", "grade", "weight", "label_source"])
    d = pd.concat(parts, ignore_index=True).sort_values("prio")
    d = d.drop_duplicates("track_id", keep="last").drop(columns="prio")
    d["grade"] = d["grade"].astype(int)
    return d.reset_index(drop=True)
