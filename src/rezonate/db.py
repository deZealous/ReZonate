"""SQLite storage for everything except dense vectors (those live in Parquet, see embed/store.py).

Track IDs are namespaced strings: ``fma:2``, ``jamendo:1234``, ``local:<sha1>``,
``lib:<sha1(artist|title)>``. A library entry (from Apple Music) and a local
audio file of the same song are linked through ``tracks.link_id`` once matched.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    track_id    TEXT PRIMARY KEY,
    source      TEXT NOT NULL,          -- library | local | fma | jamendo | bandcamp
    title       TEXT,
    artist      TEXT,
    album       TEXT,
    year        INTEGER,
    duration_s  REAL,
    audio_path  TEXT,                   -- decodable, DRM-free audio on disk (or NULL)
    popularity  REAL,                   -- source-specific raw count (FMA listens, Jamendo listens)
    license     TEXT,
    mbid        TEXT,
    link_id     TEXT,                   -- another track_id that is the same recording
    genre_hint  TEXT                    -- ground-truth genre where a dataset provides one; never a model feature
);
CREATE TABLE IF NOT EXISTS plays (
    track_id    TEXT NOT NULL,
    ts          TEXT,
    played_ms   REAL,
    media_ms    REAL,
    end_reason  TEXT,
    source      TEXT
);
CREATE INDEX IF NOT EXISTS plays_track ON plays(track_id);
CREATE TABLE IF NOT EXISTS feedback (
    track_id    TEXT NOT NULL,
    ts          REAL NOT NULL,
    grade       INTEGER NOT NULL,       -- 0 dislike, 1 meh, 2 like, 3 love
    source      TEXT NOT NULL,          -- apple_love | apple_dislike | cli | notebook | simulated
    context     TEXT                    -- seed | exploit | explore
);
CREATE INDEX IF NOT EXISTS feedback_track ON feedback(track_id);
CREATE TABLE IF NOT EXISTS tags (
    track_id    TEXT NOT NULL,
    tag         TEXT NOT NULL,
    weight      REAL NOT NULL,
    source      TEXT NOT NULL,
    PRIMARY KEY (track_id, tag, source)
);
CREATE TABLE IF NOT EXISTS lyrics (
    track_id     TEXT PRIMARY KEY,
    plain        TEXT,
    instrumental INTEGER,
    source       TEXT
);
"""

TRACK_COLS = [
    "track_id", "source", "title", "artist", "album", "year", "duration_s",
    "audio_path", "popularity", "license", "mbid", "link_id", "genre_hint",
]


def norm_key(*parts: str | None) -> str:
    """Case/whitespace-insensitive key used for matching tracks across sources."""
    return "|".join(" ".join(str(p or "").lower().split()) for p in parts)


def stable_id(prefix: str, *parts: str | None) -> str:
    return f"{prefix}:{hashlib.sha1(norm_key(*parts).encode()).hexdigest()[:16]}"


def connect(path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def upsert_tracks(con: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Insert or update tracks; NULL incoming values never overwrite existing ones."""
    df = df.copy()
    for c in TRACK_COLS:
        if c not in df:
            df[c] = None
    df = df[TRACK_COLS].astype(object).where(df[TRACK_COLS].notna(), None)
    sets = ", ".join(f"{c}=COALESCE(excluded.{c}, tracks.{c})" for c in TRACK_COLS[1:])
    sql = (
        f"INSERT INTO tracks ({', '.join(TRACK_COLS)}) VALUES ({', '.join('?' * len(TRACK_COLS))}) "
        f"ON CONFLICT(track_id) DO UPDATE SET {sets}"
    )
    con.executemany(sql, df.itertuples(index=False, name=None))
    con.commit()
    return len(df)


def add_plays(con: sqlite3.Connection, df: pd.DataFrame) -> int:
    cols = ["track_id", "ts", "played_ms", "media_ms", "end_reason", "source"]
    df = df.reindex(columns=cols).astype(object)
    df = df.where(df.notna(), None)
    con.executemany(f"INSERT INTO plays VALUES ({','.join('?' * len(cols))})",
                    df.itertuples(index=False, name=None))
    con.commit()
    return len(df)


def add_feedback(con: sqlite3.Connection, rows: Iterable[tuple[str, int, str, str | None]],
                 ts: float | None = None) -> None:
    ts = ts or time.time()
    con.executemany(
        "INSERT INTO feedback (track_id, ts, grade, source, context) VALUES (?, ?, ?, ?, ?)",
        [(tid, ts, int(g), src, ctx) for tid, g, src, ctx in rows],
    )
    con.commit()


def upsert_tags(con: sqlite3.Connection, rows: Iterable[tuple[str, str, float, str]]) -> None:
    con.executemany(
        "INSERT INTO tags VALUES (?, ?, ?, ?) ON CONFLICT DO UPDATE SET weight=excluded.weight",
        list(rows),
    )
    con.commit()


def upsert_lyrics(con: sqlite3.Connection, track_id: str, plain: str | None,
                  instrumental: bool, source: str) -> None:
    con.execute(
        "INSERT INTO lyrics VALUES (?, ?, ?, ?) ON CONFLICT(track_id) DO UPDATE SET "
        "plain=excluded.plain, instrumental=excluded.instrumental, source=excluded.source",
        (track_id, plain, int(instrumental), source),
    )
    con.commit()


def read(con: sqlite3.Connection, table: str, where: str = "", params: tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(f"SELECT * FROM {table} {where}", con, params=params)


def latest_feedback(con: sqlite3.Connection) -> pd.DataFrame:
    """One row per track: the most recent explicit grade (later ratings override earlier ones)."""
    return pd.read_sql_query(
        """SELECT f.* FROM feedback f
           JOIN (SELECT track_id, MAX(ts) AS ts FROM feedback GROUP BY track_id) m
           USING (track_id, ts)""",
        con,
    ).drop_duplicates("track_id", keep="last")
