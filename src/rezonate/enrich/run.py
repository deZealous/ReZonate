"""Batch enrichment: lyrics (LRCLIB), era + mbid + tags (MusicBrainz), tags (Last.fm).

Each step only touches tracks still missing that field, and all HTTP is cached on disk,
so re-running is cheap and an interrupted run resumes.
"""

from __future__ import annotations

import sqlite3

from rezonate import db
from rezonate.enrich.clients import LastFM, LRCLib, MusicBrainz


def _todo(con: sqlite3.Connection, sql: str, limit: int | None):
    rows = con.execute(sql + (f" LIMIT {int(limit)}" if limit else "")).fetchall()
    return [r for r in rows if r[1] and r[2]]


def lyrics(con, lrc: LRCLib, limit=None, log=print) -> dict:
    rows = _todo(con, """SELECT t.track_id, t.artist, t.title, t.album, t.duration_s FROM tracks t
                         LEFT JOIN lyrics l USING(track_id) WHERE l.track_id IS NULL""", limit)
    hits = 0
    for i, (tid, artist, title, album, dur) in enumerate(rows):
        r = lrc.get(artist, title, album, dur)
        db.upsert_lyrics(con, tid, r["plain"] if r else None, bool(r and r["instrumental"]),
                         "lrclib" if r else "lrclib_miss")
        hits += bool(r and (r["plain"] or r["instrumental"]))
        if (i + 1) % 100 == 0:
            log(f"lyrics {i + 1}/{len(rows)} hits={hits}")
    return {"queried": len(rows), "hits": hits}


def musicbrainz(con, mb: MusicBrainz, limit=None, only_missing_year=True, log=print) -> dict:
    where = "WHERE year IS NULL" if only_missing_year else "WHERE mbid IS NULL"
    rows = _todo(con, f"SELECT track_id, artist, title FROM tracks {where}", limit)
    hits = 0
    for i, (tid, artist, title) in enumerate(rows):
        r = mb.recording(artist, title)
        if r:
            hits += 1
            con.execute("UPDATE tracks SET mbid=COALESCE(mbid, ?), year=COALESCE(year, ?) "
                        "WHERE track_id=?", (r["mbid"], r["year"], tid))
            top = max((c for _, c in r["tags"]), default=1.0)
            db.upsert_tags(con, [(tid, n.lower(), c / top, "musicbrainz") for n, c in r["tags"]])
        if (i + 1) % 50 == 0:
            con.commit()
            log(f"musicbrainz {i + 1}/{len(rows)} hits={hits}")
    con.commit()
    return {"queried": len(rows), "hits": hits}


def lastfm_tags(con, lfm: LastFM, limit=None, log=print) -> dict:
    rows = _todo(con, """SELECT track_id, artist, title, mbid FROM tracks WHERE track_id NOT IN
                         (SELECT DISTINCT track_id FROM tags WHERE source='lastfm')""", limit)
    hits = 0
    for i, (tid, artist, title, mbid) in enumerate(rows):
        tags = lfm.top_tags(artist, title, mbid)
        if tags:
            hits += 1
            db.upsert_tags(con, [(tid, n, c, "lastfm") for n, c in tags])
        if (i + 1) % 100 == 0:
            log(f"lastfm {i + 1}/{len(rows)} hits={hits}")
    return {"queried": len(rows), "hits": hits}


def lastfm_popularity(con, lfm: LastFM, limit=None, log=print) -> dict:
    """Fill tracks.popularity with Last.fm listeners where no source popularity exists
    (your library and local files), so the popularity baseline is defined on real data."""
    rows = _todo(con, "SELECT track_id, artist, title, mbid FROM tracks WHERE popularity IS NULL",
                 limit)
    hits = 0
    for i, (tid, artist, title, mbid) in enumerate(rows):
        n = lfm.listeners(artist, title, mbid)
        if n is not None:
            hits += 1
            con.execute("UPDATE tracks SET popularity=? WHERE track_id=?", (n, tid))
        if (i + 1) % 100 == 0:
            con.commit()
            log(f"lastfm popularity {i + 1}/{len(rows)} hits={hits}")
    con.commit()
    return {"queried": len(rows), "hits": hits}
