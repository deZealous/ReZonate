"""Metadata enrichment clients. Endpoint facts checked against current docs (Nov 2026):

LRCLIB (lrclib.net/docs.txt): no API key; identify via User-Agent; honour Retry-After on
  429/503; identified clients get roughly 90 req/min per IP (approximate, not contractual).
  GET /api/get?track_name&artist_name[&album_name][&duration] -> best match or 404.
  Duration must be within ±2 s of LRCLIB's record, so we pass it when known.
MusicBrainz (musicbrainz.org/doc/MusicBrainz_API/Rate_Limiting): <= 1 request/second per
  application, meaningful User-Agent with contact info, 503 when throttled.
Last.fm (last.fm/api/show/track.getTopTags): method=track.getTopTags, artist+track or mbid,
  api_key required; optional autocorrect=1. Error 29 = rate limit. track.getInfo's
  ``listeners`` field is taken from client libraries (not verified against official docs).
"""

from __future__ import annotations

import re

from rezonate.enrich.http import PoliteClient


def make_client(user_agent: str, cache_dir) -> PoliteClient:
    return PoliteClient(user_agent, cache_dir=cache_dir, min_interval={
        "musicbrainz.org": 1.1,          # stay safely under 1 req/s
        "lrclib.net": 0.7,               # ~85/min, under the identified-client allowance
        "ws.audioscrobbler.com": 0.25,   # Last.fm asks for "reasonable" use; ~4/s
        "api.jamendo.com": 0.2,
    })


class LRCLib:
    API = "https://lrclib.net/api"

    def __init__(self, http: PoliteClient):
        self.http = http

    def get(self, artist: str, title: str, album: str | None = None,
            duration_s: float | None = None) -> dict | None:
        params = {"track_name": title, "artist_name": artist, "album_name": album,
                  "duration": round(duration_s) if duration_s and 1 <= duration_s <= 3600 else None}
        hit = self.http.get_json(f"{self.API}/get", params)
        if hit is None and duration_s is not None:  # exact-signature miss: fall back to search
            res = self.http.get_json(f"{self.API}/search",
                                     {"track_name": title, "artist_name": artist}) or []
            res = [r for r in res if r.get("duration") and abs(r["duration"] - duration_s) <= 3]
            hit = res[0] if res else None
        if not hit:
            return None
        return {"plain": hit.get("plainLyrics"), "instrumental": bool(hit.get("instrumental")),
                "lrclib_id": hit.get("id")}


class MusicBrainz:
    API = "https://musicbrainz.org/ws/2"

    def __init__(self, http: PoliteClient):
        self.http = http

    @staticmethod
    def _q(s: str) -> str:
        return re.sub(r'([+\-&|!(){}\[\]^"~*?:\\/])', r"\\\1", s)

    def recording(self, artist: str, title: str, min_score: int = 90) -> dict | None:
        """Best-scoring recording -> {mbid, year (earliest release), tags}."""
        q = f'recording:"{self._q(title)}" AND artist:"{self._q(artist)}"'
        data = self.http.get_json(f"{self.API}/recording", {"query": q, "fmt": "json", "limit": 25})
        def norm(s):
            return re.sub(r"\W+", "", (s or "").lower())
        recs = [r for r in (data or {}).get("recordings", [])
                if int(r.get("score", 0)) >= min_score and norm(r.get("title")) == norm(title)]
        if not recs:
            return None
        # The top hit is often a live take / remaster / compilation appearance, so the
        # song's era = earliest date across all matching recordings, not the first hit's.
        years = [int(d[:4]) for r in recs
                 for d in [r.get("first-release-date")] + [x.get("date") for x in r.get("releases", [])]
                 if d and d[:4].isdigit()]
        best = recs[0]
        tags: dict[str, float] = {}
        for r in recs:
            for t in r.get("tags", []):
                tags[t["name"]] = tags.get(t["name"], 0.0) + float(t.get("count", 1))
        return {"mbid": best["id"], "year": min(years) if years else None,
                "tags": sorted(tags.items(), key=lambda kv: -kv[1])}


class LastFM:
    API = "https://ws.audioscrobbler.com/2.0/"

    def __init__(self, api_key: str, http: PoliteClient):
        if not api_key:
            raise ValueError("Set LASTFM_API_KEY (free: last.fm/api/account/create)")
        self.key = api_key
        self.http = http

    def top_tags(self, artist: str, title: str, mbid: str | None = None,
                 min_count: int = 5, limit: int = 15) -> list[tuple[str, float]]:
        params = {"method": "track.getTopTags", "api_key": self.key, "format": "json",
                  "autocorrect": 1}
        params |= {"mbid": mbid} if mbid else {"artist": artist, "track": title}
        data = self.http.get_json(self.API, params) or {}
        if "error" in data:
            if data["error"] == 29:
                raise RuntimeError("Last.fm rate limit (error 29); slow down")
            return []
        tags = (data.get("toptags") or {}).get("tag", [])
        if isinstance(tags, dict):  # single-tag responses are not wrapped in a list
            tags = [tags]
        # Last.fm 'count' is relative weight 0-100 for the track's top tag.
        out = [(t["name"].strip().lower(), float(t.get("count", 0)) / 100) for t in tags]
        return [(n, c) for n, c in out if c * 100 >= min_count][:limit]

    def listeners(self, artist: str, title: str, mbid: str | None = None) -> float | None:
        """Global Last.fm listener count (method=track.getInfo) for the popularity baseline.
        Field names come from client libraries, not official docs (unverified); JSON
        returns them as strings, so we coerce them."""
        params = {"method": "track.getInfo", "api_key": self.key, "format": "json",
                  "autocorrect": 1}
        params |= {"mbid": mbid} if mbid else {"artist": artist, "track": title}
        data = self.http.get_json(self.API, params) or {}
        try:
            return float((data.get("track") or {})["listeners"])
        except (KeyError, TypeError, ValueError):
            return None
