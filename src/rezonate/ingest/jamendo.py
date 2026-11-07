"""Jamendo API v3 client (Creative Commons catalogue) for discovery candidates.

Verification status (Nov 2026). The docs site couldn't be fetched from the dev container,
so only the points marked [docs] were confirmed, via search snippets of
developer.jamendo.com/v3.0:
  * [docs] every call needs a ``client_id`` (free, from devportal.jamendo.com)
  * [docs] ``audiodownload_allowed`` is per track; /tracks/file returns 404 when it's false,
    so we keep only tracks where it's true.
  * [UNVERIFIED, from memory] parameter names ``fuzzytags``, ``order``, ``include=musicinfo stats``,
    ``audioformat=mp32``, ``limit`` (<=200), and the response fields ``audiodownload``,
    ``license_ccurl``, ``musicinfo.tags.{genres,instruments,vartags}``, ``stats.listened``.
    Run ``rezonate jamendo-probe`` once with your key and check the raw JSON first.
  * [UNVERIFIED, secondary sources] a ~35k requests/month quota for non-commercial apps.
Licences vary per track (``license_ccurl``); we store each track's licence with it.

The host api.jamendo.com is not on this repo's dev-container allowlist; this client is
covered by offline tests with recorded responses and is meant to run on your machine.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from rezonate.enrich.http import PoliteClient

API = "https://api.jamendo.com/v3.0"


class Jamendo:
    def __init__(self, client_id: str, http: PoliteClient):
        if not client_id:
            raise ValueError("Set JAMENDO_CLIENT_ID (free key from devportal.jamendo.com)")
        self.client_id = client_id
        self.http = http

    def search_tracks(self, fuzzytags: str | None = None, limit: int = 200, offset: int = 0,
                      order: str | None = None, **extra) -> pd.DataFrame:
        params = {"client_id": self.client_id, "format": "json", "limit": limit,
                  "offset": offset, "fuzzytags": fuzzytags, "order": order,
                  "include": "musicinfo stats", "audioformat": "mp32", **extra}
        data = self.http.get_json(f"{API}/tracks/", params) or {}
        return self.to_frame(data.get("results", []))

    @staticmethod
    def to_frame(results: list[dict]) -> pd.DataFrame:
        rows = []
        for r in results:
            if not r.get("audiodownload_allowed", False):
                continue
            mi = r.get("musicinfo") or {}
            tags = mi.get("tags") or {}
            stats = r.get("stats") or {}
            date = r.get("releasedate") or ""
            rows.append({
                "track_id": f"jamendo:{r['id']}", "source": "jamendo",
                "title": r.get("name"), "artist": r.get("artist_name"),
                "album": r.get("album_name"),
                "year": int(date[:4]) if date[:4].isdigit() else None,
                "duration_s": r.get("duration"), "license": r.get("license_ccurl"),
                "popularity": stats.get("listened"),
                "download_url": r.get("audiodownload"),
                "tags": [*tags.get("genres", []), *tags.get("instruments", []),
                         *tags.get("vartags", [])],
            })
        return pd.DataFrame(rows)

    def download(self, url: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            with requests.get(url, stream=True, timeout=60,
                              headers={"User-Agent": self.http.s.headers["User-Agent"]}) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    f.writelines(r.iter_content(1 << 16))
        return dest
