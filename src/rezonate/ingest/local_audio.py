"""Scan a folder of DRM-free audio (CD rips, Bandcamp purchases, iTunes Store "iTunes Plus"
AAC files, which have been DRM-free since 2009) and link files to library entries.

Apple Music *subscription* downloads are FairPlay-protected: they cannot be decoded and
are skipped (mutagen still reads their tags, so we detect them by codec/atom and report).
"""

from __future__ import annotations

from pathlib import Path

import mutagen
import pandas as pd

from rezonate.db import norm_key, stable_id

AUDIO_EXT = {".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wav", ".aiff", ".aif"}
DRM_EXT = {".m4p"}  # protected AAC


def _first(tags, *keys):
    for k in keys:
        try:
            v = tags.get(k)
        except Exception:  # noqa: BLE001
            v = None
        if v:
            v = v[0] if isinstance(v, list) else v
            return str(v[0] if isinstance(v, tuple) else v)
    return None


def _is_protected(f) -> bool:
    # MP4 files with a 'drms'/'drmi' sample entry are FairPlay-protected.
    info = getattr(f, "info", None)
    codec = str(getattr(info, "codec", "") or "")
    return codec.startswith("drm") or "protected" in type(info).__name__.lower()


def scan(root: Path) -> tuple[pd.DataFrame, list[Path]]:
    rows, protected = [], []
    for p in sorted(Path(root).rglob("*")):
        ext = p.suffix.lower()
        if ext in DRM_EXT:
            protected.append(p)
            continue
        if ext not in AUDIO_EXT:
            continue
        try:
            f = mutagen.File(p, easy=True)
        except Exception:  # noqa: BLE001 - unreadable file: skip
            continue
        if f is None:
            continue
        if _is_protected(mutagen.File(p)):
            protected.append(p)
            continue
        tags = f.tags or {}
        artist = _first(tags, "artist", "albumartist")
        title = _first(tags, "title") or p.stem
        date = _first(tags, "date", "originaldate")
        year = int(date[:4]) if date and date[:4].isdigit() else None
        rows.append({
            "track_id": stable_id("local", str(p.resolve())),
            "source": "local", "artist": artist, "title": title,
            "album": _first(tags, "album"), "year": year,
            "duration_s": getattr(f.info, "length", None), "audio_path": str(p.resolve()),
            "mbid": _first(tags, "musicbrainz_trackid"),
        })
    return pd.DataFrame(rows), protected


def link_to_library(local: pd.DataFrame, library: pd.DataFrame) -> pd.DataFrame:
    """Exact (normalised) artist+title match. Returns local rows with ``link_id`` set.

    Conservative on purpose: fuzzy matching across "Remastered"/"feat." variants is a
    known gap; unmatched files are still usable as candidates.
    """
    if local.empty or library.empty:
        return local.assign(link_id=None)
    key = {norm_key(a, t): tid for tid, a, t in library[["track_id", "artist", "title"]].itertuples(
        index=False, name=None)}
    local = local.copy()
    local["link_id"] = [key.get(norm_key(a, t)) for a, t in zip(local.artist, local.title)]
    return local
