"""Parse an Apple Music privacy export (privacy.apple.com -> "Apple Media Services information").

!! UNVERIFIED SCHEMA !!
Apple does not publish the column schema for these files and it has changed across
export versions. The column names below come from community write-ups and past
exports, NOT from official documentation. That's why parsing is alias-based, and
why ``rezonate inspect-export`` exists: run it on your export first; it prints every
CSV/JSON it finds, its columns, and which semantic fields were (or were not) resolved.
If something doesn't resolve, add your column name to ALIASES (or pass overrides).

Files we look for (names vary by export version):
  * "Apple Music Play Activity.csv"          per-event rows: duration played, end reason
  * "Apple Music - Play History Daily Tracks.csv"  per-day aggregates: play/skip counts
  * "Apple Music Likes and Dislikes.csv"     Loved / Disliked items
  * "Apple Music - Favorites.csv"            newer exports' favourites
  * "Apple Music Library Tracks.json"        library contents (may include loved flags)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from rezonate.db import stable_id

# semantic field -> candidate column names (case/space-insensitive match)
ALIASES: dict[str, list[str]] = {
    "title": ["Song Name", "Title", "Name", "Track Name", "Content Name"],
    "artist": ["Artist Name", "Artist", "Container Artist Name", "Album Artist"],
    "album": ["Album Name", "Album", "Container Album Name"],
    "description": ["Track Description", "Item Description", "Content Description"],
    "ts": ["Event Start Timestamp", "Event End Timestamp", "Event Received Timestamp",
           "Date Played", "Play Date UTC", "Last Played Date", "Created", "Last Modified"],
    "played_ms": ["Play Duration Milliseconds", "Play Duration Ms"],
    "media_ms": ["Media Duration In Milliseconds", "Media Duration Milliseconds",
                 "Track Duration", "Duration In Milliseconds"],
    "end_position_ms": ["End Position In Milliseconds"],
    "end_reason": ["End Reason Type", "End Reason"],
    "event_type": ["Event Type"],
    "play_count": ["Play Count", "Track Play Count"],
    "skip_count": ["Skip Count", "Track Skip Count"],
    "ignore": ["Ignore For Recommendations"],
    "preference": ["Preference", "Rating", "Favorite Type", "Track Like Rating"],
    "media_type": ["Media Type", "Media type", "Content Specific Type", "Item Type"],
}

# End reasons we treat as "the listener chose to leave early". Values seen in the wild;
# verify against ``inspect-export`` output.
SKIP_REASONS = {
    "TRACK_SKIPPED_FORWARDS",
    "TRACK_SKIPPED_BACKWARDS",
    "MANUALLY_SELECTED_PLAYBACK_OF_A_DIFF_ITEM",
}
NATURAL_END = {"NATURAL_END_OF_TRACK"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def resolve_columns(cols: list[str], overrides: dict[str, str] | None = None) -> dict[str, str]:
    lookup = {_norm(c): c for c in cols}
    out: dict[str, str] = {}
    for field, names in ALIASES.items():
        for n in names:
            if _norm(n) in lookup:
                out[field] = lookup[_norm(n)]
                break
    out.update(overrides or {})
    return out


def split_description(desc: str) -> tuple[str | None, str | None]:
    """'Artist - Title' -> (artist, title). Ambiguous when the artist contains ' - '
    (we split on the FIRST separator); flagged in inspect output."""
    if not isinstance(desc, str) or " - " not in desc:
        return None, desc if isinstance(desc, str) else None
    a, t = desc.split(" - ", 1)
    return a.strip(), t.strip()


def find_files(root: Path) -> dict[str, list[Path]]:
    root = Path(root)
    files = [p for p in root.rglob("*") if p.suffix.lower() in {".csv", ".json"}]
    kinds: dict[str, list[Path]] = {"play_activity": [], "daily_tracks": [], "likes": [],
                                    "library": [], "other": []}
    for p in files:
        n = p.name.lower()
        if "play activity" in n:
            kinds["play_activity"].append(p)
        elif "daily tracks" in n or "play history" in n:
            kinds["daily_tracks"].append(p)
        elif "likes and dislikes" in n or "favorites" in n or "favourites" in n:
            kinds["likes"].append(p)
        elif "library tracks" in n:
            kinds["library"].append(p)
        else:
            kinds["other"].append(p)
    return kinds


def _read(p: Path) -> pd.DataFrame:
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8-sig"))
        return pd.json_normalize(data if isinstance(data, list) else [data])
    return pd.read_csv(p, low_memory=False, encoding="utf-8-sig")


@dataclass
class InspectReport:
    text: str
    resolved: dict[str, dict[str, str]]


def inspect_export(root: Path, max_values: int = 15) -> InspectReport:
    lines, resolved = [], {}
    for kind, paths in find_files(root).items():
        for p in paths:
            try:
                df = _read(p)
            except Exception as e:  # noqa: BLE001 - report, don't crash, on odd files
                lines.append(f"[{kind}] {p.name}: could not read ({e})")
                continue
            m = resolve_columns(list(df.columns))
            resolved[str(p)] = m
            lines.append(f"\n[{kind}] {p.relative_to(root)}  rows={len(df):,}")
            lines.append(f"  columns: {list(df.columns)}")
            lines.append(f"  resolved: {m}")
            if kind in ("play_activity", "daily_tracks"):
                missing = [f for f in ("played_ms", "end_reason", "ts") if f not in m]
                if "title" not in m and "description" not in m:
                    missing.append("title/description")
                lines.append(f"  MISSING for implicit labels: {missing or 'none'}")
            for f in ("end_reason", "event_type", "preference", "media_type"):
                if f in m:
                    vc = df[m[f]].value_counts(dropna=False).head(max_values)
                    lines.append(f"  {f} values: {vc.to_dict()}")
            if "description" in m:
                ambiguous = df[m["description"]].astype(str).str.count(" - ").gt(1).mean()
                lines.append(f"  descriptions with >1 ' - ' (artist/title split ambiguous): "
                             f"{ambiguous:.1%}")
    return InspectReport("\n".join(lines), resolved)


def _artist_title(df: pd.DataFrame, m: dict[str, str]) -> tuple[pd.Series, pd.Series]:
    if "title" in m and "artist" in m:
        return df[m["artist"]], df[m["title"]]
    if "description" in m:
        at = df[m["description"]].map(split_description)
        return at.map(lambda x: x[0]), at.map(lambda x: x[1])
    if "title" in m:
        return pd.Series([None] * len(df), index=df.index), df[m["title"]]
    raise ValueError(f"No title/description column among {list(df.columns)}")


def parse_plays(path: Path, overrides: dict[str, str] | None = None) -> pd.DataFrame:
    """Normalise a play-activity or daily-tracks file into one row per play *event*.

    Daily aggregates are expanded: play_count completed plays + skip_count skips.
    """
    df = _read(path)
    m = resolve_columns(list(df.columns), overrides)
    if "event_type" in m:  # keep end-of-play events only; others (e.g. lyrics views) lack durations
        et = df[m["event_type"]].astype(str).str.upper()
        if et.str.contains("PLAY_END").any():
            df = df[et.eq("PLAY_END")]
    if "media_type" in m:
        mt = df[m["media_type"]].astype(str).str.lower()
        df = df[~mt.str.contains("video|radio|podcast", regex=True)]
    if "ignore" in m:
        df = df[~df[m["ignore"]].astype(str).str.lower().isin({"true", "1", "yes"})]
    artist, title = _artist_title(df, m)
    out = pd.DataFrame({
        "artist": artist.values, "title": title.values,
        "album": df[m["album"]].values if "album" in m else None,
        "ts": df[m["ts"]].astype(str).values if "ts" in m else None,
        "played_ms": pd.to_numeric(df[m["played_ms"]], errors="coerce").values
        if "played_ms" in m else None,
        "media_ms": pd.to_numeric(df[m["media_ms"]], errors="coerce").values
        if "media_ms" in m else None,
        "end_reason": df[m["end_reason"]].astype(str).str.upper().values
        if "end_reason" in m else None,
    })
    if "play_count" in m and "played_ms" not in m:  # pure aggregate: expand counts into events
        pc = pd.to_numeric(df[m["play_count"]], errors="coerce").fillna(0).astype(int).values
        sc = (pd.to_numeric(df[m["skip_count"]], errors="coerce").fillna(0).astype(int).values
              if "skip_count" in m else 0 * pc)
        base = out.drop(columns=["end_reason"])
        full = base.loc[base.index.repeat(pc)].assign(end_reason="NATURAL_END_OF_TRACK")
        skip = base.loc[base.index.repeat(sc)].assign(end_reason="TRACK_SKIPPED_FORWARDS")
        out = pd.concat([full, skip], ignore_index=True)
    elif "skip_count" in m:
        # Daily file with durations: one row per (track, day). Mark rows with skips.
        sc = pd.to_numeric(df[m["skip_count"]], errors="coerce").fillna(0).values
        pc = (pd.to_numeric(df[m["play_count"]], errors="coerce").fillna(0).values
              if "play_count" in m else 0 * sc)
        reason = pd.Series("NATURAL_END_OF_TRACK", index=out.index).where(sc <= pc,
                                                                          "TRACK_SKIPPED_FORWARDS")
        out["end_reason"] = out["end_reason"].fillna(reason) if "end_reason" in m else reason
    out = out[out["title"].notna()]
    out["track_id"] = [stable_id("lib", a, t) for a, t in zip(out["artist"], out["title"])]
    out["source"] = "apple_music"
    return out.reset_index(drop=True)


def parse_likes(path: Path, overrides: dict[str, str] | None = None) -> pd.DataFrame:
    """Loved / disliked items -> (track_id, artist, title, grade). LOVE=3, DISLIKE=0."""
    df = _read(path)
    m = resolve_columns(list(df.columns), overrides)
    if "media_type" in m:  # favourites files mix songs, albums, artists, playlists
        mt = df[m["media_type"]].astype(str).str.lower()
        keep = mt.str.contains("song|track|item", regex=True) | mt.eq("nan")
        df = df[keep]
    artist, title = _artist_title(df, m)
    pref = (df[m["preference"]].astype(str).str.upper() if "preference" in m
            else pd.Series("LOVE", index=df.index))
    grade = pref.map(lambda p: 0 if "DISLIKE" in p else (3 if any(
        k in p for k in ("LOVE", "LIKE", "FAVORITE", "SONG", "TRACK")) else None))
    out = pd.DataFrame({"artist": artist.values, "title": title.values, "grade": grade.values})
    out = out[out["title"].notna() & out["grade"].notna()]
    out["track_id"] = [stable_id("lib", a, t) for a, t in zip(out["artist"], out["title"])]
    return out.reset_index(drop=True)


def library_tracks(plays: pd.DataFrame, likes: pd.DataFrame) -> pd.DataFrame:
    """Unique library tracks (no audio: Apple Music downloads are DRM-protected)."""
    cols = ["track_id", "artist", "title"]
    parts = [plays[cols + ["album"]], likes[cols]]
    t = pd.concat(parts, ignore_index=True).drop_duplicates("track_id")
    t["source"] = "library"
    return t
