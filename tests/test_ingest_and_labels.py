import pandas as pd

from rezonate import db
from rezonate.ingest import apple_music as am
from rezonate.labels import fuse_labels, implicit_grades


def write_export(tmp_path):
    root = tmp_path / "Apple Media Services information" / "Apple Music Activity"
    root.mkdir(parents=True)
    rows = []
    # Song A: completed on 3 different days -> like
    for d in ("2024-01-01", "2024-01-05", "2024-02-01"):
        rows.append(["Song A", "Album", "Artist X", f"{d}T10:00:00Z", 200_000, 200_000,
                     "NATURAL_END_OF_TRACK", "PLAY_END"])
    # Song B: skipped early 3 times -> dislike
    for d in ("2024-01-01", "2024-01-02", "2024-01-03"):
        rows.append(["Song B", "Album", "Artist Y", f"{d}T10:00:00Z", 8_000, 210_000,
                     "TRACK_SKIPPED_FORWARDS", "PLAY_END"])
    # a non-play event that must be ignored
    rows.append(["Song B", "Album", "Artist Y", "2024-01-04T10:00:00Z", None, None, None,
                 "LYRIC_DISPLAY"])
    pd.DataFrame(rows, columns=["Song Name", "Album Name", "Artist Name", "Event Start Timestamp",
                                "Play Duration Milliseconds", "Media Duration In Milliseconds",
                                "End Reason Type", "Event Type"]).to_csv(
        root / "Apple Music Play Activity.csv", index=False)
    pd.DataFrame({"Item Description": ["Artist Z - Loved Song", "Artist Y - Song B"],
                  "Preference": ["LOVE", "DISLIKE"]}).to_csv(
        root / "Apple Music Likes and Dislikes.csv", index=False)
    return tmp_path


def test_inspect_and_parse_export(tmp_path):
    root = write_export(tmp_path)
    rep = am.inspect_export(root)
    assert "MISSING for implicit labels: none" in rep.text
    files = am.find_files(root)
    plays = am.parse_plays(files["play_activity"][0])
    assert len(plays) == 6  # LYRIC_DISPLAY dropped
    likes = am.parse_likes(files["likes"][0])
    assert set(likes.grade) == {0, 3}
    assert likes.loc[likes.grade == 3, "artist"].item() == "Artist Z"


def test_implicit_grades_and_priority(tmp_path):
    root = write_export(tmp_path)
    files = am.find_files(root)
    plays = am.parse_plays(files["play_activity"][0])
    imp = implicit_grades(plays).set_index("track_id")
    a = db.stable_id("lib", "Artist X", "Song A")
    b = db.stable_id("lib", "Artist Y", "Song B")
    assert imp.loc[a, "grade"] == 2 and imp.loc[b, "grade"] == 0
    likes = am.parse_likes(files["likes"][0])
    explicit = pd.DataFrame({"track_id": [a], "grade": [3], "source": ["cli"]})
    fused = fuse_labels(imp.reset_index(), likes, explicit).set_index("track_id")
    assert fused.loc[a, "grade"] == 3 and fused.loc[a, "label_source"] == "cli"   # explicit wins
    assert fused.loc[b, "label_source"] == "apple_pref"                         # dislike beats implicit


def test_aggregate_daily_file_expands_counts(tmp_path):
    p = tmp_path / "Apple Music - Play History Daily Tracks.csv"
    pd.DataFrame({"Track Description": ["Band - Tune"], "Play Count": [2], "Skip Count": [1],
                  "Date Played": ["20240101"]}).to_csv(p, index=False)
    plays = am.parse_plays(p)
    assert sorted(plays.end_reason) == ["NATURAL_END_OF_TRACK"] * 2 + ["TRACK_SKIPPED_FORWARDS"]


def test_db_roundtrip(tmp_path):
    con = db.connect(tmp_path / "t.db")
    db.upsert_tracks(con, pd.DataFrame({"track_id": ["x:1"], "source": ["fma"], "title": ["t"]}))
    db.upsert_tracks(con, pd.DataFrame({"track_id": ["x:1"], "source": ["fma"], "year": [2010]}))
    t = db.read(con, "tracks")
    assert t.title.item() == "t" and t.year.item() == 2010   # NULLs don't overwrite
    db.add_feedback(con, [("x:1", 1, "cli", "seed")], ts=1.0)
    db.add_feedback(con, [("x:1", 3, "cli", "exploit")], ts=2.0)
    assert db.latest_feedback(con).grade.item() == 3
