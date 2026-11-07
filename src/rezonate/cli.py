"""ReZonate command line.

  rezonate inspect-export ~/Downloads/Apple_Media_Services     # check columns FIRST
  rezonate ingest-apple   ~/Downloads/Apple_Media_Services
  rezonate ingest-local   ~/Music/DRM-free
  rezonate jamendo-fetch  --tags "postpunk indie dub" --pages 3
  rezonate enrich         --lyrics --musicbrainz [--lastfm]
  rezonate embed          --model clap
  rezonate rate --seed 150        # cold start: diverse batch to rate
  rezonate rate                   # normal batches: exploit + exploration
  rezonate recommend -n 20        # print a batch without rating
  rezonate status
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import click
import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from rezonate import db
from rezonate.config import load_config

app = typer.Typer(add_completion=False, no_args_is_help=True)
con_out = Console()
KEYS = {"0": 0, "1": 1, "2": 2, "3": 3}
LEGEND = "[bold]0[/] dislike  [bold]1[/] meh  [bold]2[/] like  [bold]3[/] love   " \
         "[bold]s[/] skip  [bold]r[/] replay  [bold]q[/] quit"


def _ctx():
    cfg = load_config()
    return cfg, db.connect(cfg.db_path)


@app.command("inspect-export")
def inspect_export(path: Path):
    """Print every file in an Apple Music export, its columns, and what we could map."""
    from rezonate.ingest.apple_music import inspect_export as insp
    con_out.print(insp(path).text, markup=False, highlight=False)


@app.command("ingest-apple")
def ingest_apple(path: Path, overrides: str = typer.Option(
        "", help='JSON column overrides, e.g. \'{"played_ms": "My Column"}\'')):
    """Load plays (implicit labels) and Loved/Disliked songs from an Apple Music export."""
    from rezonate.ingest import apple_music as am
    cfg, con = _ctx()
    ov = json.loads(overrides) if overrides else None
    files = am.find_files(path)
    plays = [am.parse_plays(p, ov) for p in files["play_activity"] or files["daily_tracks"]]
    likes = [am.parse_likes(p, ov) for p in files["likes"]]
    import pandas as pd
    plays = pd.concat(plays) if plays else pd.DataFrame(columns=["track_id", "artist", "title",
                                                                  "album"])
    likes = pd.concat(likes) if likes else pd.DataFrame(columns=["track_id", "artist", "title",
                                                                  "grade"])
    db.upsert_tracks(con, am.library_tracks(plays, likes))
    db.add_plays(con, plays)
    src = likes.grade.map(lambda g: "apple_love" if g == 3 else "apple_dislike")
    db.add_feedback(con, zip(likes.track_id, likes.grade, src, [None] * len(likes)))
    con_out.print(f"plays: {len(plays):,}  loved/disliked: {len(likes):,}  "
                  f"library tracks: {plays.track_id.nunique():,}")


@app.command("ingest-local")
def ingest_local(folder: Path):
    """Index DRM-free audio files and link them to library songs by artist+title."""
    from rezonate.ingest import local_audio
    cfg, con = _ctx()
    local, protected = local_audio.scan(folder)
    lib = db.read(con, "tracks", "WHERE source='library'")
    local = local_audio.link_to_library(local, lib)
    db.upsert_tracks(con, local)
    con_out.print(f"indexed {len(local)} files, linked {local.link_id.notna().sum()} to library; "
                  f"skipped {len(protected)} DRM-protected")


@app.command("jamendo-probe")
def jamendo_probe(tags: str = "indie"):
    """Print one raw Jamendo response so you can check field names against the client."""
    from rezonate.enrich.clients import make_client
    from rezonate.ingest.jamendo import API
    cfg, _ = _ctx()
    http = make_client(cfg.user_agent, None)
    data = http.get_json(f"{API}/tracks/", {"client_id": cfg.jamendo_client_id, "format": "json",
                                           "limit": 2, "fuzzytags": tags,
                                           "include": "musicinfo stats"}, cache=False)
    con_out.print_json(json.dumps(data))


@app.command("jamendo-fetch")
def jamendo_fetch(tags: str = typer.Option(..., help="space-separated fuzzy tags"),
                  pages: int = 2, download: bool = True):
    """Add Creative Commons tracks from Jamendo to the candidate pool (downloads audio)."""
    from rezonate.enrich.clients import make_client
    from rezonate.ingest.jamendo import Jamendo
    cfg, con = _ctx()
    j = Jamendo(cfg.jamendo_client_id, make_client(cfg.user_agent, cfg.cache_dir / "http"))
    n = 0
    for page in range(pages):
        df = j.search_tracks(fuzzytags=tags.replace(" ", "+"), offset=page * 200)
        if df.empty:
            break
        if download:
            df["audio_path"] = [str(j.download(u, cfg.audio_dir / "jamendo" / f"{t.split(':')[1]}.mp3"))
                                for t, u in zip(df.track_id, df.download_url)]
        db.upsert_tracks(con, df)
        db.upsert_tags(con, [(t, tag.lower(), 1.0, "jamendo") for t, ts in zip(df.track_id, df.tags)
                             for tag in ts])
        n += len(df)
    con_out.print(f"added {n} Jamendo tracks")


@app.command()
def enrich(lyrics: bool = True, musicbrainz: bool = True, lastfm: bool = False,
           limit: int = typer.Option(0, help="max tracks per step (0 = all)")):
    """Fetch lyrics (LRCLIB), era/mbid/tags (MusicBrainz) and tags (Last.fm, needs key)."""
    from rezonate.enrich import run
    from rezonate.enrich.clients import LastFM, LRCLib, MusicBrainz, make_client
    cfg, con = _ctx()
    http = make_client(cfg.user_agent, cfg.cache_dir / "http")
    lim = limit or None
    if musicbrainz:
        con_out.print(run.musicbrainz(con, MusicBrainz(http), lim, log=con_out.print))
    if lastfm:
        lfm = LastFM(cfg.lastfm_api_key, http)
        con_out.print(run.lastfm_tags(con, lfm, lim, log=con_out.print))
        con_out.print(run.lastfm_popularity(con, lfm, lim, log=con_out.print))
    if lyrics:
        con_out.print(run.lyrics(con, LRCLib(http), lim, log=con_out.print))


@app.command()
def embed(model: str = "clap", batch_tracks: int = 4):
    """Embed all tracks with audio (resumable; run overnight for a big library)."""
    from rezonate.embed import store
    from rezonate.embed.models import get_embedder
    cfg, con = _ctx()
    items = con.execute("SELECT track_id, audio_path FROM tracks WHERE audio_path IS NOT NULL").fetchall()
    stats = store.embed_tracks(get_embedder(model), items, store.path_for(cfg.embeddings_dir, model),
                               batch_tracks=batch_tracks, log=con_out.print)
    con_out.print({k: v for k, v in stats.items() if k != "failed"},
                  f"failed: {len(stats['failed'])}")


def _model_and_catalog(cfg, con, use_text_bridge: bool = True):
    from rezonate import service
    cat = service.load_catalog(cfg, con)
    labels = service.current_labels(con)
    extra_X = extra_g = None
    if use_text_bridge and cfg.embedding_model == "clap":
        from rezonate.embed.models import ClapEmbedder
        lib_labels = labels[~labels.track_id.isin(cat.row)]
        if len(lib_labels):
            ids, T = service.text_anchors(con, labels, cat, ClapEmbedder().embed_text)
            if len(ids):
                extra_X = T
                extra_g = labels.set_index("track_id").loc[ids, "grade"].values
    model = service.fit_bayes(cat, labels, extra_X, extra_g) if (
        len(labels[labels.track_id.isin(cat.row)]) or extra_X is not None) else None
    return cat, labels, model


def _show(cat, picks):
    t = Table("#", "artist", "title", "source", "year", "why")
    for i, p in enumerate(picks, 1):
        r = cat.tracks.iloc[p.row]
        t.add_row(str(i), str(r.artist), str(r.title), str(r.source),
                  "" if r.year != r.year or r.year is None else str(int(r.year)),
                  f"{p.context}: {p.reason}")
    con_out.print(t)


@app.command()
def recommend(n: int = typer.Option(20, "-n", "--n", help="batch size"), eps: float = typer.Option(None, help="exploration share"),
              niche_bias: float = 0.0, seed: int = 0):
    """Print a batch (exploit + exploration) without rating."""
    from rezonate import service
    cfg, con = _ctx()
    cat, labels, model = _model_and_catalog(cfg, con)
    if model is None:
        con_out.print("No ratings yet: run `rezonate rate --seed 150` first.")
        raise typer.Exit(1)
    picks = service.recommend(cat, model, labels, n=n,
                              eps=cfg.explore_fraction if eps is None else eps,
                              seed=seed, niche_bias=niche_bias)
    _show(cat, picks)


def _getkey() -> str:
    """One keypress on a terminal; falls back to reading stdin (pipes, tests, notebooks)."""
    import sys
    if sys.stdin.isatty():
        return click.getchar()
    ch = sys.stdin.read(1)
    while ch and ch.isspace():
        ch = sys.stdin.read(1)
    return ch or "q"


def _play(path: str, duration_s: float | None, clip_s: int) -> subprocess.Popen | None:
    if not shutil.which("ffplay") or not path:
        return None
    start = 0.0 if not duration_s or duration_s <= clip_s * 1.5 else 0.35 * duration_s
    return subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
                             "-ss", f"{start:.0f}", "-t", str(clip_s), path],
                            stdin=subprocess.DEVNULL)


@app.command()
def rate(seed: int = typer.Option(0, help="cold start: rate N diverse tracks (e.g. 150)"),
         n: int = typer.Option(20, "-n", "--n", help="batch size"), clip: int = 30,
         no_audio: bool = False):
    """Rate tracks one key at a time; the Bayesian model updates after every rating."""
    from rezonate import service
    cfg, con = _ctx()
    cat, labels, model = _model_and_catalog(cfg, con)
    if seed or model is None:
        rows = service.seed_selection(cat, seed or 150, known=set(labels.track_id))
        from rezonate.explore import Pick
        picks = [Pick(r, "seed", "coverage") for r in rows]
    else:
        picks = service.recommend(cat, model, labels, n=n, eps=cfg.explore_fraction)
    con_out.print(LEGEND)
    rated = []
    for i, p in enumerate(picks, 1):
        r = cat.tracks.iloc[p.row]
        before = float(model.score(cat.X[[p.row]])[0]) if model else None
        con_out.rule(f"{i}/{len(picks)}")
        con_out.print(f"[bold]{r.artist}[/] — {r.title}   [dim]({r.source}"
                      f"{', ' + str(r.license) if r.license else ''})[/]")
        proc = None if no_audio else _play(r.audio_path, r.duration_s, clip)
        while True:
            k = _getkey()
            if k == "r":
                if proc:
                    proc.kill()
                proc = None if no_audio else _play(r.audio_path, r.duration_s, clip)
                continue
            break
        if proc:
            proc.kill()
        if k == "q":
            break
        if k not in KEYS:
            con_out.print("[dim]skipped[/]")
            continue
        g = KEYS[k]
        db.add_feedback(con, [(cat.ids[p.row], g, "cli", p.context)])
        rated.append((p.context, g))
        if model is not None:
            model.update(cat.X[p.row], g)
            after = float(model.score(cat.X[[p.row]])[0])
            con_out.print(f"grade {g}  [dim]P(like) {before:.2f} -> {after:.2f}  "
                          f"({p.context}: {p.reason})[/]")
        else:
            con_out.print(f"grade {g}")
            if len(rated) >= 10:  # enough to start a model mid-session
                cat, labels, model = _model_and_catalog(cfg, con, use_text_bridge=False)
    if rated:
        ex = [g for c, g in rated if c == "explore"]
        hit = f"{np.mean([g >= 2 for g in ex]):.0%} of {len(ex)}" if ex else "n/a (none rated)"
        con_out.print(f"rated {len(rated)}; explore hit rate (like+): {hit}")


@app.command()
def evaluate(views: str = typer.Option("clap", help="comma-separated embedding views"),
             context_groups: str = "tags,era,lyrics", k: int = 10, repeats: int = 3,
             min_labels: int = 50):
    """Held-out NDCG@k on YOUR ratings: taste models vs popularity and random baselines."""
    from rezonate import service
    from rezonate.embed.views import load_view
    from rezonate.eval import protocol
    from rezonate.eval.scorers import make_scorers, pop_percentile
    from rezonate.features import context
    cfg, con = _ctx()
    vlist = [v.strip() for v in views.split(",") if v.strip()]
    ids, _ = load_view(cfg.embeddings_dir, "+".join(vlist) if len(vlist) > 1 else vlist[0])
    vecs = {v: load_view(cfg.embeddings_dir, v, ids)[1] for v in vlist}
    labels = service.current_labels(con)
    row = {t: i for i, t in enumerate(ids)}
    labels = labels[labels.track_id.isin(row)]
    if len(labels) < min_labels or labels.grade.nunique() < 2:
        con_out.print(f"Only {len(labels)} rated tracks with audio; rate more first "
                      f"(need {min_labels}+ with at least two different grades).")
        raise typer.Exit(1)
    groups = tuple(g for g in context_groups.split(",") if g)
    C, _, slices = context.build(con, ids, groups=groups + ("pop",))
    tracks = db.read(con, "tracks").set_index("track_id").reindex(ids)
    pop = pop_percentile(tracks.popularity.values, tracks.source.values)
    setup = protocol.Setup(ids=ids, views=vecs, C=C, C_slices=slices, popularity=pop,
                           lab_rows=labels.track_id.map(row).values.astype(int),
                           grades=labels.grade.values.astype(int),
                           weights=labels.weight.values.astype(float))
    ctx_sets = {"": (), "+" + "+".join(groups): groups} if groups else {"": ()}
    folds = min(5, int(np.bincount(setup.grades).max()))
    df = protocol.run(setup, make_scorers(setup, vlist, vlist[0], ctx_sets), k=k,
                      n_splits=max(2, min(folds, 5)), n_repeats=repeats, log=con_out.print)
    summ = protocol.summarise(df)
    out = cfg.data_dir / "eval_real.csv"
    summ.to_csv(out, index=False)
    cols = [c for c in summ.columns if not c.endswith("_ci") or c == f"ndcg@{k}_ci"]
    t = Table(*cols)
    for _, r in summ.iterrows():
        t.add_row(*[f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols])
    con_out.print(t)
    con_out.print(f"labels by source: {labels.label_source.value_counts().to_dict()}  "
                  f"-> {out}\n[dim]Caveat: ratings aren't a random sample of songs (selection "
                  "bias); overlapping CIs mean no evidence of a difference.[/]")


@app.command()
def status():
    """Counts of tracks, audio, embeddings, labels."""
    from rezonate import service
    from rezonate.embed import store
    cfg, con = _ctx()
    def q(s):
        return con.execute(s).fetchone()[0]
    con_out.print(f"tracks: {q('SELECT COUNT(*) FROM tracks'):,}  "
                  f"with audio: {q('SELECT COUNT(*) FROM tracks WHERE audio_path IS NOT NULL'):,}")
    for m in ("clap", "mert"):
        p = store.path_for(cfg.embeddings_dir, m)
        con_out.print(f"{m} embeddings: {len(store.load(p)[0]) if p.exists() else 0:,}")
    labels = service.current_labels(con)
    con_out.print(f"labels: {len(labels):,} by source {labels.label_source.value_counts().to_dict()}"
                  f" by grade {labels.grade.value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    app()
