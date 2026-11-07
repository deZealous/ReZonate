"""Zero-shot genre check for LAION CLAP checkpoints on the FMA comparison sample.

Why this exists: laion/larger_clap_music (the obvious choice for music) loads without any
missing-key warnings but behaves as if untrained. This script makes that claim reproducible.
Writes reports/clap_checkpoint_check.md.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

from rezonate import db
from rezonate.config import REPO_ROOT, load_config
from rezonate.embed.audio import load_audio, windows

CHECKPOINTS = {
    "laion/larger_clap_music": "a0b4534a14f58e20944452dff00a22a06ce629d1",
    "laion/larger_clap_music_and_speech": "195c3a3e68faebb3e2088b9a79e79b43ddbda76b",
}


def main(per_genre: int = 8) -> None:
    cfg = load_config()
    con = db.connect(cfg.db_path)
    ids = set((cfg.data_dir / "comparison_ids.txt").read_text().split())
    rows = [r for r in con.execute("SELECT track_id, audio_path, genre_hint FROM tracks").fetchall()
            if r[0] in ids]
    rng = np.random.default_rng(0)
    by_g: dict[str, list] = {}
    for r in rows:
        by_g.setdefault(r[2], []).append(r)
    sample = [x for g in sorted(by_g) for x in
              [by_g[g][i] for i in rng.choice(len(by_g[g]), min(per_genre, len(by_g[g])), replace=False)]]
    genres = sorted(by_g)
    y = np.array([genres.index(r[2]) for r in sample])
    wins = [windows(load_audio(r[1], 48000), 48000, n_windows=1)[0] for r in sample]
    prompts = [f"{g.lower()} music" for g in genres]
    lines = ["# CLAP checkpoint sanity check", "",
             f"{len(sample)} FMA clips ({per_genre}/genre, {len(genres)} genres); prompt "
             "\"<genre> music\"; one 10 s window per clip.", "",
             "| checkpoint | revision | logit scale | mean audio-text cos | text-text cos | zero-shot acc | chance |",
             "|---|---|---|---|---|---|---|"]
    for repo, rev in CHECKPOINTS.items():
        proc = ClapProcessor.from_pretrained(repo, revision=rev)
        model = ClapModel.from_pretrained(repo, revision=rev).eval()
        with torch.inference_mode():
            a = model.get_audio_features(**proc(audios=wins, sampling_rate=48000, return_tensors="pt"))
            t = model.get_text_features(**proc(text=prompts, return_tensors="pt", padding=True))
        a = torch.nn.functional.normalize(a, dim=-1)
        t = torch.nn.functional.normalize(t, dim=-1)
        sims = (a @ t.T).numpy()
        tt = (t @ t.T).numpy()[np.triu_indices(len(prompts), 1)].mean()
        acc = float((sims.argmax(1) == y).mean())
        lines.append(f"| {repo} | {rev[:7]} | {model.logit_scale_a.exp().item():.2f} | "
                     f"{sims.mean():.3f} | {tt:.3f} | {acc:.2f} | {1 / len(genres):.2f} |")
        print(lines[-1], flush=True)
    lines += ["", "A healthy CLAP has a logit scale in the tens, clearly positive audio-text "
              "cosines and distinct text embeddings. `larger_clap_music` fails all three, so "
              "ReZonate uses `larger_clap_music_and_speech`."]
    (REPO_ROOT / "reports" / "clap_checkpoint_check.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
