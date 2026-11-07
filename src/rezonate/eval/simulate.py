"""A *simulated* listener on FMA, so the full pipeline can be run and tested end-to-end
before your real ratings exist.

!! The numbers this produces validate the machinery; they are NOT evidence about your taste. !!

The persona maps your stated taste onto FMA sub-genres (FMA has no Rosé or Charli XCX, so
this is a stylistic analogue only):

  modes (likes): post-punk/garage, indie/lo-fi/shoegaze, alt/synth pop, soul-R&B,
                 reggae/dub, jazz-punk/free jazz, heavy alt rock (older-Linkin-Park lane)
  dislikes:      ASSUMED. You left this blank; we use old-time/historic, country,
                 easy listening and spoken word as placeholders.

Utility = best mode affinity + artist effect (some artists you just like) + a mild
popularity pull (real listeners aren't immune to it, so the popularity baseline isn't a
straw man) + noise. Grades come from utility quantiles.

Leakage caveat: liking is defined through FMA genre labels. Audio embeddings and FMA tags
both correlate with those labels, so absolute scores are optimistic. Relative comparisons
between models are more meaningful than the levels themselves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERSONA_MODES = {
    "post-punk / garage": ["Post-Punk", "Punk", "Garage", "New Wave"],
    "indie / lo-fi / shoegaze": ["Indie-Rock", "Lo-Fi", "Shoegaze"],
    "alt / synth pop": ["Experimental Pop", "Synth Pop"],
    "soul / r&b": ["Soul-RnB"],
    "reggae / dub": ["Reggae - Dub"],
    "jazz-punk / free jazz": ["Free-Jazz", "Jazz: Out", "No Wave"],
    "heavy alt rock": ["Loud-Rock", "Industrial"],
}
PERSONA_DISLIKES = ["Old-Time / Historic", "Country", "Easy Listening", "Spoken", "Bluegrass",
                    "Polka"]


def utility(meta: pd.DataFrame, seed: int = 0, artist_sd: float = 0.5, pop_w: float = 0.25,
            noise_sd: float = 0.45) -> pd.Series:
    rng = np.random.default_rng(seed)
    gsets = meta["genres_all"].map(set)
    mode = np.zeros(len(meta))
    for gs in PERSONA_MODES.values():
        mode = np.maximum(mode, gsets.map(lambda s, gs=gs: len(s & set(gs)) > 0).values.astype(float))
    dislike = gsets.map(lambda s: len(s & set(PERSONA_DISLIKES)) > 0).values.astype(float)
    artists = meta["artist"].fillna("?")
    eff = {a: rng.normal(0, artist_sd) for a in artists.unique()}
    pop = np.log1p(meta["listens"].fillna(0).values)
    pop = (pop - pop.mean()) / (pop.std() + 1e-9)
    u = 1.6 * mode - 1.2 * dislike + artists.map(eff).values + pop_w * pop + rng.normal(0, noise_sd, len(meta))
    return pd.Series(u, index=meta.index)


def grades_from_utility(u: pd.Series, q=(0.35, 0.65, 0.85)) -> pd.Series:
    """Bottom 35 % dislike, next 30 % meh, next 20 % like, top 15 % love."""
    cuts = np.quantile(u, q)
    return pd.Series(np.searchsorted(cuts, u.values), index=u.index).astype(int)
