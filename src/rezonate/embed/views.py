"""Load a named, preprocessed embedding view aligned to a list of track IDs.

View names:  "clap" | "mert:L<i>" | "mert:mean" | "mert:L<a>-<b>" (mean of a layer range)
             | "clap+mert:L<i>" (each part centered + L2-normalised, concatenated, renormalised)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rezonate.embed import store
from rezonate.embed.compare import prep


def _mert(X: np.ndarray, spec: str) -> np.ndarray:
    if spec == "mean":
        return X.mean(1)
    spec = spec.lstrip("L")
    if "-" in spec:
        a, b = map(int, spec.split("-"))
        return X[:, a:b + 1].mean(1)
    return X[:, int(spec)]


def load_view(emb_dir: Path, view: str, ids: list[str] | None = None) -> tuple[list[str], np.ndarray]:
    parts = view.split("+")
    loaded = []
    for p in parts:
        model, _, spec = p.partition(":")
        mids, X = store.load(store.path_for(emb_dir, model))
        if model == "mert":
            X = _mert(X, spec or "mean")
        loaded.append(dict(zip(mids, prep(X))))
    common = set.intersection(*[set(d) for d in loaded])
    ids = [i for i in (ids if ids is not None else sorted(common)) if i in common]
    Z = np.hstack([np.stack([d[i] for i in ids]) for d in loaded])
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    return ids, Z.astype(np.float64)
