"""Audio decoding and window selection shared by all embedders.

Decoding goes through ffmpeg (handles mp3/flac/m4a/ogg uniformly and resamples).
Every model sees the *same* windows of a track so comparisons are fair:
``n_windows`` windows of ``win_s`` seconds, spread evenly over the track while
avoiding the first/last 10 % (intros/outros are unrepresentative).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


def load_audio(path: str | Path, sr: int) -> np.ndarray:
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-f", "f32le",
           "-ac", "1", "-ar", str(sr), "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def windows(y: np.ndarray, sr: int, win_s: float = 10.0, n_windows: int = 3) -> list[np.ndarray]:
    w = int(win_s * sr)
    if len(y) <= w:
        return [np.pad(y, (0, w - len(y)))]
    lo, hi = int(0.1 * len(y)), int(0.9 * len(y)) - w
    if hi <= lo:  # short clip (e.g. FMA 30 s): use the full span
        lo, hi = 0, len(y) - w
    starts = np.linspace(lo, hi, n_windows).astype(int) if n_windows > 1 else [(lo + hi) // 2]
    return [y[s:s + w] for s in starts]
