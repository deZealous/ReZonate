"""Central configuration. Values come from defaults, then ``configs/*.toml``, then env vars.

Kept deliberately small: paths, API identity, and a few modelling knobs that you
are likely to tweak. Everything else lives as function arguments with defaults.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    data_dir: Path = REPO_ROOT / "data"
    # Contact URL/email is required by MusicBrainz' etiquette rules and used for LRCLIB too.
    user_agent: str = "ReZonate/0.1 (https://github.com/dezealous/rezonate)"
    lastfm_api_key: str | None = None
    jamendo_client_id: str | None = None
    # Embedding model used by the recommender (after you have run the comparison).
    embedding_model: str = "clap"
    # Share of each recommendation batch reserved for exploration.
    explore_fraction: float = 0.25
    batch_size: int = 20
    extra: dict = field(default_factory=dict)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "rezonate.db"

    @property
    def embeddings_dir(self) -> Path:
        return self.data_dir / "embeddings"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"


def load_config(path: str | os.PathLike | None = None) -> Config:
    cfg = Config()
    path = Path(path) if path else REPO_ROOT / "configs" / "local.toml"
    if path.exists():
        raw = tomllib.loads(path.read_text())
        names = {f.name for f in fields(Config)}
        for k, v in raw.items():
            if k in names:
                setattr(cfg, k, Path(v) if k == "data_dir" else v)
            else:
                cfg.extra[k] = v
    # Secrets are read from the environment so they never end up in git.
    cfg.lastfm_api_key = os.environ.get("LASTFM_API_KEY", cfg.lastfm_api_key)
    cfg.jamendo_client_id = os.environ.get("JAMENDO_CLIENT_ID", cfg.jamendo_client_id)
    if d := os.environ.get("REZONATE_DATA_DIR"):
        cfg.data_dir = Path(d)
    for p in (cfg.data_dir, cfg.embeddings_dir, cfg.audio_dir, cfg.cache_dir):
        p.mkdir(parents=True, exist_ok=True)
    return cfg
