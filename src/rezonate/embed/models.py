"""Pretrained audio embedders behind one tiny interface.

* ``clap``  — laion/larger_clap_music_and_speech (Apache-2.0). Contrastive audio-text model, 48 kHz,
  512-d joint space. Bonus: its *text* tower lets us embed tag descriptions into the same
  space as audio (used to seed taste for library songs we have no audio for).
  NOTE: we deliberately do NOT use ``laion/larger_clap_music``. Its Hugging Face checkpoint
  (rev a0b4534) loads without warnings but is broken: logit_scale is about 1.0 and audio-text
  cosines are about 0, so zero-shot accuracy is at chance. Its sibling checkpoints behave
  normally (see reports/clap_checkpoint_check.md).
* ``mert``  — m-a-p/MERT-v1-95M (CC-BY-NC-4.0: fine for personal use, not commercial).
  Self-supervised music model, 24 kHz, 13 hidden layers x 768. Different layers encode
  different things (timbre vs. pitch/harmony), so we keep the per-layer means and pick
  layers on the comparison set.

Revisions are pinned; MERT needs ``trust_remote_code`` (it ships its own modelling code),
which is another reason to pin the exact commit we reviewed.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch

MODEL_REVISIONS = {
    "laion/larger_clap_music_and_speech": "195c3a3e68faebb3e2088b9a79e79b43ddbda76b",
    "m-a-p/MERT-v1-95M": "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5",
}


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder(Protocol):
    name: str
    sr: int

    def embed(self, wins: list[np.ndarray]) -> np.ndarray:
        """Embed a batch of equal-length mono windows -> (n, ...) float32."""


class ClapEmbedder:
    name = "clap"
    sr = 48_000
    repo = "laion/larger_clap_music_and_speech"

    def __init__(self):
        from transformers import ClapModel, ClapProcessor

        rev = MODEL_REVISIONS[self.repo]
        self.dev = device()
        self.proc = ClapProcessor.from_pretrained(self.repo, revision=rev)
        self.model = ClapModel.from_pretrained(self.repo, revision=rev).to(self.dev).eval()

    @torch.inference_mode()
    def embed(self, wins: list[np.ndarray]) -> np.ndarray:
        inp = self.proc(audios=wins, sampling_rate=self.sr, return_tensors="pt")
        inp = {k: v.to(self.dev) for k, v in inp.items()}
        z = self.model.get_audio_features(**inp)
        return torch.nn.functional.normalize(z, dim=-1).cpu().float().numpy()

    @torch.inference_mode()
    def embed_text(self, texts: list[str]) -> np.ndarray:
        inp = self.proc(text=texts, return_tensors="pt", padding=True)
        inp = {k: v.to(self.dev) for k, v in inp.items()}
        z = self.model.get_text_features(**inp)
        return torch.nn.functional.normalize(z, dim=-1).cpu().float().numpy()


class MertEmbedder:
    name = "mert"
    sr = 24_000
    repo = "m-a-p/MERT-v1-95M"

    def __init__(self):
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        rev = MODEL_REVISIONS[self.repo]
        self.dev = device()
        self.proc = Wav2Vec2FeatureExtractor.from_pretrained(self.repo, revision=rev,
                                                             trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.repo, revision=rev,
                                               trust_remote_code=True).to(self.dev).eval()

    @torch.inference_mode()
    def embed(self, wins: list[np.ndarray]) -> np.ndarray:
        inp = self.proc(wins, sampling_rate=self.sr, return_tensors="pt")
        inp = {k: v.to(self.dev) for k, v in inp.items()}
        hs = self.model(**inp, output_hidden_states=True).hidden_states  # 13 x [B, T, 768]
        z = torch.stack([h.mean(dim=1) for h in hs], dim=1)  # [B, 13, 768]
        return z.cpu().float().numpy()


EMBEDDERS = {"clap": ClapEmbedder, "mert": MertEmbedder}


def get_embedder(name: str) -> Embedder:
    return EMBEDDERS[name]()
