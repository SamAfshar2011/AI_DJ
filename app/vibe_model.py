"""
"Vibe" understanding: a fixed-length embedding + genre/mood tags per track that
drive similarity-based ordering in the planner.

Two tiers, chosen automatically at runtime:

  1. LEARNED (preferred) — a small mel-spectrogram CNN (`GenreCNN`).  If trained
     weights exist in models_weights/model_1_genre_cnn.pt we load them and use
     the penultimate layer as a 64-d vibe embedding plus a softmax over the
     GTZAN genres.  Train it with train_models/model_1_genre_cnn.ipynb.

  2. DSP FALLBACK (always available) — a hand-crafted, L2-normalised descriptor
     built from MFCC statistics, chroma, spectral contrast, tempo and energy.
     This is genuinely good for "sounds similar" ordering and needs no weights.

The same `GenreCNN` class is imported by the training notebook so the
architecture stays in one place.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .config import ANALYSIS_SR, HOP_LENGTH, MODELS_DIR, N_FFT, N_MELS
from .utils import get_logger, safe_float, weight_signature

log = get_logger()

try:
    import librosa
except Exception as e:  # pragma: no cover
    raise RuntimeError("librosa is required") from e

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False

GTZAN_GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]

WEIGHTS_PATH = MODELS_DIR / "model_1_genre_cnn.pt"
EMBED_DIM = 64
_MEL_FRAMES = 128  # ~3 s context window at 22.05 kHz / hop 512


# ---------------------------------------------------------------------------
# Model definition (shared with the training notebook)
# ---------------------------------------------------------------------------
if _HAVE_TORCH:

    class GenreCNN(nn.Module):
        """Compact 4-block CNN over log-mel spectrograms -> 64-d embedding -> genres."""

        def __init__(self, n_classes: int = 10, n_mels: int = N_MELS, embed_dim: int = EMBED_DIM):
            super().__init__()
            self.features = nn.Sequential(
                self._block(1, 32), self._block(32, 64),
                self._block(64, 128), self._block(128, 128),
            )
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.embed = nn.Linear(128, embed_dim)
            self.dropout = nn.Dropout(0.3)
            self.classifier = nn.Linear(embed_dim, n_classes)

        @staticmethod
        def _block(cin: int, cout: int) -> "nn.Sequential":
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        def forward(self, x, return_embedding: bool = False):
            h = self.features(x)
            h = self.pool(h).flatten(1)
            emb = self.embed(h)
            if return_embedding:
                return emb
            logits = self.classifier(self.dropout(F.relu(emb)))
            return logits, emb


def log_mel(y: np.ndarray, sr: int = ANALYSIS_SR) -> np.ndarray:
    """Log-mel spectrogram used everywhere as the model input feature."""
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    return librosa.power_to_db(S + 1e-9, ref=np.max).astype(np.float32)


def mel_windows(
    y: np.ndarray, sr: int = ANALYSIS_SR, frames: int = _MEL_FRAMES, max_windows: int = 8
) -> np.ndarray:
    """
    Evenly-spaced fixed-length log-mel windows across a track, shape
    (n_windows, N_MELS, frames).  Shared by the CNN-based wrappers (vibe / mood /
    danceability) so every model sees the exact same framing used in training.
    """
    mel = log_mel(y, sr)
    T = mel.shape[1]
    n = min(max_windows, max(1, T // frames + 1))
    starts = np.linspace(0, max(0, T - frames), num=n).astype(int)
    out = []
    for s in starts:
        w = mel[:, s:s + frames]
        if w.shape[1] < frames:
            w = np.pad(w, ((0, 0), (0, frames - w.shape[1])), mode="edge")
        out.append(w)
    return np.stack(out).astype(np.float32)


class VibeModel:
    """Runtime wrapper.  Call once, reuse for the whole library."""

    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.mode = "dsp"
        self._norm = None
        if _HAVE_TORCH:
            try:
                if torch.backends.mps.is_available():
                    self.device = "mps"
                elif torch.cuda.is_available():
                    self.device = "cuda"
            except Exception:
                self.device = "cpu"
            self._try_load()

    def _try_load(self) -> None:
        if not (_HAVE_TORCH and WEIGHTS_PATH.exists()):
            log.info("Vibe model: using DSP-embedding fallback (no trained weights at %s)", WEIGHTS_PATH.name)
            return
        try:
            ckpt = torch.load(WEIGHTS_PATH, map_location="cpu")
            state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            n_classes = len(ckpt.get("genres", GTZAN_GENRES)) if isinstance(ckpt, dict) else 10
            model = GenreCNN(n_classes=n_classes)
            model.load_state_dict(state)
            model.eval().to(self.device)
            self.model = model
            self._norm = ckpt.get("norm") if isinstance(ckpt, dict) else None
            self.genres = ckpt.get("genres", GTZAN_GENRES) if isinstance(ckpt, dict) else GTZAN_GENRES
            self.mode = "cnn"
            log.info("Vibe model: loaded trained GenreCNN (%s) on %s", WEIGHTS_PATH.name, self.device)
        except Exception as e:
            log.warning("Vibe model load failed (%s) — using DSP fallback", e)
            self.model = None
            self.mode = "dsp"

    # ---- learned path -----------------------------------------------------
    def _cnn_embed(self, y: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        mel = log_mel(y, sr)
        if self._norm:
            mel = (mel - self._norm["mean"]) / (self._norm["std"] + 1e-6)
        # average embeddings/logits over several windows across the track
        embs, logits_acc = [], []
        T = mel.shape[1]
        starts = np.linspace(0, max(0, T - _MEL_FRAMES), num=min(8, max(1, T // _MEL_FRAMES + 1))).astype(int)
        with torch.no_grad():
            for s in starts:
                chunk = mel[:, s:s + _MEL_FRAMES]
                if chunk.shape[1] < _MEL_FRAMES:
                    chunk = np.pad(chunk, ((0, 0), (0, _MEL_FRAMES - chunk.shape[1])), mode="edge")
                x = torch.from_numpy(chunk)[None, None].to(self.device)
                logits, emb = self.model(x)
                embs.append(emb.cpu().numpy()[0])
                logits_acc.append(torch.softmax(logits, dim=1).cpu().numpy()[0])
        emb = np.mean(embs, axis=0)
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        probs = np.mean(logits_acc, axis=0)
        top = int(np.argmax(probs))
        tags = {
            "genre": self.genres[top],
            "genre_confidence": float(probs[top]),
            "genre_probs": {g: float(p) for g, p in zip(self.genres, probs)},
        }
        return emb.astype(np.float32), tags

    # ---- DSP fallback -----------------------------------------------------
    def _dsp_embed(self, y: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=HOP_LENGTH)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=HOP_LENGTH)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)

        parts = [
            mfcc.mean(1), mfcc.std(1),
            contrast.mean(1), chroma.mean(1),
            tonnetz.mean(1),
            [float(centroid.mean())], [float(rolloff.mean())], [float(zcr.mean())],
        ]
        vec = np.concatenate([np.atleast_1d(np.asarray(p, dtype=np.float32)) for p in parts])
        vec = np.nan_to_num(vec)
        # standardise coarsely then L2-normalise so cosine similarity is meaningful
        vec = (vec - vec.mean()) / (vec.std() + 1e-6)
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        tags = {"genre": None, "genre_confidence": 0.0, "genre_probs": {}}
        return vec.astype(np.float32), tags

    @property
    def signature(self) -> str:
        """Identity of the active vibe model for cache invalidation."""
        return f"vibe:{self.mode}:{weight_signature(WEIGHTS_PATH)}"

    # ---- public -----------------------------------------------------------
    def embed(self, y: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        if self.mode == "cnn" and self.model is not None:
            try:
                return self._cnn_embed(y, sr)
            except Exception as e:
                log.warning("CNN embed failed (%s) — DSP fallback", e)
        return self._dsp_embed(y, sr)


_SINGLETON: Optional[VibeModel] = None


def get_vibe_model() -> VibeModel:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = VibeModel()
    return _SINGLETON


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape or a.size == 0:
        return 0.0
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))
