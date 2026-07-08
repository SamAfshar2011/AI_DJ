"""
Runtime wrapper for **model 3 — the Ballroom danceability / rhythm-style CNN**.

Trained by ``train_models/model_3_danceability.ipynb`` into
``models_weights/model_3_danceability.pt``.  That notebook reuses the exact
``GenreCNN`` architecture with ``n_classes = len(STYLES)`` and stores, alongside
the weights, a ``dance_weights`` vector mapping each style to how "danceable" it
is.  Inference danceability is::

    danceability = softmax(style_logits) · dance_weights        # 0..1

The analyzer blends this model score with the existing DSP pulse-clarity proxy in
``features.py`` (or falls back to DSP alone when the model is unavailable), so the
danceability number only ever improves and never breaks.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .config import MODELS_DIR
from .utils import get_logger, weight_signature
from .vibe_model import mel_windows

log = get_logger()

try:
    import torch

    _HAVE_TORCH = True
    from .vibe_model import GenreCNN  # same architecture the notebook trained
except Exception:  # pragma: no cover
    _HAVE_TORCH = False

DANCE_WEIGHTS_PATH = MODELS_DIR / "model_3_danceability.pt"


class DanceModel:
    """Load once, reuse for the whole library."""

    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.status = "fallback"          # "loaded" | "fallback"
        self.styles: list[str] = []
        self.dance_weights: Optional[np.ndarray] = None
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
        if not (_HAVE_TORCH and DANCE_WEIGHTS_PATH.exists()):
            log.info("Dance model: fallback to DSP score (no weights at %s)",
                     DANCE_WEIGHTS_PATH.name)
            return
        try:
            ckpt = torch.load(DANCE_WEIGHTS_PATH, map_location="cpu")
            state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            styles = list(ckpt.get("styles", [])) if isinstance(ckpt, dict) else []
            dvec = ckpt.get("dance_weights") if isinstance(ckpt, dict) else None
            if not styles:
                n = state["classifier.weight"].shape[0]
                styles = [f"style_{i}" for i in range(n)]
            if dvec is None:
                dvec = [0.6] * len(styles)   # neutral if the map wasn't stored
            model = GenreCNN(n_classes=len(styles))
            model.load_state_dict(state)
            model.eval().to(self.device)
            self.model = model
            self.styles = styles
            self.dance_weights = np.asarray(dvec, dtype=np.float32)
            self.status = "loaded"
            log.info("Dance model: loaded GenreCNN (%d styles) on %s",
                     len(styles), self.device)
        except Exception as e:  # noqa: BLE001
            log.warning("Dance model load failed (%s) — DSP fallback", e)
            self.model = None
            self.status = "fallback"

    @property
    def available(self) -> bool:
        return self.status == "loaded" and self.model is not None and self.dance_weights is not None

    @property
    def signature(self) -> str:
        return f"dance:{self.status}:{weight_signature(DANCE_WEIGHTS_PATH)}"

    def score(self, y: np.ndarray, sr: int) -> Optional[float]:
        """
        Return a 0..1 danceability score, or ``None`` when the model is
        unavailable (caller keeps the DSP score).  Also returns the winning
        rhythm style via ``last_style`` for optional display.
        """
        if not self.available:
            return None
        try:
            wins = mel_windows(y, sr)
            x = torch.from_numpy(wins)[:, None].to(self.device)
            with torch.no_grad():
                logits, _ = self.model(x)
                probs = torch.softmax(logits, dim=1).mean(dim=0).cpu().numpy()
            self.last_style = self.styles[int(np.argmax(probs))]
            score = float(np.dot(probs, self.dance_weights))
            return float(np.clip(score, 0.0, 1.0))
        except Exception as e:  # noqa: BLE001
            log.warning("Dance score failed (%s) — DSP fallback", e)
            return None


_SINGLETON: Optional[DanceModel] = None


def get_dance_model() -> DanceModel:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = DanceModel()
    return _SINGLETON
