"""
Runtime wrapper for **model 2 — the MagnaTagATune mood / instrument auto-tagger**.

Trained by ``train_models/model_2_mood_tagger.ipynb`` into
``models_weights/model_2_mood_tagger.pt``.  This module mirrors that notebook's
``Tagger`` architecture *exactly* (same submodule names → the trained
``state_dict`` loads without surgery) and exposes tags to the analyzer.

Tiering, identical in spirit to ``vibe_model.py``:

  * **loaded**  — weights present and valid → per-track multi-label tag
    probabilities (mood words like *calm / happy / dark / upbeat* plus
    instrument / genre words).
  * **fallback** — weights missing or fail to load → no tags, everything else in
    the app keeps working exactly as before.

The wrapper never raises into the pipeline; any failure degrades to "fallback".
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .config import MODELS_DIR
from .utils import get_logger, weight_signature
from .vibe_model import EMBED_DIM, mel_windows

log = get_logger()

try:
    import torch
    import torch.nn as nn

    _HAVE_TORCH = True
    from .vibe_model import GenreCNN  # architecture trunk, shared in one place
except Exception:  # pragma: no cover
    _HAVE_TORCH = False

MOOD_WEIGHTS_PATH = MODELS_DIR / "model_2_mood_tagger.pt"

# tag probability >= this counts as "present"; we also always cap the count
TAG_THRESHOLD = 0.40
MAX_TAGS = 6

# MTAT tags that describe *mood / feel* (as opposed to instrument / genre).  Used
# only to split the predicted tags into a friendly "moods" list for the UI.
MOOD_WORDS = {
    "happy", "sad", "calm", "quiet", "loud", "soft", "mellow", "dark", "eerie",
    "spacey", "ambient", "upbeat", "chill", "airy", "weird", "strange", "hard",
    "heavy", "light", "deep", "fast", "slow", "funky", "epic", "dreamy", "angry",
}


# ---------------------------------------------------------------------------
# architecture — must match train_models/model_2_mood_tagger.ipynb exactly
# ---------------------------------------------------------------------------
if _HAVE_TORCH:

    class MoodTaggerNet(nn.Module):
        """GenreCNN trunk + a multi-label sigmoid head (BCE-trained)."""

        def __init__(self, n_tags: int = 50):
            super().__init__()
            base = GenreCNN(n_classes=n_tags)
            self.features, self.pool, self.embed = base.features, base.pool, base.embed
            self.head = nn.Sequential(nn.ReLU(), nn.Dropout(0.3), nn.Linear(EMBED_DIM, n_tags))

        def forward(self, x):
            h = self.pool(self.features(x)).flatten(1)
            return self.head(self.embed(h))


class MoodModel:
    """Load once, reuse for the whole library."""

    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.status = "fallback"          # "loaded" | "fallback"
        self.tags: list[str] = []
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
        if not (_HAVE_TORCH and MOOD_WEIGHTS_PATH.exists()):
            log.info("Mood model: fallback (no weights at %s)", MOOD_WEIGHTS_PATH.name)
            return
        try:
            ckpt = torch.load(MOOD_WEIGHTS_PATH, map_location="cpu")
            tags = list(ckpt.get("tags", [])) if isinstance(ckpt, dict) else []
            state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            if not tags:
                # infer count from the head weight if the tag list is absent
                n = state["head.2.weight"].shape[0]
                tags = [f"tag_{i}" for i in range(n)]
            model = MoodTaggerNet(n_tags=len(tags))
            model.load_state_dict(state)
            model.eval().to(self.device)
            self.model = model
            self.tags = tags
            self.status = "loaded"
            log.info("Mood model: loaded MoodTaggerNet (%d tags) on %s",
                     len(tags), self.device)
        except Exception as e:  # noqa: BLE001
            log.warning("Mood model load failed (%s) — fallback", e)
            self.model = None
            self.status = "fallback"

    @property
    def available(self) -> bool:
        return self.status == "loaded" and self.model is not None

    @property
    def signature(self) -> str:
        return f"mood:{self.status}:{weight_signature(MOOD_WEIGHTS_PATH)}"

    def predict(self, y: np.ndarray, sr: int) -> dict:
        """
        Return {} when unavailable, else a dict merged into ``vibe_tags``:
          {
            "tags":       [top predicted tags, strongest first],
            "moods":      [subset that are mood words],
            "tag_probs":  {tag: prob, ...}   # only the returned tags
            "mood_vec":   [p0, p1, ...]      # full aligned probability vector
          }
        The full ``mood_vec`` lets the planner measure mood similarity between any
        two tracks (same tag ordering across the whole library).
        """
        if not self.available:
            return {}
        try:
            wins = mel_windows(y, sr)                        # (nwin, mels, frames)
            x = torch.from_numpy(wins)[:, None].to(self.device)
            with torch.no_grad():
                probs = torch.sigmoid(self.model(x)).mean(dim=0).cpu().numpy()
            order = np.argsort(probs)[::-1]
            chosen = [i for i in order if probs[i] >= TAG_THRESHOLD][:MAX_TAGS]
            if not chosen:                                  # always surface the top one
                chosen = [int(order[0])]
            tags = [self.tags[i] for i in chosen]
            return {
                "tags": tags,
                "moods": [t for t in tags if t in MOOD_WORDS],
                "tag_probs": {self.tags[i]: round(float(probs[i]), 3) for i in chosen},
                "mood_vec": [round(float(p), 4) for p in probs],
            }
        except Exception as e:  # noqa: BLE001
            log.warning("Mood predict failed (%s) — skipping tags", e)
            return {}


_SINGLETON: Optional[MoodModel] = None


def get_mood_model() -> MoodModel:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = MoodModel()
    return _SINGLETON
