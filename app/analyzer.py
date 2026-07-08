"""
Per-track analysis orchestrator + on-disk cache.

Given a file path, produce a `TrackAnalysis` bundling metadata, beat grid, key,
features and a vibe embedding.  Results are cached in cache/<fingerprint>.json
keyed by (path,size,mtime) + CACHE_VERSION so unchanged files are never
re-analysed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from . import beatgrid as bg
from . import features as feat
from . import key_detection as keydet
from . import metadata as meta_mod
from .audio_io import load_audio, AudioLoadError
from .config import (
    ANALYSIS_SR,
    CACHE_DIR,
    CACHE_VERSION,
    MAX_ANALYSIS_SECONDS,
    MIN_TRACK_SECONDS,
)
from .dance_model import get_dance_model
from .mood_model import get_mood_model
from .utils import dump_json, file_fingerprint, get_logger, load_json, safe_float
from .vibe_model import get_vibe_model

log = get_logger()


def active_model_signature() -> str:
    """
    Combined identity of every ML model currently active (vibe + mood + dance).

    Written into each cached analysis; on cache read a mismatch forces a refresh.
    This is what makes trained weights take effect automatically: training or
    retraining any model changes its weight fingerprint → the signature changes →
    affected tracks are re-analysed with no manual cache clearing.
    """
    return "|".join([
        get_vibe_model().signature,
        get_mood_model().signature,
        get_dance_model().signature,
    ])


def model_status() -> dict:
    """Human-readable status of each model — surfaced in logs, /api/health, report."""
    return {
        "vibe": get_vibe_model().mode,        # "cnn" | "dsp"
        "mood": get_mood_model().status,      # "loaded" | "fallback"
        "dance": get_dance_model().status,    # "loaded" | "fallback"
    }


@dataclass
class TrackAnalysis:
    path: str
    fingerprint: str
    metadata: dict
    beatgrid: bg.BeatGrid
    key: keydet.KeyResult
    features: feat.TrackFeatures
    embedding: np.ndarray
    vibe_tags: dict
    ok: bool = True
    error: Optional[str] = None

    # convenience accessors -------------------------------------------------
    @property
    def title(self) -> str:
        return self.metadata.get("title") or Path(self.path).stem

    @property
    def artist(self) -> str:
        return self.metadata.get("artist") or "Unknown"

    @property
    def bpm(self) -> float:
        return self.beatgrid.bpm

    @property
    def duration(self) -> float:
        return self.features.duration

    def to_dict(self) -> dict:
        return {
            "version": CACHE_VERSION,
            "model_signature": active_model_signature(),
            "path": self.path,
            "fingerprint": self.fingerprint,
            "metadata": self.metadata,
            "beatgrid": self.beatgrid.to_dict(),
            "key": self.key.to_dict(),
            "features": self.features.to_dict(),
            "embedding": np.asarray(self.embedding, dtype=np.float32).tolist(),
            "vibe_tags": self.vibe_tags,
            "ok": self.ok,
            "error": self.error,
        }

    @staticmethod
    def from_dict(d: dict) -> "TrackAnalysis":
        return TrackAnalysis(
            path=d["path"],
            fingerprint=d["fingerprint"],
            metadata=d["metadata"],
            beatgrid=bg.BeatGrid.from_dict(d["beatgrid"]),
            key=keydet.KeyResult.from_dict(d["key"]),
            features=feat.TrackFeatures.from_dict(d["features"]),
            embedding=np.asarray(d["embedding"], dtype=np.float32),
            vibe_tags=d.get("vibe_tags", {}),
            ok=d.get("ok", True),
            error=d.get("error"),
        )

    def summary(self) -> dict:
        """Compact dict for the UI track list."""
        f = self.features
        return {
            "path": self.path,
            "title": self.title,
            "artist": self.artist,
            "duration": safe_float(self.duration),
            "bpm": round(safe_float(self.bpm), 1),
            "key": self.key.name,
            "camelot": self.key.camelot,
            "energy": round(safe_float(f.energy), 3),
            "danceability": round(safe_float(f.danceability), 3),
            "vocalness": round(safe_float(f.vocalness), 3),
            "lufs": round(safe_float(f.lufs), 1),
            "genre": self.vibe_tags.get("genre"),
            "moods": self.vibe_tags.get("moods", []),
            "mood_tags": self.vibe_tags.get("mood_tags", []),
            "dance_style": self.vibe_tags.get("dance_style"),
            "energy_curve": np.round(f.energy_curve[::4], 3).tolist(),  # decimated for UI
            "sources": {                       # which engine produced each signal
                "vibe": self.vibe_tags.get("vibe_source"),
                "mood": self.vibe_tags.get("mood_source"),
                "dance": self.vibe_tags.get("dance_source"),
            },
            "ok": self.ok,
            "error": self.error,
        }


def _cache_path(fingerprint: str) -> Path:
    return CACHE_DIR / f"{fingerprint}.json"


def _load_cached(fingerprint: str) -> Optional[TrackAnalysis]:
    cp = _cache_path(fingerprint)
    if not cp.exists():
        return None
    try:
        d = load_json(cp)
        if d.get("version") != CACHE_VERSION:
            return None
        # If any model was (re)trained/added/removed since this was cached, its
        # signature changes and we transparently re-analyse — no manual cache clear.
        if d.get("model_signature") != active_model_signature():
            log.info("cache stale (model set changed): %s", Path(d.get("path", cp)).name)
            return None
        return TrackAnalysis.from_dict(d)
    except Exception as e:
        log.debug("cache read failed for %s: %s", fingerprint, e)
        return None


def analyze_track(path: str, use_cache: bool = True) -> TrackAnalysis:
    """Analyse a single track (cached).  Raises AudioLoadError on hard failure."""
    fp = file_fingerprint(path)
    if use_cache:
        cached = _load_cached(fp)
        if cached is not None:
            log.info("cache hit: %s", Path(path).name)
            return cached

    log.info("analyzing: %s", Path(path).name)
    metadata = meta_mod.extract_metadata(path)

    y, sr = load_audio(path, sr=ANALYSIS_SR, mono=True, max_seconds=MAX_ANALYSIS_SECONDS)
    if y.size == 0:
        raise AudioLoadError(f"empty audio: {Path(path).name}")
    duration = len(y) / sr
    if duration < MIN_TRACK_SECONDS:
        raise AudioLoadError(f"too short after decode ({duration:.0f}s): {Path(path).name}")

    grid = bg.analyze_beatgrid(y, sr, path=path)
    key = keydet.detect_key(y, sr)
    features = feat.analyze_features(y, sr, beat_period=grid.beat_period)

    # --- model 1: vibe embedding + genre (CNN if trained, else DSP) ----------
    vibe = get_vibe_model()
    embedding, tags = vibe.embed(y, sr)
    tags = dict(tags)                       # copy so we can enrich in place
    tags["vibe_source"] = vibe.mode         # "cnn" | "dsp"

    # --- model 2: mood / instrument auto-tags (no-op if weights absent) -------
    mood = get_mood_model()
    mood_out = mood.predict(y, sr)          # {} when unavailable
    if mood_out:
        tags["mood_tags"] = mood_out.get("tags", [])
        tags["moods"] = mood_out.get("moods", [])
        tags["mood_probs"] = mood_out.get("tag_probs", {})
        if mood_out.get("mood_vec") is not None:
            tags["mood_vec"] = mood_out["mood_vec"]   # full aligned vector for planner
    tags["mood_source"] = mood.status       # "loaded" | "fallback"

    # --- model 3: danceability — blend model score with the DSP proxy ---------
    dance = get_dance_model()
    dsp_dance = float(features.danceability)
    model_dance = dance.score(y, sr)        # None when unavailable
    if model_dance is not None:
        features.danceability = float(np.clip(0.5 * dsp_dance + 0.5 * model_dance, 0.0, 1.0))
        tags["danceability_dsp"] = round(dsp_dance, 3)
        tags["danceability_model"] = round(float(model_dance), 3)
        tags["dance_style"] = getattr(dance, "last_style", None)
        tags["dance_source"] = "model+dsp"
    else:
        tags["dance_source"] = "dsp"

    analysis = TrackAnalysis(
        path=path, fingerprint=fp, metadata=metadata, beatgrid=grid, key=key,
        features=features, embedding=embedding, vibe_tags=tags, ok=True,
    )
    try:
        dump_json(_cache_path(fp), analysis.to_dict())
    except Exception as e:
        log.debug("cache write failed: %s", e)
    return analysis
