"""
Central configuration: filesystem paths, audio constants and default mix
parameters.  Everything that another module might need to know about "where
things live" or "what the sane defaults are" lives here so behaviour can be
tuned from a single place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parent.parent

APP_DIR: Path = ROOT / "app"
MODELS_DIR: Path = ROOT / "models_weights"
DATASETS_DIR: Path = ROOT / "datasets"
OUTPUTS_DIR: Path = ROOT / "outputs"
CACHE_DIR: Path = ROOT / "cache"
LOGS_DIR: Path = ROOT / "logs"
STATIC_ROOT: Path = ROOT  # index.html / style.css / app.js live at project root

for _p in (MODELS_DIR, DATASETS_DIR, OUTPUTS_DIR, CACHE_DIR, LOGS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Audio engine constants
# ---------------------------------------------------------------------------
# We analyse at a modest rate for speed, but *render* at full quality.
ANALYSIS_SR: int = 22050          # librosa-friendly analysis rate
RENDER_SR: int = 44100            # CD-quality render / output rate
HOP_LENGTH: int = 512             # analysis hop (≈23 ms @ 22.05 kHz)
N_FFT: int = 2048
N_MELS: int = 128

SUPPORTED_EXTS: tuple[str, ...] = (
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".aif",
)

# Loudness / safety
TARGET_LUFS: float = -14.0        # streaming-style integrated target for the mix
TRUE_PEAK_CEILING_DB: float = -1.0  # leave ~1 dB of true-peak headroom
LIMITER_CEILING_DB: float = -0.3    # sample-peak ceiling before dithered export

# Analysis limits (keep memory/time bounded on huge libraries)
MAX_ANALYSIS_SECONDS: float = 600.0   # only analyse first 10 min of very long files
MIN_TRACK_SECONDS: float = 20.0       # tracks shorter than this can't be mixed well

CACHE_VERSION: int = 4  # bump to invalidate all cached analyses (v4: mood/dance models)


# ---------------------------------------------------------------------------
# Mix planning defaults (exposed through the UI "advanced settings" panel)
# ---------------------------------------------------------------------------
@dataclass
class MixSettings:
    """User-tunable knobs for a single mix generation request."""

    target_minutes: float = 0.0          # 0 == use all tracks
    transition_intensity: float = 0.55   # 0..1  (short EQ cut  ->  long blended)
    effect_intensity: float = 0.45       # 0..1  amount of tasteful FX
    harmonic_priority: float = 0.6       # 0..1  weight of key compatibility
    energy_curve: str = "rising"         # rising|wave|peak|flat|descending
    preserve_quality: bool = True        # avoid aggressive time-stretch
    aggressive: bool = False             # allow bolder cuts / bigger tempo pulls
    max_tempo_stretch: float = 0.08      # ±8 % default; raised in aggressive mode
    crossfade_beats: int = 32            # nominal transition length in beats
    output_format: str = "both"          # wav|mp3|both
    normalize: bool = True

    def resolved(self) -> "MixSettings":
        """Return a copy with mode-dependent limits applied."""
        s = MixSettings(**asdict(self))
        if s.aggressive:
            s.max_tempo_stretch = max(s.max_tempo_stretch, 0.12)
        if s.preserve_quality:
            s.max_tempo_stretch = min(s.max_tempo_stretch, 0.06)
        s.transition_intensity = float(min(1.0, max(0.0, s.transition_intensity)))
        s.effect_intensity = float(min(1.0, max(0.0, s.effect_intensity)))
        s.harmonic_priority = float(min(1.0, max(0.0, s.harmonic_priority)))
        if s.energy_curve not in ("rising", "wave", "peak", "flat", "descending"):
            s.energy_curve = "rising"
        return s

    @classmethod
    def from_dict(cls, d: dict | None) -> "MixSettings":
        d = d or {}
        known = {f: d[f] for f in cls.__dataclass_fields__ if f in d}  # type: ignore[attr-defined]
        return cls(**known).resolved()


DEFAULT_SETTINGS = MixSettings()
