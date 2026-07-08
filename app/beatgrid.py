"""
Tempo / beat / downbeat / phrase estimation.

Primary engine is librosa (dynamic-programming beat tracker + PLP for a robust
global tempo).  On top of the raw beats we estimate:

  * a stable global BPM (with octave-error correction toward the dance range),
  * downbeats via a spectral-flux-per-beat autocorrelation (assume 4/4),
  * 8/16-beat *phrase* boundaries which are what DJs actually mix on.

No madmom dependency required, but if madmom is importable we use its
downbeat tracker for higher accuracy (graceful optional upgrade).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import ANALYSIS_SR, HOP_LENGTH
from .utils import get_logger, safe_float

log = get_logger()

try:
    import librosa
except Exception as e:  # pragma: no cover - librosa is required
    raise RuntimeError("librosa is required for beat analysis") from e

_HAVE_MADMOM = False
try:  # optional accuracy upgrade
    import madmom  # type: ignore  # noqa: F401

    _HAVE_MADMOM = True
except Exception:
    _HAVE_MADMOM = False


@dataclass
class BeatGrid:
    bpm: float
    beat_times: np.ndarray            # seconds
    downbeat_times: np.ndarray        # seconds (bar starts, assume 4/4)
    beats_per_bar: int
    phrase_times: np.ndarray          # seconds (musical phrase starts, ~8 bars)
    beat_period: float                # seconds per beat
    confidence: float

    def to_dict(self) -> dict:
        return {
            "bpm": safe_float(self.bpm),
            "beat_times": self.beat_times.tolist(),
            "downbeat_times": self.downbeat_times.tolist(),
            "beats_per_bar": int(self.beats_per_bar),
            "phrase_times": self.phrase_times.tolist(),
            "beat_period": safe_float(self.beat_period),
            "confidence": safe_float(self.confidence),
        }

    @staticmethod
    def from_dict(d: dict) -> "BeatGrid":
        return BeatGrid(
            bpm=d["bpm"],
            beat_times=np.asarray(d["beat_times"], dtype=np.float64),
            downbeat_times=np.asarray(d["downbeat_times"], dtype=np.float64),
            beats_per_bar=int(d.get("beats_per_bar", 4)),
            phrase_times=np.asarray(d["phrase_times"], dtype=np.float64),
            beat_period=d["beat_period"],
            confidence=d["confidence"],
        )


def _fold_tempo(bpm: float, lo: float = 70.0, hi: float = 180.0) -> float:
    """Fold an estimated tempo into a sensible dance range to fight octave errors."""
    if bpm <= 0:
        return 120.0
    while bpm < lo:
        bpm *= 2.0
    while bpm > hi:
        bpm /= 2.0
    return bpm


def estimate_downbeats(
    y: np.ndarray, sr: int, beat_frames: np.ndarray, beats_per_bar: int = 4
) -> tuple[np.ndarray, int]:
    """
    Pick the bar phase (which of every `beats_per_bar` beats is the downbeat) by
    scoring the low-band onset strength at each beat and choosing the phase whose
    beats carry the most energy — kick drums usually land on beat 1.
    """
    if len(beat_frames) < beats_per_bar:
        return np.array([]), beats_per_bar

    # low-band (bass/kick) onset envelope
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=HOP_LENGTH)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    low = S[freqs < 200].sum(axis=0)
    low = librosa.util.normalize(low + 1e-9)
    onset = librosa.onset.onset_strength(sr=sr, S=librosa.power_to_db(S + 1e-9),
                                         hop_length=HOP_LENGTH)

    beat_frames = np.clip(beat_frames, 0, len(onset) - 1)
    beat_low = low[np.clip(beat_frames, 0, len(low) - 1)]
    beat_on = onset[beat_frames]
    score_per_beat = 0.6 * beat_low + 0.4 * librosa.util.normalize(beat_on + 1e-9)

    best_phase, best_score = 0, -1.0
    for phase in range(beats_per_bar):
        idx = np.arange(phase, len(beat_frames), beats_per_bar)
        if len(idx) == 0:
            continue
        s = float(np.mean(score_per_beat[idx]))
        if s > best_score:
            best_score, best_phase = s, phase

    down_idx = np.arange(best_phase, len(beat_frames), beats_per_bar)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)
    return beat_times[down_idx], beats_per_bar


def _madmom_downbeats(path: str) -> np.ndarray | None:  # pragma: no cover - optional
    try:
        from madmom.features.downbeats import (
            RNNDownBeatProcessor,
            DBNDownBeatTrackingProcessor,
        )

        act = RNNDownBeatProcessor()(path)
        proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
        beats = proc(act)  # (n, 2) -> time, beat-in-bar
        downs = beats[beats[:, 1] == 1][:, 0]
        return downs
    except Exception as e:
        log.debug("madmom downbeat failed: %s", e)
        return None


def analyze_beatgrid(y: np.ndarray, sr: int = ANALYSIS_SR, path: str | None = None) -> BeatGrid:
    """Full beat-grid analysis for a mono analysis-rate signal."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)

    # robust global tempo via PLP-backed estimate + the beat tracker's own tempo
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH, trim=False
    )
    tempo = float(np.atleast_1d(tempo)[0])
    tempo = _fold_tempo(tempo)

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)

    # confidence: how regular are the inter-beat intervals?
    if len(beat_times) > 4:
        ibi = np.diff(beat_times)
        conf = float(np.clip(1.0 - (np.std(ibi) / (np.mean(ibi) + 1e-9)), 0.0, 1.0))
        # refine bpm from the median inter-beat interval
        med_ibi = float(np.median(ibi))
        if med_ibi > 0:
            tempo = _fold_tempo(60.0 / med_ibi)
    else:
        conf = 0.2

    beats_per_bar = 4
    downbeats = None
    if _HAVE_MADMOM and path is not None:
        downbeats = _madmom_downbeats(path)
    if downbeats is None or len(downbeats) < 2:
        downbeats, beats_per_bar = estimate_downbeats(y, sr, beat_frames, beats_per_bar)

    # phrases: DJs cut on 8-bar (32-beat) boundaries; anchor on downbeats
    if len(downbeats) >= 2:
        bars_per_phrase = 8
        phrase_times = downbeats[::bars_per_phrase]
    elif len(beat_times) >= 2:
        phrase_times = beat_times[::32]
    else:
        phrase_times = beat_times

    beat_period = 60.0 / tempo if tempo > 0 else 0.5
    return BeatGrid(
        bpm=tempo,
        beat_times=beat_times,
        downbeat_times=np.asarray(downbeats, dtype=np.float64),
        beats_per_bar=beats_per_bar,
        phrase_times=np.asarray(phrase_times, dtype=np.float64),
        beat_period=beat_period,
        confidence=conf,
    )


def nearest_beat(beat_times: np.ndarray, t: float) -> float:
    if len(beat_times) == 0:
        return t
    idx = int(np.argmin(np.abs(beat_times - t)))
    return float(beat_times[idx])


def nearest_phrase(phrase_times: np.ndarray, t: float, prefer: str = "nearest") -> float:
    """Snap a time to the nearest / previous / next phrase boundary."""
    if len(phrase_times) == 0:
        return t
    if prefer == "before":
        cand = phrase_times[phrase_times <= t + 1e-6]
        return float(cand[-1]) if len(cand) else float(phrase_times[0])
    if prefer == "after":
        cand = phrase_times[phrase_times >= t - 1e-6]
        return float(cand[0]) if len(cand) else float(phrase_times[-1])
    idx = int(np.argmin(np.abs(phrase_times - t)))
    return float(phrase_times[idx])
