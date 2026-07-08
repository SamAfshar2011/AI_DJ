"""
Musical key / scale detection + Camelot-wheel harmonic-mixing helpers.

Uses the Krumhansl-Schmuckler key-finding algorithm: correlate the track's
average chroma vector against the 24 major/minor key profiles.  Then map to the
Camelot notation DJs use so the planner can reason about harmonic compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import HOP_LENGTH
from .utils import get_logger, safe_float

log = get_logger()

try:
    import librosa
except Exception as e:  # pragma: no cover
    raise RuntimeError("librosa is required for key detection") from e

PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler tonal hierarchy profiles
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Map (pitch, mode) -> Camelot code.  Standard DJ wheel.
_CAMELOT = {
    ("C", "maj"): "8B", ("G", "maj"): "9B", ("D", "maj"): "10B", ("A", "maj"): "11B",
    ("E", "maj"): "12B", ("B", "maj"): "1B", ("F#", "maj"): "2B", ("C#", "maj"): "3B",
    ("G#", "maj"): "4B", ("D#", "maj"): "5B", ("A#", "maj"): "6B", ("F", "maj"): "7B",
    ("A", "min"): "8A", ("E", "min"): "9A", ("B", "min"): "10A", ("F#", "min"): "11A",
    ("C#", "min"): "12A", ("G#", "min"): "1A", ("D#", "min"): "2A", ("A#", "min"): "3A",
    ("F", "min"): "4A", ("C", "min"): "5A", ("G", "min"): "6A", ("D", "min"): "7A",
}


@dataclass
class KeyResult:
    key: str            # e.g. "A"
    mode: str           # "maj" | "min"
    camelot: str        # e.g. "11B"
    confidence: float
    name: str           # human "A minor"

    def to_dict(self) -> dict:
        return {
            "key": self.key, "mode": self.mode, "camelot": self.camelot,
            "confidence": safe_float(self.confidence), "name": self.name,
        }

    @staticmethod
    def from_dict(d: dict) -> "KeyResult":
        return KeyResult(d["key"], d["mode"], d["camelot"], d["confidence"], d["name"])


def detect_key(y: np.ndarray, sr: int) -> KeyResult:
    # CQT chroma is more reliable for key than STFT chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    profile = chroma.mean(axis=1)
    profile = profile / (profile.sum() + 1e-9)

    def corr(a, b):
        a = a - a.mean()
        b = b - b.mean()
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return float(np.dot(a, b) / denom)

    scores = []
    for i in range(12):
        scores.append((corr(profile, np.roll(_MAJOR, i)), PITCHES[i], "maj"))
        scores.append((corr(profile, np.roll(_MINOR, i)), PITCHES[i], "min"))
    scores.sort(key=lambda t: t[0], reverse=True)

    best = scores[0]
    second = scores[1]
    confidence = float(np.clip((best[0] - second[0]) * 3.0 + 0.4, 0.0, 1.0))
    key, mode = best[1], best[2]
    camelot = _CAMELOT.get((key, mode), "?")
    name = f"{key} {'major' if mode == 'maj' else 'minor'}"
    return KeyResult(key=key, mode=mode, camelot=camelot, confidence=confidence, name=name)


# ---------------------------------------------------------------------------
# Harmonic compatibility (Camelot wheel)
# ---------------------------------------------------------------------------
def _parse_camelot(code: str) -> tuple[int, str] | None:
    if not code or code == "?":
        return None
    try:
        return int(code[:-1]), code[-1].upper()
    except (ValueError, IndexError):
        return None


def camelot_compatibility(a: str, b: str) -> float:
    """
    Return a 0..1 harmonic-compatibility score between two Camelot codes.

    1.0  same key
    0.9  relative major/minor (same number, swapped letter)  or ±1 same letter
    0.75 energy-boost / mood shift adjacents
    ...  falling off with distance around the wheel.
    """
    pa, pb = _parse_camelot(a), _parse_camelot(b)
    if pa is None or pb is None:
        return 0.5  # unknown -> neutral
    na, la = pa
    nb, lb = pb
    if na == nb and la == lb:
        return 1.0
    # relative major/minor
    if na == nb and la != lb:
        return 0.9
    # adjacent on the wheel, same mode (classic +/-1 mix)
    ring = min((na - nb) % 12, (nb - na) % 12)
    if la == lb:
        if ring == 1:
            return 0.9
        if ring == 2:
            return 0.72
        if ring == 3:
            return 0.55
        return max(0.2, 0.5 - 0.05 * ring)
    # different mode and different number: only diagonal ±1 stays musical
    if ring == 1:
        return 0.6
    return max(0.15, 0.4 - 0.04 * ring)


def semitone_distance(a: str, b: str) -> int:
    """Approximate semitone shift needed to align key b to key a (for pitch hints)."""
    try:
        ia = PITCHES.index(a)
        ib = PITCHES.index(b)
    except ValueError:
        return 0
    d = (ia - ib) % 12
    return d - 12 if d > 6 else d
