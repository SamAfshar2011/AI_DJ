"""
Mid/high-level descriptive features used by the planner and the transition
engine: loudness (LUFS + RMS curve), spectral balance, an energy curve,
structural segmentation (sections / intro / outro), a vocal-activity proxy and
a danceability-style score.

Everything here is DSP-based and works with zero external model weights, which
keeps the app fully functional out of the box.  The learned vibe embedding lives
separately in vibe_model.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import HOP_LENGTH, N_FFT, N_MELS
from .utils import get_logger, safe_float

log = get_logger()

try:
    import librosa
except Exception as e:  # pragma: no cover
    raise RuntimeError("librosa is required for feature extraction") from e

try:
    import pyloudnorm as pyln

    _HAVE_PYLN = True
except Exception:  # pragma: no cover
    _HAVE_PYLN = False


@dataclass
class TrackFeatures:
    duration: float
    lufs: float                    # integrated loudness estimate
    rms_curve: np.ndarray          # per-frame RMS (dB), for energy display
    rms_times: np.ndarray
    energy: float                  # 0..1 overall energy
    energy_curve: np.ndarray       # smoothed 0..1 energy over time (128 pts)
    spectral_centroid: float       # Hz, "brightness"
    spectral_balance: dict         # low/mid/high fractional energy
    onset_rate: float              # onsets per second
    danceability: float            # 0..1
    vocalness: float               # 0..1 rough vocal-presence proxy
    sections: list                 # list of section start times (s)
    intro_end: float               # s
    outro_start: float             # s
    mix_in_point: float            # good place for the *next* track to start under
    mix_out_point: float           # good place to start leaving this track

    def to_dict(self) -> dict:
        return {
            "duration": safe_float(self.duration),
            "lufs": safe_float(self.lufs),
            "rms_curve": np.round(self.rms_curve, 2).tolist(),
            "rms_times": np.round(self.rms_times, 3).tolist(),
            "energy": safe_float(self.energy),
            "energy_curve": np.round(self.energy_curve, 4).tolist(),
            "spectral_centroid": safe_float(self.spectral_centroid),
            "spectral_balance": {k: safe_float(v) for k, v in self.spectral_balance.items()},
            "onset_rate": safe_float(self.onset_rate),
            "danceability": safe_float(self.danceability),
            "vocalness": safe_float(self.vocalness),
            "sections": [safe_float(s) for s in self.sections],
            "intro_end": safe_float(self.intro_end),
            "outro_start": safe_float(self.outro_start),
            "mix_in_point": safe_float(self.mix_in_point),
            "mix_out_point": safe_float(self.mix_out_point),
        }

    @staticmethod
    def from_dict(d: dict) -> "TrackFeatures":
        return TrackFeatures(
            duration=d["duration"], lufs=d["lufs"],
            rms_curve=np.asarray(d["rms_curve"], dtype=np.float32),
            rms_times=np.asarray(d["rms_times"], dtype=np.float32),
            energy=d["energy"],
            energy_curve=np.asarray(d["energy_curve"], dtype=np.float32),
            spectral_centroid=d["spectral_centroid"],
            spectral_balance=d["spectral_balance"], onset_rate=d["onset_rate"],
            danceability=d["danceability"], vocalness=d["vocalness"],
            sections=list(d["sections"]), intro_end=d["intro_end"],
            outro_start=d["outro_start"],
            mix_in_point=d.get("mix_in_point", d["intro_end"]),
            mix_out_point=d.get("mix_out_point", d["outro_start"]),
        )


def _integrated_lufs(y: np.ndarray, sr: int) -> float:
    if _HAVE_PYLN and len(y) > sr:  # pyln needs >= 400 ms
        try:
            meter = pyln.Meter(sr)
            return float(meter.integrated_loudness(y.astype(np.float64)))
        except Exception:
            pass
    # RMS-based fallback expressed on a LUFS-ish scale
    rms = np.sqrt(np.mean(y.astype(np.float64) ** 2) + 1e-12)
    return float(20.0 * np.log10(rms + 1e-12) - 3.0)


def _spectral_balance(y: np.ndarray, sr: int) -> tuple[dict, float]:
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    total = S.sum() + 1e-9
    low = S[freqs < 250].sum() / total
    mid = S[(freqs >= 250) & (freqs < 4000)].sum() / total
    high = S[freqs >= 4000].sum() / total
    centroid = float(np.mean(librosa.feature.spectral_centroid(S=np.sqrt(S), sr=sr)))
    return {"low": float(low), "mid": float(mid), "high": float(high)}, centroid


def _energy_curve(y: np.ndarray, sr: int, n_points: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-6, ref=np.max)
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=HOP_LENGTH)
    # normalised, smoothed curve resampled to n_points
    e = rms / (np.max(rms) + 1e-9)
    if len(e) >= n_points:
        idx = np.linspace(0, len(e) - 1, n_points).astype(int)
        curve = e[idx]
    else:
        curve = np.interp(np.linspace(0, 1, n_points), np.linspace(0, 1, len(e)), e)
    # light smoothing
    k = 5
    kernel = np.ones(k) / k
    curve = np.convolve(curve, kernel, mode="same")
    return rms_db.astype(np.float32), times.astype(np.float32), curve.astype(np.float32)


def _vocalness(y: np.ndarray, sr: int) -> float:
    """
    Cheap vocal-presence proxy: harmonic-percussive separation, then measure how
    much energy sits in the 300 Hz–3.4 kHz "voice" band of the harmonic part,
    combined with spectral-flatness (voiced content is less flat / noisy).
    """
    try:
        h = librosa.effects.harmonic(y, margin=3.0)
    except Exception:
        h = y
    S = np.abs(librosa.stft(h, n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    band = S[(freqs >= 300) & (freqs <= 3400)].sum() / (S.sum() + 1e-9)
    flat = float(np.mean(librosa.feature.spectral_flatness(y=h)))
    score = 0.7 * band + 0.3 * (1.0 - np.clip(flat * 4, 0, 1))
    return float(np.clip(score, 0.0, 1.0))


def _danceability(y: np.ndarray, sr: int, onset_rate: float) -> float:
    """
    Danceability proxy combining beat strength/regularity with percussive energy.
    Strong, regular beats + prominent percussion -> higher danceability.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    ac = librosa.autocorrelate(onset_env, max_size=4 * sr // HOP_LENGTH)
    ac = librosa.util.normalize(ac + 1e-9)
    # pulse clarity: height of the strongest periodic peak (skip lag 0)
    pulse = float(np.max(ac[4:])) if len(ac) > 4 else 0.0
    try:
        perc = librosa.effects.percussive(y, margin=3.0)
        perc_ratio = float(np.mean(perc ** 2) / (np.mean(y ** 2) + 1e-9))
    except Exception:
        perc_ratio = 0.3
    onset_term = np.clip(onset_rate / 6.0, 0, 1)
    score = 0.5 * pulse + 0.3 * np.clip(perc_ratio, 0, 1) + 0.2 * onset_term
    return float(np.clip(score, 0.0, 1.0))


def _sections(y: np.ndarray, sr: int, duration: float) -> list[float]:
    """
    Structural segmentation via a self-similarity/recurrence-based agglomerative
    segmentation on stacked chroma+MFCC features (librosa's approach).
    """
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH)
        feat = np.vstack([librosa.util.normalize(chroma, axis=0),
                          librosa.util.normalize(mfcc, axis=0)])
        n = max(4, min(10, int(duration // 25)))
        bounds = librosa.segment.agglomerative(feat, n)
        times = librosa.frames_to_time(bounds, sr=sr, hop_length=HOP_LENGTH)
        return sorted({0.0, *[float(t) for t in times if 0 < t < duration]})
    except Exception as e:
        log.debug("segmentation failed: %s", e)
        # fall back to even eighths
        return [duration * i / 8 for i in range(8)]


def _find_mix_points(
    energy_curve: np.ndarray, duration: float, intro_end: float, outro_start: float
) -> tuple[float, float]:
    """
    mix_out: where energy first sustains a downward trend near the end (start
    easing this track out).  mix_in: end of the low-energy intro (where the next
    track can be brought up underneath).  Both are clamped to safe regions.
    """
    n = len(energy_curve)
    t = np.linspace(0, duration, n)
    # mix_out ~ last high-energy plateau before the outro; default outro_start
    mix_out = outro_start
    # mix_in ~ intro end but not more than 25% into the song
    mix_in = min(intro_end, duration * 0.25)
    mix_in = max(mix_in, min(8.0, duration * 0.05))
    return float(mix_in), float(mix_out)


def analyze_features(y: np.ndarray, sr: int, beat_period: float | None = None) -> TrackFeatures:
    duration = len(y) / sr
    lufs = _integrated_lufs(y, sr)
    rms_db, rms_times, energy_curve = _energy_curve(y, sr)
    balance, centroid = _spectral_balance(y, sr)

    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP_LENGTH, units="time")
    onset_rate = len(onsets) / max(duration, 1e-6)

    energy = float(np.clip(np.mean(energy_curve) * 1.2, 0.0, 1.0))
    dance = _danceability(y, sr, onset_rate)
    vocal = _vocalness(y, sr)
    sections = _sections(y, sr, duration)

    # intro/outro from the energy curve: intro = first sustained rise, outro =
    # last sustained fall.
    thresh = 0.55 * float(np.max(energy_curve) + 1e-9)
    above = energy_curve >= thresh
    t_axis = np.linspace(0, duration, len(energy_curve))
    intro_end = float(t_axis[np.argmax(above)]) if above.any() else duration * 0.1
    intro_end = float(np.clip(intro_end, min(4.0, duration * 0.03), duration * 0.35))
    last_above = np.where(above)[0]
    outro_start = float(t_axis[last_above[-1]]) if len(last_above) else duration * 0.9
    outro_start = float(np.clip(outro_start, duration * 0.6, duration - min(4.0, duration * 0.05)))

    mix_in, mix_out = _find_mix_points(energy_curve, duration, intro_end, outro_start)

    return TrackFeatures(
        duration=duration, lufs=lufs, rms_curve=rms_db, rms_times=rms_times,
        energy=energy, energy_curve=energy_curve, spectral_centroid=centroid,
        spectral_balance=balance, onset_rate=onset_rate, danceability=dance,
        vocalness=vocal, sections=sections, intro_end=intro_end,
        outro_start=outro_start, mix_in_point=mix_in, mix_out_point=mix_out,
    )
