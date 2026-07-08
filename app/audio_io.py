"""
Audio input/output.

Everything internal is float32 in [-1, 1].  We keep the *render* path lossless
(soundfile / WAV) and only touch lossy codecs (mp3/m4a/…) through libsndfile +
audioread on the way *in*, and ffmpeg on the way *out*, so we never re-encode a
lossy file more than once.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

try:  # high-quality (VHQ) resampler; falls back to librosa/scipy if missing
    import soxr

    _HAVE_SOXR = True
except Exception:  # pragma: no cover
    _HAVE_SOXR = False

from .config import RENDER_SR
from .utils import get_logger

log = get_logger()

_FFMPEG = shutil.which("ffmpeg")


class AudioLoadError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_with_soundfile(path: str, sr: Optional[int], mono: bool):
    data, file_sr = sf.read(path, dtype="float32", always_2d=True)  # (n, ch)
    y = data.T  # -> (ch, n)
    if mono:
        y = y.mean(axis=0, keepdims=True)
    if sr is not None and sr != file_sr:
        y = resample(y, file_sr, sr)
        file_sr = sr
    return y, file_sr


def _load_with_ffmpeg(path: str, sr: Optional[int], mono: bool):
    """Fallback decoder for formats libsndfile can't open (m4a/aac/wma/…)."""
    if _FFMPEG is None:
        raise AudioLoadError("ffmpeg not available to decode " + path)
    target_sr = sr or RENDER_SR
    ch = 1 if mono else 2
    cmd = [
        _FFMPEG, "-v", "error", "-i", path,
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", str(ch), "-ar", str(target_sr), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise AudioLoadError(proc.stderr.decode("utf-8", "ignore")[:400] or "ffmpeg decode failed")
    raw = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)
    if ch > 1:
        raw = raw.reshape(-1, ch).T  # (ch, n)
    else:
        raw = raw.reshape(1, -1)
    return raw, target_sr


def load_audio(
    path: str | Path,
    sr: Optional[int] = None,
    mono: bool = True,
    max_seconds: Optional[float] = None,
    offset: float = 0.0,
) -> tuple[np.ndarray, int]:
    """
    Load an audio file to a float32 array.

    Returns (y, sr) where y is shape (n,) if mono else (2, n).  Tries libsndfile
    first (fast, lossless for wav/flac/ogg) then ffmpeg for everything else.
    """
    path = str(path)
    try:
        y, out_sr = _load_with_soundfile(path, sr, mono)
    except Exception as e_sf:
        try:
            y, out_sr = _load_with_ffmpeg(path, sr, mono)
        except Exception as e_ff:
            raise AudioLoadError(f"could not decode {Path(path).name}: {e_sf} / {e_ff}")

    if offset > 0:
        start = int(offset * out_sr)
        y = y[:, start:] if y.ndim == 2 else y[start:]
    if max_seconds is not None:
        n = int(max_seconds * out_sr)
        y = y[:, :n] if y.ndim == 2 else y[:n]

    if mono and y.ndim == 2:
        y = y[0]
    # sanitise
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return y, out_sr


def resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """High-quality resample.  Handles (n,) and (ch, n)."""
    if orig_sr == target_sr:
        return y.astype(np.float32, copy=False)
    was_1d = y.ndim == 1
    x = y[None, :] if was_1d else y
    if _HAVE_SOXR:
        out = np.stack([soxr.resample(ch, orig_sr, target_sr, quality="VHQ") for ch in x])
    else:  # pragma: no cover
        import librosa

        out = np.stack([librosa.resample(ch, orig_sr=orig_sr, target_sr=target_sr) for ch in x])
    out = out.astype(np.float32)
    return out[0] if was_1d else out


def to_stereo(y: np.ndarray) -> np.ndarray:
    """Return a (2, n) float32 array from mono or stereo input."""
    if y.ndim == 1:
        return np.stack([y, y]).astype(np.float32)
    if y.shape[0] == 1:
        return np.repeat(y, 2, axis=0).astype(np.float32)
    return y[:2].astype(np.float32)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def save_wav(path: str | Path, y: np.ndarray, sr: int, subtype: str = "PCM_24") -> Path:
    """Write a float array as WAV.  Accepts (n,) or (ch, n)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = y.T if y.ndim == 2 else y  # soundfile wants (n, ch)
    data = np.ascontiguousarray(np.clip(data, -1.0, 1.0), dtype=np.float32)
    sf.write(str(path), data, sr, subtype=subtype)
    return path


def wav_to_mp3(wav_path: str | Path, mp3_path: str | Path, bitrate: str = "320k") -> Optional[Path]:
    """Encode an existing WAV to MP3 via ffmpeg (single, final lossy pass)."""
    if _FFMPEG is None:
        log.warning("ffmpeg missing — cannot produce MP3")
        return None
    wav_path, mp3_path = str(wav_path), str(mp3_path)
    cmd = [
        _FFMPEG, "-y", "-v", "error", "-i", wav_path,
        "-codec:a", "libmp3lame", "-b:a", bitrate, "-q:a", "0", mp3_path,
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        log.warning("mp3 encode failed: %s", proc.stderr.decode("utf-8", "ignore")[:300])
        return None
    return Path(mp3_path)


def probe_duration(path: str | Path) -> Optional[float]:
    """Fast duration probe without fully decoding (soundfile header or ffprobe)."""
    try:
        info = sf.info(str(path))
        if info.frames and info.samplerate:
            return info.frames / info.samplerate
    except Exception:
        pass
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            out = subprocess.run(
                [ffprobe, "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=nk=1:nw=1", str(path)],
                capture_output=True, text=True, timeout=30,
            )
            return float(out.stdout.strip())
        except Exception:
            return None
    return None
