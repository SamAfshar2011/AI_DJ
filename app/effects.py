"""
DSP effects toolbox.  Everything operates on float32 stereo arrays shaped
(2, n) and is written to be *transient-safe* and *headroom-aware*: effects here
should improve a mix, never wreck it, so gains are conservative and outputs are
sanitised.

Contents:
  * equal-power fades / crossfades
  * static and time-varying (swept) biquad filters  (low/high/band)
  * 3-band shelving EQ with dB gains
  * tempo-synced echo / delay throw
  * short reverb tail (synthetic impulse response)
  * gentle tanh saturation
  * stereo-width control (mid/side) with mono-safety
  * volume automation from a breakpoint envelope
  * a look-ahead soft limiter + peak-safety utilities
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .utils import get_logger

log = get_logger()

EPS = 1e-9


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _stereo(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return np.stack([x, x]).astype(np.float32)
    return x.astype(np.float32)


def sanitize(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def equal_power_fades(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (fade_out, fade_in) equal-power curves of length n (constant power sum)."""
    t = np.linspace(0.0, 1.0, max(2, n), dtype=np.float32)
    fade_in = np.sin(t * np.pi / 2.0)
    fade_out = np.cos(t * np.pi / 2.0)
    return fade_out, fade_in


def apply_gain_envelope(x: np.ndarray, env: np.ndarray) -> np.ndarray:
    """Multiply a (2,n) signal by a length-n gain envelope."""
    x = _stereo(x)
    env = np.asarray(env, dtype=np.float32)
    if len(env) != x.shape[1]:
        env = np.interp(np.linspace(0, 1, x.shape[1]),
                        np.linspace(0, 1, len(env)), env).astype(np.float32)
    return sanitize(x * env[None, :])


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------
def _sos(kind: str, cutoff, sr: int, order: int = 4):
    nyq = sr / 2.0
    if kind in ("low", "high"):
        wn = np.clip(cutoff / nyq, 1e-4, 0.999)
        return signal.butter(order, wn, btype=kind, output="sos")
    if kind == "band":
        lo, hi = cutoff
        wn = [np.clip(lo / nyq, 1e-4, 0.998), np.clip(hi / nyq, 2e-4, 0.999)]
        return signal.butter(order, wn, btype="bandpass", output="sos")
    raise ValueError(kind)


def static_filter(x: np.ndarray, kind: str, cutoff, sr: int, order: int = 4) -> np.ndarray:
    x = _stereo(x)
    sos = _sos(kind, cutoff, sr, order)
    out = signal.sosfiltfilt(sos, x, axis=1)  # zero-phase -> no smearing
    return sanitize(out.astype(np.float32))


def sweep_filter(
    x: np.ndarray, kind: str, f_start: float, f_end: float, sr: int,
    order: int = 4, block: int = 1024,
) -> np.ndarray:
    """
    Time-varying low/high-pass sweep.  Processed in blocks with per-block Butter
    coefficients and carried filter state (zi) so there are no clicks.  Ideal for
    the classic DJ filter-sweep transition.
    """
    x = _stereo(x)
    n = x.shape[1]
    out = np.zeros_like(x)
    nyq = sr / 2.0
    # geometric cutoff trajectory (log-frequency feels natural to the ear)
    f_start = max(20.0, min(f_start, nyq * 0.98))
    f_end = max(20.0, min(f_end, nyq * 0.98))
    n_blocks = max(1, int(np.ceil(n / block)))
    freqs = np.geomspace(f_start, f_end, n_blocks)
    zi = [None, None]
    for bi in range(n_blocks):
        s, e = bi * block, min(n, (bi + 1) * block)
        sos = _sos(kind, freqs[bi], sr, order)
        for ch in range(2):
            if zi[ch] is None:
                zi[ch] = signal.sosfilt_zi(sos) * x[ch, s]
            y, zi[ch] = signal.sosfilt(sos, x[ch, s:e], zi=zi[ch])
            out[ch, s:e] = y
    return sanitize(out.astype(np.float32))


# ---------------------------------------------------------------------------
# 3-band shelving EQ (DJ-style low / mid / high)
# ---------------------------------------------------------------------------
def _peaking(f0: float, gain_db: float, q: float, sr: int):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2 * q)
    cos = np.cos(w0)
    b = [1 + alpha * A, -2 * cos, 1 - alpha * A]
    a = [1 + alpha / A, -2 * cos, 1 - alpha / A]
    return np.array(b) / a[0], np.array(a) / a[0]


def _shelf(f0: float, gain_db: float, sr: int, kind: str, s: float = 0.9):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / sr
    cos, sin = np.cos(w0), np.sin(w0)
    alpha = sin / 2 * np.sqrt((A + 1 / A) * (1 / s - 1) + 2)
    tsa = 2 * np.sqrt(A) * alpha
    if kind == "low":
        b = [A * ((A + 1) - (A - 1) * cos + tsa),
             2 * A * ((A - 1) - (A + 1) * cos),
             A * ((A + 1) - (A - 1) * cos - tsa)]
        a = [(A + 1) + (A - 1) * cos + tsa,
             -2 * ((A - 1) + (A + 1) * cos),
             (A + 1) + (A - 1) * cos - tsa]
    else:  # high
        b = [A * ((A + 1) + (A - 1) * cos + tsa),
             -2 * A * ((A - 1) + (A + 1) * cos),
             A * ((A + 1) + (A - 1) * cos - tsa)]
        a = [(A + 1) - (A - 1) * cos + tsa,
             2 * ((A - 1) - (A + 1) * cos),
             (A + 1) - (A - 1) * cos - tsa]
    b, a = np.array(b), np.array(a)
    return b / a[0], a / a[0]


def eq3(x: np.ndarray, sr: int, low_db=0.0, mid_db=0.0, high_db=0.0,
        low_f=180.0, mid_f=1200.0, high_f=5000.0) -> np.ndarray:
    """Static 3-band EQ.  dB gains; negative values cut (e.g. -24 ≈ bass kill)."""
    x = _stereo(x)
    out = x
    for (kind, f0, g, q) in (
        ("lowshelf", low_f, low_db, None),
        ("peak", mid_f, mid_db, 0.9),
        ("highshelf", high_f, high_db, None),
    ):
        if abs(g) < 0.05:
            continue
        if kind == "peak":
            b, a = _peaking(f0, g, q, sr)
        else:
            b, a = _shelf(f0, g, sr, "low" if kind == "lowshelf" else "high")
        out = signal.lfilter(b, a, out, axis=1)
    return sanitize(out.astype(np.float32))


def eq3_envelope(x: np.ndarray, sr: int, low_env, mid_env, high_env,
                 low_f=180.0, mid_f=1200.0, high_f=5000.0, block: int = 2048) -> np.ndarray:
    """
    Time-varying 3-band EQ: low/mid/high dB automation as length-n (or shorter)
    envelopes.  Used by the EQ engine to morph spectra across a transition.
    """
    x = _stereo(x)
    n = x.shape[1]

    def _resamp(env):
        env = np.asarray(env, dtype=np.float32)
        if len(env) == n:
            return env
        return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(env)), env).astype(np.float32)

    le, me, he = _resamp(low_env), _resamp(mid_env), _resamp(high_env)
    out = np.zeros_like(x)
    n_blocks = max(1, int(np.ceil(n / block)))
    for bi in range(n_blocks):
        s, e = bi * block, min(n, (bi + 1) * block)
        seg = eq3(x[:, s:e], sr,
                  low_db=float(le[s:e].mean()),
                  mid_db=float(me[s:e].mean()),
                  high_db=float(he[s:e].mean()),
                  low_f=low_f, mid_f=mid_f, high_f=high_f)
        out[:, s:e] = seg
    return sanitize(out)


# ---------------------------------------------------------------------------
# time-based effects
# ---------------------------------------------------------------------------
def echo(x: np.ndarray, sr: int, delay_s: float, feedback: float = 0.35,
         wet: float = 0.3, taps: int = 6) -> np.ndarray:
    """Feedback echo/delay.  wet mixed in; dry preserved.  Safe, bounded taps."""
    x = _stereo(x)
    n = x.shape[1]
    d = max(1, int(delay_s * sr))
    wet_sig = np.zeros_like(x)
    g = 1.0
    buf = x.copy()
    for _ in range(max(1, taps)):
        g *= feedback
        if g < 0.02:
            break
        buf = np.pad(buf, ((0, 0), (d, 0)))[:, :n]
        wet_sig += g * buf
    return sanitize(x + wet * wet_sig)


def delay_throw(x: np.ndarray, sr: int, delay_s: float, feedback: float = 0.45,
                wet: float = 0.5, taps: int = 8) -> np.ndarray:
    """
    A "throw": echoes that continue *after* the input ends, so it works as a
    transition tail.  Returns a signal that may be longer than the input.
    """
    x = _stereo(x)
    n = x.shape[1]
    d = max(1, int(delay_s * sr))
    tail = d * taps
    out = np.zeros((2, n + tail), dtype=np.float32)
    out[:, :n] = x
    g = 1.0
    src = x.copy()
    for _ in range(taps):
        g *= feedback
        if g < 0.02:
            break
        start = d
        seg = g * src
        end = start + seg.shape[1]
        out[:, start:end] += seg[:, : out.shape[1] - start]
        src = np.pad(src, ((0, 0), (d, 0)))
    return sanitize(out * (0.5 + wet))


def _synthetic_ir(sr: int, decay_s: float = 1.4, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(decay_s * sr)
    t = np.arange(n) / sr
    env = np.exp(-t / (decay_s / 4.0))
    ir = rng.standard_normal((2, n)).astype(np.float32) * env[None, :]
    # slight early-reflection cluster + gentle low-pass for warmth
    ir[:, : int(0.01 * sr)] *= 0.3
    ir /= np.max(np.abs(ir)) + EPS
    return ir


def reverb_tail(x: np.ndarray, sr: int, decay_s: float = 1.2, wet: float = 0.25) -> np.ndarray:
    """Short convolution reverb using a synthetic IR.  Output length = n + tail."""
    x = _stereo(x)
    ir = _synthetic_ir(sr, decay_s)
    wet_l = signal.fftconvolve(x[0], ir[0])[: x.shape[1] + ir.shape[1]]
    wet_r = signal.fftconvolve(x[1], ir[1])[: x.shape[1] + ir.shape[1]]
    wet_sig = np.stack([wet_l, wet_r]).astype(np.float32)
    wet_sig /= np.max(np.abs(wet_sig)) + EPS
    out = np.zeros_like(wet_sig)
    out[:, : x.shape[1]] = x
    out += wet * wet_sig
    return sanitize(out)


def saturate(x: np.ndarray, drive: float = 1.4, mix: float = 0.25) -> np.ndarray:
    """Gentle tanh saturation for glue/warmth.  Level-compensated, parallel-mixed."""
    x = _stereo(x)
    drive = max(1.0, drive)
    wet = np.tanh(x * drive) / np.tanh(np.array(drive, dtype=np.float32))
    return sanitize((1 - mix) * x + mix * wet)


def stereo_width(x: np.ndarray, width: float = 1.0) -> np.ndarray:
    """Mid/side width control.  width=1 unchanged, <1 narrows, >1 widens (capped)."""
    x = _stereo(x)
    width = float(np.clip(width, 0.0, 1.8))
    mid = (x[0] + x[1]) * 0.5
    side = (x[0] - x[1]) * 0.5 * width
    return sanitize(np.stack([mid + side, mid - side]))


def sidechain_duck(target: np.ndarray, trigger: np.ndarray, sr: int,
                   amount: float = 0.35, release_s: float = 0.18) -> np.ndarray:
    """
    Duck `target` in sympathy with `trigger`'s low-frequency envelope (pumping
    that keeps kicks punching through a pad/bass).  amount 0..1.
    """
    target = _stereo(target)
    trigger = _stereo(trigger)
    n = min(target.shape[1], trigger.shape[1])
    trig_low = static_filter(trigger[:, :n], "low", 160.0, sr, order=2)
    env = np.abs(trig_low).mean(axis=0)
    # instant-attack / exponential-release envelope, vectorised: rise follows the
    # signal, fall follows a one-pole low-pass, combined by max().
    a = float(np.exp(-1.0 / (release_s * sr)))
    released = signal.lfilter([1 - a], [1.0, -a], env)
    smoothed = np.maximum(env, released)
    smoothed /= np.max(smoothed) + EPS
    gain = 1.0 - amount * smoothed
    out = target.copy()
    out[:, :n] *= gain[None, :]
    return sanitize(out)


# ---------------------------------------------------------------------------
# safety / mastering
# ---------------------------------------------------------------------------
def soft_limiter(x: np.ndarray, sr: int, ceiling_db: float = -0.3,
                 lookahead_ms: float = 3.0, release_ms: float = 80.0) -> np.ndarray:
    """
    Fully-vectorised look-ahead limiter that provably respects the ceiling.

    Pipeline: per-sample required gain -> symmetric look-ahead *minimum* filter
    (instant, pre-emptive attack) -> exponential release smoothing via a one-pole
    that only ever *raises* gain (min with the pre-attack value guarantees we
    never exceed the required reduction) -> delay the signal by the look-ahead so
    reduction lands on the peak.  No Python sample loop, so it scales to
    hour-long mixes.  The final clip is a last-resort safety net only.
    """
    from scipy.ndimage import minimum_filter1d

    x = _stereo(x)
    ceiling = 10 ** (ceiling_db / 20.0)
    n = x.shape[1]
    la = max(1, int(lookahead_ms * 1e-3 * sr))
    peak = np.max(np.abs(x), axis=0)
    desired = np.minimum(1.0, ceiling / (peak + EPS)).astype(np.float32)
    # look-ahead: gain is already reduced within ±la of any peak
    look = minimum_filter1d(desired, size=2 * la + 1, mode="nearest").astype(np.float32)
    # exponential release; min() keeps attack instantaneous and gain <= look
    rel = float(np.exp(-1.0 / (release_ms * 1e-3 * sr)))
    smoothed = signal.lfilter([1 - rel], [1.0, -rel], look).astype(np.float32)
    gain = np.minimum(look, smoothed)
    # delay signal by look-ahead so the pre-lowered gain aligns with the peak
    xd = np.zeros_like(x)
    xd[:, la:] = x[:, : n - la]
    out = xd * gain[None, :]
    return np.clip(out, -0.9995, 0.9995).astype(np.float32)


def peak_normalize(x: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    x = _stereo(x)
    peak = np.max(np.abs(x)) + EPS
    target = 10 ** (target_db / 20.0)
    if peak > 0:
        x = x * (target / peak)
    return sanitize(x)


def dc_block(x: np.ndarray) -> np.ndarray:
    x = _stereo(x)
    return sanitize(x - x.mean(axis=1, keepdims=True))
