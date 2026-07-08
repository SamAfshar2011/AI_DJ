"""
Mix renderer: turn an ordered set + transition plans into one continuous,
mastered, quality-checked stereo mix.

Signal-flow / quality strategy
------------------------------
1. Each track is loaded at full RENDER_SR in stereo float32.
2. Beatmatching uses a *chained constant-rate* model computed in
   `transition_engine`: whole tracks are gently time-stretched (Rubber Band,
   pitch-preserving) so neighbours share a tempo.  No mid-track tempo jumps.
3. Per-track loudness is matched to a common reference so nothing lurches in
   volume, then transitions crossfade with equal-power gain + the EQ engine's
   bass-swap automation, plus tasteful, bounded effects.
4. Final bus: DC-block -> integrated-loudness normalise to TARGET_LUFS ->
   look-ahead soft limiter -> peak safety.  Then automated integrity checks.

Geometry is computed first (no audio), the output buffer is allocated once, and
tracks are streamed in one at a time to keep memory bounded.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import effects as fx
from .analyzer import TrackAnalysis
from .audio_io import load_audio, save_wav, to_stereo, wav_to_mp3
from .config import (
    LIMITER_CEILING_DB,
    OUTPUTS_DIR,
    RENDER_SR,
    TARGET_LUFS,
    TRUE_PEAK_CEILING_DB,
)
from .eq_engine import build_transition_eq, corrective_eq_gains
from .transition_engine import TransitionPlan
from .utils import get_logger, safe_float

log = get_logger()

REF_LUFS = -16.0  # per-track normalisation reference before the final master
ProgressCB = Optional[Callable[[float, str], None]]

try:
    import pyloudnorm as pyln

    _HAVE_PYLN = True
except Exception:  # pragma: no cover
    _HAVE_PYLN = False


# ---------------------------------------------------------------------------
# time-stretch
# ---------------------------------------------------------------------------
def time_stretch(y: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """Pitch-preserving constant-ratio stretch of a (2,n) signal (rate>1 = faster)."""
    if abs(rate - 1.0) < 1e-3:
        return y
    y = to_stereo(y)
    try:
        import pyrubberband as prb

        out = prb.time_stretch(y.T, sr, rate)  # expects (n, ch)
        return fx.sanitize(np.ascontiguousarray(out.T))
    except Exception as e:
        log.warning("rubberband stretch failed (%s) — librosa fallback", e)
        import librosa

        return fx.sanitize(np.stack([
            librosa.effects.time_stretch(np.ascontiguousarray(y[0]), rate=rate),
            librosa.effects.time_stretch(np.ascontiguousarray(y[1]), rate=rate),
        ]))


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
@dataclass
class Placement:
    s_use: float          # stretched-time start of used segment (s)
    e_use: float          # stretched-time end of used segment (s)
    tl_start: float       # timeline start (s)
    head_overlap: float   # blend length with previous track (s)
    tail_overlap: float   # blend length with next track (s)
    rate: float


def _source_duration(a: TrackAnalysis) -> float:
    d = a.metadata.get("duration")
    if d and d > a.features.duration - 1:
        return float(d)
    return float(a.features.duration)


def compute_geometry(
    ordered: list[TrackAnalysis], plans: list[TransitionPlan], deck_rates: list[float]
) -> list[Placement]:
    """
    Turn plans into concrete timeline placements.  Robust against short tracks and
    degenerate exit/entry points: every blend length is clamped to the material
    actually available on *both* sides, and the running timeline position is kept
    strictly increasing so segments never overlap incorrectly or index backwards.
    """
    n = len(ordered)
    dur = [_source_duration(ordered[i]) / deck_rates[i] for i in range(n)]  # stretched-time durations

    # entry points (where each track is first heard), clamped inside the track
    s_use = [0.0] * n
    for i in range(1, n):
        s = plans[i - 1].in_start / deck_rates[i]
        s_use[i] = float(min(max(0.0, s), max(0.0, dur[i] - 2.0)))

    # per-transition effective blend length, limited by available material
    eff = [0.0] * max(0, n - 1)
    e_use = [0.0] * n
    for i in range(n - 1):
        out_str = plans[i].out_start / deck_rates[i]
        out_str = float(min(max(out_str, s_use[i] + 1.0), dur[i] - 0.5))
        out_avail = dur[i] - out_str            # tail material on the outgoing track
        in_avail = dur[i + 1] - s_use[i + 1]    # material on the incoming track
        ov = min(plans[i].overlap, out_avail, in_avail)
        eff[i] = float(max(0.5, ov))
        e_use[i] = float(min(out_str + eff[i], dur[i]))
    e_use[n - 1] = dur[n - 1]
    for i in range(n):
        e_use[i] = max(e_use[i], s_use[i] + 1.0)

    places: list[Placement] = []
    tl = 0.0
    for i in range(n):
        seg_len = e_use[i] - s_use[i]
        head = eff[i - 1] if i > 0 else 0.0
        tail = eff[i] if i < n - 1 else 0.0
        # overlaps can never exceed this segment
        head = min(head, max(0.0, seg_len - 0.25))
        tail = min(tail, max(0.0, seg_len - 0.25))
        places.append(Placement(s_use[i], e_use[i], tl, head, tail, deck_rates[i]))
        tl = max(tl + 0.25, tl + seg_len - tail)  # strictly increasing timeline
    return places


# ---------------------------------------------------------------------------
# loudness helpers
# ---------------------------------------------------------------------------
def _measure_lufs(y: np.ndarray, sr: int) -> float:
    if _HAVE_PYLN and y.shape[1] > sr // 2:
        try:
            meter = pyln.Meter(sr)
            return float(meter.integrated_loudness(y.T.astype(np.float64)))
        except Exception:
            pass
    rms = np.sqrt(np.mean(y.astype(np.float64) ** 2) + 1e-12)
    return float(20 * np.log10(rms + 1e-12) - 3.0)


# ---------------------------------------------------------------------------
# main render
# ---------------------------------------------------------------------------
def render_mix(
    ordered: list[TrackAnalysis],
    plans: list[TransitionPlan],
    deck_rates: list[float],
    settings,
    output_stub: str = "aidj_mix",
    progress_cb: ProgressCB = None,
) -> dict:
    sr = RENDER_SR
    n = len(ordered)
    if n == 0:
        raise ValueError("nothing to render")

    def report(p, msg):
        if progress_cb:
            progress_cb(p, msg)

    places = compute_geometry(ordered, plans, deck_rates)
    total_s = places[-1].tl_start + (places[-1].e_use - places[-1].s_use)
    total_samples = int(np.ceil(total_s * sr)) + sr  # +1 s pad for FX tails
    log.info("rendering %d tracks -> ~%.1f min mix", n, total_s / 60)

    out = np.zeros((2, total_samples), dtype=np.float32)

    # precompute per-transition EQ automation (bass-swap etc.)
    eq_dicts = []
    for i in range(n - 1):
        L = max(2, int(plans[i].overlap * sr))
        eq_dicts.append(build_transition_eq(
            ordered[i], ordered[i + 1], L,
            intensity=settings.transition_intensity, aggressive=settings.aggressive,
        ))

    for i, a in enumerate(ordered):
        report(0.05 + 0.9 * i / n, f"Rendering {i+1}/{n}: {a.title}")
        rate = deck_rates[i]
        # load full track, stereo, at render rate
        y, _ = load_audio(a.path, sr=sr, mono=False)
        y = to_stereo(y)

        # subtle corrective tone balance (<= ~2.5 dB)
        cg = corrective_eq_gains(a)
        if any(abs(v) > 0.05 for v in cg.values()):
            y = fx.eq3(y, sr, low_db=cg["low_db"], mid_db=cg["mid_db"], high_db=cg["high_db"])

        # pitch-preserving tempo match
        if abs(rate - 1.0) > 1e-3:
            y = time_stretch(y, sr, rate)

        # per-track loudness match to the reference
        lufs = _measure_lufs(y, sr)
        gain_db = float(np.clip(REF_LUFS - lufs, -12.0, 12.0))
        y = (y * (10 ** (gain_db / 20.0))).astype(np.float32)

        pl = places[i]
        s0 = int(pl.s_use * sr)
        e0 = min(y.shape[1], int(pl.e_use * sr))
        seg = y[:, s0:e0].copy()
        del y
        gc.collect()
        if seg.shape[1] < 4:
            continue
        seg_len = seg.shape[1]

        Lh = min(int(pl.head_overlap * sr), seg_len)
        Lt = min(int(pl.tail_overlap * sr), seg_len)

        # ---- incoming head: EQ (bass-cut opening up) then equal-power fade-in
        if i > 0 and Lh > 2:
            eqd = eq_dicts[i - 1]["in"]
            head = seg[:, :Lh]
            head = fx.eq3_envelope(head, sr, eqd["low"], eqd["mid"], eqd["high"])
            _, fade_in = fx.equal_power_fades(Lh)
            head = head * fade_in[None, :]
            seg[:, :Lh] = head
        elif i == 0:
            # gentle 0.4 s open to avoid a click at the very start
            k = min(int(0.4 * sr), seg_len)
            seg[:, :k] *= np.linspace(0, 1, k, dtype=np.float32)[None, :]

        # ---- outgoing tail: EQ bass-swap -> FX -> equal-power fade-out
        if i < n - 1 and Lt > 2:
            eqd = eq_dicts[i]["out"]
            tail = seg[:, seg_len - Lt:]
            tail = fx.eq3_envelope(tail, sr, eqd["low"], eqd["mid"], eqd["high"])
            tail = _apply_tail_fx(tail, sr, plans[i], a.bpm * rate)
            fade_out, _ = fx.equal_power_fades(tail.shape[1])
            tail = tail[:, :Lt] * fade_out[None, :]
            seg[:, seg_len - Lt:] = tail
        elif i == n - 1:
            # graceful ending
            k = min(int(2.0 * sr), seg_len)
            seg[:, seg_len - k:] *= np.linspace(1, 0, k, dtype=np.float32)[None, :]

        # ---- place onto the timeline (additive; overlaps sum) ----
        start = max(0, int(pl.tl_start * sr))
        w = max(0, min(seg.shape[1], total_samples - start))
        if w > 0:
            out[:, start:start + w] += seg[:, :w]
        del seg
        gc.collect()

    # ---- master bus -----------------------------------------------------
    report(0.95, "Mastering: loudness + limiter")
    out = fx.dc_block(out)
    if settings.normalize:
        cur = _measure_lufs(out, sr)
        gain_db = float(np.clip(TARGET_LUFS - cur, -24.0, 24.0))
        out = out * (10 ** (gain_db / 20.0))
    out = fx.soft_limiter(out, sr, ceiling_db=LIMITER_CEILING_DB)
    out = fx.peak_normalize(out, target_db=TRUE_PEAK_CEILING_DB)
    out = fx.sanitize(out)

    # trim trailing silence pad
    out = _trim_tail(out, sr)

    # ---- write files ----------------------------------------------------
    report(0.98, "Writing output files")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = OUTPUTS_DIR / f"{output_stub}.wav"
    save_wav(wav_path, out, sr, subtype="PCM_24")

    result = {
        "wav": str(wav_path),
        "mp3": None,
        "duration": round(out.shape[1] / sr, 2),
        "sample_rate": sr,
        "channels": 2,
        "peak_dbfs": round(safe_float(20 * np.log10(np.max(np.abs(out)) + 1e-9)), 2),
        "lufs": round(_measure_lufs(out, sr), 2),
    }
    if settings.output_format in ("mp3", "both"):
        mp3 = wav_to_mp3(wav_path, OUTPUTS_DIR / f"{output_stub}.mp3")
        result["mp3"] = str(mp3) if mp3 else None

    checks = validate_output(wav_path, expected_min_s=total_s * 0.5)
    result["checks"] = checks
    report(1.0, "Complete")
    return result


def _apply_tail_fx(tail: np.ndarray, sr: int, plan: TransitionPlan, bpm: float) -> np.ndarray:
    beat = 60.0 / max(bpm, 60.0)
    e = plan.effects or {}
    if "outgoing_lowpass_sweep" in e:
        p = e["outgoing_lowpass_sweep"]
        tail = fx.sweep_filter(tail, "low", p.get("f_start", 18000), p.get("f_end", 350), sr)
    if "echo_throw" in e:
        p = e["echo_throw"]
        tail = fx.echo(tail, sr, delay_s=beat * p.get("beats", 1),
                       feedback=p.get("feedback", 0.35), wet=p.get("wet", 0.3))
    if "reverb_tail" in e:
        p = e["reverb_tail"]
        wet = fx.reverb_tail(tail, sr, decay_s=p.get("decay", 1.2), wet=p.get("wet", 0.2))
        tail = wet[:, : tail.shape[1]]  # crop reverb overhang to region
    if e.get("stereo_glue"):
        tail = fx.stereo_width(tail, 1.08)
    return fx.sanitize(tail)


def _trim_tail(y: np.ndarray, sr: int, thresh_db: float = -60.0) -> np.ndarray:
    env = np.max(np.abs(y), axis=0)
    thr = 10 ** (thresh_db / 20.0)
    nz = np.where(env > thr)[0]
    if len(nz) == 0:
        return y
    end = min(y.shape[1], nz[-1] + int(0.3 * sr))
    return y[:, :end]


# ---------------------------------------------------------------------------
# integrity checks
# ---------------------------------------------------------------------------
def validate_output(path: str | Path, expected_min_s: float = 20.0) -> dict:
    """Re-load the rendered file and assert it is sane.  Never raises."""
    import soundfile as sf

    checks = {"exists": False, "nonempty": False, "no_nan": False,
              "no_clip": False, "duration_ok": False, "sr_ok": False,
              "reloadable": False, "passed": False, "notes": []}
    p = Path(path)
    try:
        checks["exists"] = p.exists() and p.stat().st_size > 1024
        data, sr = sf.read(str(p), dtype="float32")
        checks["reloadable"] = True
        checks["sr_ok"] = sr in (44100, 48000, 22050, 88200, 96000)
        checks["nonempty"] = data.size > 0
        checks["no_nan"] = bool(np.isfinite(data).all())
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        checks["no_clip"] = peak <= 0.9999
        dur = len(data) / sr if sr else 0.0
        checks["duration_ok"] = dur >= max(10.0, expected_min_s * 0.5)
        checks["notes"].append(f"duration={dur:.1f}s peak={peak:.4f} sr={sr}")
    except Exception as e:
        checks["notes"].append(f"validation error: {e}")
    checks["passed"] = all(checks[k] for k in
                           ("exists", "nonempty", "no_nan", "no_clip",
                            "duration_ok", "sr_ok", "reloadable"))
    return checks
