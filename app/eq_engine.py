"""
Intelligent, context-aware EQ automation for transitions.

The single most important EQ job when mixing two tracks is the **bass swap**:
two full-energy low ends played together sum to mud and clip.  So across a
transition we hand the low band from the outgoing track to the incoming one,
crossing over near the midpoint, while keeping the combined spectrum balanced.

`build_transition_eq` returns per-track low/mid/high dB automation envelopes
(sampled over the transition region) that the render stage feeds to
`effects.eq3_envelope`.  Decisions adapt to each track's measured spectral
balance and the requested transition intensity.
"""
from __future__ import annotations

import numpy as np

from .analyzer import TrackAnalysis
from .utils import get_logger

log = get_logger()


def _smoothstep(n: int) -> np.ndarray:
    t = np.linspace(0, 1, max(2, n), dtype=np.float32)
    return t * t * (3 - 2 * t)  # C1-smooth 0->1


def build_transition_eq(
    out_track: TrackAnalysis,
    in_track: TrackAnalysis,
    n_samples: int,
    intensity: float = 0.55,
    aggressive: bool = False,
) -> dict:
    """
    Returns:
      {
        "out": {"low": env, "mid": env, "high": env},   # dB automation, len n
        "in":  {"low": env, "mid": env, "high": env},
        "notes": [...]
      }
    Envelopes are additive dB corrections applied *on top of* the volume fades.
    """
    n = max(2, n_samples)
    ramp = _smoothstep(n)          # 0 -> 1 across the transition
    inv = 1.0 - ramp
    notes = []

    # --- bass swap -------------------------------------------------------
    # incoming low starts cut, opens up; outgoing low stays then drops.
    # crossover point biased slightly past the middle so the new track's kick
    # takes over cleanly on a phrase boundary.
    cut_depth = -22.0 - 6.0 * intensity          # how hard we kill bass (dB)
    xover = 0.55                                  # fraction where bass hands over
    swap = np.clip((ramp - (1 - xover)) / xover, 0, 1)  # 0 until handover, ->1

    in_low = cut_depth * (1.0 - swap)             # incoming: cut -> 0 dB
    out_low = cut_depth * swap                    # outgoing: 0 -> cut
    notes.append("bass-swap: low band handed over near phrase midpoint")

    # --- mid clash management -------------------------------------------
    # if both tracks are mid-heavy / vocal, dip mids in the middle to avoid mud.
    mid_clash = (out_track.features.vocalness + in_track.features.vocalness) / 2.0
    mid_dip = -(3.0 + 6.0 * intensity) * mid_clash
    bell = np.sin(np.pi * ramp) ** 2             # peaks in the centre
    out_mid = mid_dip * bell * inv               # dip the one leaving more
    in_mid = mid_dip * bell * ramp * 0.6
    if mid_clash > 0.5:
        notes.append("mid-scoop to prevent vocal/mid collision")

    # --- high band -------------------------------------------------------
    # keep some air on the incoming track early (adds presence), gently roll the
    # outgoing highs as it leaves so it recedes behind the new track.
    out_high = -4.0 * intensity * ramp
    in_high = 1.5 * intensity * inv * (1 if not aggressive else 1.5)

    # avoid muddy low-mids: if incoming track already bass-heavy, trim ~250 Hz a touch
    if in_track.features.spectral_balance.get("low", 0.3) > 0.42:
        in_mid = in_mid - 2.0
        notes.append("extra low-mid trim on bass-heavy incoming track")

    return {
        "out": {
            "low": out_low.astype(np.float32),
            "mid": out_mid.astype(np.float32),
            "high": out_high.astype(np.float32),
        },
        "in": {
            "low": in_low.astype(np.float32),
            "mid": in_mid.astype(np.float32),
            "high": in_high.astype(np.float32),
        },
        "notes": notes,
    }


def corrective_eq_gains(track: TrackAnalysis) -> dict:
    """
    A subtle, static tone-balance correction for a whole track based on its
    measured spectral balance.  Conservative (<= ~2.5 dB) so we never recolour
    the artist's mix aggressively — just nudge obvious imbalances.
    """
    bal = track.features.spectral_balance
    low, mid, high = bal.get("low", 0.33), bal.get("mid", 0.34), bal.get("high", 0.33)
    gains = {"low_db": 0.0, "mid_db": 0.0, "high_db": 0.0}
    # tame boomy low end
    if low > 0.45:
        gains["low_db"] = -min(2.5, (low - 0.45) * 12)
    # lift dull / dark tracks slightly
    if high < 0.12:
        gains["high_db"] = min(2.0, (0.12 - high) * 15)
    # de-mud thick low-mids
    if mid > 0.5 and low > 0.38:
        gains["mid_db"] = -1.5
    return gains
