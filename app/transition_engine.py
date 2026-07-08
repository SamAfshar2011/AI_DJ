"""
Transition planning: for an ordered pair (A -> B) decide *how* to move between
them — where A leaves, where B enters, how long the blend is, whether a clean
beatmatch is possible, how to phase-align on beats/phrases, and which effects
serve the transition.

Quality principles baked in here:
  * Only stretch B's short overlap head, and only within the allowed limit — the
    bodies of tracks stay at native tempo (preserves quality/pitch).
  * Snap the exit and entry to phrase (8-bar) boundaries so blends are musical.
  * If a clean beatmatch isn't possible, don't force a bad time-stretch — fall
    back to a filter/echo transition on a phrase boundary.
  * Never overlap two full-energy bass sections (the EQ engine bass-swaps).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .analyzer import TrackAnalysis
from .beatgrid import nearest_beat, nearest_phrase
from .config import MixSettings
from .key_detection import camelot_compatibility, semitone_distance
from .utils import clamp, get_logger, safe_float

log = get_logger()


@dataclass
class TransitionPlan:
    # indices are into the ordered analyses list (pos, pos+1)
    from_title: str
    to_title: str
    technique: str                 # harmonic_blend | eq_blend | creative_cut | quick_cut
    out_start: float               # time in A where the blend begins (s)
    overlap: float                 # blend length (s)
    in_start: float                # time in B first heard (s, its mix-in)
    in_body_start: float           # time in B where solo body continues after blend
    stretch_ratio: float           # multiply B's head by this to beatmatch A (1.0 = none)
    beatmatched: bool
    align_offset: float            # fine offset (s) to phase-align B under A
    target_bpm: float              # tempo of the blend region
    key_compat: float
    semitone_hint: int
    effects: dict = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "from": self.from_title, "to": self.to_title,
            "technique": self.technique,
            "out_start": round(safe_float(self.out_start), 2),
            "overlap": round(safe_float(self.overlap), 2),
            "in_start": round(safe_float(self.in_start), 2),
            "in_body_start": round(safe_float(self.in_body_start), 2),
            "stretch_ratio": round(safe_float(self.stretch_ratio), 4),
            "beatmatched": self.beatmatched,
            "target_bpm": round(safe_float(self.target_bpm), 1),
            "key_compat": round(safe_float(self.key_compat), 2),
            "effects": self.effects,
            "reason": self.reason,
        }


def _phrase_exit(a: TrackAnalysis, min_tail: float) -> float:
    """
    Choose where A starts leaving: near its mix-out point, snapped to a phrase
    boundary, guaranteeing at least `min_tail` seconds remain before the track
    ends so the whole blend has material.
    """
    dur = a.duration
    ideal = min(a.features.mix_out_point, dur - min_tail - 0.5)
    ideal = max(ideal, dur * 0.4)                      # don't leave absurdly early
    exit_t = nearest_phrase(a.beatgrid.phrase_times, ideal, prefer="before")
    if exit_t + min_tail > dur:                        # not enough tail -> pull back
        exit_t = nearest_phrase(a.beatgrid.phrase_times, dur - min_tail - 0.5, prefer="before")
    if exit_t + min_tail > dur or exit_t <= 0:         # phrase grid unusable -> raw
        exit_t = max(0.0, dur - min_tail - 0.5)
    return float(exit_t)


def _phrase_entry(b: TrackAnalysis) -> float:
    """Where B is first brought in: its mix-in point snapped to a phrase/downbeat."""
    entry = b.features.mix_in_point
    snapped = nearest_phrase(b.beatgrid.phrase_times, entry, prefer="after")
    if snapped <= 0 or snapped > b.duration * 0.4:
        snapped = nearest_beat(b.beatgrid.downbeat_times, entry) if len(b.beatgrid.downbeat_times) else entry
    return float(max(0.0, snapped))


def compute_deck_rates(ordered: list[TrackAnalysis], settings: MixSettings) -> list[float]:
    """
    Chained constant-rate beatmatch: pick a whole-track playback rate for each
    track so consecutive tracks share a tempo where possible, keeping every rate
    within the allowed stretch of the track's native tempo (so quality holds and
    there are never mid-track tempo jumps).  Octave (half/double) matches allowed.
    """
    settings = settings.resolved()
    maxs = settings.max_tempo_stretch
    n = len(ordered)
    if n == 0:
        return []
    rates = [1.0] * n
    for i in range(1, n):
        a_eff = ordered[i - 1].bpm * rates[i - 1]
        b_bpm = max(1e-6, ordered[i].bpm)
        # candidate rates to land B on A's effective tempo, at octaves
        best_r, best_dev = 1.0, 1e9
        for factor in (0.5, 1.0, 2.0):
            r = (a_eff * factor) / b_bpm
            dev = abs(np.log2(max(r, 1e-6)))
            if dev < best_dev:
                best_dev, best_r = dev, r
        rates[i] = float(np.clip(best_r, 1.0 - maxs, 1.0 + maxs))
    return rates


def plan_transition(
    a: TrackAnalysis,
    b: TrackAnalysis,
    settings: MixSettings,
    a_eff_bpm: float | None = None,
    b_eff_bpm: float | None = None,
) -> TransitionPlan:
    settings = settings.resolved()
    a_eff_bpm = a_eff_bpm if a_eff_bpm else a.bpm
    b_eff_bpm = b_eff_bpm if b_eff_bpm else b.bpm

    # --- overlap length in beats -> seconds at A's effective tempo -------
    base_beats = settings.crossfade_beats
    beats = int(round(base_beats * (0.5 + settings.transition_intensity)))
    beats = int(clamp(beats, 8, 64))
    a_beat = 60.0 / max(a_eff_bpm, 60.0)
    overlap = beats * a_beat
    # keep overlap sane vs track lengths
    overlap = float(clamp(overlap, 4.0, min(a.duration * 0.5, b.duration * 0.5, 45.0)))

    out_start = _phrase_exit(a, min_tail=overlap)
    in_start = _phrase_entry(b)

    # --- beatmatch feasibility (using chained effective tempos) ----------
    # The render pre-stretches whole decks to these effective tempos, so no extra
    # stretch happens inside the blend.  Tracks are "beatmatched" when their
    # effective tempos already coincide (within ~2%) at some octave.
    octave_dev = min(abs(np.log2(max((a_eff_bpm * f) / max(b_eff_bpm, 1e-6), 1e-6)))
                     for f in (0.5, 1.0, 2.0))
    beatmatched = bool(octave_dev <= np.log2(1.02))
    stretch_ratio = 1.0                    # deck-level stretch already applied
    target_bpm = a_eff_bpm

    key_compat = camelot_compatibility(a.key.camelot, b.key.camelot)
    semitone_hint = semitone_distance(a.key.key, b.key.key)

    # --- fine phase alignment (align a B downbeat under an A beat) -------
    # after stretching, B's head beat period == a_beat; align first B downbeat
    # (>= in_start) so it lands exactly on A's beat at the blend start.
    align_offset = 0.0
    if beatmatched and len(b.beatgrid.downbeat_times):
        db = b.beatgrid.downbeat_times
        cand = db[db >= in_start]
        if len(cand):
            in_start = float(cand[0])

    # --- technique + effects selection ----------------------------------
    fx_amt = settings.effect_intensity
    effects: dict = {"eq_bass_swap": True}
    if beatmatched and key_compat >= 0.7:
        technique = "harmonic_blend"
        reason = f"Beatmatched harmonic blend over {beats} beats ({a.key.camelot}→{b.key.camelot})."
        if fx_amt > 0.5:
            effects["reverb_tail"] = {"decay": 1.2, "wet": 0.15 * fx_amt}
    elif beatmatched:
        technique = "eq_blend"
        overlap *= 0.85
        reason = f"Beatmatched EQ blend with extra low/mid separation (keys {a.key.camelot}/{b.key.camelot})."
        effects["outgoing_lowpass_sweep"] = {"f_start": 20000, "f_end": 400}
        if fx_amt > 0.4:
            effects["echo_throw"] = {"beats": 1, "feedback": 0.35, "wet": 0.3 * fx_amt}
    else:
        # tempo too far apart to beatmatch cleanly -> creative phrase transition
        overlap = float(clamp(overlap * 0.5, 3.0, 12.0))
        if settings.aggressive or fx_amt < 0.3:
            technique = "quick_cut"
            reason = "Tempos incompatible — clean cut on a phrase boundary."
        else:
            technique = "creative_cut"
            reason = "Tempos incompatible — filter-sweep + echo throw over a phrase boundary."
            effects["outgoing_lowpass_sweep"] = {"f_start": 18000, "f_end": 250}
            effects["echo_throw"] = {"beats": 2, "feedback": 0.45, "wet": 0.4 + 0.3 * fx_amt}
            effects["reverb_tail"] = {"decay": 1.4, "wet": 0.2 + 0.2 * fx_amt}
        # re-snap exit with the shorter tail requirement
        out_start = _phrase_exit(a, min_tail=overlap)

    if settings.effect_intensity > 0.6 and technique in ("harmonic_blend", "eq_blend"):
        effects["stereo_glue"] = True

    in_body_start = in_start + overlap / max(stretch_ratio, 1e-6)

    return TransitionPlan(
        from_title=a.title, to_title=b.title, technique=technique,
        out_start=out_start, overlap=overlap, in_start=in_start,
        in_body_start=in_body_start, stretch_ratio=stretch_ratio,
        beatmatched=beatmatched, align_offset=align_offset, target_bpm=target_bpm,
        key_compat=key_compat, semitone_hint=semitone_hint, effects=effects,
        reason=reason,
    )


def plan_all_transitions(
    ordered: list[TrackAnalysis], settings: MixSettings
) -> tuple[list[TransitionPlan], list[float]]:
    """Return (transition_plans, deck_rates) for an ordered set."""
    deck_rates = compute_deck_rates(ordered, settings)
    plans = []
    for i in range(len(ordered) - 1):
        a_eff = ordered[i].bpm * deck_rates[i]
        b_eff = ordered[i + 1].bpm * deck_rates[i + 1]
        plans.append(plan_transition(ordered[i], ordered[i + 1], settings,
                                     a_eff_bpm=a_eff, b_eff_bpm=b_eff))
    return plans, deck_rates
