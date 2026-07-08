"""
AI set planner: decide the *order* of tracks so the mix tells a coherent musical
story.

We build a pairwise "how well does A flow into B" cost from four musically
meaningful terms — tempo compatibility, harmonic (Camelot) compatibility, vibe
similarity (embedding cosine) and energy step — then search for an ordering that
is both smoothly mixable and follows the user's desired energy curve.

Search = energy-guided greedy construction + 2-opt refinement.  This is a small,
well-behaved TSP-like problem (libraries are usually tens of tracks), so exact
optimality isn't needed; musical smoothness is.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .analyzer import TrackAnalysis
from .config import MixSettings
from .key_detection import camelot_compatibility
from .utils import get_logger, safe_float
from .vibe_model import cosine_similarity

log = get_logger()


# ---------------------------------------------------------------------------
# Pairwise musical cost
# ---------------------------------------------------------------------------
def tempo_compatibility(bpm_a: float, bpm_b: float, max_stretch: float) -> tuple[float, float]:
    """
    Return (cost 0..1, stretch_ratio) for moving from tempo A to tempo B, allowing
    half/double-time matching.  stretch_ratio is what B must be multiplied by to
    land on A's grid (nearest octave).
    """
    if bpm_a <= 0 or bpm_b <= 0:
        return 0.5, 1.0
    best_cost, best_ratio = 1.0, 1.0
    for factor in (0.5, 1.0, 2.0):
        target = bpm_a * factor          # B should sound like this tempo
        ratio = target / bpm_b           # multiply B by this
        rel = abs(ratio - 1.0)           # fractional stretch needed
        # cost grows with required stretch; cheap within max_stretch, steep past it
        if rel <= max_stretch:
            cost = 0.15 * (rel / (max_stretch + 1e-9))
        else:
            cost = 0.4 + 2.0 * (rel - max_stretch)
        if cost < best_cost:
            best_cost, best_ratio = cost, ratio
    return float(min(1.0, best_cost)), float(best_ratio)


def transition_cost(
    a: TrackAnalysis, b: TrackAnalysis, settings: MixSettings
) -> tuple[float, dict]:
    """Directional cost of playing b right after a, plus a breakdown for reasons."""
    tempo_c, ratio = tempo_compatibility(a.bpm, b.bpm, settings.max_tempo_stretch)
    key_c = 1.0 - camelot_compatibility(a.key.camelot, b.key.camelot)
    sim = cosine_similarity(a.embedding, b.embedding)  # -1..1
    vibe_c = float(np.clip((1.0 - sim) / 2.0, 0.0, 1.0))
    # small preference for not slamming a very dense/vocal track onto another
    vocal_clash = 0.0
    if a.features.vocalness > 0.6 and b.features.vocalness > 0.6:
        vocal_clash = 0.1

    # optional mood-continuity nudge (model 2): only active when *both* tracks
    # carry a mood-tag vector, so behaviour is unchanged when the mood model is
    # untrained/absent.
    mood_c = 0.0
    mood_sim = None
    mva, mvb = a.vibe_tags.get("mood_vec"), b.vibe_tags.get("mood_vec")
    if mva and mvb:
        mood_sim = cosine_similarity(np.asarray(mva, dtype=np.float32),
                                     np.asarray(mvb, dtype=np.float32))
        mood_c = float(np.clip(1.0 - mood_sim, 0.0, 1.0))

    hp = settings.harmonic_priority
    cost = (
        0.34 * tempo_c
        + (0.30 * hp + 0.10) * key_c
        + (0.34 - 0.12 * (hp - 0.5)) * vibe_c
        + 0.12 * mood_c
        + vocal_clash
    )
    breakdown = {
        "tempo_cost": tempo_c,
        "key_cost": key_c,
        "vibe_cost": vibe_c,
        "vibe_similarity": sim,
        "mood_cost": mood_c,
        "mood_similarity": mood_sim,
        "stretch_ratio": ratio,
        "key_compat": camelot_compatibility(a.key.camelot, b.key.camelot),
    }
    return float(cost), breakdown


# ---------------------------------------------------------------------------
# Energy target curve
# ---------------------------------------------------------------------------
def energy_target(style: str, n: int) -> np.ndarray:
    x = np.linspace(0, 1, max(n, 1))
    if style == "rising":
        y = 0.25 + 0.7 * x
    elif style == "descending":
        y = 0.95 - 0.7 * x
    elif style == "peak":  # build up then come down (festival arc)
        y = 0.3 + 0.65 * np.sin(np.pi * x)
    elif style == "wave":
        y = 0.55 + 0.35 * np.sin(2.0 * np.pi * x - np.pi / 2)
    else:  # flat / steady groove
        y = np.full_like(x, 0.6)
    return np.clip(y, 0.05, 1.0)


# ---------------------------------------------------------------------------
# Ordering search
# ---------------------------------------------------------------------------
@dataclass
class MixPlan:
    order: list[int]                      # indices into the analyses list
    analyses: list[TrackAnalysis]
    reasons: list[dict]                   # per-transition breakdown (len = n-1)
    settings: MixSettings
    energy_style: str
    est_duration: float = 0.0

    def ordered(self) -> list[TrackAnalysis]:
        return [self.analyses[i] for i in self.order]

    def to_dict(self) -> dict:
        tracks = []
        for pos, idx in enumerate(self.order):
            a = self.analyses[idx]
            tracks.append({
                "position": pos,
                "title": a.title,
                "artist": a.artist,
                "bpm": round(safe_float(a.bpm), 1),
                "key": a.key.name,
                "camelot": a.key.camelot,
                "energy": round(safe_float(a.features.energy), 3),
                "duration": round(safe_float(a.duration), 1),
            })
        return {
            "tracks": tracks,
            "transitions": self.reasons,
            "energy_style": self.energy_style,
            "est_duration": round(self.est_duration, 1),
            "n_tracks": len(self.order),
        }


def _cost_matrix(analyses: list[TrackAnalysis], settings: MixSettings):
    n = len(analyses)
    C = np.zeros((n, n), dtype=np.float32)
    B: list[list[dict]] = [[{} for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                C[i, j] = 0.0
                continue
            c, bd = transition_cost(analyses[i], analyses[j], settings)
            C[i, j] = c
            B[i][j] = bd
    return C, B


def _greedy_order(
    analyses: list[TrackAnalysis], C: np.ndarray, target: np.ndarray, energy_lambda: float
) -> list[int]:
    n = len(analyses)
    energies = np.array([a.features.energy for a in analyses], dtype=np.float32)
    # start near the first energy target (e.g. lowest-energy track for "rising")
    start = int(np.argmin(np.abs(energies - target[0])))
    order = [start]
    used = {start}
    while len(order) < n:
        pos = len(order)
        prev = order[-1]
        best, best_c = -1, math.inf
        for j in range(n):
            if j in used:
                continue
            energy_pen = energy_lambda * (energies[j] - target[pos]) ** 2
            c = C[prev, j] + energy_pen
            if c < best_c:
                best_c, best = c, j
        order.append(best)
        used.add(best)
    return order


def _plan_cost(order: list[int], C: np.ndarray, energies, target, energy_lambda) -> float:
    total = 0.0
    for pos, idx in enumerate(order):
        total += energy_lambda * (energies[idx] - target[pos]) ** 2
        if pos > 0:
            total += C[order[pos - 1], idx]
    return total


def _two_opt(order, C, energies, target, energy_lambda, max_iter=2000):
    """Local 2-opt improvement (segment reversal) on the combined objective."""
    n = len(order)
    if n < 4:
        return order
    best = order[:]
    best_cost = _plan_cost(best, C, energies, target, energy_lambda)
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                cand = best[:i] + best[i:k + 1][::-1] + best[k + 1:]
                cost = _plan_cost(cand, C, energies, target, energy_lambda)
                if cost + 1e-6 < best_cost:
                    best, best_cost = cand, cost
                    improved = True
                    it += 1
                    if it >= max_iter:
                        break
            if it >= max_iter:
                break
    return best


def plan_set(analyses: list[TrackAnalysis], settings: MixSettings) -> MixPlan:
    """Produce an ordered mix plan from analysed tracks."""
    settings = settings.resolved()
    n = len(analyses)
    if n == 0:
        raise ValueError("no tracks to plan")
    if n == 1:
        return MixPlan(order=[0], analyses=analyses, reasons=[], settings=settings,
                       energy_style=settings.energy_curve,
                       est_duration=analyses[0].duration)

    C, B = _cost_matrix(analyses, settings)
    target = energy_target(settings.energy_curve, n)
    energies = np.array([a.features.energy for a in analyses], dtype=np.float32)
    # how strongly to enforce the energy curve vs pure mixability
    energy_lambda = 0.9

    order = _greedy_order(analyses, C, target, energy_lambda)
    order = _two_opt(order, C, energies, target, energy_lambda)

    # optionally trim to a target length (drop tracks that hurt flow the least)
    order = _apply_length_target(order, analyses, settings, C)

    reasons = []
    for pos in range(len(order) - 1):
        i, j = order[pos], order[pos + 1]
        bd = dict(B[i][j])
        bd["from"] = analyses[i].title
        bd["to"] = analyses[j].title
        bd["reason"] = _describe(analyses[i], analyses[j], bd)
        reasons.append(bd)

    est = _estimate_duration(order, analyses, settings)
    plan = MixPlan(order=order, analyses=analyses, reasons=reasons, settings=settings,
                   energy_style=settings.energy_curve, est_duration=est)
    log.info("planned %d-track set (~%.1f min, %s energy)", len(order), est / 60, settings.energy_curve)
    return plan


def _apply_length_target(order, analyses, settings, C):
    if settings.target_minutes <= 0:
        return order
    target_s = settings.target_minutes * 60.0
    # greedily keep the front of the set until we exceed target, but always keep
    # at least 2 tracks; account for overlap shortening (~20 s per transition).
    kept, acc = [], 0.0
    for pos, idx in enumerate(order):
        dur = analyses[idx].duration
        overlap = 20.0 if kept else 0.0
        if acc + dur - overlap > target_s and len(kept) >= 2:
            break
        kept.append(idx)
        acc += dur - overlap
    return kept if len(kept) >= 2 else order[:2]


def _estimate_duration(order, analyses, settings) -> float:
    total = sum(analyses[i].duration for i in order)
    # each transition overlaps the two tracks by roughly the crossfade length
    beats = settings.crossfade_beats
    for pos in range(len(order) - 1):
        bpm = analyses[order[pos]].bpm or 120
        overlap = beats * (60.0 / bpm) * (0.5 + settings.transition_intensity * 0.5)
        total -= overlap
    return max(0.0, total)


def _describe(a: TrackAnalysis, b: TrackAnalysis, bd: dict) -> str:
    bits = []
    if bd["key_compat"] >= 0.85:
        bits.append(f"harmonically matched ({a.key.camelot}→{b.key.camelot})")
    elif bd["key_compat"] >= 0.6:
        bits.append(f"near-key blend ({a.key.camelot}→{b.key.camelot})")
    else:
        bits.append(f"key change {a.key.camelot}→{b.key.camelot} (handled with EQ/filter)")
    db = b.bpm - a.bpm
    if abs(db) < 1.5:
        bits.append(f"tempo locked ~{a.bpm:.0f} BPM")
    else:
        bits.append(f"tempo {a.bpm:.0f}→{b.bpm:.0f} BPM (beatmatched)")
    de = b.features.energy - a.features.energy
    if de > 0.08:
        bits.append("energy lift")
    elif de < -0.08:
        bits.append("energy release")
    else:
        bits.append("energy held")
    if bd["vibe_similarity"] > 0.5:
        bits.append("similar vibe")
    return "; ".join(bits)
