"""
End-to-end job runner: folder -> scan -> analyse -> plan -> transitions ->
render, driving the ProgressManager the whole way so the UI stays live.

Designed to run in a background thread (see api.py).  Every stage is guarded so
a single bad track degrades gracefully instead of killing the job.
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path

from .analyzer import analyze_track, TrackAnalysis
from .config import MixSettings, OUTPUTS_DIR
from .planner import plan_set
from .progress import PROGRESS
from .render import render_mix
from .scanner import scan_folder
from .transition_engine import plan_all_transitions
from .utils import dump_json, get_logger

log = get_logger()


def run_mix_job(folder: str, settings: MixSettings, output_stub: str = "aidj_mix") -> dict:
    """Run the full pipeline.  Returns the render result dict; updates PROGRESS."""
    t0 = time.time()
    try:
        settings = settings.resolved()

        # report which analysis models are active (CNN/loaded vs DSP/fallback)
        from .analyzer import model_status
        ms = model_status()
        log.info("active models -> vibe=%s  mood=%s  dance=%s",
                 ms["vibe"], ms["mood"], ms["dance"])

        # ---- 1. scan ----------------------------------------------------
        PROGRESS.update(stage="scanning", stage_progress=0.1, message="Scanning folder…")
        tracks, warnings = scan_folder(folder, validate=True)
        for w in warnings:
            PROGRESS.add_warning(w)
        if len(tracks) < 2:
            PROGRESS.fail(f"Need at least 2 usable tracks (found {len(tracks)}).")
            return {}
        PROGRESS.update(stage="scanning", stage_progress=1.0,
                        message=f"Found {len(tracks)} tracks")

        # ---- 2. analyse -------------------------------------------------
        PROGRESS.update(stage="analysis", stage_progress=0.0,
                        message="Analysing tracks (BPM, key, energy, vibe)…")
        analyses: list[TrackAnalysis] = []
        n = len(tracks)
        for i, tr in enumerate(tracks):
            try:
                a = analyze_track(tr.path)
                analyses.append(a)
                PROGRESS.add_track(a.summary())
            except Exception as e:  # noqa: BLE001 — one bad track must not kill the job
                PROGRESS.add_warning(f"Skipped {Path(tr.path).name}: {e}")
                log.warning("analysis failed for %s: %s", tr.path, e)
            PROGRESS.update(stage="analysis", stage_progress=(i + 1) / n,
                            message=f"Analysed {i+1}/{n}")

        if len(analyses) < 2:
            PROGRESS.fail(f"Only {len(analyses)} track(s) analysed successfully; need 2+.")
            return {}

        # ---- 3. plan set order -----------------------------------------
        PROGRESS.update(stage="planning", stage_progress=0.3,
                        message="AI planning the set order…")
        plan = plan_set(analyses, settings)
        ordered = plan.ordered()
        PROGRESS.update(stage="planning", stage_progress=1.0, plan=plan.to_dict(),
                        message=f"Set ordered ({plan.energy_style} energy)")

        # ---- 4. transitions --------------------------------------------
        PROGRESS.update(stage="transitions", stage_progress=0.3,
                        message="Designing beatmatched transitions…")
        plans, deck_rates = plan_all_transitions(ordered, settings)
        plan_dict = plan.to_dict()
        plan_dict["transition_details"] = [p.to_dict() for p in plans]
        plan_dict["deck_rates"] = [round(r, 4) for r in deck_rates]
        PROGRESS.update(stage="transitions", stage_progress=1.0, plan=plan_dict,
                        message=f"{len(plans)} transitions planned")

        # persist the analysis/plan report
        try:
            report = {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "models": ms,       # vibe/mood/dance: which engine produced the analysis
                "settings": settings.__dict__,
                "plan": plan_dict,
                "tracks": [a.summary() for a in ordered],
            }
            dump_json(OUTPUTS_DIR / f"{output_stub}_report.json", report)
        except Exception as e:
            log.debug("report write failed: %s", e)

        # ---- 5. render --------------------------------------------------
        PROGRESS.update(stage="rendering", stage_progress=0.0,
                        message="Rendering continuous mix…")

        def _cb(p, msg):
            PROGRESS.update(stage="rendering", stage_progress=p, message=msg)

        result = render_mix(ordered, plans, deck_rates, settings,
                            output_stub=output_stub, progress_cb=_cb)

        # ---- 6. finalize ------------------------------------------------
        PROGRESS.update(stage="finalizing", stage_progress=0.6,
                        message="Verifying output integrity…")
        result["elapsed"] = round(time.time() - t0, 1)
        result["report"] = str(OUTPUTS_DIR / f"{output_stub}_report.json")
        result["n_tracks"] = len(ordered)
        result["models"] = ms       # {"vibe": cnn|dsp, "mood": loaded|fallback, "dance": ...}
        if not result.get("checks", {}).get("passed", False):
            PROGRESS.add_warning("Output integrity checks reported issues — see result.checks")

        PROGRESS.update(stage="complete", stage_progress=1.0, result=result,
                        message="Mix complete!")
        log.info("job complete in %.1fs -> %s", result["elapsed"], result.get("wav"))
        return result

    except Exception as e:  # noqa: BLE001
        log.error("pipeline crashed: %s\n%s", e, traceback.format_exc())
        PROGRESS.fail(f"Pipeline error: {e}")
        return {}
