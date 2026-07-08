"""
Thread-safe job/progress state shared between the worker thread that builds a
mix and the API endpoints (polling + WebSocket) that report it to the UI.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .utils import get_logger

log = get_logger()

STAGES = [
    "idle", "scanning", "metadata", "analysis", "planning",
    "transitions", "rendering", "finalizing", "complete", "error",
]

# rough weights so the overall bar advances sensibly across stages
_STAGE_WEIGHT = {
    "scanning": 0.03, "metadata": 0.02, "analysis": 0.55, "planning": 0.05,
    "transitions": 0.05, "rendering": 0.25, "finalizing": 0.05,
}
_STAGE_ORDER = ["scanning", "metadata", "analysis", "planning",
                "transitions", "rendering", "finalizing"]


@dataclass
class JobState:
    job_id: str
    stage: str = "idle"
    stage_progress: float = 0.0   # 0..1 within the current stage
    overall: float = 0.0          # 0..1 across all stages
    message: str = ""
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    eta_seconds: Optional[float] = None
    error: Optional[str] = None
    warnings: list = field(default_factory=list)
    tracks: list = field(default_factory=list)     # summaries as they analyse
    plan: Optional[dict] = None
    result: Optional[dict] = None                  # final output info
    done: bool = False

    def snapshot(self) -> dict:
        elapsed = time.time() - self.started_at
        return {
            "job_id": self.job_id,
            "stage": self.stage,
            "stage_progress": round(self.stage_progress, 4),
            "overall": round(self.overall, 4),
            "message": self.message,
            "elapsed": round(elapsed, 1),
            "eta_seconds": None if self.eta_seconds is None else round(self.eta_seconds, 1),
            "error": self.error,
            "warnings": self.warnings[-20:],
            "tracks": self.tracks,
            "plan": self.plan,
            "result": self.result,
            "done": self.done,
        }


class ProgressManager:
    """One live job at a time (single-user local app), guarded by a lock."""

    def __init__(self):
        self._lock = threading.Lock()
        self._job: Optional[JobState] = None

    def start(self, job_id: str) -> JobState:
        with self._lock:
            self._job = JobState(job_id=job_id, stage="scanning", message="Starting…")
            return self._job

    @property
    def job(self) -> Optional[JobState]:
        return self._job

    def _recompute_overall(self, job: JobState) -> None:
        base = 0.0
        for s in _STAGE_ORDER:
            if s == job.stage:
                base += _STAGE_WEIGHT[s] * job.stage_progress
                break
            base += _STAGE_WEIGHT[s]
        job.overall = min(0.999, base) if job.stage != "complete" else 1.0
        # ETA from elapsed / overall
        elapsed = time.time() - job.started_at
        if 0.02 < job.overall < 0.999:
            total_est = elapsed / job.overall
            job.eta_seconds = max(0.0, total_est - elapsed)

    def update(
        self,
        stage: Optional[str] = None,
        stage_progress: Optional[float] = None,
        message: Optional[str] = None,
        **extra: Any,
    ) -> None:
        with self._lock:
            job = self._job
            if job is None:
                return
            if stage is not None:
                job.stage = stage
                if stage_progress is None:
                    stage_progress = 0.0
            if stage_progress is not None:
                job.stage_progress = max(0.0, min(1.0, stage_progress))
            if message is not None:
                job.message = message
            for k, v in extra.items():
                if hasattr(job, k):
                    setattr(job, k, v)
            job.updated_at = time.time()
            if stage == "complete":
                job.overall = 1.0
                job.done = True
                job.eta_seconds = 0.0
            elif stage == "error":
                job.done = True
                job.error = message
            else:
                self._recompute_overall(job)

    def add_track(self, summary: dict) -> None:
        with self._lock:
            if self._job is not None:
                self._job.tracks.append(summary)

    def add_warning(self, msg: str) -> None:
        with self._lock:
            if self._job is not None:
                self._job.warnings.append(msg)
                log.warning("job warning: %s", msg)

    def fail(self, msg: str) -> None:
        self.update(stage="error", message=msg)
        log.error("job failed: %s", msg)

    def snapshot(self) -> dict:
        with self._lock:
            if self._job is None:
                return {"stage": "idle", "overall": 0.0, "done": False, "tracks": []}
            return self._job.snapshot()


# process-wide singleton
PROGRESS = ProgressManager()
