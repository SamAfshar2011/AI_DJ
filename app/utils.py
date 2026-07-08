"""
Small, dependency-light helpers shared across the code base: logging setup,
content hashing for the analysis cache, JSON-safe serialisation and numeric
guards.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from .config import LOGS_DIR

_LOG_CONFIGURED = False


def get_logger(name: str = "aidj") -> logging.Logger:
    """Return a process-wide logger that writes to both stderr and logs/aidj.log."""
    global _LOG_CONFIGURED
    logger = logging.getLogger(name)
    if not _LOG_CONFIGURED:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        try:
            fh = logging.FileHandler(LOGS_DIR / "aidj.log", encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except Exception:  # pragma: no cover - logging must never crash the app
            pass
        logger.propagate = False
        _LOG_CONFIGURED = True
    return logger


log = get_logger()


def file_fingerprint(path: str | Path) -> str:
    """
    Cheap, stable identity for a track based on absolute path + size + mtime.
    Used as the cache key so an unchanged file is never re-analysed.
    """
    p = Path(path)
    st = p.stat()
    raw = f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:20]


def weight_signature(path: str | Path) -> str:
    """
    Short identity of a model-weight file (size + mtime) or "none" if absent.

    Used by the analysis cache: when a model is (re)trained the weight file
    changes, its signature changes, and any cached analysis produced with the
    old model is automatically refreshed — no manual cache clearing required.
    """
    p = Path(path)
    if not p.exists():
        return "none"
    try:
        st = p.stat()
        raw = f"{st.st_size}|{int(st.st_mtime)}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()[:10]
    except OSError:
        return "none"


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def amp_to_db(amp: float, floor_db: float = -120.0) -> float:
    amp = max(abs(amp), 1e-12)
    return max(floor_db, 20.0 * math.log10(amp))


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy / Path / set types into JSON-serialisable data."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        return safe_float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        return safe_float(obj)
    return obj


def dump_json(path: str | Path, data: Any) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(data), indent=2), encoding="utf-8")
    tmp.replace(path)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def human_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
