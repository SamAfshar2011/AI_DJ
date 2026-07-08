#!/usr/bin/env python3
"""
Environment & pipeline self-check.

    python validate.py            # checks deps, folders, weights, ffmpeg
    python validate.py --audio    # also renders a tiny 2-track test mix

Exits non-zero if a hard requirement is missing.  Nothing here is destructive.
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

OK, WARN, BAD = "\033[92m✓\033[0m", "\033[93m•\033[0m", "\033[91m✗\033[0m"


def line(sym, msg):
    print(f"  {sym} {msg}")


def check_deps() -> bool:
    print("\nDependencies")
    hard = ["numpy", "scipy", "librosa", "soundfile", "fastapi", "uvicorn", "mutagen"]
    soft = ["pyloudnorm", "soxr", "pyrubberband", "torch", "pydub", "sklearn"]
    ok = True
    for m in hard:
        try:
            importlib.import_module(m); line(OK, m)
        except Exception as e:
            line(BAD, f"{m} MISSING ({e})"); ok = False
    for m in soft:
        try:
            importlib.import_module(m); line(OK, f"{m} (optional)")
        except Exception:
            line(WARN, f"{m} missing — a fallback will be used")
    return ok


def check_binaries():
    print("\nExternal tools")
    line(OK if shutil.which("ffmpeg") else BAD,
         f"ffmpeg {'found' if shutil.which('ffmpeg') else 'MISSING (required for mp3/m4a & MP3 export)'}")
    line(OK if shutil.which("rubberband") else WARN,
         f"rubberband {'found' if shutil.which('rubberband') else 'missing (librosa stretch fallback)'}")


def check_layout():
    print("\nProject layout")
    for name in ["app", "train_models", "models_weights", "datasets",
                 "outputs", "cache", "logs", "index.html", "style.css", "app.js",
                 "server.ipynb", "run_server.py"]:
        p = ROOT / name
        line(OK if p.exists() else BAD, f"{name} {'' if p.exists() else 'MISSING'}")


def check_weights():
    print("\nModel weights (optional)")
    from app.config import MODELS_DIR
    for f, note in [("model_1_genre_cnn.pt", "vibe embedding + genre"),
                    ("model_2_mood_tagger.pt", "mood tags"),
                    ("model_3_danceability.pt", "danceability")]:
        p = MODELS_DIR / f
        line(OK if p.exists() else WARN,
             f"{f} — {'present' if p.exists() else 'not trained (DSP fallback active)'} [{note}]")

    # live runtime status: which engine each model actually resolves to
    try:
        from app.analyzer import model_status
        st = model_status()
        line(OK if st["vibe"] == "cnn" else WARN, f"runtime vibe  = {st['vibe']}   (cnn=trained, dsp=fallback)")
        line(OK if st["mood"] == "loaded" else WARN, f"runtime mood  = {st['mood']} (loaded=trained, fallback=off)")
        line(OK if st["dance"] == "loaded" else WARN, f"runtime dance = {st['dance']} (loaded=trained, fallback=DSP)")
    except Exception as e:  # noqa: BLE001
        line(WARN, f"could not query runtime model status: {e}")


def check_import():
    print("\nBackend import")
    try:
        from app import api  # noqa: F401
        line(OK, "app.api imported (FastAPI app builds)")
        return True
    except Exception as e:
        line(BAD, f"import failed: {e}")
        return False


def check_audio_pipeline():
    print("\nAudio pipeline (synthetic 2-track render)")
    import numpy as np
    import soundfile as sf
    from app.config import MixSettings
    from app.progress import PROGRESS
    from app.pipeline import run_mix_job

    tmp = Path(tempfile.mkdtemp(prefix="aidj_val_"))
    sr = 44100
    for i, bpm in enumerate((120, 124)):
        n = sr * 25
        t = np.arange(n) / sr
        spb = int(sr * 60 / bpm)
        y = 0.2 * np.sin(2 * np.pi * (220 + 40 * i) * t)
        for k in range(0, n, spb):
            e = np.exp(-np.arange(min(spb, n - k)) / (0.08 * sr))
            y[k:k + len(e)] += 0.7 * np.sin(2 * np.pi * 60 * np.arange(len(e)) / sr) * e
        y = (y / (np.abs(y).max() + 1e-9) * 0.9).astype("float32")
        sf.write(tmp / f"t{i}.wav", np.stack([y, y]).T, sr)

    PROGRESS.start("validate")
    res = run_mix_job(str(tmp), MixSettings.from_dict({"transition_intensity": 0.5}),
                      output_stub="validate_mix")
    ok = bool(res) and res.get("checks", {}).get("passed")
    if ok:
        line(OK, f"rendered {res['duration']}s mix, LUFS {res['lufs']}, peak {res['peak_dbfs']} dBFS")
        line(OK, "integrity checks passed")
    else:
        line(BAD, f"render failed or checks failed: {res.get('checks') if res else 'no result'}")
    shutil.rmtree(tmp, ignore_errors=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", action="store_true", help="also run a synthetic render test")
    args = ap.parse_args()

    print("=" * 52)
    print("  AI DJ — validation")
    print("=" * 52)
    deps_ok = check_deps()
    check_binaries()
    check_layout()
    import_ok = check_import()
    if import_ok:
        check_weights()
    audio_ok = True
    if args.audio and import_ok:
        try:
            audio_ok = check_audio_pipeline()
        except Exception as e:
            line(BAD, f"audio test crashed: {e}"); audio_ok = False

    print("\n" + "=" * 52)
    hard_ok = deps_ok and import_ok and audio_ok
    print(f"  RESULT: {'READY ✅' if hard_ok else 'ISSUES FOUND ⚠️  (see above)'}")
    print("=" * 52 + "\n")
    sys.exit(0 if hard_ok else 1)


if __name__ == "__main__":
    main()
