"""
FastAPI backend.

Serves the liquid-glass UI, exposes the analysis/render pipeline and streams
progress over both a WebSocket and a polling endpoint.  Runs a single mix job at
a time in a background thread (this is a local, single-user studio app).

Endpoints
  GET  /                     -> index.html
  GET  /style.css /app.js    -> static assets
  POST /api/pick-folder      -> native OS folder chooser (macOS/most Linux)
  POST /api/scan             -> quick folder scan (counts + filenames)
  POST /api/generate         -> start a mix job (background)
  GET  /api/progress         -> latest progress snapshot (polling)
  WS   /ws/progress          -> live progress stream
  POST /api/upload           -> webkitdirectory upload fallback (no path access)
  GET  /api/download         -> download a rendered file
  GET  /outputs/<file>       -> serve rendered audio to the player
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import CACHE_DIR, MixSettings, OUTPUTS_DIR, STATIC_ROOT
from .pipeline import run_mix_job
from .progress import PROGRESS
from .utils import get_logger

log = get_logger()

_JOB_LOCK = threading.Lock()
_JOB_THREAD: threading.Thread | None = None
UPLOAD_ROOT = CACHE_DIR / "uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# request models
# ---------------------------------------------------------------------------
class ScanRequest(BaseModel):
    folder: str


class GenerateRequest(BaseModel):
    folder: str
    settings: dict | None = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _native_folder_dialog() -> str | None:
    """Open an OS-native 'choose folder' dialog.  Returns a path or None."""
    import sys

    # macOS: AppleScript is reliable and needs no GUI main-thread juggling
    if sys.platform == "darwin":
        try:
            script = 'POSIX path of (choose folder with prompt "Select your music folder")'
            out = subprocess.run(["osascript", "-e", script],
                                 capture_output=True, text=True, timeout=120)
            path = out.stdout.strip()
            return path or None
        except Exception as e:
            log.debug("osascript picker failed: %s", e)
    # Linux: try zenity / kdialog if present
    for tool, args in (("zenity", ["--file-selection", "--directory"]),
                       ("kdialog", ["--getexistingdirectory", str(Path.home())])):
        if shutil.which(tool):
            try:
                out = subprocess.run([tool, *args], capture_output=True, text=True, timeout=120)
                path = out.stdout.strip()
                if path:
                    return path
            except Exception:
                continue
    return None


def _job_running() -> bool:
    return _JOB_THREAD is not None and _JOB_THREAD.is_alive()


def _safe_output(path: str) -> Path:
    p = Path(path).resolve()
    if OUTPUTS_DIR.resolve() not in p.parents and p.parent != OUTPUTS_DIR.resolve():
        raise HTTPException(403, "path outside outputs directory")
    if not p.exists():
        raise HTTPException(404, "file not found")
    return p


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(title="AI DJ", version="1.0.0")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        idx = STATIC_ROOT / "index.html"
        if not idx.exists():
            return HTMLResponse("<h1>AI DJ</h1><p>index.html missing.</p>", status_code=500)
        return HTMLResponse(idx.read_text(encoding="utf-8"))

    @app.get("/style.css")
    async def style():
        return FileResponse(STATIC_ROOT / "style.css", media_type="text/css")

    @app.get("/app.js")
    async def appjs():
        return FileResponse(STATIC_ROOT / "app.js", media_type="application/javascript")

    @app.get("/api/health")
    async def health():
        from .analyzer import model_status

        return {"ok": True, "job_running": _job_running(), "version": "1.0.0",
                "models": model_status()}

    @app.get("/api/models")
    async def models():
        """Which analysis models are active right now (CNN/loaded vs DSP/fallback)."""
        from .analyzer import active_model_signature, model_status

        return {"ok": True, "models": model_status(), "signature": active_model_signature()}

    @app.post("/api/pick-folder")
    async def pick_folder():
        path = await asyncio.to_thread(_native_folder_dialog)
        if not path:
            return JSONResponse({"ok": False, "error": "No folder chosen or native picker unavailable."})
        if not Path(path).is_dir():
            return JSONResponse({"ok": False, "error": "Chosen path is not a folder."})
        return {"ok": True, "folder": path}

    @app.post("/api/scan")
    async def scan(req: ScanRequest):
        from .scanner import scan_folder

        try:
            tracks, warnings = await asyncio.to_thread(scan_folder, req.folder, True)
        except Exception as e:
            raise HTTPException(400, str(e))
        return {
            "ok": True,
            "count": len(tracks),
            "warnings": warnings,
            "tracks": [{"filename": t.filename, "ext": t.ext,
                        "duration": t.duration, "size_bytes": t.size_bytes}
                       for t in tracks[:500]],
        }

    @app.post("/api/generate")
    async def generate(req: GenerateRequest):
        global _JOB_THREAD
        with _JOB_LOCK:
            if _job_running():
                raise HTTPException(409, "A mix is already being generated.")
            folder = req.folder
            if not folder or not Path(folder).is_dir():
                raise HTTPException(400, "Folder not found. Provide a valid folder path.")
            settings = MixSettings.from_dict(req.settings or {})
            job_id = uuid.uuid4().hex[:12]
            PROGRESS.start(job_id)
            stub = f"aidj_mix_{job_id}"

            def _worker():
                run_mix_job(folder, settings, output_stub=stub)

            _JOB_THREAD = threading.Thread(target=_worker, name="mixjob", daemon=True)
            _JOB_THREAD.start()
            return {"ok": True, "job_id": job_id}

    @app.get("/api/progress")
    async def progress():
        return PROGRESS.snapshot()

    @app.websocket("/ws/progress")
    async def ws_progress(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                snap = PROGRESS.snapshot()
                await ws.send_json(snap)
                if snap.get("done"):
                    # send a couple final frames then close
                    await asyncio.sleep(0.3)
                    await ws.send_json(PROGRESS.snapshot())
                    break
                await asyncio.sleep(0.4)
        except WebSocketDisconnect:
            pass
        except Exception as e:  # noqa: BLE001
            log.debug("ws closed: %s", e)

    @app.post("/api/upload")
    async def upload(files: list[UploadFile] = File(...)):
        """Fallback for browsers/sandboxes that can't hand us a folder path."""
        sess = UPLOAD_ROOT / uuid.uuid4().hex[:10]
        sess.mkdir(parents=True, exist_ok=True)
        saved = 0
        for f in files:
            name = Path(f.filename or "track").name
            if not name:
                continue
            dest = sess / name
            try:
                with dest.open("wb") as fh:
                    while chunk := await f.read(1 << 20):
                        fh.write(chunk)
                saved += 1
            except Exception as e:
                log.warning("upload save failed for %s: %s", name, e)
        return {"ok": True, "folder": str(sess), "saved": saved}

    @app.get("/api/download")
    async def download(path: str):
        p = _safe_output(path)
        return FileResponse(str(p), filename=p.name, media_type="application/octet-stream")

    # serve rendered audio to the <audio> player
    app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

    return app


app = create_app()
