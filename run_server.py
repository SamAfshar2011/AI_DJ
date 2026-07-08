#!/usr/bin/env python3
"""
Convenience launcher for the AI DJ backend.

    python run_server.py [--host 127.0.0.1] [--port 8000] [--open]

Then open http://127.0.0.1:8000 in your browser.  The same app object is used by
`server.ipynb` for people who prefer notebooks.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

# ensure project root on path when run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    ap = argparse.ArgumentParser(description="AI DJ local server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--open", action="store_true", help="open the UI in a browser")
    ap.add_argument("--reload", action="store_true", help="dev auto-reload")
    args = ap.parse_args()

    import uvicorn

    url = f"http://{args.host}:{args.port}"
    print("\n  ▶  AI DJ studio starting…")
    print(f"  ▶  Open  {url}\n")

    if args.open:
        def _open():
            time.sleep(1.5)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run("app.api:app", host=args.host, port=args.port,
                reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()
