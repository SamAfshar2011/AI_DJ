"""
Recursive folder scanning + audio-file discovery.

Walks a directory tree, keeps only supported audio extensions, skips hidden /
system junk, and does a light validity probe so obviously-broken files are
filtered before the (expensive) analysis stage.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .audio_io import probe_duration
from .config import MIN_TRACK_SECONDS, SUPPORTED_EXTS
from .utils import file_fingerprint, get_logger

log = get_logger()

_SKIP_DIRS = {"__macosx", ".git", ".svn", "node_modules", ".venv", "cache", "outputs"}


@dataclass
class DiscoveredTrack:
    path: str
    filename: str
    ext: str
    size_bytes: int
    fingerprint: str
    duration: float | None = None
    valid: bool = True
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "filename": self.filename,
            "ext": self.ext,
            "size_bytes": self.size_bytes,
            "fingerprint": self.fingerprint,
            "duration": self.duration,
            "valid": self.valid,
            "error": self.error,
        }


def _is_hidden(name: str) -> bool:
    return name.startswith(".") or name.startswith("._")


def scan_folder(
    folder: str | Path,
    validate: bool = True,
    max_files: int = 5000,
) -> tuple[list[DiscoveredTrack], list[str]]:
    """
    Return (valid_tracks, warnings).

    ``validate`` does a cheap duration probe and drops files that can't be read
    or are shorter than MIN_TRACK_SECONDS.  Warnings are human-readable strings
    surfaced to the UI.
    """
    folder = Path(folder).expanduser()
    warnings: list[str] = []
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")

    found: list[DiscoveredTrack] = []
    seen = 0
    for root, dirs, files in os.walk(folder):
        # prune noise directories in-place
        dirs[:] = [d for d in dirs if d.lower() not in _SKIP_DIRS and not _is_hidden(d)]
        for name in sorted(files):
            if _is_hidden(name):
                continue
            ext = Path(name).suffix.lower()
            if ext not in SUPPORTED_EXTS:
                continue
            seen += 1
            if seen > max_files:
                warnings.append(f"Stopped after {max_files} files (folder is very large).")
                return found, warnings
            fpath = Path(root) / name
            try:
                size = fpath.stat().st_size
                fp = file_fingerprint(fpath)
            except OSError as e:
                warnings.append(f"Skipped unreadable file {name}: {e}")
                continue
            if size < 4096:
                warnings.append(f"Skipped tiny/empty file: {name}")
                continue

            track = DiscoveredTrack(
                path=str(fpath.resolve()),
                filename=name,
                ext=ext,
                size_bytes=size,
                fingerprint=fp,
            )
            if validate:
                dur = probe_duration(fpath)
                track.duration = dur
                if dur is None:
                    # header couldn't be read but ffmpeg may still decode it later;
                    # keep it but flag it so analysis can decide.
                    track.error = "duration probe failed (will retry on decode)"
                elif dur < MIN_TRACK_SECONDS:
                    track.valid = False
                    track.error = f"too short ({dur:.0f}s)"
                    warnings.append(f"Skipped short track ({dur:.0f}s): {name}")
            found.append(track)

    valid = [t for t in found if t.valid]
    if not valid:
        warnings.append("No usable audio tracks were found in the selected folder.")
    log.info("Scanned %s: %d candidate files, %d usable", folder, len(found), len(valid))
    return valid, warnings
