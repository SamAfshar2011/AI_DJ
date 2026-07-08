"""
Tag / container metadata extraction via mutagen with an ffprobe fallback.

Returns a clean dict of title/artist/album/duration/sample_rate/channels/
bitrate.  Missing values degrade gracefully (title falls back to the filename).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from .utils import get_logger

log = get_logger()

try:
    import mutagen
    from mutagen import File as MutagenFile

    _HAVE_MUTAGEN = True
except Exception:  # pragma: no cover
    _HAVE_MUTAGEN = False

_FFPROBE = shutil.which("ffprobe")

# common tag keys across ID3 / MP4 / Vorbis
_TITLE_KEYS = ("title", "TIT2", "\xa9nam", "TITLE")
_ARTIST_KEYS = ("artist", "TPE1", "\xa9ART", "ARTIST", "albumartist", "TPE2")
_ALBUM_KEYS = ("album", "TALB", "\xa9alb", "ALBUM")


def _first_tag(tags: Any, keys) -> str | None:
    if tags is None:
        return None
    for k in keys:
        try:
            if k in tags:
                v = tags[k]
                if isinstance(v, (list, tuple)) and v:
                    v = v[0]
                v = str(v).strip()
                if v:
                    return v
        except Exception:
            continue
    return None


def _ffprobe_meta(path: str) -> dict:
    if not _FFPROBE:
        return {}
    try:
        import json

        out = subprocess.run(
            [_FFPROBE, "-v", "error", "-show_format", "-show_streams",
             "-of", "json", path],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(out.stdout or "{}")
    except Exception:
        return {}
    res: dict = {}
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {})
    if audio:
        res["sample_rate"] = int(audio.get("sample_rate", 0)) or None
        res["channels"] = audio.get("channels")
    fmt = data.get("format", {})
    if fmt.get("bit_rate"):
        res["bitrate"] = int(fmt["bit_rate"])
    if fmt.get("duration"):
        try:
            res["duration"] = float(fmt["duration"])
        except ValueError:
            pass
    tags = {**fmt.get("tags", {}), **audio.get("tags", {})}
    lower = {k.lower(): v for k, v in tags.items()}
    res["title"] = lower.get("title")
    res["artist"] = lower.get("artist") or lower.get("album_artist")
    res["album"] = lower.get("album")
    return res


def extract_metadata(path: str | Path) -> dict:
    path = str(path)
    name = Path(path).stem
    meta: dict = {
        "title": None, "artist": None, "album": None,
        "duration": None, "sample_rate": None, "channels": None, "bitrate": None,
    }

    if _HAVE_MUTAGEN:
        try:
            mf = MutagenFile(path)
            if mf is not None:
                meta["title"] = _first_tag(mf.tags, _TITLE_KEYS)
                meta["artist"] = _first_tag(mf.tags, _ARTIST_KEYS)
                meta["album"] = _first_tag(mf.tags, _ALBUM_KEYS)
                if mf.info is not None:
                    meta["duration"] = getattr(mf.info, "length", None)
                    meta["sample_rate"] = getattr(mf.info, "sample_rate", None)
                    meta["channels"] = getattr(mf.info, "channels", None)
                    br = getattr(mf.info, "bitrate", None)
                    meta["bitrate"] = int(br) if br else None
        except Exception as e:
            log.debug("mutagen failed on %s: %s", name, e)

    # fill gaps with ffprobe
    if not all([meta["duration"], meta["sample_rate"], meta["channels"]]):
        for k, v in _ffprobe_meta(path).items():
            if meta.get(k) in (None, 0):
                meta[k] = v

    # sensible fallbacks
    if not meta["title"]:
        # strip common "artist - title" filename convention
        if " - " in name:
            parts = name.split(" - ", 1)
            if not meta["artist"]:
                meta["artist"] = parts[0].strip()
            meta["title"] = parts[1].strip()
        else:
            meta["title"] = name
    if not meta["artist"]:
        meta["artist"] = "Unknown Artist"
    return meta
