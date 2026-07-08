<div align="center">

# 🎧 AI DJ — Autonomous Mix Studio

**Point it at a folder of music. It analyses every track, plans a musical
journey, beatmatches and blends them with tasteful EQ + effects, and renders a
continuous, mastered, download-ready DJ mix — all locally, in a liquid-glass UI.**

librosa · Rubber Band · PyTorch · scipy · pyloudnorm · FastAPI

</div>

---

## What it does

1. **Discovers** audio in a folder (recursive) — mp3, wav, flac, m4a, aac, ogg…
2. **Analyses** each track: BPM, beat grid, downbeats, phrase boundaries, musical
   key (Camelot), integrated loudness (LUFS), spectral balance, energy curve,
   onset rate, danceability, a vocal-presence proxy, structural sections, and a
   64-d **vibe embedding**.
3. **Plans the set** — orders tracks so the mix flows: tempo-compatible,
   harmonically related (Camelot wheel), vibe-similar, and shaped to a chosen
   **energy journey** (rising / wave / peak / steady / wind-down).
4. **Designs each transition** — exit/entry on **phrase boundaries**, chained
   constant-rate **beatmatching** (pitch-preserving Rubber Band stretch, bounded
   so quality holds), and a technique per pair (harmonic blend / EQ blend /
   creative filter-echo / clean cut).
5. **Applies EQ + effects** — the signature move is a **bass-swap** so two low
   ends never clash; plus filter sweeps, echo throws, reverb tails, gentle glue
   saturation and stereo width — all conservative and headroom-aware.
6. **Masters & renders** — per-track loudness match → equal-power crossfades →
   DC-block → integrated-loudness normalise to **−14 LUFS** → look-ahead soft
   **limiter** (provably respects a −0.3 dBFS ceiling) → −1 dBFS peak safety →
   automated integrity checks → **WAV (24-bit) + MP3 (320k)** in `outputs/`.

Everything runs on your machine. Nothing is uploaded anywhere.

---

## Quick start

```bash
# 1. from the project folder, with the bundled virtualenv:
source .venv/bin/activate            # or use .venv/bin/python directly

# (fresh machine instead? python3.12 -m venv .venv && pip install -r requirements.txt)

# 2. make sure ffmpeg is installed (and, ideally, rubberband):
#    macOS:   brew install ffmpeg rubberband
#    Ubuntu:  sudo apt-get install ffmpeg rubberband-cli

# 3. sanity-check the environment (optional but recommended):
python validate.py --audio

# 4. launch:
python run_server.py --open
#    → open http://127.0.0.1:8000
```

Prefer notebooks? Open **`server.ipynb`** and run the cells — it starts the same
server (auto-picking a free port) and can also render a mix headless.

> The launcher honours a `PORT` env var and `--port` flag if 8000 is taken.

---

## Using the UI

1. **Select your music** — click **Browse folder…** (native OS picker), paste an
   absolute path + **Scan**, or **Upload folder** (browser `webkitdirectory`
   fallback for sandboxes that can't share a path).
2. *(Optional)* open **Mix direction** to set the energy journey, transition
   length, effect intensity, harmonic priority, target length, and quality mode.
3. Hit **Generate the mix**. Watch live stages, a liquid progress bar, an
   animated elapsed/ETA timer, per-track analysis cards, and the AI's planned
   set with the reason for every transition.
4. When it's done, play it in the **custom player** (with a real-time
   spectrum visualizer) and **download** the WAV or MP3.

---

## How the beatmatching stays clean (the honest version)

Time-stretching is where auto-DJs usually get crunchy. The approach here:

* A **chained constant-rate** model picks *one* playback rate per track so
  neighbours share a tempo, and **every rate is clamped** to ±6 % (±12 % in
  aggressive mode) of the track's native tempo. Whole tracks are stretched by a
  constant ratio with Rubber Band, so there are **no mid-track tempo jumps** and
  no formant smearing from segment-wise stretching.
* When two tracks are simply too far apart to beatmatch within the limit, the
  engine **does not force a bad stretch** — it switches to a musical
  filter-sweep + echo transition on a phrase boundary, or a clean cut.
* Exits and entries snap to **8-bar phrase boundaries** derived from the beat
  grid + estimated downbeats, which is where blends actually sound intentional.

This won't produce a flawless pro set for *every* arbitrary pair of songs (no
automatic system can) — but it makes musically-justified, technically-clean
choices and tells you what it did.

---

## Project structure

```
AI DJ/
├── app/                     # backend package
│   ├── config.py            # paths, audio constants, MixSettings
│   ├── audio_io.py          # float32 load/save, soxr resample, mp3 export
│   ├── scanner.py           # recursive discovery + validity probe
│   ├── metadata.py          # mutagen + ffprobe tags
│   ├── beatgrid.py          # tempo / beats / downbeats / phrases
│   ├── key_detection.py     # Krumhansl key + Camelot compatibility
│   ├── features.py          # LUFS, energy, structure, danceability, vocalness
│   ├── vibe_model.py        # GenreCNN wrapper + DSP-embedding fallback
│   ├── planner.py           # set ordering (cost matrix + 2-opt + energy curve)
│   ├── transition_engine.py # per-pair timing, beatmatch, technique, deck rates
│   ├── eq_engine.py         # bass-swap + adaptive transition EQ
│   ├── effects.py           # filters, echo, reverb, saturation, limiter…
│   ├── render.py            # assemble, master, integrity-check the mix
│   ├── progress.py          # thread-safe job/progress state
│   ├── pipeline.py          # end-to-end job runner
│   └── api.py               # FastAPI app + WebSocket + static serving
├── train_models/            # optional training notebooks (model_1/2/3)
├── models_weights/          # trained checkpoints (auto-loaded if present)
├── datasets/                # datasets for training (see its README)
├── outputs/                 # rendered mixes + JSON reports
├── cache/                   # per-track analysis cache (fingerprint-keyed)
├── logs/                    # aidj.log
├── index.html · style.css · app.js   # liquid-glass frontend
├── server.ipynb · run_server.py       # launchers
├── validate.py · requirements.txt · README.md
```

---

## Models

| Model | Role | Default (no weights) | Trained upgrade | Runtime wrapper |
|-------|------|---------|---------|---------|
| **Model 1 — Vibe/Genre** (`GenreCNN`) | genre tag + 64-d similarity vector that orders the set | **DSP embedding** (MFCC/chroma/contrast/tonnetz stats) | train `model_1_genre_cnn.ipynb` on **GTZAN** | `app/vibe_model.py` (`mode`: `cnn`/`dsp`) |
| **Model 2 — Mood tagger** (`MoodTaggerNet`) | mood/instrument tags + a mood-similarity term in the planner | **none** (no mood tags; ordering unaffected) | train `model_2_mood_tagger.ipynb` on **MagnaTagATune** | `app/mood_model.py` (`status`: `loaded`/`fallback`) |
| **Model 3 — Danceability** (`GenreCNN` on styles) | danceability score + rhythm style | **DSP** pulse-clarity proxy in `app/features.py` | train `model_3_danceability.ipynb` on **Ballroom** | `app/dance_model.py` (`status`: `loaded`/`fallback`) |
| **Beat / downbeat** | grid + phrases | **librosa** DP tracker + spectral-flux downbeat phase | drop in **madmom** (auto-used if importable) | `app/beatgrid.py` |
| **Key** | harmonic mixing | **Krumhansl-Schmuckler** on CQT chroma | — | `app/key_detection.py` |
| **Loudness** | mastering | **pyloudnorm** (ITU-R BS.1770) | — | `app/features.py` / `app/render.py` |

Training is **entirely optional** — the app is fully functional with zero
weights. See `train_models/`, `models_weights/README.md` and `datasets/README.md`
for datasets and exact folder layouts. No fake download links: only official
dataset pages are referenced.

### Trained models are used automatically — what each does and how to verify

All three CNN models are loaded by fixed path from `models_weights/` and are
**used automatically** the moment the weight file exists. Behaviour is identical
whether you train one, two, all three, or none.

| Trained model | Weight file (exact) | What it changes in the app when present |
|---|---|---|
| Model 1 | `models_weights/model_1_genre_cnn.pt` | per-track **genre** label + a learned 64-d **vibe embedding** replaces the DSP embedding, giving smoother "sounds-alike" ordering. `vibe_tags.vibe_source` → `cnn`. |
| Model 2 | `models_weights/model_2_mood_tagger.pt` | per-track **mood/instrument tags** (`mood_tags`, `moods`, `mood_probs`) **and** a mood-continuity term in `planner.transition_cost` (active only when *both* tracks have a mood vector). `vibe_tags.mood_source` → `loaded`. |
| Model 3 | `models_weights/model_3_danceability.pt` | **danceability** becomes a 50/50 blend of the model score and the DSP proxy, plus a detected **rhythm style** (`dance_style`). `vibe_tags.dance_source` → `model+dsp`. |

**After training you do NOT edit any code or config.** Concretely:

1. **Restart needed?** Yes — the model wrappers are process-level singletons
   (`get_vibe_model` / `get_mood_model` / `get_dance_model`), loaded once per
   server process. Restart `run_server.py` / the `server.ipynb` kernel so the new
   `.pt` file is picked up. (A browser refresh alone is not enough.)
2. **Clear the cache?** No. The analysis cache stores a `model_signature`
   (weight file size+mtime for all three models). When you add, retrain, or
   remove any weight file the signature changes, and `analyzer._load_cached`
   transparently re-analyses affected tracks. Deleting `cache/` still works but
   is unnecessary.
3. **How to verify which engine is active** — any of:
   * **Logs** at server/job start: `active models -> vibe=cnn  mood=loaded  dance=fallback`, plus per-model load lines (`Mood model: loaded MoodTaggerNet (50 tags) on mps`).
   * **`GET /api/models`** → `{"models": {"vibe": "...","mood": "...","dance": "..."}, "signature": "..."}` (also included in `GET /api/health`).
   * **Output JSON**: the render result and `outputs/<mix>_report.json` both carry a top-level `"models"` block, and every track in the report has a `"sources"` field (`{"vibe","mood","dance"}`).
   * **`python validate.py`** prints the active model status.

---

## Audio-quality safeguards

* Internal processing in **float32**; lossless WAV render path; a single final
  lossy pass to MP3 (no repeated re-encoding).
* High-quality resampling (**soxr VHQ**) and pitch-preserving stretch (**Rubber
  Band**).
* Per-track loudness matching (no volume lurches) before crossfades.
* **Equal-power** crossfades; bass-swap EQ so lows never sum to mud/clipping.
* **Look-ahead soft limiter** that mathematically cannot exceed its ceiling,
  then a −1 dBFS true-peak-style safety margin.
* Automated post-render checks: file exists · non-empty · no NaN · no clipping ·
  plausible duration · valid sample rate · reloads successfully — surfaced as
  badges in the UI and in `outputs/<mix>_report.json`.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| MP3s/m4a won't decode, or no MP3 output | Install **ffmpeg** and ensure it's on `PATH`. |
| Beatmatching sounds slightly loose on some pairs | Those tracks were too far apart in tempo to match within the stretch limit — enable **Aggressive remix** for a wider range, or curate more tempo-similar tracks. |
| "Need at least 2 usable tracks" | The folder had <2 decodable tracks ≥20 s. Check formats / lengths. |
| Native folder picker does nothing | macOS uses AppleScript, Linux uses zenity/kdialog. If unavailable, **paste a path** or **Upload folder**. |
| Port 8000 in use | `python run_server.py --port 8010` or set `PORT=8010`. |
| Re-analysis every run | The cache keys on path+size+mtime; editing/moving files invalidates it. Delete `cache/` to force a rebuild. |
| Apple-Silicon torch warnings | Harmless; the vibe model uses **MPS** when available, else CPU. |

Logs: `logs/aidj.log`. Per-mix analysis/plan report: `outputs/<mix>_report.json`.

---

## Limitations & what would push quality further

* **Key/beat detection are estimates.** They're solid on 4/4 dance/pop material,
  weaker on rubato, live, or highly syncopated music. Adding **madmom** (downbeats)
  and a CNN key model would help.
* **No stem separation.** Bass-swap works on full-band EQ. Adding **Demucs**
  source separation would enable true stem-level swaps (e.g. keep the incoming
  vocal, drop the outgoing one) and cleaner vocal-clash avoidance.
* **Beatmatch is bounded by design** (quality-first). A per-beat dynamic tempo
  warp (elastic beat-grid alignment) would let very different tempos meet, at
  some artifact risk.
* **Structure detection is generic.** A trained boundary/segment model (e.g. on
  SALAMI) would place drops/breakdowns more precisely for smarter mix points.
* **Vibe model** improves with training (GTZAN → MagnaTagATune → your own
  library). CLAP/OpenL3 embeddings could be swapped into `vibe_model.py` for
  even better "sounds like" ordering.

Built to be honest about all of the above — the UI shows the technique and
reasoning for every transition so you can hear *and see* what it decided.
