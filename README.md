# Webcam 3D Mapper

Turn an ordinary webcam into a 3D scanner. Point it at something, move slowly around it, and
get back an interactive 3D point cloud — reconstructed locally by [COLMAP](https://colmap.github.io/).

```
WEBCAM  →  SCAN  →  COLMAP  →  INTERACTIVE POINT CLOUD
```

Everything runs on your machine. No cloud, no accounts, no database, no telemetry. The server
binds to `127.0.0.1` and nothing leaves the computer.

---

## What it is

A small FastAPI app that serves a single-page browser UI. The browser owns the webcam, the
preview and the WebGL viewer; Python owns frame capture, quality filtering, the scan workspace
and the COLMAP pipeline.

- **Capture** — ~2.5 candidate frames/second at 1280×720, quality-filtered on the way in.
- **Reconstruct** — `feature_extractor` → `sequential_matcher` → `mapper` → `model_converter`.
- **View** — Three.js point cloud plus the reconstructed camera trajectory, with orbit, zoom,
  pan, reset, point-size control and an orientation gizmo.

### What it is not

This is **structure-from-motion photogrammetry**, not depth sensing. A webcam has no depth
sensor: 3D structure is inferred from how the scene shifts as the camera moves. That means:

- A single-camera reconstruction has **no true real-world scale**. Distances in the viewer are
  relative, not metric. Do not use it to measure anything.
- Version 1 stops at a **sparse** point cloud. No dense stereo, no mesh, no texture.

---

## Requirements

| | |
|---|---|
| **Python** | 3.11 or newer (developed and tested on 3.14) |
| **COLMAP** | 3.8+ or 4.x — see [Installing COLMAP](#installing-colmap) |
| **Webcam** | any device the browser can open |
| **Browser** | Chrome, Edge or Firefox (needs `getUserMedia` + WebGL2 + import maps) |
| **GPU** | not required; the CPU-only COLMAP build works fine |

---

## Setup and running

**Windows**

```powershell
scripts\run.bat
```

**macOS / Linux**

```bash
./scripts/run.sh
```

The script verifies Python, creates `.venv`, installs dependencies on first run, starts the
server and opens <http://127.0.0.1:8765>.

Doing it by hand:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt   # Windows
# .venv/bin/python -m pip install -r requirements.txt     # macOS / Linux

.venv\Scripts\python app.py --open
```

Useful flags: `--port 9000`, `--host` (defaults to `127.0.0.1`; anything else exposes the app
beyond this machine), `--reload` for development.

---

## Installing COLMAP

The app never installs COLMAP for you. It searches `PATH`, the usual install directories, the
`COLMAP_PATH` environment variable and `colmap_path.txt`, then reports exactly where it looked.

**Windows** — download `colmap-x64-windows-nocuda.zip` (or `-cuda` if you have an NVIDIA card)
from the [releases page](https://github.com/colmap/colmap/releases) and extract it to
`C:\COLMAP`, or to `vendor\colmap\` inside this project. Both are searched automatically.

**macOS** — `brew install colmap`

**Linux** — `sudo apt install colmap`, or build from source.

If it still is not found, open **Settings** in the app and paste the full path to `COLMAP.bat`
(Windows) or the `colmap` binary. The choice is saved to `colmap_path.txt`. The header shows a
green **3D Engine** dot once it is working.

> COLMAP's option names changed between 3.x and 4.x. The app reads `colmap <command> -h` at
> startup and adapts, so both generations work without configuration.

---

## How to scan

1. **Pick a good subject.** Textured, matte, stationary things: a desk, a robot, a machine,
   a plant, a room corner. Avoid blank walls, glass, mirrors, glossy plastic and anything that
   moves.
2. **Light it well.** More light means sharper frames and more features.
3. Press **Start Scan**.
4. **Move slowly around it**, keeping the subject in frame the whole time. Sideways motion is
   what creates 3D structure — turning on the spot does not.
5. Keep a **large overlap** between viewpoints. Neighbouring frames should share most of what
   they see.
6. Scan for **30–60 seconds** (roughly 60–150 frames).
7. Press **Finish Scan** and watch the reconstruction stages.
8. Drag to rotate, scroll to zoom, right-drag to pan. **Export .PLY** saves the cloud;
   **Scan folder** opens the raw files.

**Scan quality** (Fast / Balanced) trades detail for speed. Fast downscales to 1024 px and is
the default; Balanced uses 1600 px and roughly doubles the time.

---

## Troubleshooting

**"No camera detected"** — plug in a webcam and press Retry. If you have several, pick one from
the Camera dropdown.

**"Camera access blocked"** — click the camera icon in the address bar, choose Allow, press
Retry. `127.0.0.1` counts as a secure origin, so no HTTPS setup is needed.

**"Camera is in use"** — close Teams/Zoom/OBS and press Retry.

**"3D reconstruction engine not found"** — see [Installing COLMAP](#installing-colmap). The
Settings panel lists every path that was searched.

**"Not enough images"** — scan for longer. Below 12 accepted frames the app will not even try.

**"Could not build a 3D map"** — COLMAP registered too few images. Almost always the subject or
the motion, not the software:
- move more slowly, with more overlap
- add light
- choose something with visible surface detail
- orbit *around* the subject rather than rotating in place

**Reconstruction is slow** — expected on CPU. Roughly 15–60 s for 40 frames on Fast; Balanced
and larger scans take proportionally longer. A CUDA build plus an NVIDIA GPU is much faster,
and the app switches to GPU SIFT automatically when both are present.

**Nothing visible in the viewer** — the view is auto-framed on load; press **Reset view** if you
have orbited away. Very small clouds may need the **Point size** slider.

---

## Project layout

```
webcam-3d-mapper/
├── app.py                  FastAPI server + API routes
├── backend/
│   ├── config.py           paths, thresholds, quality presets
│   ├── models.py           scan status enum, stats, errors
│   ├── image_quality.py    blur / exposure / novelty filter
│   ├── colmap_runner.py    discovery, `-h` introspection, subprocess execution
│   ├── reconstruction.py   the pipeline + COLMAP model readers
│   └── scan_manager.py     scan sessions and the on-disk workspace
├── frontend/
│   ├── index.html          the whole UI (one page, five states)
│   ├── styles.css
│   ├── app.js              camera, capture loop, state machine
│   ├── viewer.js           Three.js point-cloud viewer
│   └── vendor/             Three.js, OrbitControls, PLYLoader (pinned, local)
├── scans/                  one folder per scan (gitignored)
├── scripts/
│   ├── run.bat / run.sh    launchers
│   ├── make_test_scene.py  renders a synthetic multi-view sequence
│   └── ui_smoke.py         headless-browser smoke test
└── tests/test_mapper.py
```

Each scan gets its own folder:

```
scans/2026-09-24_143001_ab12/
├── images/frame_000001.jpg …
├── database.db
├── sparse/0/               COLMAP model
├── result/map.ply          exported point cloud
├── result/cameras.json     reconstructed camera trajectory
├── logs/colmap.log         raw COLMAP output
└── scan.json               status and statistics
```

No database — the filesystem is the state.

---

## API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/system` | COLMAP readiness, capture settings, presets |
| `POST` | `/api/system/colmap` | set the COLMAP executable path |
| `POST` | `/api/system/colmap/detect` | re-run detection |
| `GET` | `/api/scans` | recent scans |
| `POST` | `/api/scans` | start a scan session |
| `POST` | `/api/scans/{id}/frames` | upload and quality-check one frame |
| `POST` | `/api/scans/{id}/finish` | stop capture, start reconstruction |
| `GET` | `/api/scans/{id}/status` | stage, progress, stats, error |
| `GET` | `/api/scans/{id}/result` | result summary + asset URLs |
| `GET` | `/api/scans/{id}/map.ply` | the point cloud |
| `GET` | `/api/scans/{id}/cameras.json` | camera trajectory |
| `GET` | `/api/scans/{id}/log` | raw COLMAP log |
| `POST` | `/api/scans/{id}/cancel` | cancel a running reconstruction |
| `POST` | `/api/scans/{id}/reveal` | open the scan folder |
| `DELETE` | `/api/scans/{id}` | delete a scan |
| `POST` | `/api/dev/import` | reconstruct a folder of existing images |

Scan IDs are validated against a strict pattern and every resolved path is checked to stay
inside `scans/`, so nothing a browser sends can escape the workspace or reach a command line.

---

## Development

```bash
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest -q
```

Reconstruct without a webcam — either paste a folder path into **Settings → Developer**, or
generate a synthetic scene:

```bash
python scripts/make_test_scene.py --out .cache/testscene --frames 40
```

Full UI smoke test (needs `pip install playwright && python -m playwright install chromium`,
with the server already running):

```bash
python scripts/ui_smoke.py --images .cache/testscene --screenshots .cache/shots
```

---

## Roadmap

Deliberately out of scope for v1: dense stereo, meshing, texturing, loop closure, metric scale,
NeRF / Gaussian splatting, SLAM, depth cameras.

Next most useful additions: an optional dense/mesh step, scan history in the UI, and a live
feature-count indicator during capture so a bad scan is obvious before it finishes.

---

## Licence

Your project's licence applies to this code. Three.js (`frontend/vendor/`) is MIT — see
`frontend/vendor/LICENSE`. COLMAP is a separate installation under its own licence.
