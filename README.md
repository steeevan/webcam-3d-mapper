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
- **Live tracking** — every frame is checked for texture, overlap with recent views, and
  parallax. A HUD, a keypoint overlay and plain-language guidance tell you *while scanning*
  when the scan is heading for trouble (see [Live tracking](#live-tracking)).
- **Live 3D preview** — while you scan, a small, fast COLMAP run rebuilds the map in the
  background and shows the growing point cloud plus **"X of Y frames placed"**, straight from
  COLMAP (see [Live 3D preview](#live-3d-preview)).
- **Coverage guide** — a compass around the subject shows which sides are covered and which
  way to walk next, based on the cameras of the latest preview.
- **Reconstruct** — `feature_extractor` → `sequential_matcher` → `mapper` → `model_converter`.
- **View** — Three.js point cloud, loaded upright, with the reconstructed cameras drawn as
  frustums. Orbit, zoom, pan, auto-spin, point-size control and an orientation gizmo.
- **Library** — every scan stays on disk and is listed in a sidebar. Click one to reopen it,
  delete the ones you don't need, or press **Compare** and pick two to view them side by side
  with a stats table (see [Comparing scans](#comparing-scans)).
- **Find record** — find number, site, context, material, date, recorder and notes, entered
  before or after scanning. The library shows the find number and searches by it (see
  [Find records](#find-records)).
- **Real-world scale** — print the marker board, scan the find on it, and the model gets
  millimetres, a scale bar and a two-click measuring tool (see
  [Real-world scale](#real-world-scale-the-marker-board)).
- **Find report** — a two-page PDF: top, front, side and bottom views with mm scale bars, a
  photograph, the record and the capture log (see [Find report](#find-report)).
- **Dense cloud (NVIDIA GPU, opt-in)** — after the scan is complete, COLMAP's patch-match
  stereo can add a dense point cloud in the same model frame, shown with a Sparse / Dense
  toggle (see [Dense cloud](#dense-cloud-nvidia-gpu)).

### What it is not

This is **structure-from-motion photogrammetry**, not depth sensing. A webcam has no depth
sensor: 3D structure is inferred from how the scene shifts as the camera moves. That means:

- A single-camera reconstruction has **no real-world scale of its own**. Only a scan made with
  the printed [marker board](#real-world-scale-the-marker-board) in view gets millimetres;
  every other scan is relative and says so ("unscaled") in the viewer and the report.
- It makes a **sparse** point cloud, plus an optional dense one on a computer with an NVIDIA
  GPU. No mesh, no texture. Measurements are between reconstructed points, not on a surface.

---

## Requirements

| | |
|---|---|
| **Python** | 3.11 or newer (developed and tested on 3.14) |
| **COLMAP** | 3.8+ or 4.x — see [Installing COLMAP](#installing-colmap) |
| **Webcam** | any device the browser can open |
| **Browser** | Chrome, Edge or Firefox (needs `getUserMedia` + WebGL2 + import maps) |
| **GPU** | not required; the CPU-only COLMAP build works fine. With an NVIDIA GPU and the CUDA build, feature detection and matching run on the GPU automatically, and the optional dense cloud becomes available |

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

The app never downloads COLMAP by itself. It searches `PATH`, the usual install directories,
`vendor/colmap/` in this project, the `COLMAP_PATH` environment variable and `colmap_path.txt`,
then reports exactly where it looked. `vendor/` is gitignored, so every computer needs its own
copy.

**Windows** — from the project folder:

```powershell
.venv\Scripts\python scripts\get_colmap.py
```

It downloads the official release into `vendor\colmap\`: the CUDA build if `nvidia-smi` finds
an NVIDIA GPU, otherwise the CPU-only build (`--cuda` / `--no-cuda` override the choice,
`--force` replaces an existing copy). It then checks that COLMAP actually starts. To do it by
hand, download `colmap-x64-windows-cuda.zip` or `-nocuda.zip` from the
[releases page](https://github.com/colmap/colmap/releases) and extract it to `vendor\colmap\`
or `C:\COLMAP`.

With the CUDA build and an NVIDIA GPU, the final reconstruction runs feature detection and
matching on the GPU (the live preview always stays on the CPU). If a GPU stage fails, for
example because the driver is too old, the stage is run again on the CPU and the server stays
on the CPU until you press **Re-detect** in Settings or restart the app.

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

**Scan quality** trades detail for speed:

| Preset | What it does | Measured on a 56-frame webcam scan (CPU) |
|---|---|---|
| **Fast** (default) | 1024 px, 4k SIFT features | quickest preview |
| **Balanced** | 1600 px, 8k features | 55/56 frames, 2,441 points, 82 s |
| **Detailed** | 1600 px, 12k DSP-SIFT features with affine shapes, guided matching | 56/56 frames, 4,150 points, 165 s |

> **Dense cloud:** Detailed is the densest result the CPU pipeline can produce. COLMAP's
> dense stereo (`patch_match_stereo`) needs CUDA; on a CPU-only build it stops with *"Dense
> stereo reconstruction requires CUDA"* (checked with 3.8, 3.11.1 and 4.2.0). With the CUDA
> build and an NVIDIA GPU, tick **Dense cloud** before scanning — see
> [Dense cloud](#dense-cloud-nvidia-gpu).

### Live tracking

A scan can be full of sharp, distinct frames and still fail. The tracking HUD watches for the
three causes, using ORB features on each frame (~16 ms):

| HUD says | Meaning | What to do |
|---|---|---|
| **Low detail** | fewer than ~120 keypoints | aim at something textured, add light |
| **Lost track** | too few verified matches with the frame 4 steps back | go back a little, move slower |
| **No depth** | a single homography explains the motion (rolling H/F inlier ratio ≥ 0.96) | you are turning in place; step sideways |

"No depth" is the subtle one: rotating on the spot gives crisp, overlapping frames but no
parallax, so COLMAP cannot triangulate anything. On a real scan that placed only 8 of 30 frames,
the tracker flagged 12 frames (7 of them as rotation). A good 55/56 scan got 4 flags.

After reconstruction the result bar shows a per-frame tracking timeline, and scans where fewer
than 60% of frames could be placed get a warning explaining the likely cause.

### Live 3D preview

The tracking HUD guesses from ORB features. The live preview asks COLMAP itself: while you
scan, the frames so far go through a low-resolution copy of the pipeline and the capture screen
shows the resulting cloud and how many frames it could place. Once a round has seen 20+ frames,
a low placed count replaces the HUD tips as the main warning.

How it works:

- **Its own workspace.** Everything lives in `scans/<id>/preview/`, with its own
  `database.db`. The final reconstruction's database and `sparse/` are never touched. After
  Finish, the preview database and models are deleted; the last cloud and log stay.
- **640 px, 2,048 SIFT features, 4 CPU threads.** At the Fast preset's 1024 px / 4,096, a
  56-frame round took 37 s on this CPU. At 640 px it takes 2–13 s, and a round that uses fewer
  threads left uploads unaffected (median 35–45 ms per frame during rounds). Because there are
  half as many features, COLMAP's minimum inlier count for placing a frame is halved to 15.
  At the default of 30, the good scan's first 15 frames placed 2/15 instead of 15/15.
- **Incremental, because COLMAP is.** Re-using the preview database, `feature_extractor` skips
  images it already has (`IMAGE_EXISTS`) and `sequential_matcher` skips pairs already matched:
  a re-run with nothing new takes 0.5 s. With `single_camera`, though, every run created a
  *new* camera (4 rounds → 4 sets of intrinsics, and the model fell apart), so later rounds
  pass `ImageReader.existing_camera_id`.
- **The mapper continues the last model** (`mapper --input_path`) instead of starting over:
  registering 11 new frames took 1.6 s, versus 12–22 s for a fresh map of all 56. If the last
  model placed fewer than half its frames, the next round rebuilds from scratch, so a bad
  first guess is not locked in.
- **Only finished frames reach COLMAP.** Frames are written to a `.tmp` name and published by
  atomic rename, and COLMAP gets an explicit image list (`--image_list_path`), never a folder
  scan.
- **One round at a time.** The first round starts at 12 frames (~5 s into the scan). Rounds
  then start every 15 s, or every 7.5 s until a round has seen 20 frames, because webcam scans
  are short: the two real scans here lasted 11 s and 24 s. A round still running at the next
  tick makes that tick be skipped, not queued. Finish, Cancel, Delete and server shutdown
  kill the running round's whole process tree.
- **A failed round is normal early on** (no good initial pair yet) and shows "Waiting for
  enough views", never an error. The last good round stays on screen.

Measured with the real scans replayed through the capture endpoint at their original pace:

| Scan | Round (frames) | Placed | Round time |
|---|---|---|---|
| good (56 frames, 24 s) | 1 (12) · 2 (30) · 3 (48) | 12/12 · 30/30 · cancelled by Finish | 2.9 s · 4.1 s |
| synthetic orbit (52 frames, 22 s) | 1 (12) · 2 (31) · 3 (46) | 12/12 · 31/31 · 46/46 | 2.7 s · 4.2 s · 4.1 s |

**Limits.** Readings under 20 frames are noisy: the good scan's 12-frame round gave 4/12 in one
run and 12/12 in the next, so early rounds are labelled "early estimate" and never raise
a warning. The mapper is not deterministic between runs, so the preview placed count is a
trend, not an exact figure. The same applies to the final reconstruction: re-running the Fast
pipeline on the "8 of 30" scan below placed 23/30 three times out of three.

### Coverage guide

A compass on the capture screen shows which directions around the subject are covered and
points to the nearest open side next to the covered arc. It updates once per preview round,
shows how old its data is, and says "Waiting for first preview" until a round exists (there is
no IMU to guess from).

- **Up** is the average camera "up", the same gravity estimate that loads models upright.
- **The subject** is where the cameras' view rays meet (a least-squares intersection), not the
  cloud median, which background points pull off-centre.
- **Azimuth** is measured around that point from the first camera. The compass turns so you
  stand at the bottom facing the subject, so its left and right are yours.

It needs a scan that *circles* something. When the cameras pan across a room instead, their
view rays never meet in front of them and the guide says so, rather than drawing a
meaningless compass. On a synthetic 80° orbit, the recovered arc (68.9° for the 46 frames
used) and the direction to walk matched the ground truth.

### Comparing scans

Press **Compare** in the library and pick two finished scans. Both open side by side. With
**Sync orbit** on, dragging one view turns the other to the same angle and relative zoom. Each
scan has its own arbitrary scale and origin, so the sync is relative to each model's framing,
not absolute camera positions.

The table below the viewers highlights the better value in each row: points, frames placed,
reconstruction time, weak links, rotation-only frames, mean track length (how many views see
each point) and mean reprojection error (pixels). The last two are read from `points3D.bin`.
Scans made before live tracking existed show "—" for its counters, not zero. Leaving compare
mode disposes both viewers and releases their WebGL contexts.

---

## Find records

Every scan can carry the record of the find it documents: **find number** (required), site
code, context or unit number, material (a fixed list plus "Other" with a description), date
found, recorder and notes. Enter it with **Find** before pressing Start Scan, or with **Find
record** on a finished scan; it can be edited at any time and is stored in `scan.json` under
`"record"`. Scans made before records existed load as before, without one.

The server is strict about what it stores (`backend/find_record.py`):

- unknown fields are refused with 422 and the field named, never silently dropped;
- control characters and invisible formatting characters (right-to-left overrides, zero-width
  spaces, soft hyphens) are removed, and text is normalised to NFC, so a find number reads the
  same everywhere; lone surrogates, which JSON allows but UTF-8 cannot store, are removed;
- over-long values are refused, not cut (find number, site and context 40 characters, notes
  2,000), so what is stored is exactly what was typed; dates must be real `YYYY-MM-DD` dates
  and not in the future.

The browser only ever shows these values with `textContent`. The library card shows the find
number as its title, and the search box above the library matches find number, site, context
and material (value or label, so "stone" finds lithics) across **all** scans on disk, not only
the most recent 20. Every word typed must match somewhere.

The **capture log** (`GET /api/scans/{id}/capture-log`, and page 2 of the report) is filled in
automatically: the camera's name as the browser reports it, the frame size measured from the
first accepted frame, the quality preset, the COLMAP version, frames captured / accepted /
placed, points, mean reprojection error, mean track length, reconstruction time and the scale.
Anything a scan did not record (older scans have no camera name) is shown as "not recorded".

## Real-world scale: the marker board

A webcam cannot tell a large thing far away from a small thing nearby. The printed board fixes
that: it is a ring of 14 ArUco markers (4×4 bits, `DICT_4X4_50`, ids 0–13) with 30 mm black
squares, around an empty area where the find goes.

**Printing and using it**

1. Settings → **Marker board** → *Board · A4* or *Board · Letter* (`/api/marker-board.pdf`). The
   PDF is vector, at exact size.
2. Print at **100% / "Actual size"**, with "Fit to page" off. Printers rescale pages without
   asking, so check the **100 mm bar** on the sheet with a ruler.
3. If it is not exactly 100 mm, measure the edge of one black marker square and enter that in
   Settings → *Printed marker size*. New scans use it; *Re-measure open scan* applies it to the
   scan in the viewer. Everything on the board scales with the page, so the marker size is all
   that is needed.
4. Tape the sheet flat, put the find inside the dashed area, and circle it slowly with the
   camera, keeping several markers in view. The capture HUD says **board · N markers** for
   every frame (a 3 ms check per frame; the upload round trip stayed at a 27 ms median).

**How the scale is measured** (`backend/scale.py`), after reconstruction, in about 2 s:

- Markers are found in every registered frame, corners refined to sub-pixel, and converted to
  COLMAP's pixel convention (OpenCV's pixel centres are half a pixel off).
- Each corner is undistorted with the camera's SIMPLE_RADIAL coefficient from `cameras.bin` and
  triangulated from all views with COLMAP's own poses, so it lands in exactly the frame of
  `map.ply`. A view is dropped while its reprojection error exceeds 2 px; at least 3 views and
  5° of triangulation angle must remain.
- **Known distances are between markers, not across one.** Distances between the same corner
  of two markers equal the printed spacing. Against ground truth, a marker's *own* edges came
  out 0.41% short with sub-pixel refinement (1.2% and 1.7% with the other fast refinements):
  refinement pulls the corners of a blurred black square inwards. The pull is the same for
  every marker, so it cancels between corresponding corners, which stayed within 0.02–0.09%.
- Each marker's estimate is the median over its pairings; the scale is the median over
  markers (at least 3). The **±** shown everywhere is the larger of two spreads: the robust
  standard deviation between markers, and a bootstrap over frames (the frames are resampled 40
  times), which also catches camera-pose errors that move all markers together.
- Two checks produce a warning instead of a quiet number: marker edges disagreeing with marker
  spacing by more than 2%, and a board-fit RMS over 1 mm (the sheet was not flat).

The result is stored in `scan.json` under `"scale"`: mm per model unit, the spreads, markers
and frames used, reprojection error, the checks and the board pose. The viewer then shows a
**scale bar** (exact at the orbit centre; a perspective view has no single scale) and a
**Measure** button: click two points of the cloud for the distance in mm, with the ± from the
scale alone. Without a scale, the viewer says "Unscaled: print the marker board" and offers
neither.

**What was measured, and what was not.** Everything below is synthetic
(`scripts/make_test_scene.py --scene find`: a 64×42×28 mm box on the A4 board, 60 frames at
1280×720, one circle at ~43 cm; `scripts/scale_accuracy.py --mode poses`, 10 runs per row, each
in a new random model frame):

| Camera poses given to the estimator | median error | max error | error within ± |
|---|---|---|---|
| exact | 0.002% | 0.002% | 0 of 10 (within 2×: 10) |
| noise 0.5 mm / 0.05° per camera | 0.020% | 0.053% | 9 of 10 |
| noise 1 mm / 0.1° | 0.050% | 0.21% | 9 of 10 (within 2×: 10) |
| noise 2 mm / 0.2° | 0.25% | 0.71% | 7 of 10 (within 2×: 10) |

Detection, measured with the real webcam's intrinsics (f = 696 px, blur σ 1.6 px, JPEG 90):
30 mm markers were found in 100% of views up to 45 cm and 60° from face-on; 20 mm markers
dropped to 25–50% at 45 cm, which is why the board uses 30 mm. None of the 334 frames of the
existing real scans produced a false detection.

**Full COLMAP reconstructions** of the same synthetic scene (`scripts/scale_accuracy.py --runs 5`,
Fast preset, CPU-only COLMAP 4.2.0, 2026-09-25): all 60 frames placed in every run, median
error 0.13%, max 0.23%, **every run reading slightly small** (−0.08% to −0.23%). The reported ±
was 0.007–0.013%, so the error fell within it in 0 of 5 runs, not even within 2×: on a COLMAP
model the ± understates the real error by about ten times. The marker edge check read +0.42–0.43%
in every run.

**Not measured:** any scan of a real printed board. Scan an object of known size on the board
and compare. The ± cannot show errors that affect every marker equally — a wrongly entered
marker size, or a model-wide distortion from poorly estimated intrinsics (the likely cause of
the consistent bias above) — so treat it as a lower bound until a real check exists.

## Find report

**Find report** on a finished scan opens `/api/scans/{id}/report.pdf`, built on request (0.3 s
for the synthetic scan; nothing is written to the scan folder):

- **Page 1:** find number, site and context; orthographic **top, front, side and bottom**
  views; each with a mm scale bar and a conventional drawing ratio (1:1, 2:1 … at 100% on A4)
  when the scan is scaled, or marked **not to scale** when it is not.
- **Page 2:** one photograph (the middle frame), the find record, the capture log with its
  quality numbers, how the scale was obtained and its checks.

The views are rendered server-side (`backend/ortho.py`) by splatting the sparse points into a
depth buffer with numpy: deterministic (the same scan gives the same bytes), testable (a 40 mm
test cube spans exactly its size in pixels), and no WebGL needed. They use the viewer's upright
frame — the average camera up becomes +Y — then turn the model about +Y so its longest
horizontal axis runs left to right. For a scaled scan, only points standing on the sheet inside
the marker ring are drawn, which isolates the find from the table and the room; otherwise the
points within the 90th-percentile radius. When the scan has a dense cloud that passed the
same-frame check (see [Dense cloud](#dense-cloud-nvidia-gpu)), the views are drawn from it
instead (at most 400,000 of its points, evenly strided, 1-pixel dots), and the caption and the
capture log say so. The PDF itself comes from a small writer in
`backend/pdf.py` (Helvetica, vector shapes, embedded images) rather than a new dependency. Its
limit: text outside Windows-1252 (e.g. Greek or Cyrillic site names) prints as "?".

## Dense cloud (NVIDIA GPU)

An opt-in step after the sparse reconstruction, for computers with an NVIDIA GPU and the CUDA
build of COLMAP (`python scripts/get_colmap.py --cuda --force`):

```
sparse model (sparse/N) → image_undistorter → patch_match_stereo → stereo_fusion → result/dense.ply
```

**Offered only where it can run.** The **Dense cloud · NVIDIA GPU** box in the capture bar is
enabled only when COLMAP is a CUDA build, `nvidia-smi` reports a GPU, and no GPU stage has
failed since the server started — the same conditions under which SIFT runs on the GPU.
Otherwise it is greyed out with the reason ("CPU-only COLMAP build", "no NVIDIA GPU found",
"GPU failed earlier"), and Settings → Reconstruction engine gives the full sentence. A finished
scan can also get a dense cloud later: **Make dense cloud** in the viewer's corner.

**It cannot break a scan.** The scan is marked complete — sparse cloud, scale, report — before
the dense step starts, and the dense step only ever writes its own note in `scan.json`
(`"dense"`). A failure, a timeout (60 minutes per stage) or **Cancel** ends as
*"Dense cloud failed: …"* / *"cancelled"* in the viewer, with the reason, and the scan stays
complete. A CUDA error in patch-match keeps the server on the CPU afterwards, like a GPU failure
during SIFT (Settings → Re-detect resets it).

**Settings per preset.** The undistorted images, and with them the depth maps, are limited to
the preset's size; the other patch-match and fusion settings are COLMAP's defaults, with
geometric consistency on (photometric pass, then geometric pass, then fusion of the filtered
geometric maps).

| Preset | Dense image size (longest side) | the real 57-frame scan's 1280×720 frames became |
|---|---|---|
| Fast | 640 px | 640×352 |
| Balanced | 1024 px | 1024×562 |
| Detailed | 1600 px | 1320×725 (full size: undistorting that lens widened the frame a little) |

**Progress** comes from COLMAP's own lines: "Undistorting image [i/N]", "Processing view i / N"
(printed twice per image, once per pass) and "Fusing image [i/N]".

**Same frame as the sparse model — checked on every run.** `image_undistorter` copies the
sparse model's poses into `dense/sparse`, and `stereo_fusion` back-projects every depth pixel
with those poses (`src/colmap/mvs/fusion.cc`), so `dense.ply` is in the sparse model's frame
exactly when those cameras are. Each run compares the camera centres of the two models; the
largest difference, relative to the camera spread, is stored as `frameOffset`. Only when it is
below 10⁻⁶ (`sameFrame`) is the dense cloud used for the scale bar, the **Measure** tool and the
report's orthographic views; otherwise it is shown but not measured. Measured with COLMAP 4.2.0:
the offset was exactly 0.0 for three real webcam scans (55, 57 and 86 placed frames) and for
the synthetic find at all three presets.

**Disk.** Depth and normal maps are big: for every image, a photometric and a geometric depth
map (4 bytes per pixel) and normal map (12 bytes per pixel). After a successful fusion the
whole `dense/` workspace is deleted — also after a failure or cancel — and only
`result/dense.ply` (27 bytes per point) and the log stay. Peak size of the workspace while it
runs:

| Scan | Fast | Balanced | Detailed |
|---|---|---|---|
| synthetic find, 60 frames (measured, ideal depth maps, see below) | 478 MB | 1.19 GB | 1.85 GB |
| real webcam scan, 57 frames, scan folder 55 MB (maps computed from their fixed file size) | 0.42 GB | 1.07 GB | 1.77 GB |

Scans can have up to 180 frames, so Detailed can need ~5.5 GB free while it runs.

**What was measured without a GPU, and how.** This was built on a computer without an NVIDIA
GPU, so `patch_match_stereo` itself has **not** been run. `scripts/dense_check.py --ideal` runs
the app's own dense stage with the real `image_undistorter` and `stereo_fusion` (COLMAP 4.2.0),
but replaces patch-match with exact depth and normal maps ray-cast from the synthetic scene's
true geometry. That tests the frame, the scale, the file handling and the measuring — not the
depth estimation. Points are taken into the board's frame with the marker scale and board pose
measured on the sparse model, exactly as the viewer's scale bar and Measure tool do. The box's
top is the median of the points within ±0.5 mm of the height histogram's peak (a loose band,
everything above half the height, takes in the side faces and reads 24–27 mm):

| Preset | dense points | top of box above the sheet (true 28 mm) | top face (true 64 × 42 mm) | fused points vs true surface: median / 95% |
|---|---|---|---|---|
| Fast | 83,805 | 27.997 mm | 63.99 × 42.00 mm | 0.012 / 0.054 mm |
| Balanced | 220,118 | 27.998 mm | 64.14 × 42.02 mm | 0.007 / 0.034 mm |
| Detailed | 336,803 | 27.999 mm | 64.10 × 42.03 mm | 0.006 / 0.027 mm |

The sparse ground-truth points measured with the same code: 28.001 mm, 64.00 × 42.00 mm.

**Not measured yet (needs an NVIDIA GPU):** patch-match's time, GPU memory, point count and
accuracy, on the synthetic find and on a real scan. Run, on such a machine:

```bash
python scripts/dense_check.py --images .cache/findscene --preset fast --runs 3
python scripts/dense_check.py --images .cache/findscene --preset balanced --runs 3
python scripts/dense_check.py --scan scans/<a finished scan>        # a copy; the scan is not touched
```

Until then, treat the dense cloud's accuracy as unknown: it inherits the sparse model's scale
and its ± (which covers the scale only), but the depth maps can add their own noise and
outliers, especially on shiny or untextured surfaces.

---

## Troubleshooting

**"No camera detected"** — plug in a webcam and press Retry. If you have several, pick one from
the Camera dropdown.

**"Camera access blocked"** — click the camera icon in the address bar, choose Allow, press
Retry. `127.0.0.1` counts as a secure origin, so no HTTPS setup is needed.

**"Camera is in use"** — close Teams/Zoom/OBS and press Retry.

**"3D reconstruction engine not found"** — see [Installing COLMAP](#installing-colmap). The
Settings panel lists every path that was searched.

**"COLMAP was found but could not start … Windows blocked it (exit code 0xC0E90002)"** —
Windows **Smart App Control** (or an application control policy) refused to load part of
COLMAP; on this machine it blocks the unsigned `libcurl.dll` in `vendor\colmap\bin`. The
CodeIntegrity event log (Event Viewer → Applications and Services → Microsoft → Windows →
CodeIntegrity → Operational) names the file. Smart App Control cannot allow single programs;
turning it off is a system-wide decision (and cannot be undone without reinstalling Windows),
so it is yours to make. Until COLMAP can run, scans can be captured but not reconstructed.

**"Not enough images"** — scan for longer. Below 12 accepted frames the app will not even try.

**"Could not build a 3D map"** — COLMAP registered too few images. Almost always the subject or
the motion, not the software:
- move more slowly, with more overlap
- add light
- choose something with visible surface detail
- orbit *around* the subject rather than rotating in place

**Reconstruction is slow** — expected on CPU. Roughly 15–60 s for 40 frames on Fast; Balanced
and larger scans take proportionally longer. A CUDA build plus an NVIDIA GPU is much faster,
and the app switches to GPU SIFT automatically when both are present (`get_colmap.py` picks the
CUDA build on such machines). Detailed uses CPU-only SIFT refinements, so it stays on the CPU
for feature detection either way.

**Dense cloud not available** — the capture bar says why. "CPU-only COLMAP build": run
`python scripts/get_colmap.py --cuda --force`. "no NVIDIA GPU found": `nvidia-smi` must work from
a command prompt. "GPU failed earlier": see the log of the scan that failed, then Settings →
Re-detect.

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
│   ├── image_quality.py    blur / exposure / novelty filter, thumbnails
│   ├── tracking.py         live texture / continuity / parallax estimator
│   ├── preview.py          live low-res reconstruction rounds while scanning
│   ├── coverage.py         view-ray subject centre, azimuth sectors, where to walk next
│   ├── colmap_runner.py    discovery, `-h` introspection, subprocess execution
│   ├── reconstruction.py   the pipeline + COLMAP model readers (cameras, images, points)
│   ├── find_record.py      find record validation, cleaning and search
│   ├── markers.py          the printable marker board and marker detection
│   ├── scale.py            triangulating the board: millimetres per model unit
│   ├── ortho.py            orthographic views by point splatting
│   ├── pdf.py              minimal PDF writer
│   ├── report.py           capture log and the find report
│   └── scan_manager.py     scan sessions and the on-disk workspace
├── frontend/
│   ├── index.html          the whole UI (one page, five states)
│   ├── styles.css
│   ├── app.js              camera, capture loop, state machine
│   ├── viewer.js           Three.js point-cloud viewer
│   └── vendor/             Three.js, OrbitControls, PLYLoader (pinned, local)
├── scans/                  one folder per scan (gitignored)
├── vendor/colmap/          local COLMAP install (gitignored, see Installing COLMAP)
├── scripts/
│   ├── run.bat / run.sh    launchers
│   ├── get_colmap.py       downloads the right COLMAP build into vendor/colmap/ (Windows)
│   ├── make_test_scene.py  renders a synthetic room or find-on-the-board sequence (+ ground truth)
│   ├── scale_accuracy.py   scale error against ground truth
│   ├── dense_check.py      dense cloud: points, time, disk, accuracy (also without a GPU: --ideal)
│   └── ui_smoke.py         headless-browser smoke test
└── tests/test_mapper.py
```

Each scan gets its own folder:

```
scans/2026-09-24_143001_ab12/
├── images/frame_000001.jpg …
├── database.db
├── sparse/0/               COLMAP model
├── preview/                live-preview workspace (own database while capturing; last cloud after)
├── dense/                  dense workspace, only while the dense step runs
├── result/map.ply          exported point cloud
├── result/dense.ply        dense point cloud (optional, NVIDIA GPU)
├── result/cameras.json     reconstructed cameras (position, forward, up)
├── result/thumbnail.jpg    library preview
├── logs/colmap.log         raw COLMAP output
└── scan.json               status, statistics, find record, capture details, scale
```

No database — the filesystem is the state.

---

## API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/system` | COLMAP readiness, capture settings, presets |
| `POST` | `/api/system/colmap` | set the COLMAP executable path |
| `POST` | `/api/system/colmap/detect` | re-run detection |
| `GET` | `/api/scans?q=` | recent scans, or every scan whose find record matches `q` |
| `POST` | `/api/scans` | start a scan session (optional `record`, `camera`, `markerSizeMm`) |
| `GET` | `/api/scans/{id}/record` | the find record |
| `PUT` | `/api/scans/{id}/record` | replace the find record (validated; 422 names the field) |
| `GET` | `/api/scans/{id}/capture-log` | camera, resolution, preset, COLMAP version, frames, quality, scale |
| `GET` | `/api/scans/{id}/scale` | the marker-board scale, or `not_measured` |
| `POST` | `/api/scans/{id}/scale` | measure the board again, e.g. with `{"markerSizeMm": 29.5}` |
| `GET` | `/api/scans/{id}/report.pdf` | the two-page find report |
| `GET` | `/api/marker-board.pdf?paper=a4\|letter` | the printable marker board |
| `POST` | `/api/scans/{id}/frames` | upload and quality-check one frame; reports board markers seen |
| `GET` | `/api/scans/{id}/preview` | latest live-preview round: state, placed/total, age, coverage, asset URLs |
| `GET` | `/api/scans/{id}/preview/{round}/map.ply` | a preview round's point cloud |
| `GET` | `/api/scans/{id}/preview/{round}/cameras.json` | a preview round's cameras |
| `POST` | `/api/scans/{id}/finish` | stop capture and the preview, start reconstruction |
| `GET` | `/api/scans/{id}/status` | stage, progress, stats, error |
| `GET` | `/api/scans/{id}/metrics` | comparison numbers, incl. mean track length and reprojection error |
| `GET` | `/api/scans/{id}/result` | result summary + asset URLs |
| `GET` | `/api/scans/{id}/map.ply` | the point cloud |
| `POST` | `/api/scans/{id}/dense` | make (or remake) the dense cloud of a complete scan; 409 with the reason if unavailable |
| `GET` | `/api/scans/{id}/dense.ply` | the dense point cloud |
| `GET` | `/api/scans/{id}/cameras.json` | camera trajectory |
| `GET` | `/api/scans/{id}/thumbnail.jpg` | library preview (after capture ends) |
| `GET` | `/api/scans/{id}/log` | raw COLMAP log |
| `POST` | `/api/scans/{id}/cancel` | cancel a running reconstruction, or the dense step of a complete scan |
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
python scripts/make_test_scene.py --scene find --out .cache/findscene --frames 60 --model
```

`--scene find` renders a 64×42×28 mm box on the A4 marker board, in millimetres, and writes
`scene.json` with the true camera poses. `--model` adds the ground-truth COLMAP model (in a
random frame, like a real one), which exercises scaling, the viewer and the report without
running COLMAP. Measure the scale error:

```bash
python scripts/scale_accuracy.py --images .cache/findscene --runs 5            # full COLMAP runs
python scripts/scale_accuracy.py --images .cache/findscene --mode poses --position-noise 1 --rotation-noise 0.1
```

`MAPPER_SCANS_DIR=<folder>` points a second server at its own workspace, e.g. for UI tests
next to a running app.

Full UI smoke test (needs `pip install playwright && python -m playwright install chromium`,
with the server already running):

```bash
python scripts/ui_smoke.py --images .cache/testscene --screenshots .cache/shots
```

---

## Roadmap

Deliberately out of scope: dense stereo without CUDA, meshing, texturing, loop closure,
NeRF / Gaussian splatting, SLAM, depth cameras. Metric scale exists only through the printed marker board.

Next most useful additions: a real printed-board accuracy check, measuring the dense step on
an NVIDIA GPU (see [Dense cloud](#dense-cloud-nvidia-gpu)), a Unicode font for the report, and
an elevation (up/down) dimension for the coverage guide.

---

## Licence

Your project's licence applies to this code. Three.js (`frontend/vendor/`) is MIT — see
`frontend/vendor/LICENSE`. COLMAP is a separate installation under its own licence.
