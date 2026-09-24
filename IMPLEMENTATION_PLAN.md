# Implementation Plan — Webcam 3D Mapper (v1)

## Environment findings (2026-09-24)

| Item | Result |
|---|---|
| OS | Windows 11 Home 10.0.26200 |
| Python | 3.14.7 (only interpreter; `py -0p` lists just 3.14-64), pip 26.2.1 |
| Wheels | `opencv-python-headless` 5.0.0.93 ships `cp37-abi3` wheels → works on 3.14. `numpy` 2.5.3 has cp314 wheels. |
| COLMAP | Not installed initially. **COLMAP 4.2.0 (no-CUDA Windows build) has since been extracted to `vendor/colmap/`**, which the runner searches automatically. |
| GPU | Intel(R) Graphics (integrated) only. No `nvidia-smi`, no CUDA. |
| Webcam | Cannot be enumerated from the shell; enumeration happens in the browser via `navigator.mediaDevices`. |

### Consequences for the design

1. **COLMAP must be optional at startup.** The server has to boot, serve the UI, and show a
   readiness indicator instead of crashing. A configurable path (`COLMAP_PATH` env var or
   `colmap_path.txt` in the app dir) lets the user point at it after install.
2. **CPU-only reconstruction.** No CUDA → `--SiftExtraction.use_gpu 0` and
   `--SiftMatching.use_gpu 0`. GPU is auto-detected, never assumed.
3. **Never hardcode CLI flags.** Because no COLMAP binary exists here to test against, the runner
   parses `colmap <command> -h` at runtime, caches the set of supported `--Option.Name` tokens,
   and filters the desired argument list against it. Unsupported options are dropped with a log
   line rather than crashing the pipeline.
4. **A dev image-import path is required** to exercise the reconstruction pipeline without a
   webcam scan (`POST /api/dev/import`).

## Architecture

```
Browser (getUserMedia → canvas → JPEG)      FastAPI (127.0.0.1:8765)
        │  POST /api/scans/{id}/frames             │
        │                                          ├─ image_quality  (OpenCV: blur/brightness/similarity)
        │  POST .../finish                         ├─ scan_manager   (filesystem session state)
        │  GET  .../status  (poll ~700 ms)         ├─ colmap_runner  (discovery + -h introspection + exec)
        │  GET  /scans/{id}/result/map.ply         └─ reconstruction (async subprocess pipeline)
        ▼                                                    │
   Three.js viewer  ◀──────── result/map.ply ────────────────┘
```

Quality checks run in a thread (`asyncio.to_thread`); COLMAP runs via
`asyncio.create_subprocess_exec` with stdout streamed to `logs/colmap.log`, so the event loop
is never blocked.

## Pipeline

`feature_extractor` (SIMPLE_RADIAL, `--ImageReader.single_camera 1`)
→ `sequential_matcher` → `mapper` → pick best `sparse/N` → `model_converter --output_type PLY`.

Best model is chosen by reading the `uint64` count at the head of `images.bin` / `points3D.bin`
(stable across COLMAP versions, no extra subprocess needed). The same parse yields camera
centres for the trajectory overlay.

## Phases

1. Skeleton: FastAPI + static frontend + `/api/system`. ✅
2. Webcam: device enumeration, preview, camera picker. ✅
3. Capture: 400 ms sampling → JPEG → upload. ✅
4. Scan manager: scan folders, `scan.json`, state machine. ✅
5. COLMAP: discovery, `-h` introspection, 4-stage pipeline, PLY export. ✅
6. Progress: stage-based status polling + technical log. ✅
7. Viewer: Three.js PLY + OrbitControls + camera path + fit/reset. ✅
8. Polish: dark UI, error states, empty states. ✅
9. Test: pipeline unit tests + live server smoke test + dev import. ✅

## Non-goals for v1

Dense stereo, meshing, texturing, SLAM, NeRF/3DGS, metric scale, cloud anything.


---

## Post-implementation notes

**COLMAP 4.2 renamed options.** The installed build advertises `FeatureExtraction.max_image_size`,
`FeatureExtraction.use_gpu`, `FeatureMatching.use_gpu` and `Mapper.ba_global_frames_ratio` where
3.x used `SiftExtraction.*`, `SiftMatching.use_gpu` and `Mapper.ba_global_images_ratio`. Plain
`-h` filtering would have silently dropped all four and left the pipeline untuned, so
`filter_options` takes a tuple of aliases per logical option and picks the first spelling the
build advertises. Both generations are covered by tests.

**The build reports its own GPU support.** COLMAP's banner ends in `without GPU support` for the
no-CUDA release, so `use_gpu` is forced to `0` even if an NVIDIA card is present.

**Three.js ships split.** `three.module.min.js` imports a sibling `three.core.min.js`; vendoring
only the first breaks the viewer entirely. A test now walks every relative import in every
shipped script and asserts the server serves it.

**COLMAP can hard-crash on textureless input.** The headless-browser run produced exit code
`0xC0000409` from `sequential_matcher` on Chromium's flat synthetic camera feed. Crash exit codes
are now distinguished from ordinary failures in the user-facing guidance.

## Verified end to end (2026-09-24)

40 synthetic frames -> 40/40 images registered, 1512 points, 13.2 s on CPU, PLY exported with
colour, camera trajectory recovered as a smooth arc, rendered and manipulated in the browser.
