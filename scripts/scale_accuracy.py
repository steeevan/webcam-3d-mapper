"""Measure how well the marker board recovers real-world scale, against synthetic ground truth.

    python scripts/make_test_scene.py --scene find --out .cache/findscene --frames 60
    python scripts/scale_accuracy.py --images .cache/findscene --runs 5

Two modes:

* ``--mode colmap`` (default): every run is a full, fresh reconstruction by the app's own
  pipeline (feature extraction, matching, mapping, then the marker scale estimate) in a
  temporary workspace, never in ``scans/``. COLMAP's mapper is not deterministic, so one run
  proves little. The true scale does not come from the markers: COLMAP's camera centres are
  fitted to the true camera positions in ``scene.json`` (millimetres) with a similarity
  transform, whose scale is the true millimetres per model unit.
* ``--mode poses``: no COLMAP. Each run writes a model from the *true* poses in a random
  similarity frame, optionally with independent noise on camera positions, orientations and
  focal length, and measures the rendered frames with it. This isolates marker detection,
  triangulation and the estimator; it is a sensitivity test, not COLMAP's accuracy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_test_scene import write_colmap_model  # noqa: E402

from backend.models import ScanStatus  # noqa: E402
from backend.reconstruction import ReconstructionPipeline, read_camera_centers  # noqa: E402
from backend.report import scale_uncertainty_pct  # noqa: E402
from backend.scale import estimate_scale, fit_similarity  # noqa: E402
from backend.scan_manager import ScanManager  # noqa: E402


def true_mm_per_unit(model_dir: Path, truth: dict) -> tuple[float, float, int]:
    """(mm per model unit, RMS camera position residual in mm, cameras used)."""
    eyes = {frame["name"]: frame["eye"] for frame in truth["frames"]}
    pairs = [(camera["position"], eyes[camera["name"]]) for camera in read_camera_centers(model_dir)
             if camera["name"] in eyes]
    model = np.array([p for p, _ in pairs])
    world = np.array([e for _, e in pairs])
    scale, rotation, translation = fit_similarity(model, world)
    residual = (scale * (rotation @ model.T).T + translation) - world
    return scale, float(np.sqrt((residual**2).sum(axis=1).mean())), len(pairs)


def run_once(images: list[Path], workspace: Path, preset: str) -> dict:
    manager = ScanManager(workspace)
    session = manager.create(preset=preset, source="import")
    for index, source in enumerate(images, start=1):
        shutil.copyfile(source, session.images_dir / f"frame_{index:06d}.jpg")
    session.stats.captured = session.stats.accepted = len(images)
    started = time.time()
    asyncio.run(ReconstructionPipeline(session).run())
    return {"session": session, "seconds": time.time() - started}


def poses_mode(folder: Path, truth: dict, args) -> int:
    """Scale from the rendered frames with true (optionally perturbed) poses; no COLMAP."""
    rows = []
    print(
        f"poses mode: position noise {args.position_noise} mm, rotation noise "
        f"{args.rotation_noise} deg, {args.runs} runs\n"
    )
    print(" run   mm/unit(board)  mm/unit(true)    error   spread  bootstrap  edge-check  markers")
    for run in range(1, args.runs + 1):
        with tempfile.TemporaryDirectory(prefix="scale-poses-") as tmp:
            frame = write_colmap_model(
                Path(tmp), truth, np.empty((0, 3)), np.empty((0, 3), np.uint8), seed=run,
                position_noise=args.position_noise, rotation_noise_deg=args.rotation_noise,
            )
            scale = estimate_scale(Path(tmp), folder, truth["board"]["markerMm"])
        true_scale = 1.0 / frame["unitsPerScene"]
        if scale.get("status") != "scaled":
            print(f"{run:>4}  scale status: {scale.get('status')}")
            continue
        error = 100.0 * (scale["mmPerUnit"] / true_scale - 1.0)
        rows.append({"run": run, "errorPct": error, "spreadPct": scale["spreadPct"],
                     "bootstrapPct": scale["bootstrapPct"],
                     "uncertaintyPct": scale_uncertainty_pct(scale)})
        print(
            f"{run:>4}  {scale['mmPerUnit']:>15.6f}  {true_scale:>13.6f}  {error:+7.3f}%  "
            f"{scale['spreadPct']:5.3f}%  {scale['bootstrapPct']:8.3f}%  "
            f"{scale['edgeCheckPct']:+9.3f}%  {scale['markersUsed']:>7}"
        )
    return summarise(rows, args)


def summarise(rows: list[dict], args) -> int:
    errors = [abs(row["errorPct"]) for row in rows if "errorPct" in row]
    if errors:
        print(
            f"\n|error| over {len(errors)} scaled runs: median {statistics.median(errors):.3f}%, "
            f"max {max(errors):.3f}%"
        )
        scaled = [r for r in rows if "errorPct" in r]
        for key in ("spreadPct", "bootstrapPct", "uncertaintyPct"):
            if all(r.get(key) is not None for r in scaled):
                inside = sum(abs(r["errorPct"]) <= r[key] for r in scaled)
                within2 = sum(abs(r["errorPct"]) <= 2 * r[key] for r in scaled)
                print(f"error within {key}: {inside} of {len(scaled)} runs (within 2x: {within2})")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0 if errors else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="folder written by make_test_scene.py --scene find")
    parser.add_argument("--mode", choices=("colmap", "poses"), default="colmap")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--preset", default="fast")
    parser.add_argument("--position-noise", type=float, default=0.0, help="poses mode: mm, 1 sigma")
    parser.add_argument("--rotation-noise", type=float, default=0.0, help="poses mode: degrees, 1 sigma")
    parser.add_argument("--focal-error", type=float, default=0.0,
                        help="poses mode: focal length error in %%, alternating sign per run")
    parser.add_argument("--json", default="", help="also write the per-run results here")
    args = parser.parse_args()

    folder = Path(args.images)
    truth = json.loads((folder / "scene.json").read_text(encoding="utf-8"))
    if truth.get("units") != "mm":
        sys.exit("scene.json is not in millimetres: render it with --scene find")
    if args.mode == "poses":
        return poses_mode(folder, truth, args)
    images = sorted(folder.glob("frame_*.jpg"))

    rows = []
    print(f"{len(images)} frames, preset {args.preset}, {args.runs} runs\n")
    print(" run  placed  points   mm/unit(board)  mm/unit(true)   error   spread  edge-check  markers  frames  pose-rms   time")
    for run in range(1, args.runs + 1):
        with tempfile.TemporaryDirectory(prefix="scale-accuracy-") as tmp:
            outcome = run_once(images, Path(tmp), args.preset)
            session = outcome["session"]
            if session.status != ScanStatus.COMPLETE:
                print(f"{run:>4}  reconstruction {session.status.value}: {session.error}")
                rows.append({"run": run, "status": session.status.value})
                continue
            model_dir = session.sparse_dir / str(session.result.model_index)
            true_scale, pose_rms, _ = true_mm_per_unit(model_dir, truth)
            scale = session.scale or {}
            row = {
                "run": run,
                "status": scale.get("status"),
                "placed": session.result.registered_images,
                "points": session.result.points,
                "trueMmPerUnit": true_scale,
                "poseRmsMm": pose_rms,
                "seconds": round(outcome["seconds"], 1),
                **{k: scale.get(k) for k in ("mmPerUnit", "spreadPct", "edgeCheckPct",
                                             "markersUsed", "framesUsed", "reprojectionErrorPx")},
            }
            if scale.get("status") == "scaled":
                row["errorPct"] = 100.0 * (scale["mmPerUnit"] / true_scale - 1.0)
                print(
                    f"{run:>4}  {row['placed']:>3}/{len(images):<3} {row['points']:>6}   "
                    f"{scale['mmPerUnit']:>14.5f}  {true_scale:>13.5f}  {row['errorPct']:+6.3f}%  "
                    f"{scale['spreadPct']:5.3f}%  {scale['edgeCheckPct']:+9.3f}%  "
                    f"{scale['markersUsed']:>7}  {scale['framesUsed']:>6}  {pose_rms:6.3f} mm  "
                    f"{row['seconds']:5.0f} s"
                )
            else:
                print(f"{run:>4}  {row['placed']:>3}/{len(images):<3} scale status: {scale.get('status')}")
            rows.append(row)
    return summarise(rows, args)


if __name__ == "__main__":
    sys.exit(main())
