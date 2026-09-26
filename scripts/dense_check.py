"""Measure the optional dense cloud: points, time, disk, and accuracy on the synthetic find.

    python scripts/make_test_scene.py --scene find --out .cache/findscene --frames 60 --model

    # Full COLMAP, sparse then dense (needs a CUDA build of COLMAP and an NVIDIA GPU):
    python scripts/dense_check.py --images .cache/findscene --preset fast --runs 3

    # Dense on a copy of a finished real scan (the scan itself is not touched; NVIDIA only):
    python scripts/dense_check.py --scan scans/2026-09-25_144444_861c

    # Any machine: the app's own dense stage with the real image_undistorter and
    # stereo_fusion, but patch_match_stereo replaced by exact depth and normal maps ray-cast
    # from the scene's true geometry. It checks the coordinate frame, the scale, the file
    # handling and this script's measuring - and says nothing about patch-match's accuracy.
    python scripts/dense_check.py --images .cache/findscene --preset fast --ideal

Accuracy on the find scene (true box 64 x 42 x 28 mm on the A4 board): points are taken into
the board's frame with the marker scale and board pose the app measured on the sparse model -
exactly what the viewer's scale bar and Measure tool use. The height of the box's top above
the sheet is the median of the points in a +-0.5 mm band around the histogram peak above 5 mm
(a loose band takes in the side faces and reads low). Its size in the plane is the minimum-area
rectangle around those top-face points.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import math
import shutil
import statistics
import struct
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_test_scene import build_find_scene  # noqa: E402

from backend import config, markers  # noqa: E402
from backend.colmap_runner import ColmapRunner  # noqa: E402
from backend.models import ReconstructionResult, ScanStatus  # noqa: E402
from backend.reconstruction import (  # noqa: E402
    ReconstructionPipeline,
    dense_support,
    read_cameras,
    read_image_poses,
    read_ply_points,
    read_points3d,
)
from backend.scale import measure_session  # noqa: E402
from backend.scan_manager import ScanManager, ScanSession  # noqa: E402

#: Heights below this are the sheet and its noise, not the box's top.
MIN_TOP_MM = 5.0
TOP_BAND_MM = 0.5


# --------------------------------------------------------------------------------------
# Measuring the box
# --------------------------------------------------------------------------------------


def board_frame(xyz: np.ndarray, scale: dict) -> np.ndarray:
    """Model points -> board millimetres (x, y on the sheet, z = height above it), using the
    marker scale (mm per unit) and board pose the app measured on the sparse model."""
    pose = scale["boardPose"]
    local = (xyz - np.array(pose["translation"])) @ np.array(pose["rotation"])
    local *= float(scale["mmPerUnit"])
    local[:, 2] *= float(pose.get("normalSign", 1.0))
    return local


def measure_box(xyz: np.ndarray, scale: dict) -> dict:
    """Height of the top face above the sheet and its size in the plane, in millimetres."""
    board = board_frame(xyz, scale)
    factor = float(scale.get("markerSizeMm", markers.MARKER_MM)) / markers.MARKER_MM
    mm_per_board_unit = float(scale["mmPerUnit"]) * float(scale["boardPose"]["scale"])
    x0, y0, x1, y1 = (v * factor * mm_per_board_unit for v in markers.INNER_AREA)
    inside = (board[:, 0] > x0) & (board[:, 0] < x1) & (board[:, 1] > y0) & (board[:, 1] < y1)
    tall = board[inside & (board[:, 2] > MIN_TOP_MM) & (board[:, 2] < 200)]
    if len(tall) < 10:
        return {"topPoints": len(tall)}
    bins = np.arange(MIN_TOP_MM, tall[:, 2].max() + 0.2, 0.1)
    counts, edges = np.histogram(tall[:, 2], bins=bins)
    peak = (edges[counts.argmax()] + edges[counts.argmax() + 1]) / 2
    band = tall[np.abs(tall[:, 2] - peak) <= TOP_BAND_MM]
    centre = float(np.median(band[:, 2]))
    band = tall[np.abs(tall[:, 2] - centre) <= TOP_BAND_MM]
    height = float(np.median(band[:, 2]))
    (_, _), (w, h), _ = cv2.minAreaRect(band[:, :2].astype(np.float32))
    loose = float(np.median(tall[tall[:, 2] > 0.5 * height][:, 2]))
    return {
        "topPoints": int(len(band)),
        "heightMm": height,
        "heightMadMm": float(np.median(np.abs(band[:, 2] - height))),
        "lengthMm": float(max(w, h)),
        "widthMm": float(min(w, h)),
        "looseBandHeightMm": loose,
    }


def tree_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


# --------------------------------------------------------------------------------------
# Ideal depth maps (stand-in for patch_match_stereo)
# --------------------------------------------------------------------------------------


def _write_mat(path: Path, array: np.ndarray) -> None:
    """COLMAP's depth/normal map format: ``W&H&C&`` then float32, one [row][col] plane per
    channel (src/colmap/mvs/mat.cc)."""
    height, width = array.shape[:2]
    channels = 1 if array.ndim == 2 else array.shape[2]
    planes = array.reshape(height, width, channels).transpose(2, 0, 1)
    with path.open("wb") as handle:
        handle.write(f"{width}&{height}&{channels}&".encode())
        handle.write(np.ascontiguousarray(planes, dtype="<f4").tobytes())


def ideal_maps(workspace: Path, truth_frame: dict, line_callback=None) -> int:
    """Ray-cast the true scene into every undistorted view of ``workspace`` and write the
    photometric and geometric depth and normal maps patch-match would have written.

    Pixel (col, row) is back-projected as ``K^-1 (col, row, 1)``, the convention
    stereo_fusion uses (mvs/fusion.cc), so an exact map fuses onto the true surfaces.
    """
    faces, _ = build_find_scene()
    # The table's tiles and the sheet all lie in z = 0: one quad over the tiles' extent
    # gives the same depths, ~10x faster than casting against each tile.
    table = np.array([c for c, _texture, layer in faces if layer == 0]).reshape(-1, 3)
    low, high = table.min(axis=0), table.max(axis=0)
    quads = [(low, np.array([high[0] - low[0], 0.0, 0.0]), np.array([0.0, high[1] - low[1], 0.0]))]
    quads += [(c[0], c[1] - c[0], c[3] - c[0]) for c, _texture, layer in faces if layer == 2]
    s = float(truth_frame["unitsPerScene"])
    r_s = np.array(truth_frame["rotation"])
    t_s = np.array(truth_frame["translation"])

    model = workspace / "sparse"
    cameras = read_cameras(model)
    poses = read_image_poses(model)
    out = workspace / "stereo"
    (out / "depth_maps").mkdir(parents=True, exist_ok=True)
    (out / "normal_maps").mkdir(parents=True, exist_ok=True)
    for index, pose in enumerate(poses, start=1):
        camera = cameras[pose["camera_id"]]
        assert camera["model"] == "PINHOLE", camera["model"]
        fx, fy, cx, cy = camera["params"]
        width, height = int(camera["width"]), int(camera["height"])
        r = np.array(pose["rotation"])
        t = np.array(pose["translation"])
        cols, rows = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
        rays_cam = np.stack([(cols - cx) / fx, (rows - cy) / fy, np.ones_like(cols)], axis=-1).reshape(-1, 3)
        # Camera -> model -> world (mm): model = s R_s world + T_s.
        centre_model = -r.T @ t
        centre = r_s.T @ (centre_model - t_s) / s
        rays = (rays_cam @ r) @ r_s  # R^T per ray, then R_s^T; scale does not change direction

        best = np.full(len(rays), np.inf)
        normal_world = np.zeros((len(rays), 3))
        for origin, edge_u, edge_v in quads:
            n = np.cross(edge_u, edge_v)
            denom = rays @ n
            with np.errstate(divide="ignore", invalid="ignore"):
                dist = ((origin - centre) @ n) / denom
            hit = centre + dist[:, None] * rays
            a = (hit - origin) @ edge_u / (edge_u @ edge_u)
            b = (hit - origin) @ edge_v / (edge_v @ edge_v)
            ok = (dist > 0) & (a >= 0) & (a <= 1) & (b >= 0) & (b <= 1) & (dist < best)
            best[ok] = dist[ok]
            normal_world[ok] = n / np.linalg.norm(n)
        valid = np.isfinite(best)
        # World hit -> camera depth: X_cam = R (s R_s X + T_s) + t; the ray is K^-1 (u, v, 1),
        # so depth (z) is the ray parameter measured in model units.
        depth = np.where(valid, best * s, 0.0)
        normals = normal_world @ r_s.T @ r.T
        facing = np.sum(normals * rays_cam, axis=1) > 0
        normals[facing] *= -1  # COLMAP's normals point towards the camera
        normals[~valid] = 0.0
        depth_map = depth.reshape(height, width)
        normal_map = normals.reshape(height, width, 3)
        for kind in ("photometric", "geometric"):
            _write_mat(out / "depth_maps" / f"{pose['name']}.{kind}.bin", depth_map)
            _write_mat(out / "normal_maps" / f"{pose['name']}.{kind}.bin", normal_map)
        if line_callback:
            for pass_number in (1, 2):
                line_callback(f"Processing view {index} / {len(poses)} for {pose['name']}")
    return len(poses)


class IdealStereoRunner(ColmapRunner):
    """The real COLMAP for everything but patch_match_stereo, which gets ideal maps.

    It reports itself as a GPU build so the app's gating lets the dense stage run; nothing
    in it touches a GPU.
    """

    def __init__(self, truth_frame: dict) -> None:
        super().__init__()
        self.truth_frame = truth_frame

    def detect(self, force: bool = False):
        info = super().detect(force)
        return dataclasses.replace(info, gpu_build=True, gpu="none (ideal depth maps)")

    async def run(self, command, args, log_sink=None, line_callback=None, timeout=config.COLMAP_STAGE_TIMEOUT_S):
        if command != "patch_match_stereo":
            return await super().run(command, args, log_sink, line_callback, timeout)
        workspace = Path(args[args.index("--workspace_path") + 1])
        if log_sink:
            log_sink("\n[ideal] patch_match_stereo replaced by ray-cast ground-truth maps\n")
        await asyncio.to_thread(ideal_maps, workspace, self.truth_frame, line_callback)
        return ""


# --------------------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------------------


def consistent_tracks(model: Path) -> np.ndarray:
    """Give make_test_scene's ground-truth model real tracks, in place.

    Its points3D.bin lists observations that its images.bin does not have (0 per image), which
    COLMAP 4.2 refuses to load: image_undistorter crashed with 0xC0000409. And stereo_fusion
    picks each view's neighbours from shared tracks (GetMaxOverlappingImages), so empty
    tracks fuse nothing. Every true point is projected into every view where it lands in
    front of the camera and inside the image, and both files are rewritten to agree.
    """
    xyz, rgb = read_points3d(model)
    camera = next(iter(read_cameras(model).values()))
    f, cx, cy = camera["params"][:3]  # SIMPLE_RADIAL with k = 0
    width, height = camera["width"], camera["height"]
    raw = (model / "images.bin").read_bytes()
    (count,) = struct.unpack_from("<Q", raw, 0)
    offset, headers = 8, []
    for _ in range(count):
        header = raw[offset:offset + 64]
        end = raw.index(b"\x00", offset + 64)
        name = raw[offset + 64:end + 1]
        (points2d,) = struct.unpack_from("<Q", raw, end + 1)
        offset = end + 9 + 24 * points2d
        headers.append((header, name))
    poses = {pose["image_id"]: pose for pose in read_image_poses(model)}

    tracks: list[list[tuple[int, int]]] = [[] for _ in range(len(xyz))]
    with (model / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", count))
        for header, name in headers:
            image_id = struct.unpack_from("<I", header, 0)[0]
            pose = poses[image_id]
            cam = xyz @ np.array(pose["rotation"]).T + np.array(pose["translation"])
            with np.errstate(divide="ignore", invalid="ignore"):
                u, v = f * cam[:, 0] / cam[:, 2] + cx, f * cam[:, 1] / cam[:, 2] + cy
            seen = np.flatnonzero((cam[:, 2] > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height))
            handle.write(header + name + struct.pack("<Q", len(seen)))
            for index, point in enumerate(seen):
                handle.write(struct.pack("<ddq", u[point], v[point], point + 1))
                tracks[point].append((image_id, index))
    with (model / "points3D.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(xyz)))
        for point, (position, colour) in enumerate(zip(xyz, rgb)):
            track = tracks[point]
            handle.write(struct.pack("<Q3d3BdQ", point + 1, *position, *map(int, colour), 0.5, len(track)))
            handle.write(b"".join(struct.pack("<II", image_id, index) for image_id, index in track))
    return xyz


def new_session(workspace: Path, preset: str, images: list[Path]) -> ScanSession:
    session = ScanManager(workspace).create(preset=preset, source="import", dense=True)
    for index, source in enumerate(images, start=1):
        shutil.copyfile(source, session.images_dir / f"frame_{index:06d}.jpg")
    session.stats.captured = session.stats.accepted = len(images)
    return session


def ideal_session(folder: Path, workspace: Path, preset: str) -> ScanSession:
    """A complete scan whose sparse model is the ground truth written by --model."""
    images = sorted(folder.glob("frame_*.jpg"))
    session = new_session(workspace, preset, images)
    model = session.sparse_dir / "0"
    shutil.copytree(folder / "model", model)
    xyz = consistent_tracks(model)
    session.result = ReconstructionResult(points=len(xyz), registered_images=len(images),
                                          input_images=len(images), model_index=0)
    session.scale = measure_session(session)
    session.set_status(ScanStatus.COMPLETE)
    return session


def describe(session: ScanSession, truth: dict | None, seconds_sparse: float | None) -> dict:
    dense = session.dense or {}
    model = session.sparse_dir / str(session.result.model_index)
    row = {
        "preset": session.preset,
        "placed": session.result.registered_images,
        "sparsePoints": session.result.points,
        "sparseSeconds": seconds_sparse,
        "sparseBytes": tree_bytes(model) + (session.ply_path.stat().st_size if session.ply_path.is_file() else 0),
        "dense": {k: dense.get(k) for k in ("status", "reason", "points", "bytes", "workspaceBytes",
                                            "seconds", "totalSeconds", "frameOffset", "sameFrame",
                                            "maxImageSize")},
        "leftInWorkspace": tree_bytes(session.dense_dir),
    }
    scale = session.scale or {}
    if truth is not None and scale.get("status") == "scaled":
        true_mm = truth["find"]["sizeMm"]
        row["scale"] = {k: scale.get(k) for k in ("mmPerUnit", "spreadPct", "markersUsed")}
        row["sparseBox"] = measure_box(read_points3d(model)[0], scale)
        if dense.get("status") == "complete":
            row["denseBox"] = measure_box(read_ply_points(session.dense_ply_path)[0], scale)
        row["trueBox"] = {"lengthMm": true_mm[0], "widthMm": true_mm[1], "heightMm": true_mm[2]}
    return row


def surface_error(session: ScanSession, folder: Path) -> dict:
    """Ideal mode: distance of every fused point from the true surfaces, in millimetres."""
    frame = json.loads((folder / "model" / "truth.json").read_text(encoding="utf-8"))
    xyz, _ = read_ply_points(session.dense_ply_path)
    world = (xyz - np.array(frame["translation"])) @ np.array(frame["rotation"]) / frame["unitsPerScene"]
    faces, truth = build_find_scene()
    length, breadth, tall = truth["find"]["sizeMm"]
    # Nearest face: the sheet/table plane or the box's top and sides.
    distance = np.abs(world[:, 2])
    centre = np.array(truth["find"]["centre"])
    yaw = math.radians(truth["find"]["yawDeg"])
    ax, ay = np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])
    rel = world[:, :2] - centre[:2]
    u, v = rel @ ax, rel @ ay
    on_box = (np.abs(u) <= length / 2 + 0.5) & (np.abs(v) <= breadth / 2 + 0.5) & (world[:, 2] > 0.5)
    box_distance = np.minimum.reduce([
        np.abs(world[:, 2] - tall),
        np.abs(np.abs(u) - length / 2),
        np.abs(np.abs(v) - breadth / 2),
    ])
    distance = np.where(on_box, box_distance, distance)
    return {
        "points": int(len(distance)),
        "medianMm": float(np.median(distance)),
        "p95Mm": float(np.percentile(distance, 95)),
        "maxMm": float(distance.max()),
    }


def print_row(run: int, row: dict) -> None:
    dense = row["dense"]
    seconds = dense.get("seconds") or {}
    print(
        f"{run:>3}  {row['preset']:<8} {row['placed']:>4}  sparse {row['sparsePoints']:>7,} pts "
        f"{(row['sparseSeconds'] or 0):6.0f} s {row['sparseBytes'] / 1e6:7.1f} MB | dense "
        f"{dense.get('status')}: {(dense.get('points') or 0):>9,} pts "
        f"{(dense.get('totalSeconds') or 0):6.0f} s ({', '.join(f'{k} {v:.0f}' for k, v in seconds.items())}) "
        f"ply {(dense.get('bytes') or 0) / 1e6:7.1f} MB, workspace {(dense.get('workspaceBytes') or 0) / 1e6:7.1f} MB, "
        f"left {row['leftInWorkspace']} B, frame offset {dense.get('frameOffset')}"
    )
    if dense.get("reason"):
        print(f"     dense reason: {dense['reason']}")
    for key in ("sparseBox", "denseBox"):
        box = row.get(key)
        if box and "heightMm" in box:
            print(
                f"     {key[:-3]:<6} top {box['heightMm']:7.3f} mm (MAD {box['heightMadMm']:.3f}, "
                f"{box['topPoints']:,} pts; loose band {box['looseBandHeightMm']:.2f})  "
                f"plane {box['lengthMm']:6.2f} x {box['widthMm']:6.2f} mm"
            )
        elif box:
            print(f"     {key[:-3]:<6} too few top-face points ({box['topPoints']})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", type=Path, help="folder written by make_test_scene.py --scene find")
    source.add_argument("--scan", type=Path, help="a finished scan folder (copied, never modified)")
    parser.add_argument("--preset", default=None, help="default: fast (or the scan's own preset)")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--ideal", action="store_true",
                        help="replace patch_match_stereo by exact depth maps (any machine)")
    parser.add_argument("--json", default="", help="also write the per-run results here")
    args = parser.parse_args()

    rows = []
    truth = None
    if args.images:
        truth = json.loads((args.images / "scene.json").read_text(encoding="utf-8"))
        if truth.get("units") != "mm":
            sys.exit("scene.json is not in millimetres: render it with --scene find")
    frame = None
    if args.ideal:
        if not args.images or not (args.images / "model" / "truth.json").is_file():
            sys.exit("--ideal needs --images from make_test_scene.py --scene find --model")
        frame = json.loads((args.images / "model" / "truth.json").read_text(encoding="utf-8"))
        runner = IdealStereoRunner(frame)
    else:
        runner = ColmapRunner()
    available, _, reason = dense_support(runner)
    if not available:
        print(f"Dense cloud not available on this machine: {reason}")
        return 2
    print(f"COLMAP {runner.detect().version} at {runner.detect().path}; GPU: {runner.detect().gpu}\n")

    for run in range(1, args.runs + 1):
        with tempfile.TemporaryDirectory(prefix="dense-check-") as tmp:
            workspace = Path(tmp)
            sparse_seconds = None
            if args.ideal:
                session = ideal_session(args.images, workspace, args.preset or "fast")
                pipeline = ReconstructionPipeline(session, runner=runner)
                asyncio.run(pipeline.run_dense())
            elif args.images:
                session = new_session(workspace, args.preset or "fast", sorted(args.images.glob("frame_*.jpg")))
                started = time.time()
                pipeline = ReconstructionPipeline(session, runner=runner)
                asyncio.run(pipeline.run())
                sparse_seconds = session.result.duration_seconds if session.result else None
                if session.status != ScanStatus.COMPLETE:
                    print(f"{run:>3}  reconstruction {session.status.value}: {session.error}")
                    continue
                del started
            else:
                root = workspace / args.scan.name
                shutil.copytree(args.scan, root, ignore=shutil.ignore_patterns("preview", "dense", "database.db*"))
                session = ScanSession.load(root)
                if session.status != ScanStatus.COMPLETE:
                    sys.exit(f"{args.scan} is not a complete scan")
                if args.preset:
                    session.preset = args.preset
                (root / "result" / "dense.ply").unlink(missing_ok=True)
                session.dense = None
                session.dense_requested = True
                sparse_seconds = session.result.duration_seconds
                asyncio.run(ReconstructionPipeline(session, runner=runner).run_dense())
            row = describe(session, truth, sparse_seconds)
            if args.ideal and (session.dense or {}).get("status") == "complete":
                row["surfaceError"] = surface_error(session, args.images)
            row["run"] = run
            rows.append(row)
            print_row(run, row)
            if "surfaceError" in row:
                error = row["surfaceError"]
                print(f"     fused points vs true surfaces: median {error['medianMm']:.4f} mm, "
                      f"95th pct {error['p95Mm']:.4f} mm, max {error['maxMm']:.4f} mm")

    boxes = [row["denseBox"] for row in rows if row.get("denseBox", {}).get("heightMm")]
    if len(boxes) > 1:
        heights = [box["heightMm"] for box in boxes]
        print(f"\ndense top height over {len(boxes)} runs: median {statistics.median(heights):.3f} mm, "
              f"range {min(heights):.3f}-{max(heights):.3f} mm (true 28 mm)")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
