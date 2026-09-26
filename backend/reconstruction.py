"""The sparse reconstruction pipeline.

    feature_extractor -> sequential_matcher -> mapper -> model_converter (PLY)

All four run as async subprocesses so the FastAPI event loop stays responsive. Progress is
stage-based; inside a stage we refine it from COLMAP's own counters ("Processed file [12/94]",
"Registering image #31 (24)") rather than inventing a percentage.

The module also contains a small reader for COLMAP's binary model format. Reading it directly
avoids an extra subprocess and gives us the camera trajectory for the viewer; the layout has
been stable across COLMAP releases.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import struct
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import config
from .colmap_runner import ColmapCommandError, ColmapNotFoundError, ColmapRunner
from .colmap_runner import runner as default_runner
from .models import ReconstructionResult, ScanError, ScanStatus
from .scan_manager import ScanSession

logger = logging.getLogger(__name__)

_PROCESSED_RE = re.compile(r"Processed file \[(\d+)/(\d+)\]")
_MATCH_BLOCK_RE = re.compile(r"Matching block \[(\d+)/(\d+)")
_REGISTERING_RE = re.compile(r"Registering image #(\d+) \((\d+)\)")


# --------------------------------------------------------------------------------------
# COLMAP model readers
# --------------------------------------------------------------------------------------


def _read_u64(path: Path) -> int:
    """Both images.bin and points3D.bin begin with a uint64 element count."""
    with path.open("rb") as handle:
        data = handle.read(8)
    if len(data) < 8:
        return 0
    return int(struct.unpack("<Q", data)[0])


def _count_text_entries(path: Path, fields_per_entry: int) -> int:
    """Count records in a COLMAP .txt model file (comments start with '#')."""
    lines = [
        line
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return len(lines) // fields_per_entry


def model_stats(model_dir: Path) -> tuple[int, int]:
    """Return ``(registered_images, points)`` for a COLMAP model directory."""
    images = 0
    points = 0

    if (model_dir / "images.bin").is_file():
        images = _read_u64(model_dir / "images.bin")
    elif (model_dir / "images.txt").is_file():
        # Each image occupies two lines (pose line + observations line).
        images = _count_text_entries(model_dir / "images.txt", 2)

    if (model_dir / "points3D.bin").is_file():
        points = _read_u64(model_dir / "points3D.bin")
    elif (model_dir / "points3D.txt").is_file():
        points = _count_text_entries(model_dir / "points3D.txt", 1)

    return images, points


def _read_string(handle) -> str:
    chars: list[bytes] = []
    while True:
        char = handle.read(1)
        if not char or char == b"\x00":
            break
        chars.append(char)
    return b"".join(chars).decode("utf-8", errors="replace")


def _rotation(qw: float, qx: float, qy: float, qz: float) -> tuple[tuple[float, ...], ...]:
    """Rotation matrix (rows) from a quaternion (w, x, y, z), normalised first."""
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) or 1.0
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return (
        (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)),
        (2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)),
        (2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)),
    )


def read_image_poses(model_dir: Path) -> list[dict[str, Any]]:
    """Every registered image in ``images.bin``: name, camera id and world-to-camera pose.

    ``rotation`` (rows) and ``translation`` map a world point X to camera coordinates
    ``R X + t``, COLMAP's convention. Sorted by image name, which is capture order here.
    """
    path = model_dir / "images.bin"
    if not path.is_file():
        return []

    poses: list[dict[str, Any]] = []
    try:
        with path.open("rb") as handle:
            (count,) = struct.unpack("<Q", handle.read(8))
            for _ in range(count):
                header = handle.read(64)
                if len(header) < 64:
                    break
                image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = struct.unpack("<I7dI", header)
                name = _read_string(handle)
                (num_points2d,) = struct.unpack("<Q", handle.read(8))
                handle.seek(num_points2d * 24, 1)
                poses.append(
                    {
                        "image_id": image_id,
                        "camera_id": camera_id,
                        "name": name,
                        "rotation": _rotation(qw, qx, qy, qz),
                        "translation": (tx, ty, tz),
                    }
                )
    except (OSError, struct.error) as exc:
        logger.warning("Could not parse camera poses from %s: %s", path, exc)
        return []

    poses.sort(key=lambda pose: pose["name"])
    return poses


def read_camera_centers(model_dir: Path) -> list[dict[str, Any]]:
    """Extract camera positions and view directions from ``images.bin``.

    COLMAP stores world-to-camera poses, so the camera centre is ``C = -R^T * t``, the viewing
    direction is the third row of ``R`` expressed in world space, and "up" is the negated second
    row (image y points down). Averaging ``up`` over a scan tells the viewer which way gravity
    was, so models load upright instead of in COLMAP's arbitrary y-down frame.
    """
    cameras: list[dict[str, Any]] = []
    for pose in read_image_poses(model_dir):
        r, t = pose["rotation"], pose["translation"]
        centre = [-sum(r[row][col] * t[row] for row in range(3)) for col in range(3)]
        forward = [r[2][col] for col in range(3)]
        up = [-r[1][col] for col in range(3)]
        cameras.append({"name": pose["name"], "position": centre, "forward": forward, "up": up})
    return cameras


#: COLMAP camera model ids -> (name, parameter count), as written to ``cameras.bin``.
CAMERA_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
    11: ("RAD_TAN_THIN_PRISM_FISHEYE", 16),
}

#: camera_id (uint32), model_id (int32), width, height (uint64). Parameters follow as doubles.
_CAMERA_HEADER = struct.Struct("<IiQQ")


def read_cameras(model_dir: Path) -> dict[int, dict[str, Any]]:
    """Intrinsics from ``cameras.bin``, keyed by camera id.

    Each entry has ``model`` (e.g. ``SIMPLE_RADIAL``), ``width``, ``height`` and ``params`` in
    COLMAP's order (SIMPLE_RADIAL: f, cx, cy, k). Pixel coordinates in COLMAP put the centre of
    the top-left pixel at (0.5, 0.5).
    """
    path = model_dir / "cameras.bin"
    if not path.is_file():
        return {}
    cameras: dict[int, dict[str, Any]] = {}
    try:
        data = path.read_bytes()
        (count,) = struct.unpack_from("<Q", data, 0)
        offset = 8
        for _ in range(count):
            camera_id, model_id, width, height = _CAMERA_HEADER.unpack_from(data, offset)
            offset += _CAMERA_HEADER.size
            if model_id not in CAMERA_MODELS:
                logger.warning("Unknown COLMAP camera model %d in %s", model_id, path)
                break
            name, num_params = CAMERA_MODELS[model_id]
            params = struct.unpack_from(f"<{num_params}d", data, offset)
            offset += 8 * num_params
            cameras[camera_id] = {
                "model": name,
                "width": width,
                "height": height,
                "params": list(params),
            }
    except (OSError, struct.error) as exc:
        logger.warning("Could not parse %s: %s", path, exc)
    return cameras


#: point3D_id, xyz, rgb, reprojection error, track length. The track itself follows as
#: ``track_length`` pairs of (image_id uint32, point2D_idx uint32).
_POINT3D_HEADER = struct.Struct("<Q3d3BdQ")


def _walk_points3d(data: bytes):
    """Yield each complete ``points3D.bin`` record's header fields; stops at a truncated one."""
    (count,) = struct.unpack_from("<Q", data, 0)
    offset = 8
    for _ in range(count):
        fields = _POINT3D_HEADER.unpack_from(data, offset)
        offset += _POINT3D_HEADER.size + 8 * fields[8]
        if offset > len(data):
            break
        yield fields


def read_point_stats(model_dir: Path) -> dict[str, float] | None:
    """Mean track length and mean reprojection error from ``points3D.bin``.

    Track length is how many images observe a point: longer tracks mean views overlapped more
    and the point is better constrained. Reprojection error (pixels) is how far each point lands
    from where it was detected once projected back; lower is a tighter fit. The per-point
    records are variable length, so the file is walked rather than read as one array.
    """
    path = model_dir / "points3D.bin"
    if not path.is_file():
        return None
    total_track = 0
    total_error = 0.0
    read = 0
    try:
        for fields in _walk_points3d(path.read_bytes()):
            total_track += fields[8]
            total_error += fields[7]
            read += 1
    except (OSError, struct.error) as exc:
        logger.warning("Could not parse %s: %s", path, exc)
        return None
    if read == 0:
        return {"points": 0, "mean_track_length": 0.0, "mean_reprojection_error": 0.0}
    return {
        "points": read,
        "mean_track_length": total_track / read,
        "mean_reprojection_error": total_error / read,
    }


def read_points3d(model_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Positions ``(N, 3)`` float64 and colours ``(N, 3)`` uint8 (RGB) from ``points3D.bin``.

    The same coordinates ``model_converter`` writes to ``map.ply``, read without a second
    parser for the PLY header (which COLMAP writes with CRLF line endings on Windows).
    """
    path = model_dir / "points3D.bin"
    xyz: list[tuple[float, float, float]] = []
    rgb: list[tuple[int, int, int]] = []
    if path.is_file():
        try:
            for fields in _walk_points3d(path.read_bytes()):
                xyz.append(fields[1:4])
                rgb.append(fields[4:7])
        except (OSError, struct.error) as exc:
            logger.warning("Could not parse %s: %s", path, exc)
    return (
        np.array(xyz, dtype=np.float64).reshape(-1, 3),
        np.array(rgb, dtype=np.uint8).reshape(-1, 3),
    )


def read_model(model_dir: Path) -> tuple[Path | None, int, int]:
    """A mapper output directory: either one model in place or numbered components under it.

    ``mapper --input_path`` (continuing an existing model) writes the result straight into
    ``output_path``; a fresh run writes ``output_path/0``, ``/1``, ...
    """
    if (model_dir / "images.bin").is_file():
        images, points = model_stats(model_dir)
        return (model_dir if images else None), images, points
    return select_best_model(model_dir)


def select_best_model(sparse_dir: Path) -> tuple[Path | None, int, int]:
    """Pick the reconstructed component with the most registered images, then most points."""
    best: tuple[Path | None, int, int] = (None, 0, 0)
    if not sparse_dir.is_dir():
        return best

    for child in sorted(sparse_dir.iterdir()):
        if not child.is_dir():
            continue
        images, points = model_stats(child)
        if images == 0 and points == 0:
            continue
        if (images, points) > (best[1], best[2]):
            best = (child, images, points)
    return best


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------


class ReconstructionPipeline:
    """Runs the four COLMAP stages for one scan and records the outcome on the session."""

    def __init__(self, session: ScanSession, runner: ColmapRunner | None = None) -> None:
        self.session = session
        self.runner = runner or default_runner
        self.preset = config.PRESETS.get(session.preset, config.PRESETS[config.DEFAULT_PRESET])

    # -- argument construction -------------------------------------------------

    @property
    def use_gpu(self) -> str:
        """SIFT on the GPU needs both a CUDA device and a CUDA-enabled COLMAP build."""
        info = self.runner.detect()
        usable = info.gpu_build and info.gpu not in ("unknown", "CPU-only build")
        return "1" if usable and not self.runner.gpu_failed else "0"

    def feature_extractor_args(self) -> list[str]:
        # Each entry lists every spelling we know of; the runner picks the one this build
        # advertises. COLMAP 4.x moved several SiftExtraction.* options to FeatureExtraction.*.
        return self.runner.filter_options(
            "feature_extractor",
            [
                ("database_path", self.session.database_path),
                ("image_path", self.session.images_dir),
                # Every frame comes from the same webcam, so one shared intrinsic model.
                ("ImageReader.single_camera", 1),
                ("ImageReader.camera_model", config.CAMERA_MODEL),
                (
                    ("FeatureExtraction.max_image_size", "SiftExtraction.max_image_size"),
                    self.preset.max_image_size,
                ),
                (
                    ("SiftExtraction.max_num_features", "FeatureExtraction.max_num_features"),
                    self.preset.max_num_features,
                ),
                (("FeatureExtraction.use_gpu", "SiftExtraction.use_gpu"), self.use_gpu),
                *(
                    [
                        ("SiftExtraction.estimate_affine_shape", 1),
                        # DSP-SIFT is CPU-only in COLMAP, which suits the no-CUDA build.
                        ("SiftExtraction.domain_size_pooling", 1),
                    ]
                    if self.preset.refined_features
                    else []
                ),
            ],
        )

    def sequential_matcher_args(self) -> list[str]:
        return self.runner.filter_options(
            "sequential_matcher",
            [
                ("database_path", self.session.database_path),
                # Frames arrive in capture order, so neighbours in the sequence overlap.
                ("SequentialMatching.overlap", self.preset.sequential_overlap),
                (("FeatureMatching.use_gpu", "SiftMatching.use_gpu"), self.use_gpu),
                *(
                    [(("FeatureMatching.guided_matching", "SiftMatching.guided_matching"), 1)]
                    if self.preset.refined_features
                    else []
                ),
            ],
        )

    def mapper_args(self) -> list[str]:
        return self.runner.filter_options(
            "mapper",
            [
                ("database_path", self.session.database_path),
                ("image_path", self.session.images_dir),
                ("output_path", self.session.sparse_dir),
                (
                    ("Mapper.ba_global_frames_ratio", "Mapper.ba_global_images_ratio"),
                    self.preset.mapper_ba_global_images_ratio,
                ),
            ],
        )

    def model_converter_args(self, model_dir: Path) -> list[str]:
        return self.runner.filter_options(
            "model_converter",
            [
                ("input_path", model_dir),
                ("output_path", self.session.ply_path),
                ("output_type", "PLY"),
            ],
        )

    # -- progress ---------------------------------------------------------------

    def _extraction_progress(self, line: str) -> None:
        match = _PROCESSED_RE.search(line)
        if match:
            done, total = int(match.group(1)), max(int(match.group(2)), 1)
            self.session.set_progress(30 + int(24 * done / total), f"{done} of {total} images")

    def _matching_progress(self, line: str) -> None:
        match = _MATCH_BLOCK_RE.search(line)
        if match:
            done, total = int(match.group(1)), max(int(match.group(2)), 1)
            self.session.set_progress(55 + int(24 * done / total))

    def _mapping_progress(self, line: str) -> None:
        match = _REGISTERING_RE.search(line)
        if match:
            registered = int(match.group(2))
            total = max(self.session.stats.accepted, registered, 1)
            self.session.set_progress(
                80 + int(14 * registered / total), f"{registered} images placed"
            )

    # -- execution ---------------------------------------------------------------

    async def run(self) -> None:
        """Execute the pipeline. Failures are translated into a ScanError on the session."""
        started = time.time()
        self.session.started_processing_at = started
        log = self.session.append_log

        try:
            info = self.runner.detect()
            if not info.available:
                raise ColmapNotFoundError(info.error or "COLMAP not available")

            self.session.colmap_version = info.version
            image_count = len(list(self.session.images_dir.glob("frame_*.jpg")))
            log(
                f"=== Scan {self.session.id} ===\n"
                f"COLMAP: {info.path} (version {info.version})\n"
                f"GPU: {info.gpu} | SIFT on GPU: {self.use_gpu}\n"
                f"Preset: {self.preset.name} | Images: {image_count}\n"
            )

            if image_count < config.MIN_FRAMES_FOR_RECONSTRUCTION:
                self.session.fail(
                    ScanError(
                        code="too_few_frames",
                        title="Not enough images",
                        message=(
                            "We need more images to build a reliable map. "
                            f"This scan captured {image_count} usable frames."
                        ),
                        hints=[
                            "Scan for at least 20-30 seconds",
                            "Move slowly and keep the subject in frame",
                            "Make sure the scene is well lit",
                        ],
                    )
                )
                return

            self.session.set_status(ScanStatus.PREPARING, f"{image_count} images")
            await asyncio.sleep(0)  # let the first status poll observe this stage

            self.session.set_status(ScanStatus.EXTRACTING)
            await self._run_sift_stage(
                "feature_extractor", self.feature_extractor_args, self._extraction_progress
            )

            self.session.set_status(ScanStatus.MATCHING)
            await self._run_sift_stage(
                "sequential_matcher", self.sequential_matcher_args, self._matching_progress
            )

            self.session.set_status(ScanStatus.MAPPING)
            self.session.sparse_dir.mkdir(parents=True, exist_ok=True)
            await self.runner.run(
                "mapper",
                self.mapper_args(),
                log_sink=log,
                line_callback=self._mapping_progress,
            )

            self.session.set_status(ScanStatus.EXPORTING)
            model_dir, registered, points = select_best_model(self.session.sparse_dir)
            if model_dir is None or points == 0:
                self.session.fail(_mapping_failed_error())
                return

            log(f"\nSelected model {model_dir.name}: {registered} images, {points} points\n")
            await self.runner.run(
                "model_converter", self.model_converter_args(model_dir), log_sink=log
            )

            if not self.session.ply_path.is_file() or self.session.ply_path.stat().st_size == 0:
                self.session.fail(
                    ScanError(
                        code="export_failed",
                        title="Could not export the point cloud",
                        message="COLMAP built a model but the PLY export produced no data.",
                        hints=["Check Technical Details for the raw COLMAP output"],
                    )
                )
                return

            cameras = read_camera_centers(model_dir)
            if cameras:
                self.session.cameras_path.write_text(
                    json.dumps({"cameras": cameras}), encoding="utf-8"
                )

            self.session.result = ReconstructionResult(
                points=points,
                registered_images=registered,
                input_images=image_count,
                ply_path=str(self.session.ply_path.relative_to(self.session.root)),
                model_index=int(model_dir.name) if model_dir.name.isdigit() else None,
                duration_seconds=round(time.time() - started, 1),
            )
            await self._measure_scale(log)
            self.session.result.duration_seconds = round(time.time() - started, 1)
            self.session.set_status(ScanStatus.COMPLETE)
            log(f"\nDone in {self.session.result.duration_seconds}s\n")

        except ColmapNotFoundError as exc:
            self.session.fail(
                ScanError(
                    code="colmap_missing",
                    title="3D reconstruction engine not found",
                    message="COLMAP must be installed to build a map. " + str(exc),
                    hints=["Set the COLMAP path in Settings, then run the scan again"],
                )
            )
        except ColmapCommandError as exc:
            log(f"\n[error] {exc}\n{exc.tail}\n")
            self.session.fail(_translate_command_error(exc))
        except asyncio.CancelledError:
            self.session.set_status(ScanStatus.CANCELLED)
            raise
        except Exception as exc:  # pragma: no cover - unexpected failure path
            logger.exception("Reconstruction failed for %s", self.session.id)
            log(f"\n[error] {exc!r}\n")
            self.session.fail(
                ScanError(
                    code="internal_error",
                    title="Reconstruction failed",
                    message=str(exc),
                    hints=["See Technical Details for the raw log"],
                )
            )


    async def _run_sift_stage(
        self, command: str, build_args: Callable[[], list[str]], progress: Callable[[str], None]
    ) -> None:
        """Run a SIFT stage; if it fails on the GPU, switch to the CPU and run it again.

        A CUDA build can still fail on a particular machine (old driver, out of GPU memory).
        The CPU path always exists, so a GPU failure costs time, not the scan.
        """
        log = self.session.append_log
        on_gpu = self.use_gpu == "1"
        try:
            await self.runner.run(command, build_args(), log_sink=log, line_callback=progress)
        except ColmapCommandError as exc:
            if not (on_gpu and _is_gpu_failure(exc)):
                raise
            self.runner.gpu_failed = True
            log(f"\n[warning] GPU failed during {command}; running it again on the CPU.\n")
            if command == "feature_extractor":
                # COLMAP skips images already in the database, including half-written ones.
                for suffix in ("", "-wal", "-shm"):
                    Path(f"{self.session.database_path}{suffix}").unlink(missing_ok=True)
            await self.runner.run(command, build_args(), log_sink=log, line_callback=progress)

    async def _measure_scale(self, log) -> None:
        """Look for the printed marker board and derive millimetres per model unit.

        Runs before the scan is marked complete so the result view never shows a stale
        "unscaled". Detection plus triangulation is ~1-2 s for 60 frames, all local Python.
        """
        from .scale import measure_session  # scale.py reads models through this module

        self.session.set_progress(96, "Looking for the scale board")
        scale = await asyncio.to_thread(measure_session, self.session)
        self.session.scale = scale
        if scale.get("status") == "scaled":
            log(
                f"\nScale: {scale['mmPerUnit']:.5g} mm per unit (spread {scale['spreadPct']}%) "
                f"from {scale['markersUsed']} markers in {scale['framesUsed']} frames; "
                f"marker edges {scale['edgeCheckPct']:+.2f}%, board fit RMS "
                f"{scale['boardResidualMm']} mm\n"
            )
            for warning in scale.get("warnings", []):
                log(f"Scale warning: {warning}\n")
        else:
            log(f"\nScale: {scale.get('status')} {scale.get('message', '')}\n")


def _mapping_failed_error() -> ScanError:
    return ScanError(
        code="mapping_failed",
        title="Could not build a 3D map",
        message="The images did not contain enough matching detail to reconstruct the scene.",
        hints=[
            "Move the camera more slowly",
            "Add more light",
            "Scan something with visible texture, not a blank or shiny surface",
            "Keep a large overlap between consecutive viewpoints",
        ],
    )


def _crashed(returncode: int) -> bool:
    """Windows reports a crash as an NTSTATUS (0xC0000005, 0xC0000409, ...), POSIX as -SIG."""
    return returncode < 0 or returncode >= 0x8000_0000


def _is_gpu_failure(exc: ColmapCommandError) -> bool:
    tail = exc.tail.lower()
    return any(token in tail for token in ("cuda", "no gpu", "opengl", "siftgpu", "sift_gpu"))


def _translate_command_error(exc: ColmapCommandError) -> ScanError:
    """Turn a non-zero exit code into guidance a person can act on."""
    # COLMAP can hard-crash on degenerate input (large flat areas, almost no texture).
    # Saying so is more honest than blaming the user's camera technique alone.
    crash_hint = (
        ["The engine stopped unexpectedly, which usually means the frames had very "
         "little texture to work with"]
        if _crashed(exc.returncode)
        else []
    )

    if _is_gpu_failure(exc):
        # GPU runs are retried on the CPU before this point, so this is not a one-off.
        return ScanError(
            code="gpu_error",
            title="GPU feature extraction failed",
            message="COLMAP reported a GPU or graphics error while analysing the frames.",
            hints=[
                "Update the graphics driver",
                "Check Technical Details for the raw COLMAP output",
            ],
        )

    if exc.command == "mapper":
        error = _mapping_failed_error()
        error.hints = crash_hint + error.hints
        return error

    if exc.command == "feature_extractor":
        return ScanError(
            code="feature_extraction_failed",
            title="Could not read the captured images",
            message="COLMAP failed while reading or analysing the captured frames.",
            hints=crash_hint + [
                "Try another scan",
                "Check Technical Details for the raw COLMAP output",
            ],
        )

    if exc.command == "sequential_matcher":
        return ScanError(
            code="matching_failed",
            title="Could not match the views",
            message="COLMAP failed while comparing the captured frames to each other.",
            hints=crash_hint + [
                "Move the camera more slowly so consecutive frames overlap",
                "Check Technical Details for the raw COLMAP output",
            ],
        )

    return ScanError(
        code="colmap_error",
        title="Reconstruction failed",
        message=f"COLMAP stage '{exc.command}' stopped unexpectedly (exit code {exc.returncode}).",
        hints=crash_hint + ["See Technical Details for the raw log"],
    )
