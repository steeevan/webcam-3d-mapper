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
from typing import Any

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


def read_camera_centers(model_dir: Path) -> list[dict[str, Any]]:
    """Extract camera positions and view directions from ``images.bin``.

    COLMAP stores world-to-camera poses, so the camera centre is ``C = -R^T * t`` and the
    viewing direction is the third row of ``R`` expressed in world space.
    """
    path = model_dir / "images.bin"
    if not path.is_file():
        return []

    cameras: list[dict[str, Any]] = []
    try:
        with path.open("rb") as handle:
            (count,) = struct.unpack("<Q", handle.read(8))
            for _ in range(count):
                header = handle.read(64)
                if len(header) < 64:
                    break
                _image_id, qw, qx, qy, qz, tx, ty, tz, _camera_id = struct.unpack(
                    "<I7dI", header
                )
                name = _read_string(handle)
                (num_points2d,) = struct.unpack("<Q", handle.read(8))
                handle.seek(num_points2d * 24, 1)

                # Rotation matrix from the unit quaternion (w, x, y, z).
                norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) or 1.0
                qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
                r = (
                    (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)),
                    (2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)),
                    (2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)),
                )
                t = (tx, ty, tz)
                centre = [-sum(r[row][col] * t[row] for row in range(3)) for col in range(3)]
                forward = [r[2][col] for col in range(3)]
                cameras.append({"name": name, "position": centre, "forward": forward})
    except (OSError, struct.error) as exc:
        logger.warning("Could not parse camera poses from %s: %s", path, exc)
        return []

    cameras.sort(key=lambda camera: camera["name"])
    return cameras


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
        return "1" if info.gpu_build and info.gpu not in ("unknown", "CPU-only build") else "0"

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
            await self.runner.run(
                "feature_extractor",
                self.feature_extractor_args(),
                log_sink=log,
                line_callback=self._extraction_progress,
            )

            self.session.set_status(ScanStatus.MATCHING)
            await self.runner.run(
                "sequential_matcher",
                self.sequential_matcher_args(),
                log_sink=log,
                line_callback=self._matching_progress,
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


def _translate_command_error(exc: ColmapCommandError) -> ScanError:
    """Turn a non-zero exit code into guidance a person can act on."""
    tail = exc.tail.lower()
    # COLMAP can hard-crash on degenerate input (large flat areas, almost no texture).
    # Saying so is more honest than blaming the user's camera technique alone.
    crash_hint = (
        ["The engine stopped unexpectedly, which usually means the frames had very "
         "little texture to work with"]
        if _crashed(exc.returncode)
        else []
    )

    if any(token in tail for token in ("cuda", "no gpu", "opengl", "siftgpu", "sift_gpu")):
        return ScanError(
            code="gpu_error",
            title="GPU feature extraction failed",
            message="COLMAP could not use the GPU for feature detection on this machine.",
            hints=["Reconstruction will fall back to CPU - try the scan again"],
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
