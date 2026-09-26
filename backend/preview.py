"""Live 3D preview: a small, fast COLMAP reconstruction that grows while the scan is captured.

Every ``PREVIEW_INTERVAL_S`` the frames captured so far go through a low-resolution copy of the
pipeline in ``scans/<id>/preview/``, a workspace of its own - the final reconstruction's
``database.db`` and ``sparse/`` are never touched. The capture screen shows the growing cloud and
"X of Y frames placed", which comes from COLMAP itself and is a better early warning than the
ORB heuristics in ``tracking.py``.

What the design rests on (measured on real scans, CPU-only COLMAP 4.2, see README):

* **Re-using the database is incremental.** ``feature_extractor`` skips images it already has
  (``IMAGE_EXISTS``) and ``sequential_matcher`` skips pairs already in the database: a re-run
  with nothing new costs ~0.5 s of process start-up.
* **...except for the camera.** With ``single_camera`` every run creates a *new* camera, so four
  rounds gave four sets of intrinsics and the model fell apart. From the second round on the
  extractor is pointed at the existing camera (``ImageReader.existing_camera_id``).
* **The mapper is the cost.** A fresh map of 56 frames took 12-22 s. Continuing from the
  previous round's model (``mapper --input_path``) only registers the new frames: 1.6-9 s. A bad
  first model would be locked in by continuing, so a round whose model placed under half its
  frames makes the next round rebuild from scratch.
* **Only fully written frames reach COLMAP.** The image list is the session's record of
  completed writes, and frames are published by atomic rename (``ScanSession.write_frame``).

At most one round runs per scan. Rounds start on a fixed grid; a round still running when the
next tick arrives makes that tick be skipped rather than queued. The grid is twice as fine until
a round has covered ``PREVIEW_TRUSTED_FRAMES``, because early readings are noisy and many scans
are over in under half a minute.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import config
from .colmap_runner import ColmapCommandError, ColmapNotFoundError, ColmapRunner
from .colmap_runner import runner as default_runner
from .coverage import compute_coverage
from .reconstruction import read_camera_centers, read_model
from .scan_manager import ScanSession

logger = logging.getLogger(__name__)


@dataclass
class PreviewResult:
    """The latest round that produced a model."""

    round: int
    placed: int
    total: int
    points: int
    finished_at: float
    duration: float
    continued: bool
    coverage: dict[str, Any] | None
    coverage_reason: str | None


class LivePreview:
    """Background preview rounds for one capturing scan."""

    def __init__(
        self,
        session: ScanSession,
        runner: ColmapRunner | None = None,
        interval: float = config.PREVIEW_INTERVAL_S,
    ) -> None:
        self.session = session
        self.runner = runner or default_runner
        self.interval = interval
        self.result: PreviewResult | None = None
        self.rounds = 0
        #: Grid ticks that arrived while a round was still running.
        self.skipped = 0
        self.last_round_failed = False
        #: One row per round: [round, frames, placed, seconds, "continued"/"rebuilt"/"failed"].
        self.history: list[list[Any]] = []
        self._frames_seen = 0
        self._model_dir: Path | None = None
        self._model_round: int | None = None
        self._model_healthy = False
        self._loop_task: asyncio.Task[None] | None = None
        self._round_task: asyncio.Task[None] | None = None

    # -- paths -----------------------------------------------------------------

    @property
    def root(self) -> Path:
        return self.session.preview_dir

    @property
    def database_path(self) -> Path:
        return self.root / "database.db"

    def round_dir(self, number: int) -> Path:
        """Published output of one round: ``map.ply`` and ``cameras.json``."""
        return self.root / "rounds" / str(number)

    # -- scheduling --------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._round_task is not None and not self._round_task.done()

    def start(self) -> None:
        if self._loop_task is None:
            self._loop_task = asyncio.create_task(
                self._loop(), name=f"preview:{self.session.id}"
            )

    async def stop(self) -> None:
        """Cancel the schedule and any running round; COLMAP's process tree is killed."""
        tasks = [task for task in (self._round_task, self._loop_task) if task and not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _loop(self) -> None:
        # The first round fires as soon as there is something to reconstruct: a webcam scan is
        # often over in 15-25 s, so waiting a full interval would mean no preview at all.
        while len(self.session.frame_names) < config.PREVIEW_MIN_FRAMES:
            await asyncio.sleep(0.5)
        while True:
            self.tick()
            await asyncio.sleep(self.next_delay())

    def next_delay(self) -> float:
        """Half-interval ticks until a round has seen enough frames to be worth trusting."""
        trusted = self.result is not None and self.result.total >= config.PREVIEW_TRUSTED_FRAMES
        return self.interval if trusted else min(self.interval, config.PREVIEW_EARLY_INTERVAL_S)

    def tick(self) -> bool:
        """Start a round unless one is still running (skip, never queue) or nothing is new."""
        if self.running:
            self.skipped += 1
            logger.info("Preview %s: round still running, skipping this tick", self.session.id)
            return False
        frames = list(self.session.frame_names)
        if len(frames) < config.PREVIEW_MIN_FRAMES or len(frames) == self._frames_seen:
            return False
        self._frames_seen = len(frames)
        self._round_task = asyncio.create_task(
            self.run_round(frames), name=f"preview-round:{self.session.id}"
        )
        return True

    # -- arguments ---------------------------------------------------------------

    def _existing_camera_id(self) -> int | None:
        if not self.database_path.is_file():
            return None
        try:
            connection = sqlite3.connect(f"{self.database_path.as_uri()}?mode=ro", uri=True)
            try:
                row = connection.execute("SELECT MIN(camera_id) FROM cameras").fetchone()
            finally:
                connection.close()
        except sqlite3.Error as exc:
            logger.warning("Could not read the preview camera id: %s", exc)
            return None
        return int(row[0]) if row and row[0] is not None else None

    def feature_extractor_args(self, image_list: Path) -> list[str]:
        options: list[tuple[Any, object]] = [
            ("database_path", self.database_path),
            ("image_path", self.session.images_dir),
            ("image_list_path", image_list),
            ("ImageReader.single_camera", 1),
            ("ImageReader.camera_model", config.CAMERA_MODEL),
            (
                ("FeatureExtraction.max_image_size", "SiftExtraction.max_image_size"),
                config.PREVIEW_MAX_IMAGE_SIZE,
            ),
            (
                ("SiftExtraction.max_num_features", "FeatureExtraction.max_num_features"),
                config.PREVIEW_MAX_FEATURES,
            ),
            # The preview stays on the CPU and off most of it, whatever the machine has.
            (("FeatureExtraction.use_gpu", "SiftExtraction.use_gpu"), 0),
            (
                ("FeatureExtraction.num_threads", "SiftExtraction.num_threads"),
                config.PREVIEW_NUM_THREADS,
            ),
        ]
        camera_id = self._existing_camera_id()
        if camera_id is not None:
            options.append(("ImageReader.existing_camera_id", camera_id))
        return self.runner.filter_options("feature_extractor", options)

    def sequential_matcher_args(self) -> list[str]:
        return self.runner.filter_options(
            "sequential_matcher",
            [
                ("database_path", self.database_path),
                ("SequentialMatching.overlap", config.PREVIEW_SEQUENTIAL_OVERLAP),
                (("FeatureMatching.use_gpu", "SiftMatching.use_gpu"), 0),
                (
                    ("FeatureMatching.num_threads", "SiftMatching.num_threads"),
                    config.PREVIEW_NUM_THREADS,
                ),
            ],
        )

    def mapper_args(self, output: Path, continue_from: Path | None) -> list[str]:
        return self.runner.filter_options(
            "mapper",
            [
                ("database_path", self.database_path),
                ("image_path", self.session.images_dir),
                ("output_path", output),
                *([("input_path", continue_from)] if continue_from else []),
                ("Mapper.num_threads", config.PREVIEW_NUM_THREADS),
                ("Mapper.abs_pose_min_num_inliers", config.PREVIEW_ABS_POSE_MIN_INLIERS),
                # A preview needs the shape, not the last tenth of a pixel: fewer and shorter
                # bundle adjustments.
                (("Mapper.ba_global_frames_ratio", "Mapper.ba_global_images_ratio"), 2.0),
                ("Mapper.ba_global_max_num_iterations", 20),
                ("Mapper.ba_local_max_num_iterations", 10),
            ],
        )

    # -- one round -------------------------------------------------------------------

    async def run_round(self, frames: list[str]) -> None:
        self.rounds += 1
        number = self.rounds
        started = time.time()
        continue_from = self._model_dir if self._model_healthy else None
        output = self.root / "sparse" / str(number)
        self.root.mkdir(parents=True, exist_ok=True)
        image_list = self.root / "images.txt"
        image_list.write_text("\n".join(frames) + "\n", encoding="utf-8")

        stage_options = {"timeout": config.PREVIEW_STAGE_TIMEOUT_S}
        with (self.root / "colmap.log").open("w", encoding="utf-8", errors="replace") as log:
            log.write(
                f"=== Preview round {number}: {len(frames)} frames, "
                f"{'continuing from round ' + str(self._model_round) if continue_from else 'fresh map'} ===\n"
            )
            try:
                await self.runner.run(
                    "feature_extractor",
                    self.feature_extractor_args(image_list),
                    log_sink=log.write,
                    **stage_options,
                )
                await self.runner.run(
                    "sequential_matcher",
                    self.sequential_matcher_args(),
                    log_sink=log.write,
                    **stage_options,
                )
                output.mkdir(parents=True, exist_ok=True)
                await self.runner.run(
                    "mapper",
                    self.mapper_args(output, continue_from),
                    log_sink=log.write,
                    **stage_options,
                )
                model_dir, placed, points = await asyncio.to_thread(read_model, output)
                if model_dir is None or points == 0:
                    self._round_failed(number, len(frames), started, "no model yet")
                    return

                published = self.round_dir(number)
                published.mkdir(parents=True, exist_ok=True)
                await self.runner.run(
                    "model_converter",
                    self.runner.filter_options(
                        "model_converter",
                        [
                            ("input_path", model_dir),
                            ("output_path", published / "map.ply"),
                            ("output_type", "PLY"),
                        ],
                    ),
                    log_sink=log.write,
                    **stage_options,
                )
                cameras, (coverage, reason) = await asyncio.to_thread(
                    self._publish_cameras, model_dir, published
                )
            except (ColmapCommandError, ColmapNotFoundError) as exc:
                # Common early in a scan: no usable initial pair yet, or a crash on flat frames.
                log.write(f"\n[preview round failed] {exc}\n")
                self._round_failed(number, len(frames), started, str(exc))
                return
            except Exception as exc:  # pragma: no cover - a preview must never take capture down
                logger.exception("Preview round %d failed for %s", number, self.session.id)
                self._round_failed(number, len(frames), started, repr(exc))
                return

        finished = time.time()
        self.result = PreviewResult(
            round=number,
            placed=placed,
            total=len(frames),
            points=points,
            finished_at=finished,
            duration=round(finished - started, 1),
            continued=continue_from is not None,
            coverage=coverage.to_dict() if coverage else None,
            coverage_reason=reason,
        )
        self.last_round_failed = False
        self._model_dir = model_dir
        self._model_round = number
        self._model_healthy = placed >= config.PREVIEW_CONTINUE_RATIO * len(frames)
        self.history.append(
            [number, len(frames), placed, self.result.duration,
             "continued" if continue_from else "rebuilt"]
        )
        logger.info(
            "Preview %s round %d: %d/%d placed, %d points, %.1fs (%s)",
            self.session.id, number, placed, len(frames), points, self.result.duration,
            "continued" if continue_from else "fresh",
        )
        await asyncio.to_thread(self._prune)

    def _publish_cameras(self, model_dir: Path, published: Path):
        cameras = read_camera_centers(model_dir)
        (published / "cameras.json").write_text(json.dumps({"cameras": cameras}), encoding="utf-8")
        return cameras, compute_coverage(cameras)

    def _round_failed(self, number: int, frames: int, started: float, why: str) -> None:
        self.last_round_failed = True
        self.history.append([number, frames, 0, round(time.time() - started, 1), "failed"])
        logger.info("Preview %s round %d produced no model (%s)", self.session.id, number, why)

    def _prune(self) -> None:
        """Keep the model the next round continues from and the last two published rounds.

        A file still being served cannot be deleted on Windows; ``ignore_errors`` leaves it for
        the next round's prune.
        """
        keep_models = {str(self._model_round)}
        published = sorted(
            (int(p.name) for p in (self.root / "rounds").glob("*") if p.name.isdigit()),
            reverse=True,
        )
        keep_rounds = {str(n) for n in published[:2]}
        for parent, keep in ((self.root / "sparse", keep_models), (self.root / "rounds", keep_rounds)):
            for child in parent.glob("*"):
                if child.is_dir() and child.name not in keep:
                    shutil.rmtree(child, ignore_errors=True)

    def discard_workspace(self) -> None:
        """After capture: drop the bulky parts (database, models), keep the last cloud and log."""
        for name in ("database.db", "database.db-wal", "database.db-shm"):
            try:
                (self.root / name).unlink(missing_ok=True)
            except OSError:
                pass
        shutil.rmtree(self.root / "sparse", ignore_errors=True)

    # -- API ---------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        frames = len(self.session.frame_names)
        if self.result is not None:
            state = "ready"
        elif frames < config.PREVIEW_MIN_FRAMES:
            state = "collecting"
        else:
            state = "waiting"

        payload: dict[str, Any] = {
            "state": state,
            "running": self.running,
            "frames": frames,
            "minFrames": config.PREVIEW_MIN_FRAMES,
            "trustedFrames": config.PREVIEW_TRUSTED_FRAMES,
            "interval": self.interval,
            "rounds": self.rounds,
            "skipped": self.skipped,
            "lastRoundFailed": self.last_round_failed,
            "history": self.history[-12:],
            "serverTime": time.time(),
        }
        result = self.result
        if result is not None:
            base = f"/api/scans/{self.session.id}/preview/{result.round}"
            payload.update(
                {
                    "round": result.round,
                    "placed": result.placed,
                    "total": result.total,
                    "points": result.points,
                    "updatedAt": result.finished_at,
                    "duration": result.duration,
                    "continued": result.continued,
                    "plyUrl": f"{base}/map.ply",
                    "camerasUrl": f"{base}/cameras.json",
                    "coverage": result.coverage,
                    "coverageReason": result.coverage_reason,
                }
            )
        return payload
