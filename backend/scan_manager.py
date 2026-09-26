"""Scan session lifecycle and on-disk workspace.

One scan == one directory under ``scans/``. The filesystem *is* the database:

    scans/2026-09-24_143001_ab12/
        images/          frame_000001.jpg ...
        database.db      COLMAP feature/match database
        sparse/          COLMAP models (0/, 1/, ...)
        preview/         live-preview workspace (own database, models, per-round output)
        dense/           dense workspace while the optional dense stage runs (then removed)
        result/          map.ply, cameras.json, dense.ply (optional)
        logs/colmap.log  raw COLMAP output
        scan.json        status, statistics, find record, capture details, scale

Scan IDs are validated against a strict pattern before ever touching a path, and every
resolved path is checked to be inside ``scans/`` — nothing a browser sends can escape the
workspace or reach a COLMAP command line.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from . import config
from .find_record import clean_text, load_record, record_matches
from .image_quality import FrameQualityFilter
from .models import (
    STAGE_PROGRESS,
    FrameDecision,
    FrameStats,
    ReconstructionResult,
    ScanError,
    ScanStatus,
)

logger = logging.getLogger(__name__)

#: Dense-stage states in which COLMAP is (or is about to be) running.
DENSE_RUNNING = frozenset({"queued", "undistorting", "stereo", "fusing"})

#: ``2026-09-24_143001_ab12`` - date, time, 4 hex chars. Anything else is rejected outright.
SCAN_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}_[0-9a-f]{4}$")


class ScanNotFoundError(KeyError):
    """No scan with that ID exists in the workspace."""


class InvalidScanIdError(ValueError):
    """The supplied scan ID does not match the expected format."""


class ScanSession:
    """In-memory view of one scan, backed by ``scan.json``."""

    def __init__(self, scan_id: str, root: Path, preset: str = config.DEFAULT_PRESET) -> None:
        self.id = scan_id
        self.root = root
        self.preset = preset
        self.status: ScanStatus = ScanStatus.CAPTURING
        self.stats = FrameStats()
        self.error: ScanError | None = None
        self.result: ReconstructionResult | None = None
        self.created_at: float = time.time()
        self.updated_at: float = self.created_at
        self.started_processing_at: float | None = None
        self.stage_detail: str = ""
        self.progress: int = 0
        self.source: str = "webcam"
        #: One entry per accepted frame: ``[tracking score, state]``, for the result timeline.
        self.timeline: list[list[Any]] = []
        #: The find record (see find_record.py), or None if nobody has filled one in.
        self.record: dict[str, str] | None = None
        #: Measured at capture: the browser's camera label and the pixel size of the frames.
        self.capture: dict[str, Any] = {}
        #: COLMAP version that built the model, for the capture log.
        self.colmap_version: str | None = None
        #: Printed marker size to use for scaling; None means the board's nominal size.
        self.marker_size_mm: float | None = None
        #: Outcome of the marker-board scale estimate (see scale.py), or None if never run.
        self.scale: dict[str, Any] | None = None
        #: The person asked for a dense cloud (NVIDIA GPU) when starting the scan.
        self.dense_requested: bool = False
        #: State of the optional dense stage (see reconstruction.run_dense), None if never run.
        self.dense: dict[str, Any] | None = None

        # Capture-time state, never persisted.
        self.quality_filter = FrameQualityFilter()
        self._next_frame_index = 1
        #: Names of frames that are completely on disk, in capture order. The live preview
        #: hands exactly this list to COLMAP, never a directory listing.
        self.frame_names: list[str] = []

    # -- paths ----------------------------------------------------------------

    @property
    def images_dir(self) -> Path:
        return self.root / "images"

    @property
    def sparse_dir(self) -> Path:
        return self.root / "sparse"

    @property
    def result_dir(self) -> Path:
        return self.root / "result"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def database_path(self) -> Path:
        return self.root / "database.db"

    @property
    def log_path(self) -> Path:
        return self.logs_dir / "colmap.log"

    @property
    def ply_path(self) -> Path:
        return self.result_dir / "map.ply"

    @property
    def cameras_path(self) -> Path:
        return self.result_dir / "cameras.json"

    @property
    def manifest_path(self) -> Path:
        return self.root / "scan.json"

    @property
    def preview_dir(self) -> Path:
        """Live-preview workspace. Nothing in here is read by the final reconstruction."""
        return self.root / "preview"

    @property
    def dense_dir(self) -> Path:
        """Dense workspace (undistorted images, depth and normal maps); removed after fusion."""
        return self.root / "dense"

    @property
    def dense_ply_path(self) -> Path:
        return self.result_dir / "dense.ply"

    # -- mutation --------------------------------------------------------------

    def set_status(
        self,
        status: ScanStatus,
        detail: str = "",
        progress: int | None = None,
    ) -> None:
        self.status = status
        self.stage_detail = detail
        self.progress = STAGE_PROGRESS.get(status, 0) if progress is None else progress
        self.updated_at = time.time()
        self.save()

    def set_progress(self, progress: int, detail: str = "") -> None:
        """Nudge progress inside the current stage without changing status."""
        self.progress = max(self.progress, min(progress, 99))
        if detail:
            self.stage_detail = detail
        self.updated_at = time.time()

    def fail(self, error: ScanError) -> None:
        self.error = error
        self.set_status(ScanStatus.FAILED)

    def set_dense(self, save: bool = True, **fields: Any) -> None:
        """Update the dense stage's state. It never touches ``status``: a scan whose dense
        cloud failed is still a complete scan with its sparse cloud."""
        self.dense = {**(self.dense or {}), **fields}
        self.updated_at = time.time()
        if save:
            self.save()

    def append_log(self, text: str) -> None:
        try:
            with self.log_path.open("a", encoding="utf-8", errors="replace") as handle:
                handle.write(text)
        except OSError as exc:  # pragma: no cover - logging must never break the pipeline
            logger.warning("Could not write scan log: %s", exc)

    def read_log(self, max_chars: int = 20000) -> str:
        if not self.log_path.exists():
            return ""
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return text[-max_chars:]

    def next_frame_path(self) -> Path:
        path = self.images_dir / f"frame_{self._next_frame_index:06d}.jpg"
        self._next_frame_index += 1
        return path

    def write_frame(self, data: bytes) -> Path:
        """Store an accepted frame so that no reader ever sees it half written.

        The live preview runs COLMAP on this folder while capture is still writing to it. The
        bytes go to a ``.tmp`` name that ``frame_*.jpg`` does not match, then an atomic rename
        publishes the finished file.
        """
        path = self.next_frame_path()
        partial = path.with_name(path.name + ".tmp")
        try:
            partial.write_bytes(data)
            os.replace(partial, path)
        except OSError:
            partial.unlink(missing_ok=True)
            raise
        self.frame_names.append(path.name)
        return path

    def record_decision(self, decision: FrameDecision) -> None:
        self.stats.captured += 1
        if decision.accepted:
            self.stats.accepted += 1
            if decision.tracking:
                state = decision.tracking.state
                self.stats.weak_links += state == "lost"
                self.stats.rotation_frames += state == "rotating"
                self.timeline.append([decision.tracking.score, state])
        elif decision.reason == "blurry":
            self.stats.rejected_blurry += 1
        elif decision.reason == "duplicate":
            self.stats.rejected_duplicate += 1
        elif decision.reason in ("dark", "bright"):
            self.stats.rejected_dark += 1
        else:
            self.stats.rejected_error += 1
        self.updated_at = time.time()

    @property
    def elapsed_processing(self) -> float:
        if self.started_processing_at is None:
            return 0.0
        return time.time() - self.started_processing_at

    # -- serialisation ----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status.value,
            "progress": self.progress,
            "stageDetail": self.stage_detail,
            "preset": self.preset,
            "source": self.source,
            "capturedFrames": self.stats.captured,
            "acceptedFrames": self.stats.accepted,
            "stats": self.stats.to_dict(),
            "registeredImages": self.result.registered_images if self.result else 0,
            "points": self.result.points if self.result else 0,
            "result": self.result.to_dict() if self.result else None,
            "error": self.error.to_dict() if self.error else None,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "elapsedProcessing": round(self.elapsed_processing, 1),
            "hasPly": self.ply_path.exists(),
            "timeline": self.timeline,
            "record": self.record,
            "capture": self.capture,
            "colmapVersion": self.colmap_version,
            "markerSizeMm": self.marker_size_mm,
            "scale": self.scale,
            "denseRequested": self.dense_requested,
            "dense": self.dense,
        }

    def save(self) -> None:
        try:
            self.manifest_path.write_text(
                json.dumps(self.to_dict(), indent=2), encoding="utf-8"
            )
        except OSError as exc:  # pragma: no cover
            logger.warning("Could not persist scan.json for %s: %s", self.id, exc)

    @classmethod
    def load(cls, root: Path) -> "ScanSession":
        """Rehydrate a session from disk (used for scans from an earlier server run)."""
        manifest = root / "scan.json"
        data: dict[str, Any] = {}
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Corrupt scan.json in %s: %s", root, exc)

        session = cls(root.name, root, data.get("preset", config.DEFAULT_PRESET))
        try:
            session.status = ScanStatus(data.get("status", ScanStatus.FAILED.value))
        except ValueError:
            session.status = ScanStatus.FAILED

        # A scan that was mid-processing when the server stopped cannot be resumed.
        if session.status.is_processing or session.status == ScanStatus.CAPTURING:
            session.status = ScanStatus.CANCELLED

        stats = data.get("stats") or {}
        session.stats = FrameStats(**{k: int(v) for k, v in stats.items() if k in FrameStats.__annotations__})
        session.progress = int(data.get("progress", 0))
        session.stage_detail = str(data.get("stageDetail", ""))
        session.source = str(data.get("source", "webcam"))
        session.created_at = float(data.get("createdAt", time.time()))
        session.updated_at = float(data.get("updatedAt", session.created_at))
        session.timeline = [list(entry) for entry in data.get("timeline") or []]
        # Scans made before find records, capture details or scaling existed have none of these.
        session.record = load_record(data.get("record"))
        capture = data.get("capture")
        session.capture = dict(capture) if isinstance(capture, dict) else {}
        version = data.get("colmapVersion")
        session.colmap_version = str(version) if version else None
        size = data.get("markerSizeMm")
        session.marker_size_mm = float(size) if isinstance(size, (int, float)) and size > 0 else None
        scale = data.get("scale")
        session.scale = scale if isinstance(scale, dict) else None
        session.dense_requested = data.get("denseRequested") is True
        dense = data.get("dense")
        session.dense = dict(dense) if isinstance(dense, dict) else None
        if session.dense and session.dense.get("status") in DENSE_RUNNING:
            # Like an interrupted sparse run: it cannot be resumed. The sparse scan is intact.
            session.dense.update(status="cancelled", reason="the server stopped while it ran")

        if data.get("result"):
            session.result = ReconstructionResult(**data["result"])
        if data.get("error"):
            session.error = ScanError(**data["error"])

        existing = sorted(session.images_dir.glob("frame_*.jpg")) if session.images_dir.exists() else []
        session._next_frame_index = len(existing) + 1
        session.frame_names = [path.name for path in existing]
        return session


class ScanManager:
    """Creates, tracks and deletes scans. The only component that touches ``scans/``."""

    def __init__(self, scans_dir: Path = config.SCANS_DIR) -> None:
        self.scans_dir = scans_dir
        self.scans_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, ScanSession] = {}

    # -- id / path safety --------------------------------------------------------

    @staticmethod
    def validate_id(scan_id: str) -> str:
        if not SCAN_ID_RE.match(scan_id or ""):
            raise InvalidScanIdError(f"Invalid scan id: {scan_id!r}")
        return scan_id

    def path_for(self, scan_id: str) -> Path:
        """Resolve a scan directory, refusing anything that escapes the workspace."""
        self.validate_id(scan_id)
        root = (self.scans_dir / scan_id).resolve()
        workspace = self.scans_dir.resolve()
        if root != workspace and workspace not in root.parents:
            raise InvalidScanIdError(f"Scan path escapes workspace: {scan_id!r}")
        return root

    # -- lifecycle -----------------------------------------------------------------

    def create(
        self,
        preset: str = config.DEFAULT_PRESET,
        source: str = "webcam",
        record: dict[str, str] | None = None,
        camera: str = "",
        marker_size_mm: float | None = None,
        dense: bool = False,
    ) -> ScanSession:
        """Start a scan. ``record`` must already be validated (``validate_record``)."""
        if preset not in config.PRESETS:
            preset = config.DEFAULT_PRESET

        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        scan_id = f"{stamp}_{secrets.token_hex(2)}"
        while (self.scans_dir / scan_id).exists():  # pragma: no cover - collision is unlikely
            scan_id = f"{stamp}_{secrets.token_hex(2)}"

        root = self.scans_dir / scan_id
        session = ScanSession(scan_id, root, preset)
        session.source = source
        session.record = record
        session.marker_size_mm = marker_size_mm
        session.dense_requested = bool(dense)
        camera = clean_text(camera)[:120]
        if camera:
            session.capture["camera"] = camera
        for directory in (session.images_dir, session.sparse_dir, session.result_dir, session.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)

        session.set_status(ScanStatus.CAPTURING)
        self._sessions[scan_id] = session
        logger.info("Created scan %s (preset=%s, source=%s)", scan_id, preset, source)
        return session

    def get(self, scan_id: str) -> ScanSession:
        if scan_id in self._sessions:
            return self._sessions[scan_id]
        root = self.path_for(scan_id)
        if not root.is_dir():
            raise ScanNotFoundError(scan_id)
        session = ScanSession.load(root)
        self._sessions[scan_id] = session
        return session

    def delete(self, scan_id: str) -> None:
        root = self.path_for(scan_id)
        if not root.is_dir():
            raise ScanNotFoundError(scan_id)
        self._sessions.pop(scan_id, None)
        shutil.rmtree(root, ignore_errors=True)
        logger.info("Deleted scan %s", scan_id)

    def list_scans(self, limit: int = 20, query: str = "") -> list[dict[str, Any]]:
        """Most recent scans first. Directory names sort chronologically by construction.

        With a ``query``, every scan on disk is searched by its find record (find number, site,
        context, material), not just the most recent ``limit``.
        """
        entries: list[dict[str, Any]] = []
        for root in sorted(self._iter_scan_dirs(), reverse=True):
            if len(entries) >= limit:
                break
            try:
                session = self.get(root.name)
            except (ScanNotFoundError, InvalidScanIdError):
                continue
            if query and not record_matches(session.record, query):
                continue
            entries.append(session.to_dict())
        return entries

    def _iter_scan_dirs(self) -> Iterator[Path]:
        if not self.scans_dir.is_dir():
            return
        for child in self.scans_dir.iterdir():
            if child.is_dir() and SCAN_ID_RE.match(child.name):
                yield child


#: Module-level singleton used by the API layer.
manager = ScanManager()
