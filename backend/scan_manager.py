"""Scan session lifecycle and on-disk workspace.

One scan == one directory under ``scans/``. The filesystem *is* the database:

    scans/2026-09-24_143001_ab12/
        images/          frame_000001.jpg ...
        database.db      COLMAP feature/match database
        sparse/          COLMAP models (0/, 1/, ...)
        result/          map.ply, cameras.json
        logs/colmap.log  raw COLMAP output
        scan.json        status + statistics

Scan IDs are validated against a strict pattern before ever touching a path, and every
resolved path is checked to be inside ``scans/`` — nothing a browser sends can escape the
workspace or reach a COLMAP command line.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from . import config
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

        # Capture-time state, never persisted.
        self.quality_filter = FrameQualityFilter()
        self._next_frame_index = 1

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

    def record_decision(self, decision: FrameDecision) -> None:
        self.stats.captured += 1
        if decision.accepted:
            self.stats.accepted += 1
        elif decision.reason == "blurry":
            self.stats.rejected_blurry += 1
        elif decision.reason == "duplicate":
            self.stats.rejected_duplicate += 1
        elif decision.reason == "dark":
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

        if data.get("result"):
            session.result = ReconstructionResult(**data["result"])
        if data.get("error"):
            session.error = ScanError(**data["error"])

        existing = sorted(session.images_dir.glob("frame_*.jpg")) if session.images_dir.exists() else []
        session._next_frame_index = len(existing) + 1
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

    def create(self, preset: str = config.DEFAULT_PRESET, source: str = "webcam") -> ScanSession:
        if preset not in config.PRESETS:
            preset = config.DEFAULT_PRESET

        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        scan_id = f"{stamp}_{secrets.token_hex(2)}"
        while (self.scans_dir / scan_id).exists():  # pragma: no cover - collision is unlikely
            scan_id = f"{stamp}_{secrets.token_hex(2)}"

        root = self.scans_dir / scan_id
        session = ScanSession(scan_id, root, preset)
        session.source = source
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

    def list_scans(self, limit: int = 20) -> list[dict[str, Any]]:
        """Most recent scans first. Directory names sort chronologically by construction."""
        entries: list[dict[str, Any]] = []
        for root in sorted(self._iter_scan_dirs(), reverse=True)[:limit]:
            try:
                entries.append(self.get(root.name).to_dict())
            except (ScanNotFoundError, InvalidScanIdError):
                continue
        return entries

    def _iter_scan_dirs(self) -> Iterator[Path]:
        if not self.scans_dir.is_dir():
            return
        for child in self.scans_dir.iterdir():
            if child.is_dir() and SCAN_ID_RE.match(child.name):
                yield child


#: Module-level singleton used by the API layer.
manager = ScanManager()
