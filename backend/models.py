"""Data models shared between the scan manager, the reconstruction pipeline and the API."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .tracking import TrackingSample


class ScanStatus(str, Enum):
    """Lifecycle of a single scan session."""

    QUEUED = "queued"
    CAPTURING = "capturing"
    PREPARING = "preparing"
    EXTRACTING = "extracting"
    MATCHING = "matching"
    MAPPING = "mapping"
    EXPORTING = "exporting"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (ScanStatus.COMPLETE, ScanStatus.FAILED, ScanStatus.CANCELLED)

    @property
    def is_processing(self) -> bool:
        return self in (
            ScanStatus.PREPARING,
            ScanStatus.EXTRACTING,
            ScanStatus.MATCHING,
            ScanStatus.MAPPING,
            ScanStatus.EXPORTING,
        )


#: Honest, stage-based progress. COLMAP does not report a reliable percentage, so these are
#: the floor of each stage; within `mapping` we interpolate using registered-image counts.
STAGE_PROGRESS: dict[ScanStatus, int] = {
    ScanStatus.QUEUED: 0,
    ScanStatus.CAPTURING: 0,
    ScanStatus.PREPARING: 10,
    ScanStatus.EXTRACTING: 30,
    ScanStatus.MATCHING: 55,
    ScanStatus.MAPPING: 80,
    ScanStatus.EXPORTING: 95,
    ScanStatus.COMPLETE: 100,
    ScanStatus.FAILED: 100,
    ScanStatus.CANCELLED: 100,
}

#: User-facing stage list rendered by the processing screen.
STAGE_SEQUENCE: list[tuple[str, str]] = [
    ("preparing", "Preparing images"),
    ("extracting", "Detecting features"),
    ("matching", "Matching views"),
    ("mapping", "Building 3D map"),
    ("exporting", "Preparing viewer"),
]


@dataclass
class FrameStats:
    """Counters for the capture-time quality filter."""

    captured: int = 0
    accepted: int = 0
    rejected_blurry: int = 0
    rejected_duplicate: int = 0
    rejected_dark: int = 0
    rejected_error: int = 0
    #: Accepted frames whose views no longer overlapped the recent past (tracking "lost").
    weak_links: int = 0
    #: Accepted frames captured while the camera was only rotating (no parallax).
    rotation_frames: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class ScanError:
    """A failure translated into something a human can act on."""

    code: str
    title: str
    message: str
    hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReconstructionResult:
    """Outcome of a successful sparse reconstruction."""

    points: int = 0
    registered_images: int = 0
    input_images: int = 0
    ply_path: str | None = None
    model_index: int | None = None
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FrameDecision:
    """Result of running one uploaded frame through the quality filter."""

    accepted: bool
    reason: str
    blur: float = 0.0
    brightness: float = 0.0
    difference: float = 0.0
    tracking: TrackingSample | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "blur": round(self.blur, 2),
            "brightness": round(self.brightness, 2),
            "difference": round(self.difference, 3),
            "tracking": self.tracking.to_dict() if self.tracking else None,
        }
