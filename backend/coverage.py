"""Which directions around the subject have been covered, and where to walk next.

Works from the camera poses of the latest live-preview model; there is no IMU, so before the
first preview there is no honest answer and the UI says so.

* **Up** is the mean camera "up" vector, the same gravity estimate the viewer uses to load
  models upright.
* **The subject** is the point the cameras look at: the least-squares intersection of the view
  rays. The median of the point cloud is pulled off-centre by background points; the rays are
  not, because the person scanning keeps the subject in the middle of the frame.
* **Azimuth** is measured around that point in the horizontal plane, starting at the first
  camera and increasing towards the scanner's right (counter-clockwise seen from above).

Everything here is plain numpy on a few hundred numbers - microseconds per preview round.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from . import config

#: Below this, the view rays are close to parallel and have no well-defined meeting point
#: (e.g. panning along a wall). Compared against the smallest eigenvalue of the normal matrix
#: divided by the ray count, which is sin^2 of the typical angle between rays: ~5 degrees.
_MIN_RAY_SPREAD = math.sin(math.radians(5)) ** 2


@dataclass
class Coverage:
    sectors: list[int]
    current: float
    current_sector: int
    target: int | None
    turn: str | None
    centre: list[float]
    camera_angles: list[float] = field(default_factory=list)

    @property
    def covered(self) -> int:
        return sum(1 for count in self.sectors if count)

    @property
    def complete(self) -> bool:
        return self.covered == len(self.sectors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sectors": self.sectors,
            "sectorCount": len(self.sectors),
            "covered": self.covered,
            "complete": self.complete,
            "current": round(self.current, 1),
            "currentSector": self.current_sector,
            "target": self.target,
            "turn": self.turn,
            "centre": [round(value, 4) for value in self.centre],
            "cameraAngles": [round(angle, 1) for angle in self.camera_angles],
        }


def up_axis(ups: np.ndarray) -> np.ndarray:
    """Mean camera up; COLMAP's y-down convention (world -Y) if the views give no answer."""
    mean = ups.sum(axis=0) if len(ups) else np.zeros(3)
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm > 1e-8 else np.array([0.0, -1.0, 0.0])


def look_at_point(positions: np.ndarray, forwards: np.ndarray) -> np.ndarray | None:
    """The point closest to every view ray, in the least-squares sense.

    Each ray contributes the projector ``I - d d^T``, which measures distance perpendicular to
    it; summing them gives a 3x3 system ``A x = b``. Returns ``None`` when the rays are too
    close to parallel for the intersection to mean anything.
    """
    directions = forwards / np.linalg.norm(forwards, axis=1, keepdims=True)
    projectors = np.eye(3)[None, :, :] - directions[:, :, None] * directions[:, None, :]
    a = projectors.sum(axis=0)
    b = np.einsum("nij,nj->i", projectors, positions)
    if np.linalg.eigvalsh(a)[0] / len(positions) < _MIN_RAY_SPREAD:
        return None
    return np.linalg.solve(a, b)


def _wrap(angle: float) -> float:
    """Signed difference folded into (-180, 180]."""
    return (angle + 180.0) % 360.0 - 180.0


def compute_coverage(
    cameras: Sequence[dict[str, Any]], sector_count: int = config.COVERAGE_SECTORS
) -> tuple[Coverage | None, str | None]:
    """Return ``(coverage, None)`` or ``(None, reason)`` for the given reconstructed cameras.

    ``cameras`` are in capture order, as ``read_camera_centers`` returns them; the last one is
    where the person scanning is standing now.
    """
    usable = [c for c in cameras if c.get("forward") and c.get("up") and c.get("position")]
    if len(usable) < config.COVERAGE_MIN_CAMERAS:
        return None, "too_few_cameras"

    positions = np.array([c["position"] for c in usable], dtype=float)
    forwards = np.array([c["forward"] for c in usable], dtype=float)
    ups = np.array([c["up"] for c in usable], dtype=float)

    centre = look_at_point(positions, forwards)
    if centre is None:
        return None, "no_common_subject"
    # Cameras pointing *away* from the intersection (a room scanned from its middle) have no
    # subject to walk around.
    in_front = np.einsum("ij,ij->i", centre - positions, forwards) > 0
    if in_front.mean() < 0.6:
        return None, "no_common_subject"

    up = up_axis(ups)
    offsets = positions - centre
    flat = offsets - np.outer(offsets @ up, up)
    radii = np.linalg.norm(flat, axis=1)
    if radii.max() < 1e-9:
        return None, "no_common_subject"

    # Zero azimuth at the first camera that is not straight above the subject.
    reference = flat[int(np.argmax(radii > 0.05 * radii.max()))]
    e1 = reference / np.linalg.norm(reference)
    # up x e1 is the scanner's right-hand side when standing at e1 facing the subject, so
    # increasing azimuth means walking right (counter-clockwise seen from above).
    e2 = np.cross(up, e1)
    angles = np.degrees(np.arctan2(flat @ e2, flat @ e1)) % 360.0

    width = 360.0 / sector_count
    sector_of = lambda angle: int(((angle + width / 2) % 360.0) // width)  # noqa: E731
    sectors = [0] * sector_count
    for angle, radius in zip(angles, radii):
        if radius > 1e-9:
            sectors[sector_of(angle)] += 1

    current = float(angles[-1])
    current_sector = sector_of(current)
    target, turn = _next_sector(sectors, current_sector, _recent_motion(angles))
    return (
        Coverage(
            sectors=sectors,
            current=current,
            current_sector=current_sector,
            target=target,
            turn=turn,
            centre=[float(v) for v in centre],
            camera_angles=[float(a) for a in angles],
        ),
        None,
    )


def _recent_motion(angles: np.ndarray, window: int = 6) -> float:
    """Signed azimuth travelled over the last few views: >0 walking right, <0 walking left."""
    recent = angles[-window:]
    return float(sum(_wrap(b - a) for a, b in zip(recent, recent[1:])))


def _next_sector(sectors: list[int], current: int, motion: float) -> tuple[int | None, str | None]:
    """The empty sector next to the covered arc the scanner is standing in, nearest first.

    Walking the arc outwards from ``current`` in both directions finds its two open ends. The
    closer end wins; on a tie, keep going the way the scanner is already moving.
    """
    count = len(sectors)
    if all(sectors):
        return None, None

    def open_end(step: int) -> tuple[int, int]:
        for distance in range(1, count):
            index = (current + step * distance) % count
            if not sectors[index]:
                return index, distance
        return current, count  # pragma: no cover - unreachable while any sector is empty

    right, right_distance = open_end(+1)
    left, left_distance = open_end(-1)
    if right_distance < left_distance or (right_distance == left_distance and motion >= 0):
        return right, "right"
    return left, "left"
