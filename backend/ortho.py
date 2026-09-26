"""Orthographic views of a sparse cloud for the find report: top, bottom, front and side.

Server-side point splatting with a depth buffer, in numpy. No WebGL, no randomness: the same
model always gives the same pixels, which is what makes the report reproducible and testable.

The frame is the viewer's: the average camera "up" becomes +Y (the same shortest-arc rotation
``viewer.js`` applies), so "top" means what the person scanning thought was up. The model is
then turned about Y so the longest horizontal axis of the find runs left-right, the way a find
is usually drawn; that turn is the only freedom the viewer leaves open.

Which points: for a scaled scan the board pose is known, so only points standing on the sheet
inside the marker ring are drawn - that is the find, without the table or the room. Otherwise
the points within the 90th-percentile radius around the median are used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import markers

#: Name -> rows mapping an upright point (x, y up, z) to (u right, v up, depth towards viewer).
VIEWS: dict[str, np.ndarray] = {
    "top": np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]]),
    "front": np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]]),
    "side": np.array([[0, 0, -1.0], [0, 1, 0], [1, 0, 0]]),
    "bottom": np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]]),
}
VIEW_LABELS = {"top": "Top", "front": "Front", "side": "Right side", "bottom": "Bottom"}

#: Drawing ratios offered for scaled views (page mm per real mm), largest first.
RATIOS: tuple[tuple[float, str], ...] = (
    (5.0, "5:1"), (4.0, "4:1"), (3.0, "3:1"), (2.0, "2:1"), (1.5, "3:2"), (1.0, "1:1"),
    (0.5, "1:2"), (1 / 3, "1:3"), (0.25, "1:4"), (0.2, "1:5"), (0.1, "1:10"), (0.05, "1:20"),
    (0.02, "1:50"), (0.01, "1:100"),
)

#: Points of the sheet itself (and its noise) are not the find.
SHEET_CLEARANCE_MM = 0.8
FIND_MAX_HEIGHT_MM = 300.0


def upright_rotation(ups: np.ndarray) -> np.ndarray:
    """Rotation taking the mean camera up to +Y, exactly as three.js ``setFromUnitVectors``.

    Falls back to COLMAP's y-down convention when there are no cameras, like the viewer.
    """
    up = np.asarray(ups, dtype=np.float64).reshape(-1, 3).sum(axis=0)
    if float(up @ up) < 1e-16:
        up = np.array([0.0, -1.0, 0.0])
    up = up / np.linalg.norm(up)
    target = np.array([0.0, 1.0, 0.0])
    r = float(up @ target) + 1.0
    if r < 1e-6:  # opposite vectors: three.js picks an axis perpendicular to `up`
        axis = np.array([-up[1], up[0], 0.0]) if abs(up[0]) > abs(up[2]) else np.array([0.0, -up[2], up[1]])
        q = np.array([0.0, *axis])
    else:
        q = np.array([r, *np.cross(up, target)])
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def yaw_alignment(points: np.ndarray) -> np.ndarray:
    """Rotation about +Y putting the main horizontal axis of ``points`` (upright) along X.

    The sign is fixed so the result is deterministic: the axis points towards the side where
    the points reach further.
    """
    if len(points) < 3:
        return np.eye(3)
    flat = points[:, [0, 2]] - np.median(points[:, [0, 2]], axis=0)
    _, vectors = np.linalg.eigh(np.cov(flat.T))
    major = vectors[:, -1]
    projected = flat @ major
    if abs(projected.max()) < abs(projected.min()):
        major = -major
    angle = math.atan2(major[1], major[0])  # angle of the major axis from +X towards +Z
    c, s = math.cos(angle), math.sin(angle)
    # Rotating by -angle about Y maps the major axis onto +X.
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def find_points(xyz: np.ndarray, scale: dict[str, Any] | None) -> tuple[np.ndarray, str]:
    """Indices of the points to draw, and how they were chosen (for the report's caption)."""
    if len(xyz) == 0:
        return np.arange(0), "no points"
    pose = (scale or {}).get("boardPose") if (scale or {}).get("status") == "scaled" else None
    if pose:
        s = float(pose["scale"])
        rotation = np.array(pose["rotation"])
        translation = np.array(pose["translation"])
        board = (xyz - translation) @ rotation / s  # sheet millimetres, z along the normal
        height = board[:, 2] * float(pose.get("normalSign", 1.0))
        factor = float(scale.get("markerSizeMm", markers.MARKER_MM)) / markers.MARKER_MM
        x0, y0, x1, y1 = (value * factor for value in markers.INNER_AREA)
        inside = (
            (board[:, 0] > x0) & (board[:, 0] < x1) & (board[:, 1] > y0) & (board[:, 1] < y1)
            & (height > SHEET_CLEARANCE_MM) & (height < FIND_MAX_HEIGHT_MM)
        )
        if inside.sum() >= 20:
            return np.flatnonzero(inside), "points standing on the sheet inside the marker ring"
    centre = np.median(xyz, axis=0)
    distance = np.linalg.norm(xyz - centre, axis=1)
    keep = distance <= np.percentile(distance, 90)
    return np.flatnonzero(keep), "points within the 90th-percentile radius of the cloud"


@dataclass
class Layout:
    """Where the model sits in every view: one common scale so the views compare directly."""

    ratio: float            # page millimetres per model millimetre (or per unit if unscaled)
    ratio_label: str | None  # "1:1" etc., None when not to scale
    centre: np.ndarray       # upright model centre, drawn at the middle of every view


def choose_layout(points: np.ndarray, box_mm: tuple[float, float], scaled: bool) -> Layout:
    """One ratio for all four views that fits the largest of them into ``box_mm`` (w, h).

    Scaled views snap down to a conventional drawing ratio (1:1, 2:1, 1:2 ...); unscaled
    views just fill the box.
    """
    if len(points) == 0:
        return Layout(1.0, "1:1" if scaled else None, np.zeros(3))
    low = np.percentile(points, 0.5, axis=0)
    high = np.percentile(points, 99.5, axis=0)
    centre = (low + high) / 2
    extent = np.maximum(high - low, 1e-9)
    fit = math.inf
    for rows in VIEWS.values():
        width, height = np.abs(rows[0]) @ extent, np.abs(rows[1]) @ extent
        fit = min(fit, 0.9 * box_mm[0] / width, 0.9 * box_mm[1] / height)
    if not scaled:
        return Layout(fit, None, centre)
    for ratio, label in RATIOS:
        if ratio <= fit:
            return Layout(ratio, label, centre)
    return Layout(RATIOS[-1][0], RATIOS[-1][1], centre)


def render_view(
    points: np.ndarray,
    colours: np.ndarray,
    view: str,
    layout: Layout,
    size_px: tuple[int, int],
    px_per_mm: float,
    radius_px: int = 2,
    background: int = 255,
) -> np.ndarray:
    """Splat upright points into an ``(H, W, 3)`` RGB image for one orthographic view.

    A model millimetre covers ``layout.ratio * px_per_mm`` pixels. Each point is a disc of
    ``radius_px``; where discs overlap, the point nearest the viewer wins (depth buffer), ties
    broken by input order, so the output is fully deterministic. Nearer points are drawn a
    little brighter, which is the only depth cue a sparse cloud has.
    """
    width, height = size_px
    image = np.full((height, width, 3), background, dtype=np.uint8)
    if len(points) == 0:
        return image
    uvd = (points - layout.centre) @ VIEWS[view].T
    k = layout.ratio * px_per_mm
    px = uvd[:, 0] * k + width / 2
    py = height / 2 - uvd[:, 1] * k
    depth = uvd[:, 2]

    offsets = [(dx, dy) for dx in range(-radius_px, radius_px + 1)
               for dy in range(-radius_px, radius_px + 1) if dx * dx + dy * dy <= radius_px * radius_px + 0.5]
    cx, cy = np.floor(px).astype(np.int64), np.floor(py).astype(np.int64)
    all_x = np.concatenate([cx + dx for dx, _ in offsets])
    all_y = np.concatenate([cy + dy for _, dy in offsets])
    owner = np.tile(np.arange(len(points)), len(offsets))
    valid = (all_x >= 0) & (all_x < width) & (all_y >= 0) & (all_y < height)
    all_x, all_y, owner = all_x[valid], all_y[valid], owner[valid]
    if len(owner) == 0:
        return image

    pixel = all_y * width + all_x
    # Sort by pixel, then nearest first (largest depth towards the viewer), then input order.
    order = np.lexsort((owner, -depth[owner], pixel))
    pixel, owner = pixel[order], owner[order]
    first = np.ones(len(pixel), dtype=bool)
    first[1:] = pixel[1:] != pixel[:-1]
    pixel, owner = pixel[first], owner[first]

    span = float(np.ptp(depth)) or 1.0
    shade = 0.7 + 0.3 * (depth - depth.min()) / span
    shaded = np.clip(colours.astype(np.float64) * shade[:, None], 0, 255).astype(np.uint8)
    image.reshape(-1, 3)[pixel] = shaded[owner]
    return image


def nice_length(max_mm: float) -> float:
    """The largest 1, 2 or 5 x 10^n not above ``max_mm``: a scale bar people can read."""
    if max_mm <= 0:
        return 0.0
    exponent = math.floor(math.log10(max_mm))
    for step in (5, 2, 1):
        value = step * 10**exponent
        if value <= max_mm:
            return float(value)
    return float(10 ** (exponent - 1) * 5)
