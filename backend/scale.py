"""Millimetres per model unit, from the printed marker board.

After reconstruction every registered frame is searched for board markers (``markers.py``).
Each marker corner seen in several frames is triangulated with COLMAP's own camera poses and
intrinsics, so the corners land in exactly the coordinate frame of ``map.ply``. The known edge
length of a marker divided by its triangulated length is one estimate of the scale.

How the number is made trustworthy:

* **Undistorted first.** Corners are undistorted with the camera's SIMPLE_RADIAL coefficient
  before triangulation; COLMAP's poses were estimated on undistorted geometry too.
* **Outlier views are dropped.** A corner is re-projected into every view it was seen in; the
  worst view is removed while any exceeds ``MAX_REPROJECTION_PX``. At least ``MIN_VIEWS`` views
  and ``MIN_RAY_ANGLE_DEG`` of triangulation angle must remain.
* **Distances between markers, not across one.** The known distances used are between the same
  corner of two different markers (e.g. top-left to top-left), which equals the printed marker
  spacing. Measured against synthetic ground truth, a marker's own edges came out 0.4% short
  with sub-pixel refinement (1.2-1.7% with the other fast methods): corner refinement pulls
  the corners of a blurred black square inwards. That bias is the same for every marker, so it
  cancels between corresponding corners, which stayed within 0.1% for every method.
* **Median of markers.** Each marker's estimate is the median over its pairings with the other
  markers; the scale is the median over markers, and the **spread** is the robust standard
  deviation between markers (1.4826 x MAD). That spread is what the UI and the report show as
  "±": the disagreement between markers, which grows if the sheet is not flat.
* **Cross-checked.** The marker-edge estimate and a similarity fit of the whole board are kept
  as checks; a large disagreement produces a warning rather than a silent number.

Nothing here claims real-world accuracy by itself; the README lists what was measured.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import markers
from .reconstruction import read_cameras, read_image_poses

logger = logging.getLogger(__name__)

MIN_VIEWS = 3
MAX_REPROJECTION_PX = 2.0
MIN_RAY_ANGLE_DEG = 5.0
MIN_MARKERS = 3
#: A marker whose triangulated edges differ by more than this is not square: dropped.
MAX_EDGE_RATIO = 1.15
#: Beyond these, the result carries a warning (marker edges vs spacing, and board-fit RMS).
#: Edges are expected to read slightly short (see above), so the first is generous.
EDGE_CHECK_WARN_PCT = 2.0
BOARD_RESIDUAL_WARN_MM = 1.0
#: Frame resamples for the bootstrap uncertainty.
BOOTSTRAP_ROUNDS = 40
#: Robust standard deviation from the median absolute deviation, for normal data.
MAD_TO_SIGMA = 1.4826

SUPPORTED_MODELS = ("SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL")


class UnsupportedCameraError(ValueError):
    """A COLMAP camera model this module cannot undistort."""


# --------------------------------------------------------------------------------------
# Camera geometry
# --------------------------------------------------------------------------------------


def _intrinsics(camera: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    """(fx, fy, cx, cy, k1, k2) for the supported COLMAP models."""
    model, p = camera["model"], camera["params"]
    if model == "SIMPLE_PINHOLE":
        return p[0], p[0], p[1], p[2], 0.0, 0.0
    if model == "PINHOLE":
        return p[0], p[1], p[2], p[3], 0.0, 0.0
    if model == "SIMPLE_RADIAL":
        return p[0], p[0], p[1], p[2], p[3], 0.0
    if model == "RADIAL":
        return p[0], p[0], p[1], p[2], p[3], p[4]
    raise UnsupportedCameraError(f"Camera model {model} is not supported for scaling")


def distort(normalized: np.ndarray, camera: dict[str, Any]) -> np.ndarray:
    """Normalised image coordinates ``(N, 2)`` -> COLMAP pixels, applying radial distortion.

    COLMAP's radial model: ``u_d = u (1 + k1 r^2 + k2 r^4)`` with ``r^2 = u^2 + v^2``.
    """
    fx, fy, cx, cy, k1, k2 = _intrinsics(camera)
    xy = np.asarray(normalized, dtype=np.float64).reshape(-1, 2)
    r2 = (xy**2).sum(axis=1, keepdims=True)
    xy = xy * (1.0 + k1 * r2 + k2 * r2 * r2)
    return np.column_stack([fx * xy[:, 0] + cx, fy * xy[:, 1] + cy])


def undistort(pixels: np.ndarray, camera: dict[str, Any], iterations: int = 30) -> np.ndarray:
    """COLMAP pixels ``(N, 2)`` -> undistorted normalised coordinates, by fixed-point iteration.

    Inverts ``distort``: solve ``u = u_d / (1 + k1 r(u)^2 + k2 r(u)^4)``. For webcam-sized
    coefficients (|k| ~ 0.03) this converges to ~1e-12 in under ten steps.
    """
    fx, fy, cx, cy, k1, k2 = _intrinsics(camera)
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    distorted = np.column_stack([(pixels[:, 0] - cx) / fx, (pixels[:, 1] - cy) / fy])
    xy = distorted.copy()
    for _ in range(iterations):
        r2 = (xy**2).sum(axis=1, keepdims=True)
        xy = distorted / (1.0 + k1 * r2 + k2 * r2 * r2)
    return xy


@dataclass
class View:
    """One registered image: pose (world -> camera), camera centre and intrinsics."""

    name: str
    rotation: np.ndarray
    translation: np.ndarray
    camera: dict[str, Any]

    @property
    def centre(self) -> np.ndarray:
        return -self.rotation.T @ self.translation

    def project(self, point: np.ndarray) -> tuple[np.ndarray, float]:
        """Pixel position of a world point, and its depth in front of the camera."""
        local = self.rotation @ point + self.translation
        return distort(local[:2] / local[2], self.camera)[0], float(local[2])


def _dlt(rotations: np.ndarray, translations: np.ndarray, normalized: np.ndarray) -> np.ndarray:
    projections = np.concatenate([rotations, translations[:, :, None]], axis=2)  # (n, 3, 4)
    rows = np.concatenate(
        [
            normalized[:, 0:1] * projections[:, 2] - projections[:, 0],
            normalized[:, 1:2] * projections[:, 2] - projections[:, 1],
        ]
    )
    _, _, vt = np.linalg.svd(rows)
    homogeneous = vt[-1]
    return homogeneous[:3] / homogeneous[3]


def triangulate(views: list[View], normalized: np.ndarray) -> np.ndarray:
    """Linear (DLT) triangulation of one point from its undistorted observations."""
    return _dlt(
        np.stack([view.rotation for view in views]),
        np.stack([view.translation for view in views]),
        np.asarray(normalized, dtype=np.float64).reshape(-1, 2),
    )


def _by_camera(views: list[View], values: np.ndarray, function) -> np.ndarray:
    """Apply ``function(values, camera)`` per camera: usually all views share one."""
    out = np.empty_like(values)
    groups: dict[int, list[int]] = {}
    for index, view in enumerate(views):
        groups.setdefault(id(view.camera), []).append(index)
    for indices in groups.values():
        out[indices] = function(values[indices], views[indices[0]].camera)
    return out


def triangulate_robust(
    views: list[View], pixels: np.ndarray
) -> tuple[np.ndarray, list[float], int] | None:
    """Triangulate, dropping the worst view until every reprojection error is acceptable.

    Returns ``(point, reprojection errors in px, views used)``, or ``None`` if too few
    consistent views remain or the rays meet at too shallow an angle to fix depth.
    """
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    rotations = np.stack([view.rotation for view in views])
    translations = np.stack([view.translation for view in views])
    normalized = _by_camera(views, pixels, undistort)
    active = np.arange(len(views))
    while len(active) >= MIN_VIEWS:
        point = _dlt(rotations[active], translations[active], normalized[active])
        if not np.isfinite(point).all():
            return None
        local = np.einsum("nij,j->ni", rotations[active], point) + translations[active]
        if (local[:, 2] <= 0).any():
            return None
        chosen = [views[i] for i in active]
        projected = _by_camera(chosen, local[:, :2] / local[:, 2:3], distort)
        errors = np.linalg.norm(projected - pixels[active], axis=1)
        worst = int(np.argmax(errors))
        if errors[worst] <= MAX_REPROJECTION_PX:
            centres = -np.einsum("nji,nj->ni", rotations[active], translations[active])
            rays = centres - point
            rays /= np.linalg.norm(rays, axis=1, keepdims=True)
            widest = math.degrees(math.acos(max(-1.0, min(1.0, float((rays @ rays.T).min())))))
            if widest < MIN_RAY_ANGLE_DEG:
                return None
            return point, errors.tolist(), len(active)
        active = np.delete(active, worst)
    return None


# --------------------------------------------------------------------------------------
# Board geometry
# --------------------------------------------------------------------------------------


def fit_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares ``target ~ s R source + t`` (Umeyama). Returns ``(s, R, t)``.

    Works for planar ``source`` (the board, z = 0): the missing third direction only decides
    the sign of one axis, which the determinant correction resolves to a proper rotation.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
    s0, t0 = source - mu_s, target - mu_t
    covariance = t0.T @ s0 / len(source)
    u, sigma, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(u) * np.linalg.det(vt)) or 1.0
    rotation = u @ correction @ vt
    variance = (s0**2).sum() / len(source)
    scale = float(np.trace(np.diag(sigma) @ correction) / variance)
    translation = mu_t - scale * rotation @ mu_s
    return scale, rotation, translation


def robust_spread(values: np.ndarray) -> float:
    """1.4826 x median absolute deviation: a standard deviation that ignores outliers."""
    values = np.asarray(values, dtype=np.float64)
    return float(MAD_TO_SIGMA * np.median(np.abs(values - np.median(values))))


def edge_estimate(corners: list[np.ndarray], marker_mm: float) -> float | None:
    """mm per model unit from one marker's own 4 edges and 2 diagonals (median), or ``None``
    if the triangulated marker is not square enough to be the printed marker at all."""
    sides = [float(np.linalg.norm(corners[k] - corners[(k + 1) % 4])) for k in range(4)]
    diagonals = [float(np.linalg.norm(corners[0] - corners[2])),
                 float(np.linalg.norm(corners[1] - corners[3]))]
    if min(sides) <= 0 or max(sides) / min(sides) > MAX_EDGE_RATIO:
        return None
    if max(diagonals) / min(diagonals) > MAX_EDGE_RATIO:
        return None
    ratios = [marker_mm / side for side in sides]
    ratios += [marker_mm * math.sqrt(2.0) / diagonal for diagonal in diagonals]
    return float(np.median(ratios))


def spacing_estimates(
    corners: dict[int, list[np.ndarray]], marker_mm: float
) -> dict[int, float]:
    """Per-marker mm per model unit from corresponding corners of every other marker.

    Markers are printed unrotated, so corner k of marker i and corner k of marker j are exactly
    the markers' layout offset apart. For each pair the median over the four corners is taken;
    a marker's estimate is the median over its pairs.
    """
    factor = marker_mm / markers.MARKER_MM
    ids = sorted(corners)
    pairs: dict[int, list[float]] = {marker_id: [] for marker_id in ids}
    for index, i in enumerate(ids):
        for j in ids[index + 1 :]:
            known = factor * math.dist(markers.MARKERS[i], markers.MARKERS[j])
            ratios = [
                known / distance
                for k in range(4)
                if (distance := float(np.linalg.norm(corners[i][k] - corners[j][k]))) > 0
            ]
            if ratios:
                estimate = float(np.median(ratios))
                pairs[i].append(estimate)
                pairs[j].append(estimate)
    return {marker_id: float(np.median(values)) for marker_id, values in pairs.items() if values}


# --------------------------------------------------------------------------------------
# The whole estimate
# --------------------------------------------------------------------------------------


def _load_views(model_dir: Path) -> list[View]:
    cameras = read_cameras(model_dir)
    views = []
    for pose in read_image_poses(model_dir):
        camera = cameras.get(pose["camera_id"])
        if camera is None:
            continue
        _intrinsics(camera)  # raises for models we cannot undistort
        views.append(
            View(
                name=pose["name"],
                rotation=np.array(pose["rotation"], dtype=np.float64),
                translation=np.array(pose["translation"], dtype=np.float64),
                camera=camera,
            )
        )
    return views


def measure_session(session) -> dict[str, Any]:
    """Run ``estimate_scale`` on a finished scan's selected model, with its marker size.

    Never raises: an unexpected failure is reported as ``status: error`` so that a scale problem
    can never turn a good reconstruction into a failed one.
    """
    marker_mm = session.marker_size_mm or markers.MARKER_MM
    index = session.result.model_index if session.result else None
    model_dir = session.sparse_dir / str(index) if index is not None else session.sparse_dir
    try:
        result = estimate_scale(model_dir, session.images_dir, marker_mm)
    except Exception as exc:  # pragma: no cover - defensive; estimate_scale handles known cases
        logger.exception("Scale estimate failed for %s", session.id)
        result = {"status": "error", "message": repr(exc), "markerSizeMm": marker_mm}
    result["modelIndex"] = index
    return result


@dataclass
class _Solution:
    corners: dict[tuple[int, int], np.ndarray]
    errors: list[float]
    frames: set[int]
    edges: dict[int, float]
    per_marker: dict[int, float]


def _solve(
    views: list[View],
    observations: dict[tuple[int, int], list[tuple[int, np.ndarray]]],
    marker_mm: float,
    counts: np.ndarray | None = None,
) -> _Solution:
    """Triangulate every corner and estimate per marker. ``counts`` (times each view is used)
    re-weights the frames, which is how the bootstrap resamples them."""
    corners3d: dict[tuple[int, int], np.ndarray] = {}
    errors: list[float] = []
    used: set[int] = set()
    for key, seen in observations.items():
        if counts is not None:
            seen = [(i, p) for i, p in seen for _ in range(int(counts[i]))]
        if len(seen) < MIN_VIEWS:
            continue
        result = triangulate_robust([views[i] for i, _ in seen], np.array([p for _, p in seen]))
        if result is None:
            continue
        corners3d[key] = result[0]
        errors.extend(result[1])
        used.update(i for i, _ in seen)

    # Markers with all four corners, and square enough to be the printed marker.
    complete: dict[int, list[np.ndarray]] = {}
    edges: dict[int, float] = {}
    for marker_id in sorted({marker_id for marker_id, _ in corners3d}):
        corners = [corners3d.get((marker_id, k)) for k in range(4)]
        if any(corner is None for corner in corners):
            continue
        edge = edge_estimate(corners, marker_mm)  # type: ignore[arg-type]
        if edge is not None:
            complete[marker_id] = corners  # type: ignore[assignment]
            edges[marker_id] = edge
    per_marker = spacing_estimates(complete, marker_mm) if len(complete) >= 2 else {}
    return _Solution(corners3d, errors, used, edges, per_marker)


def bootstrap_frames(
    views: list[View],
    observations: dict[tuple[int, int], list[tuple[int, np.ndarray]]],
    marker_mm: float,
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = 0,
) -> float | None:
    """Robust standard deviation of the scale when the frames are resampled with replacement.

    Every marker is measured with every camera pose, so an error in one pose moves all markers
    together, and the between-marker spread cannot see it. Resampling the frames can: it
    re-measures the board with different subsets of poses. Seeded, so reruns agree.
    """
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(rounds):
        counts = np.bincount(rng.integers(0, len(views), len(views)), minlength=len(views))
        solution = _solve(views, observations, marker_mm, counts)
        if len(solution.per_marker) >= MIN_MARKERS:
            estimates.append(float(np.median(list(solution.per_marker.values()))))
    if len(estimates) < rounds // 2:
        return None
    return robust_spread(np.array(estimates))


def estimate_scale(
    model_dir: Path, images_dir: Path, marker_mm: float = markers.MARKER_MM
) -> dict[str, Any]:
    """Detect, triangulate and measure the board. Always returns a dict with a ``status``:

    ``scaled`` (with ``mmPerUnit`` and ``spreadPct``), ``no_markers``, ``insufficient`` or
    ``error``. It never raises for a bad scan; a scan without a board is the normal case.
    """
    started = time.time()
    base: dict[str, Any] = {
        "board": markers.BOARD_NAME,
        "markerSizeMm": marker_mm,
        "computedAt": started,
    }
    try:
        views = _load_views(model_dir)
    except UnsupportedCameraError as exc:
        return {**base, "status": "error", "message": str(exc)}
    if not views:
        return {**base, "status": "error", "message": "The model has no registered images."}

    # (marker id, corner) -> [(view index, pixel)]
    observations: dict[tuple[int, int], list[tuple[int, np.ndarray]]] = {}
    frames_with_markers = 0
    for index, view in enumerate(views):
        gray = cv2.imread(str(images_dir / view.name), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        found = markers.detect(gray)
        frames_with_markers += bool(found)
        for marker_id, corners in found.items():
            for k in range(4):
                observations.setdefault((marker_id, k), []).append((index, corners[k]))

    base.update({"framesChecked": len(views), "framesWithMarkers": frames_with_markers})
    if not observations:
        return {**base, "status": "no_markers", "seconds": round(time.time() - started, 2)}

    solved = _solve(views, observations, marker_mm)
    corners3d, errors, used_frames = solved.corners, solved.errors, solved.frames
    edges, per_marker = solved.edges, solved.per_marker

    base.update(
        {
            "markersSeen": sorted({marker_id for marker_id, _ in observations}),
            "markerIds": sorted(per_marker),
            "markersUsed": len(per_marker),
            "framesUsed": len(used_frames),
            "reprojectionErrorPx": round(float(np.median(errors)), 3) if errors else None,
            "seconds": round(time.time() - started, 2),
        }
    )
    if len(per_marker) < MIN_MARKERS:
        return {**base, "status": "insufficient"}

    values = np.array(list(per_marker.values()))
    mm_per_unit = float(np.median(values))
    spread = robust_spread(values)
    bootstrap = bootstrap_frames(views, observations, marker_mm)
    edge_check = 100.0 * (float(np.median(list(edges.values()))) / mm_per_unit - 1.0)

    # The whole board as a rigid, scaled sheet: its pose (for the report) and how well a flat
    # sheet explains the corners at all.
    layout = markers.corner_positions(marker_mm)
    keys = [(marker_id, k) for marker_id in per_marker for k in range(4)]
    board_mm = np.array([[*layout[key], 0.0] for key in keys])
    model = np.array([corners3d[key] for key in keys])
    units_per_mm, rotation, translation = fit_similarity(board_mm, model)
    residual = (units_per_mm * (rotation @ board_mm.T).T + translation) - model
    residual_mm = float(np.sqrt((residual**2).sum(axis=1).mean())) / units_per_mm

    warnings = []
    if abs(edge_check) > EDGE_CHECK_WARN_PCT:
        warnings.append(
            f"Marker size and marker spacing disagree by {edge_check:+.1f}%. Check that the "
            "sheet was printed without scaling and that the entered marker size is right."
        )
    if residual_mm > BOARD_RESIDUAL_WARN_MM:
        warnings.append(
            f"The markers do not lie on a flat sheet (RMS {residual_mm:.1f} mm). "
            "Tape the sheet flat and scan again."
        )

    # Which side of the sheet the cameras are on decides the board's "up" (towards them).
    normal = rotation[:, 2]
    centres = np.array([view.centre for view in views])
    above = float(np.median((centres - translation) @ normal))
    return {
        **base,
        "status": "scaled",
        "method": "marker spacing (corresponding corners)",
        "mmPerUnit": mm_per_unit,
        "spreadPct": round(100.0 * spread / mm_per_unit, 3),
        "bootstrapPct": round(100.0 * bootstrap / mm_per_unit, 3) if bootstrap is not None else None,
        "perMarker": {str(k): round(v, 6) for k, v in per_marker.items()},
        "edgeCheckPct": round(edge_check, 3),
        "boardResidualMm": round(residual_mm, 3),
        "warnings": warnings,
        "seconds": round(time.time() - started, 2),
        "boardPose": {
            "scale": units_per_mm,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "normalSign": 1.0 if above >= 0 else -1.0,
        },
    }
