"""The find report: a two-page PDF documenting one scanned find, and the capture log behind it.

Page 1 is the find: header with find number, site and context; orthographic top, front, side
and bottom views of the model, each with a millimetre scale bar or clearly marked "not to
scale". Page 2 is the evidence: a photograph from the scan, the find record, the capture log
with its quality numbers, how the scale was obtained, and what the views do not show.

Every number printed comes from the scan itself (``scan.json``, the COLMAP model, the frames)
or is a stated constant of the app. A number that was not measured is printed as "not
recorded", never guessed.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import cv2
import numpy as np

from . import markers
from .find_record import material_label
from .ortho import (
    VIEW_LABELS,
    choose_layout,
    find_points,
    nice_length,
    render_view,
    upright_rotation,
    yaw_alignment,
)
from .pdf import A4_MM, PdfDocument
from .reconstruction import read_camera_centers, read_point_stats, read_points3d

NOT_RECORDED = "not recorded"
_LOG_VERSION_RE = re.compile(r"\(version ([0-9][\w.]*)\)")

GREY = (0.45, 0.45, 0.45)
LIGHT = (0.85, 0.85, 0.85)
WARN = (0.7, 0.2, 0.1)


def _model_dir(session):
    index = session.result.model_index if session.result else None
    return session.sparse_dir / str(index) if index is not None else None


def _colmap_version(session) -> str | None:
    """From scan.json, or for older scans from the first lines of the COLMAP log (read-only)."""
    if session.colmap_version:
        return session.colmap_version
    try:
        with session.log_path.open(encoding="utf-8", errors="replace") as handle:
            head = handle.read(2000)
    except OSError:
        return None
    match = _LOG_VERSION_RE.search(head)
    return match.group(1) if match else None


def _resolution(session) -> tuple[int, int] | None:
    """Measured at capture, or read from the first frame for scans made before that."""
    capture = session.capture or {}
    if capture.get("width") and capture.get("height"):
        return int(capture["width"]), int(capture["height"])
    frames = sorted(session.images_dir.glob("frame_*.jpg")) if session.images_dir.is_dir() else []
    if frames:
        image = cv2.imread(str(frames[0]), cv2.IMREAD_UNCHANGED)
        if image is not None:
            return int(image.shape[1]), int(image.shape[0])
    return None


def capture_log(session, point_stats: dict[str, float] | None = None) -> dict[str, Any]:
    """Automatic capture log for the report and ``GET /api/scans/{id}/capture-log``.

    ``None`` means "not recorded for this scan" (e.g. scans made before a field existed).
    """
    result = session.result
    model_dir = _model_dir(session)
    if point_stats is None and model_dir is not None and (model_dir / "points3D.bin").is_file():
        point_stats = read_point_stats(model_dir)
    resolution = _resolution(session)
    scale = session.scale or {}
    return {
        "scanId": session.id,
        "createdAt": session.created_at,
        "source": session.source,
        "camera": (session.capture or {}).get("camera"),
        "resolution": list(resolution) if resolution else None,
        "preset": session.preset,
        "colmapVersion": _colmap_version(session),
        "framesCaptured": session.stats.captured,
        "framesAccepted": session.stats.accepted,
        "framesPlaced": result.registered_images if result else None,
        "points": result.points if result else None,
        "meanReprojectionErrorPx": point_stats["mean_reprojection_error"] if point_stats else None,
        "meanTrackLength": point_stats["mean_track_length"] if point_stats else None,
        "reconstructionSeconds": result.duration_seconds if result else None,
        "scaleStatus": scale.get("status", "not_measured"),
        "mmPerUnit": scale.get("mmPerUnit"),
        "scaleUncertaintyPct": scale_uncertainty_pct(scale),
        "markersUsed": scale.get("markersUsed"),
        "scaleFramesUsed": scale.get("framesUsed"),
        "markerSizeMm": scale.get("markerSizeMm"),
    }


def scale_uncertainty_pct(scale: dict[str, Any] | None) -> float | None:
    """The ± shown everywhere: the larger of the between-marker and frame-bootstrap spreads."""
    if not scale or scale.get("status") != "scaled":
        return None
    values = [v for v in (scale.get("spreadPct"), scale.get("bootstrapPct")) if v is not None]
    return max(values) if values else None


def format_pct(value: float | None) -> str:
    """Two significant figures: "0.0021", "0.35", "1.2" - never a misleading "0.00"."""
    return "?" if value is None else f"{value:.2g}"


def scale_summary(scale: dict[str, Any] | None) -> str:
    """One line for humans, used by the report and mirrored by the viewer."""
    scale = scale or {}
    status = scale.get("status")
    if status == "scaled":
        return (
            f"Scaled: {scale['mmPerUnit']:.4g} mm per model unit, "
            f"± {format_pct(scale_uncertainty_pct(scale))}% (from {scale['markersUsed']} markers "
            f"in {scale['framesUsed']} frames)"
        )
    if status == "insufficient":
        return (
            f"Not to scale: the marker board was seen ({len(scale.get('markersSeen', []))} "
            f"markers) but fewer than 3 could be measured reliably."
        )
    if status == "no_markers":
        return "Not to scale: no marker board in the frames. Print the board and scan the find on it."
    if status == "error":
        return f"Not to scale: {scale.get('message', 'the scale could not be measured')}."
    return "Not to scale: this scan was made before scaling existed, or was never measured."


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


def _fmt(value: Any, digits: int | None = None, unit: str = "") -> str:
    if value is None or value == "":
        return NOT_RECORDED
    if digits is not None and isinstance(value, (int, float)):
        return f"{value:.{digits}f}{unit}"
    return f"{value}{unit}"


def _photo(session) -> tuple[bytes, int, int, str] | None:
    """The middle frame (the library thumbnail's frame), re-encoded at ~1000 px."""
    frames = sorted(session.images_dir.glob("frame_*.jpg")) if session.images_dir.is_dir() else []
    if not frames:
        return None
    path = frames[len(frames) // 2]
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    scale = min(1.0, 1000 / image.shape[1])
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        return None
    return jpeg.tobytes(), small.shape[1], small.shape[0], path.name


def _upright_points(session) -> tuple[np.ndarray, np.ndarray, str, bool]:
    """Selected cloud in the upright, yaw-aligned frame; millimetres when scaled."""
    model_dir = _model_dir(session)
    if model_dir is None:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8), "no model", False
    xyz, rgb = read_points3d(model_dir)
    scale = session.scale or {}
    chosen, how = find_points(xyz, scale)
    xyz, rgb = xyz[chosen], rgb[chosen]
    cameras = read_camera_centers(model_dir)
    rotation = upright_rotation(np.array([c["up"] for c in cameras]) if cameras else np.zeros((0, 3)))
    upright = xyz @ rotation.T
    upright = upright @ yaw_alignment(upright).T
    scaled = scale.get("status") == "scaled"
    if scaled:
        upright = upright * float(scale["mmPerUnit"])
    return upright, rgb, how, scaled


def _scale_bar(page, x: float, y: float, ratio: float, max_page_mm: float) -> None:
    """A bar of a round number of real millimetres, drawn at the view's ratio."""
    real = nice_length(max_page_mm / ratio)
    length = real * ratio
    page.rect(x, y, length, 1.0, fill=(0, 0, 0))
    half = length / 2
    page.rect(x + half, y, half, 1.0, fill=(1, 1, 1), stroke=(0, 0, 0), width=0.2)
    for tick in (0.0, half, length):
        page.line(x + tick, y - 1.2, x + tick, y + 1.0, width=0.2)
    label = f"{real:g} mm"
    page.text(x + length + 2, y + 1.1, label, 7.5)


def _header(page, session, generated: datetime, version: str, number: int, total: int) -> float:
    record = session.record or {}
    width = A4_MM[0]
    page.text(15, 16, "FIND REPORT", 8, bold=True, color=GREY)
    page.text(width - 15, 16, f"Page {number} of {total}", 8, color=GREY, align="right")
    title = record.get("findNumber") or "No find number recorded"
    page.text(15, 26, title, 18, bold=True, color=(0, 0, 0) if record.get("findNumber") else WARN)
    site = " · ".join(
        part
        for part in (
            f"Site {record['siteCode']}" if record.get("siteCode") else "",
            f"Context {record['context']}" if record.get("context") else "",
        )
        if part
    )
    page.text(15, 33, site or "Site and context not recorded", 10, color=(0, 0, 0) if site else GREY)
    page.line(15, 37, width - 15, 37, width=0.4)
    page.text(
        15,
        A4_MM[1] - 9,
        f"Generated {generated:%Y-%m-%d %H:%M} · Webcam 3D Mapper {version} · scan {session.id}",
        7,
        color=GREY,
    )
    return 44.0


def build_report(session, version: str, generated: datetime | None = None) -> bytes:
    """Render the two-page A4 report for a completed scan."""
    generated = generated or datetime.now()
    record = session.record or {}
    scale = session.scale or {}
    doc = PdfDocument(
        title=f"Find report {record.get('findNumber') or session.id}",
        producer=f"Webcam 3D Mapper {version}",
        created=generated,
    )

    # ── page 1: the views ─────────────────────────────────────────────────────────
    page = doc.add_page(A4_MM)
    y = _header(page, session, generated, version, 1, 2)
    scaled = scale.get("status") == "scaled"
    page.text(15, y, scale_summary(scale), 9, bold=not scaled, color=(0, 0, 0) if scaled else WARN)
    y += 4

    points, colours, how, scaled = _upright_points(session)
    cell_w, cell_h = 87.0, 104.0
    image_w, image_h = 87.0, 88.0
    layout = choose_layout(points, (image_w, image_h), scaled)
    px_per_mm = 8.0
    if scaled:
        page.text(15, y + 4, f"Drawing scale {layout.ratio_label} when printed at 100% on A4. "
                             "Scale bars stay correct at any print size.", 8, color=GREY)
    else:
        page.text(15, y + 4, "The four views share one relative unit. Do not measure them.",
                  8, color=GREY)
    top = y + 9
    for index, view in enumerate(("top", "front", "side", "bottom")):
        x = 15 + (index % 2) * (cell_w + 6)
        cy = top + (index // 2) * (cell_h + 4)
        page.text(x, cy + 3.5, VIEW_LABELS[view], 9, bold=True)
        image = render_view(
            points, colours, view, layout,
            (round(image_w * px_per_mm), round(image_h * px_per_mm)), px_per_mm,
        )
        doc.image_rgb(page, image, x, cy + 5, image_w, image_h)
        page.rect(x, cy + 5, image_w, image_h, stroke=LIGHT, width=0.2)
        if scaled:
            _scale_bar(page, x + 2, cy + 5 + image_h + 5, layout.ratio, 40.0)
        else:
            page.text(x + 2, cy + 5 + image_h + 6, "not to scale", 8, bold=True, color=WARN)
    note_y = top + 2 * (cell_h + 4) + 2
    on_sheet = "on the sheet" in how
    page.paragraph(
        15, note_y,
        f"Orthographic views of the sparse point cloud ({len(points):,} {how}). Up is the "
        "average camera up during the scan; the longest horizontal axis runs left to right. "
        "Only surfaces the camera saw are reconstructed"
        + (": the underside of a find lying on the sheet is missing, so the bottom view shows "
           "the upper surfaces from below." if on_sheet else "."),
        180, size=7.5, color=GREY,
    )

    # ── page 2: photo, record, capture log ───────────────────────────────────────
    page = doc.add_page(A4_MM)
    y = _header(page, session, generated, version, 2, 2)
    photo = _photo(session)
    if photo:
        data, width, height, name = photo
        w = 100.0
        h = w * height / width
        doc.image_jpeg(page, data, width, height, False, 15, y, w, h)
        page.text(15, y + h + 4, f"Frame {name} from the scan (not to scale).", 7.5, color=GREY)
        record_x, record_w = 15 + w + 6, 180 - w - 6
    else:
        page.text(15, y + 4, "No photograph: the scan has no frames.", 8, color=GREY)
        record_x, record_w = 15, 180
        h = 0.0

    ry = y + 3
    page.text(record_x, ry, "Find record", 10, bold=True)
    ry += 6
    rows = [
        ("Find number", record.get("findNumber")),
        ("Site", record.get("siteCode")),
        ("Context", record.get("context")),
        ("Material", material_label(record) if record.get("material") else None),
        ("Date found", record.get("dateFound")),
        ("Recorder", record.get("recorder")),
    ]
    for label, value in rows:
        page.text(record_x, ry, label, 8, color=GREY)
        ry = page.paragraph(record_x + 24, ry, value or NOT_RECORDED, record_w - 24, size=8.5,
                            color=(0, 0, 0) if value else GREY)
        ry += 0.6

    y = max(y + h + 10, ry + 4)
    if record.get("notes"):
        page.text(15, y, "Notes", 10, bold=True)
        y = page.paragraph(15, y + 5.5, record["notes"], 180, size=8.5) + 3

    log = capture_log(session)
    page.text(15, y, "Capture log", 10, bold=True)
    y += 6
    resolution = log["resolution"]
    placed = log["framesPlaced"]
    entries = [
        ("Scanned", datetime.fromtimestamp(log["createdAt"]).strftime("%Y-%m-%d %H:%M")),
        ("Source", "webcam" if log["source"] == "webcam" else "imported images"),
        ("Camera", log["camera"]),
        ("Resolution", f"{resolution[0]} x {resolution[1]} px" if resolution else None),
        ("Quality preset", log["preset"]),
        ("COLMAP version", log["colmapVersion"]),
        ("Frames", f"{log['framesCaptured']} captured, {log['framesAccepted']} accepted"),
        ("Placed in 3D", f"{placed} of {log['framesAccepted']} accepted frames" if placed is not None else None),
        ("Points", f"{log['points']:,}" if log["points"] is not None else None),
        ("Mean reprojection error", _fmt(log["meanReprojectionErrorPx"], 3, " px")),
        ("Mean track length", _fmt(log["meanTrackLength"], 2, " views per point")),
        ("Reconstruction time", _fmt(log["reconstructionSeconds"], 0, " s")),
    ]
    for label, value in entries:
        page.text(15, y, label, 8, color=GREY)
        page.text(60, y, value if value not in (None, "") else NOT_RECORDED, 8.5,
                  color=(0, 0, 0) if value not in (None, "", NOT_RECORDED) else GREY)
        y += 4.6

    y += 3
    page.text(15, y, "Scale", 10, bold=True)
    y += 5.5
    y = page.paragraph(15, y, scale_summary(scale), 180, size=8.5, bold=not scaled,
                       color=(0, 0, 0) if scaled else WARN)
    if scaled:
        details = (
            f"Method: printed board {scale.get('board', markers.BOARD_NAME)}, "
            f"{scale.get('markerSizeMm', markers.MARKER_MM):g} mm markers; distances between "
            f"corresponding corners of different markers, triangulated with the reconstructed "
            f"cameras. Median over markers; the ± is the larger of the between-marker spread "
            f"({scale.get('spreadPct')}%) and a frame bootstrap ({scale.get('bootstrapPct')}%). "
            f"Median corner reprojection error {scale.get('reprojectionErrorPx')} px; board "
            f"flatness RMS {scale.get('boardResidualMm')} mm; marker-edge check "
            f"{scale.get('edgeCheckPct'):+.2f}%. The ± cannot show errors that affect every "
            f"marker equally, such as a wrongly entered marker size."
        )
        y = page.paragraph(15, y + 1, details, 180, size=7.5, color=GREY)
        for warning in scale.get("warnings", []):
            y = page.paragraph(15, y + 1, f"Warning: {warning}", 180, size=8, bold=True, color=WARN)
    return doc.to_bytes()


def report_filename(session) -> str:
    """A safe download name: the find number when there is one, else the scan id."""
    number = (session.record or {}).get("findNumber") or ""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", number).strip("._")[:60]
    return f"find-report-{safe or session.id}.pdf"
