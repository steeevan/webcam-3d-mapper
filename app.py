"""Webcam 3D Mapper - local FastAPI server.

Serves the single-page frontend and a small JSON API. Everything stays on this machine:
the server binds to loopback, frames are written to ``scans/``, and reconstruction shells out
to a locally installed COLMAP.

Run with ``python app.py`` or ``scripts/run.bat`` / ``scripts/run.sh``.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import logging
import shutil
import subprocess
import sys
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from backend import config, markers
from backend.colmap_runner import ColmapNotFoundError
from backend.colmap_runner import runner as colmap
from backend.find_record import FIELD_LIMITS, MATERIALS, RecordError, validate_record
from backend.image_quality import decode_jpeg, make_thumbnail
from backend.models import STAGE_SEQUENCE, FrameDecision, ScanStatus
from backend.preview import LivePreview
from backend.reconstruction import ReconstructionPipeline, read_camera_centers, read_point_stats
from backend.report import build_report, capture_log, report_filename, scale_uncertainty_pct
from backend.scale import measure_session
from backend.scan_manager import InvalidScanIdError, ScanNotFoundError, ScanSession
from backend.scan_manager import manager as scans

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mapper")

#: Running reconstruction tasks, keyed by scan id, so they can be cancelled on shutdown.
_tasks: dict[str, asyncio.Task[None]] = {}

#: Live previews of scans that are still capturing, keyed by scan id.
_previews: dict[str, LivePreview] = {}

#: points3D.bin statistics, keyed by (scan id, file mtime) - the walk is not free on big clouds.
_point_stats: dict[tuple[str, float], dict[str, float] | None] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    info = colmap.detect()
    banner = [
        "",
        "  Webcam 3D Mapper",
        f"  http://{config.HOST}:{config.PORT}",
        "",
        f"  Python      {sys.version.split()[0]}",
        (
            f"  COLMAP      detected  -  {info.path} (version {info.version})"
            if info.available
            else "  COLMAP      NOT FOUND  -  reconstruction disabled until it is installed"
        ),
        f"  GPU         {info.gpu}",
        f"  Workspace   {config.SCANS_DIR}",
        "",
    ]
    logger.info("\n".join(banner))
    try:
        yield
    finally:
        for scan_id in list(_previews):
            await _stop_preview(scan_id)
        for task in list(_tasks.values()):
            task.cancel()
        if _tasks:
            await asyncio.gather(*_tasks.values(), return_exceptions=True)


app = FastAPI(title="Webcam 3D Mapper", version="1.1.0", lifespan=lifespan)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _session(scan_id: str) -> ScanSession:
    try:
        return scans.get(scan_id)
    except InvalidScanIdError:
        raise HTTPException(status_code=400, detail="Invalid scan id")
    except ScanNotFoundError:
        raise HTTPException(status_code=404, detail="Scan not found")


def _status_payload(session: ScanSession) -> dict[str, Any]:
    payload = session.to_dict()
    payload["stages"] = [
        {"key": key, "label": label} for key, label in STAGE_SEQUENCE
    ]
    payload["scaleUncertaintyPct"] = scale_uncertainty_pct(session.scale)
    return payload


def _record_or_422(payload: Any) -> dict[str, str]:
    try:
        return validate_record(payload)
    except RecordError as exc:
        raise HTTPException(status_code=422, detail={"message": str(exc), "field": exc.field})


def _marker_size(value: Any) -> float | None:
    """A measured marker size in millimetres, or None for the board's nominal size."""
    if value in (None, ""):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HTTPException(status_code=422, detail="markerSizeMm must be a number")
    if not config.MARKER_SIZE_RANGE_MM[0] <= float(value) <= config.MARKER_SIZE_RANGE_MM[1]:
        low, high = config.MARKER_SIZE_RANGE_MM
        raise HTTPException(status_code=422, detail=f"markerSizeMm must be between {low} and {high}")
    return float(value)


async def _stop_preview(scan_id: str, discard: bool = False) -> None:
    """Cancel a scan's live preview, killing any COLMAP it is running."""
    preview = _previews.pop(scan_id, None)
    if preview is None:
        return
    await preview.stop()
    if discard:
        await asyncio.to_thread(preview.discard_workspace)


# --------------------------------------------------------------------------------------
# System
# --------------------------------------------------------------------------------------


@app.get("/api/system")
async def system_info() -> dict[str, Any]:
    """Readiness for the UI's status pills, plus capture settings the frontend obeys."""
    info = colmap.detect()
    return {
        "colmap": info.to_dict(),
        "capture": {
            "intervalMs": config.CAPTURE_INTERVAL_MS,
            "maxFrames": config.MAX_ACCEPTED_FRAMES,
            "minFrames": config.MIN_FRAMES_FOR_RECONSTRUCTION,
            "recommendedFrames": config.RECOMMENDED_FRAMES,
            "width": 1280,
            "height": 720,
            "jpegQuality": 0.9,
        },
        "presets": [
            {"name": preset.name, "label": preset.label, "description": preset.description}
            for preset in config.PRESETS.values()
        ],
        "defaultPreset": config.DEFAULT_PRESET,
        "record": {
            "materials": [{"value": value, "label": label} for value, label in MATERIALS],
            "limits": FIELD_LIMITS,
        },
        "markerBoard": {
            "markerMm": markers.MARKER_MM,
            "papers": sorted(markers.PAPERS),
            "sizeRangeMm": list(config.MARKER_SIZE_RANGE_MM),
        },
        "python": sys.version.split()[0],
        "version": app.version,
    }


@app.post("/api/system/colmap")
async def set_colmap_path(payload: dict[str, str]) -> dict[str, Any]:
    """Point the app at a COLMAP executable and persist the choice."""
    path = (payload or {}).get("path", "").strip()
    if not path:
        raise HTTPException(status_code=400, detail="A path is required")
    try:
        info = await asyncio.to_thread(colmap.set_path, path)
    except ColmapNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"colmap": info.to_dict()}


@app.post("/api/system/colmap/detect")
async def redetect_colmap() -> dict[str, Any]:
    info = await asyncio.to_thread(colmap.detect, True)
    return {"colmap": info.to_dict()}


# --------------------------------------------------------------------------------------
# Scans
# --------------------------------------------------------------------------------------


@app.get("/api/scans")
async def list_scans(q: str = "") -> dict[str, Any]:
    """Recent scans, or with ``q`` every scan whose find record matches (see find_record.py)."""
    query = q[:200]
    limit = 100 if query.strip() else 20
    return {"scans": await asyncio.to_thread(scans.list_scans, limit, query), "query": query}


@app.post("/api/scans")
async def create_scan(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Start a scan. Optional: ``record`` (the find record), ``camera`` (the browser's camera
    label, for the capture log) and ``markerSizeMm`` (the printed board as measured)."""
    payload = payload or {}
    preset = payload.get("preset", config.DEFAULT_PRESET)
    record = _record_or_422(payload["record"]) if payload.get("record") else None
    camera = payload.get("camera")
    session = scans.create(
        preset=str(preset),
        record=record,
        camera=camera if isinstance(camera, str) else "",
        marker_size_mm=_marker_size(payload.get("markerSizeMm")),
    )
    _start_preview(session)
    return _status_payload(session)


@app.get("/api/scans/{scan_id}/record")
async def get_record(scan_id: str) -> dict[str, Any]:
    return {"id": scan_id, "record": _session(scan_id).record}


@app.put("/api/scans/{scan_id}/record")
async def put_record(scan_id: str, payload: Any = Body(...)) -> dict[str, Any]:
    """Replace the find record. Allowed at any time: before, during and after reconstruction."""
    session = _session(scan_id)
    session.record = _record_or_422(payload)
    session.save()
    return {"id": session.id, "record": session.record}


@app.get("/api/scans/{scan_id}/capture-log")
async def get_capture_log(scan_id: str) -> dict[str, Any]:
    """What the report prints about how the scan was made; ``null`` means not recorded."""
    session = _session(scan_id)
    return await asyncio.to_thread(capture_log, session, await _cached_point_stats(session))


@app.get("/api/scans/{scan_id}/scale")
async def get_scale(scan_id: str) -> dict[str, Any]:
    session = _session(scan_id)
    scale = session.scale or {"status": "not_measured"}
    return {**scale, "uncertaintyPct": scale_uncertainty_pct(scale)}


@app.post("/api/scans/{scan_id}/scale")
async def remeasure_scale(scan_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Measure the board again, e.g. with the marker size the printout really has."""
    session = _session(scan_id)
    if session.status != ScanStatus.COMPLETE:
        raise HTTPException(status_code=409, detail="Scan is not complete")
    session.marker_size_mm = _marker_size((payload or {}).get("markerSizeMm"))
    session.scale = await asyncio.to_thread(measure_session, session)
    session.save()
    return {**session.scale, "uncertaintyPct": scale_uncertainty_pct(session.scale)}


@app.get("/api/scans/{scan_id}/report.pdf")
async def find_report(scan_id: str) -> Response:
    """The two-page find report, built on request (nothing is written to the scan folder)."""
    session = _session(scan_id)
    if session.status != ScanStatus.COMPLETE:
        raise HTTPException(status_code=409, detail="Scan is not complete")
    pdf = await asyncio.to_thread(build_report, session, app.version)
    return Response(
        pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{report_filename(session)}"'},
    )


@app.get("/api/marker-board.pdf")
async def marker_board(paper: str = "a4") -> Response:
    """The printable scale board, vector, at exact size. ``paper`` is ``a4`` or ``letter``."""
    if paper not in markers.PAPERS:
        raise HTTPException(status_code=400, detail="paper must be a4 or letter")
    return Response(
        _board_pdf(paper),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="scale-board-{paper}.pdf"'},
    )


@functools.cache
def _board_pdf(paper: str) -> bytes:
    return markers.board_pdf(paper)


async def _cached_point_stats(session: ScanSession) -> dict[str, float] | None:
    """points3D.bin statistics for a complete scan, cached by file modification time."""
    result = session.result
    if session.status != ScanStatus.COMPLETE or not result or result.model_index is None:
        return None
    points_file = session.sparse_dir / str(result.model_index) / "points3D.bin"
    if not points_file.is_file():
        return None
    key = (session.id, points_file.stat().st_mtime)
    if key not in _point_stats:
        _point_stats[key] = await asyncio.to_thread(read_point_stats, points_file.parent)
    return _point_stats[key]


def _start_preview(session: ScanSession) -> None:
    if not colmap.detect().available:
        return
    preview = LivePreview(session)
    _previews[session.id] = preview
    preview.start()


@app.post("/api/scans/{scan_id}/frames")
async def upload_frame(scan_id: str, frame: UploadFile = File(...)) -> dict[str, Any]:
    """Quality-check one captured frame and keep it if it is usable."""
    session = _session(scan_id)
    if session.status != ScanStatus.CAPTURING:
        raise HTTPException(status_code=409, detail="This scan is no longer capturing")

    if session.stats.accepted >= config.MAX_ACCEPTED_FRAMES:
        return {
            "accepted": False,
            "reason": "limit",
            "acceptedFrames": session.stats.accepted,
            "capturedFrames": session.stats.captured,
            "limitReached": True,
        }

    data = await frame.read()
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Frame too large")

    decision, board = await asyncio.to_thread(_evaluate_and_store, session, data)
    session.record_decision(decision)

    return {
        **decision.to_dict(),
        "board": board,
        "acceptedFrames": session.stats.accepted,
        "capturedFrames": session.stats.captured,
        "limitReached": session.stats.accepted >= config.MAX_ACCEPTED_FRAMES,
    }


def _evaluate_and_store(session: ScanSession, data: bytes) -> tuple[FrameDecision, int]:
    """Runs on a worker thread: decode, score, look for the board, write accepted frames.

    Returns the decision and how many board markers the frame shows (a quick count without
    sub-pixel refinement, ~2 ms at 1280x720; the scale itself is measured after reconstruction).
    """
    image = decode_jpeg(data)
    if image is None:
        return FrameDecision(False, "error"), 0

    decision = session.quality_filter.evaluate(image)
    board = markers.count(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    if decision.accepted:
        try:
            session.write_frame(data)
        except OSError as exc:
            logger.warning("Could not write a frame for %s: %s", session.id, exc)
            return FrameDecision(False, "error"), board
        if "width" not in session.capture:
            # The size COLMAP will see, measured from the frame rather than asked for.
            session.capture["width"], session.capture["height"] = image.shape[1], image.shape[0]
    return decision, board


@app.post("/api/scans/{scan_id}/finish")
async def finish_scan(scan_id: str) -> dict[str, Any]:
    """Stop capturing and start reconstruction in the background."""
    session = _session(scan_id)
    if session.status != ScanStatus.CAPTURING:
        return _status_payload(session)

    # The preview's COLMAP would otherwise compete with the real run for the CPU.
    await _stop_preview(scan_id, discard=True)
    session.set_status(ScanStatus.QUEUED)
    session.save()
    _start_reconstruction(session)
    return _status_payload(session)


def _start_reconstruction(session: ScanSession) -> None:
    async def job() -> None:
        try:
            await ReconstructionPipeline(session).run()
        finally:
            session.save()
            _tasks.pop(session.id, None)

    _tasks[session.id] = asyncio.create_task(job(), name=f"reconstruct:{session.id}")


@app.get("/api/scans/{scan_id}/status")
async def scan_status(scan_id: str) -> dict[str, Any]:
    return _status_payload(_session(scan_id))


@app.get("/api/scans/{scan_id}/preview")
async def scan_preview(scan_id: str) -> dict[str, Any]:
    """Latest live-preview round of a capturing scan: counts, coverage and asset URLs."""
    session = _session(scan_id)
    preview = _previews.get(session.id)
    if preview is None:
        return {"state": "off"}
    return preview.to_dict()


@app.get("/api/scans/{scan_id}/preview/{round_number}/{asset}")
async def scan_preview_asset(scan_id: str, round_number: int, asset: str) -> FileResponse:
    """A published preview round's point cloud or cameras. Round numbers are integers only."""
    session = _session(scan_id)
    media_types = {"map.ply": "application/octet-stream", "cameras.json": "application/json"}
    if asset not in media_types or round_number < 1:
        raise HTTPException(status_code=404, detail="Not found")
    path = session.preview_dir / "rounds" / str(round_number) / asset
    if not path.is_file():
        raise HTTPException(status_code=404, detail="That preview round is gone")
    # Published rounds never change, so the browser may keep them.
    return FileResponse(
        path, media_type=media_types[asset], headers={"Cache-Control": "max-age=600"}
    )


@app.get("/api/scans/{scan_id}/metrics")
async def scan_metrics(scan_id: str) -> dict[str, Any]:
    """Numbers for comparing scans side by side, including model statistics from points3D.bin."""
    session = _session(scan_id)
    result = session.result
    point_stats = await _cached_point_stats(session)

    # Scans captured before live tracking existed have no timeline: their counters mean
    # "not measured", not zero.
    tracked = bool(session.timeline)
    return {
        "id": session.id,
        "status": session.status.value,
        "preset": session.preset,
        "createdAt": session.created_at,
        "points": result.points if result else 0,
        "placed": result.registered_images if result else 0,
        "accepted": session.stats.accepted,
        "captured": session.stats.captured,
        "durationSeconds": result.duration_seconds if result else None,
        "weakLinks": session.stats.weak_links if tracked else None,
        "rotationFrames": session.stats.rotation_frames if tracked else None,
        "meanTrackLength": point_stats["mean_track_length"] if point_stats else None,
        "meanReprojectionError": point_stats["mean_reprojection_error"] if point_stats else None,
    }


@app.get("/api/scans/{scan_id}/result")
async def scan_result(scan_id: str) -> dict[str, Any]:
    session = _session(scan_id)
    if session.status != ScanStatus.COMPLETE:
        raise HTTPException(status_code=409, detail="Scan is not complete")
    payload = _status_payload(session)
    payload["plyUrl"] = f"/api/scans/{session.id}/map.ply"
    payload["camerasUrl"] = (
        f"/api/scans/{session.id}/cameras.json" if session.cameras_path.exists() else None
    )
    return payload


@app.get("/api/scans/{scan_id}/map.ply")
async def scan_ply(scan_id: str) -> FileResponse:
    session = _session(scan_id)
    if not session.ply_path.is_file():
        raise HTTPException(status_code=404, detail="No point cloud for this scan")
    return FileResponse(
        session.ply_path,
        media_type="application/octet-stream",
        filename=f"{session.id}.ply",
    )


@app.get("/api/scans/{scan_id}/cameras.json")
async def scan_cameras(scan_id: str) -> FileResponse:
    session = _session(scan_id)
    if not session.cameras_path.is_file():
        raise HTTPException(status_code=404, detail="No camera poses for this scan")
    await asyncio.to_thread(_backfill_camera_up, session)
    return FileResponse(session.cameras_path, media_type="application/json")


def _backfill_camera_up(session: ScanSession) -> None:
    """Scans made before cameras carried an ``up`` vector get it re-read from the model."""
    try:
        cameras = json.loads(session.cameras_path.read_text(encoding="utf-8")).get("cameras", [])
    except (OSError, json.JSONDecodeError):
        return
    if not cameras or "up" in cameras[0]:
        return
    index = session.result.model_index if session.result else None
    if index is None:
        return
    fresh = read_camera_centers(session.sparse_dir / str(index))
    if fresh:
        session.cameras_path.write_text(json.dumps({"cameras": fresh}), encoding="utf-8")


@app.get("/api/scans/{scan_id}/thumbnail.jpg")
async def scan_thumbnail(scan_id: str) -> FileResponse:
    """A small preview for the scan library: a frame from the middle of the capture."""
    session = _session(scan_id)
    if session.status == ScanStatus.CAPTURING:
        # The middle frame is still moving; caching now would freeze the wrong one.
        raise HTTPException(status_code=409, detail="Scan is still capturing")
    target = session.result_dir / "thumbnail.jpg"
    if not target.is_file():
        frames = sorted(session.images_dir.glob("frame_*.jpg"))
        if not frames:
            raise HTTPException(status_code=404, detail="This scan has no frames")
        made = await asyncio.to_thread(
            make_thumbnail, frames[len(frames) // 2], target, config.LIBRARY_THUMBNAIL_WIDTH
        )
        if not made:
            raise HTTPException(status_code=404, detail="Could not build a thumbnail")
    return FileResponse(target, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.get("/api/scans/{scan_id}/log")
async def scan_log(scan_id: str) -> dict[str, Any]:
    session = _session(scan_id)
    return {"id": session.id, "log": session.read_log()}


@app.post("/api/scans/{scan_id}/reveal")
async def reveal_scan(scan_id: str) -> dict[str, Any]:
    """Open the scan folder in the OS file manager. Local convenience only.

    The path comes from the validated scan workspace, never from the request body.
    """
    session = _session(scan_id)
    target = session.root
    try:
        if sys.platform == "win32":
            subprocess.Popen(["explorer.exe", str(target)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not open folder: {exc}")
    return {"opened": str(target)}


@app.delete("/api/scans/{scan_id}")
async def delete_scan(scan_id: str) -> dict[str, Any]:
    # COLMAP must be gone before its files can be deleted on Windows.
    await _stop_preview(scan_id)
    task = _tasks.pop(scan_id, None)
    if task:
        task.cancel()
    try:
        scans.delete(scan_id)
    except InvalidScanIdError:
        raise HTTPException(status_code=400, detail="Invalid scan id")
    except ScanNotFoundError:
        raise HTTPException(status_code=404, detail="Scan not found")
    return {"deleted": scan_id}


@app.post("/api/scans/{scan_id}/cancel")
async def cancel_scan(scan_id: str) -> dict[str, Any]:
    session = _session(scan_id)
    await _stop_preview(scan_id, discard=True)
    task = _tasks.pop(scan_id, None)
    if task:
        task.cancel()
    if not session.status.is_terminal:
        session.set_status(ScanStatus.CANCELLED)
    return _status_payload(session)


# --------------------------------------------------------------------------------------
# Developer tools
# --------------------------------------------------------------------------------------


@app.post("/api/dev/import")
async def dev_import(payload: dict[str, Any]) -> dict[str, Any]:
    """Build a scan from a folder of existing images.

    Developer-only: it lets the reconstruction path be exercised without a webcam. The folder
    is read with Python and the images are *copied* into the scan workspace, so no
    browser-supplied path ever reaches a COLMAP command line.
    """
    folder = Path(str((payload or {}).get("folder", "")).strip().strip('"')).expanduser()
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a folder: {folder}")

    extensions = {".jpg", ".jpeg", ".png"}
    sources = sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in extensions
    )
    if not sources:
        raise HTTPException(status_code=400, detail="No .jpg/.jpeg/.png images in that folder")

    session = scans.create(
        preset=str((payload or {}).get("preset", config.DEFAULT_PRESET)),
        source="import",
        marker_size_mm=_marker_size((payload or {}).get("markerSizeMm")),
    )
    for index, source in enumerate(sources[: config.MAX_ACCEPTED_FRAMES], start=1):
        await asyncio.to_thread(
            shutil.copyfile, source, session.images_dir / f"frame_{index:06d}.jpg"
        )

    session.stats.captured = len(sources)
    session.stats.accepted = min(len(sources), config.MAX_ACCEPTED_FRAMES)
    session.set_status(ScanStatus.QUEUED)
    _start_reconstruction(session)
    return _status_payload(session)


# --------------------------------------------------------------------------------------
# Frontend
# --------------------------------------------------------------------------------------


@app.exception_handler(404)
async def not_found(request: Request, exc: Exception) -> JSONResponse | FileResponse:
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": getattr(exc, "detail", "Not found")}, status_code=404)
    return FileResponse(config.FRONTEND_DIR / "index.html")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(config.FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=config.FRONTEND_DIR), name="static")


def main() -> None:
    parser = argparse.ArgumentParser(description="Webcam 3D Mapper")
    parser.add_argument("--host", default=config.HOST, help="bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=config.PORT)
    parser.add_argument("--open", action="store_true", help="open the browser on startup")
    parser.add_argument("--reload", action="store_true", help="auto-reload (development)")
    args = parser.parse_args()

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning("Binding to %s exposes this app beyond this machine.", args.host)

    if args.open:
        url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}"
        # Delay slightly so the server is listening before the tab opens.
        import threading

        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    import uvicorn

    uvicorn.run(
        "app:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
