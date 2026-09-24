"""Webcam 3D Mapper - local FastAPI server.

Serves the single-page frontend and a small JSON API. Everything stays on this machine:
the server binds to loopback, frames are written to ``scans/``, and reconstruction shells out
to a locally installed COLMAP.

Run with ``python app.py`` or ``scripts/run.bat`` / ``scripts/run.sh``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import subprocess
import sys
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend import config
from backend.colmap_runner import ColmapNotFoundError
from backend.colmap_runner import runner as colmap
from backend.image_quality import decode_jpeg
from backend.models import STAGE_SEQUENCE, FrameDecision, ScanStatus
from backend.reconstruction import ReconstructionPipeline
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
        for task in list(_tasks.values()):
            task.cancel()
        if _tasks:
            await asyncio.gather(*_tasks.values(), return_exceptions=True)


app = FastAPI(title="Webcam 3D Mapper", version="1.0.0", lifespan=lifespan)


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
    return payload


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
async def list_scans() -> dict[str, Any]:
    return {"scans": scans.list_scans()}


@app.post("/api/scans")
async def create_scan(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    preset = (payload or {}).get("preset", config.DEFAULT_PRESET)
    session = scans.create(preset=str(preset))
    return _status_payload(session)


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

    decision = await asyncio.to_thread(_evaluate_and_store, session, data)
    session.record_decision(decision)

    return {
        **decision.to_dict(),
        "acceptedFrames": session.stats.accepted,
        "capturedFrames": session.stats.captured,
        "limitReached": session.stats.accepted >= config.MAX_ACCEPTED_FRAMES,
    }


def _evaluate_and_store(session: ScanSession, data: bytes) -> FrameDecision:
    """Runs on a worker thread: decode, score, and write accepted frames to disk."""
    image = decode_jpeg(data)
    if image is None:
        return FrameDecision(False, "error")

    decision = session.quality_filter.evaluate(image)
    if decision.accepted:
        path = session.next_frame_path()
        try:
            path.write_bytes(data)
        except OSError as exc:
            logger.warning("Could not write frame %s: %s", path, exc)
            return FrameDecision(False, "error")
    return decision


@app.post("/api/scans/{scan_id}/finish")
async def finish_scan(scan_id: str) -> dict[str, Any]:
    """Stop capturing and start reconstruction in the background."""
    session = _session(scan_id)
    if session.status != ScanStatus.CAPTURING:
        return _status_payload(session)

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
    return FileResponse(session.cameras_path, media_type="application/json")


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
        preset=str((payload or {}).get("preset", config.DEFAULT_PRESET)), source="import"
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
