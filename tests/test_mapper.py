"""Tests for the parts that do not need a webcam or a COLMAP install.

Run with:  .venv/Scripts/python -m pytest -q
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as application  # noqa: E402
from backend import config  # noqa: E402
from backend.colmap_runner import ColmapCommandError, ColmapInfo, ColmapRunner  # noqa: E402
from backend.image_quality import FrameQualityFilter, decode_jpeg  # noqa: E402
from backend.models import ScanStatus  # noqa: E402
from backend.reconstruction import (  # noqa: E402
    ReconstructionPipeline,
    model_stats,
    read_camera_centers,
    select_best_model,
)
from backend.scan_manager import InvalidScanIdError, ScanManager  # noqa: E402


# --------------------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------------------


def textured_frame(seed: int = 0, size: tuple[int, int] = (720, 1280)) -> np.ndarray:
    """A sharp, well-lit, structured image - the kind the filter should always keep.

    Pure random noise is a bad stand-in for a photo: downsampling averages it away, so every
    frame looks identical to the similarity check. This builds large-scale blocks (which
    survive the 64x36 thumbnail) plus fine grain (which gives the Laplacian something to find).
    """
    rng = np.random.default_rng(seed)
    blocks = rng.integers(30, 225, (12, 20, 3), dtype=np.uint8)
    image = cv2.resize(blocks, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    grain = rng.integers(-18, 18, (*size, 3), dtype=np.int16)
    return np.clip(image.astype(np.int16) + grain, 0, 255).astype(np.uint8)


def jpeg_bytes(image: np.ndarray, quality: int = 90) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok
    return buffer.tobytes()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A TestClient whose scan workspace is a temp directory."""
    manager = ScanManager(tmp_path / "scans")
    monkeypatch.setattr(application, "scans", manager)
    with TestClient(application.app) as test_client:
        test_client.manager = manager  # type: ignore[attr-defined]
        yield test_client


# --------------------------------------------------------------------------------------
# Image quality
# --------------------------------------------------------------------------------------


def test_sharp_frame_is_accepted():
    decision = FrameQualityFilter().evaluate(textured_frame(1))
    assert decision.accepted, decision


def test_blurry_frame_is_rejected():
    quality = FrameQualityFilter()
    quality.evaluate(textured_frame(2))  # establish a reference frame
    blurred = cv2.GaussianBlur(textured_frame(3), (61, 61), 0)
    decision = quality.evaluate(blurred)
    assert not decision.accepted and decision.reason == "blurry", decision


def test_dark_frame_is_rejected():
    dark = np.full((480, 640, 3), 6, dtype=np.uint8)
    decision = FrameQualityFilter().evaluate(dark)
    assert not decision.accepted and decision.reason == "dark", decision


def test_identical_frames_are_rejected_as_duplicates():
    quality = FrameQualityFilter()
    frame = textured_frame(4)
    assert quality.evaluate(frame).accepted
    decision = quality.evaluate(frame.copy())
    assert not decision.accepted and decision.reason == "duplicate", decision


def test_first_frame_is_never_a_duplicate():
    """Nothing to compare against yet, so the reference frame must always be kept."""
    flat = np.full((480, 640, 3), 128, dtype=np.uint8)
    flat[::4] = 255  # enough edges to clear the blur threshold
    assert FrameQualityFilter().evaluate(flat).accepted


def test_moving_scene_keeps_frames():
    """A slow pan must not be filtered out - overlap matters more than novelty."""
    quality = FrameQualityFilter()
    base = textured_frame(5)
    shifts = range(0, 600, 60)
    kept = sum(1 for shift in shifts if quality.evaluate(np.roll(base, shift, axis=1)).accepted)
    assert kept >= len(list(shifts)) - 1


def test_decode_rejects_garbage():
    assert decode_jpeg(b"not an image") is None
    assert decode_jpeg(b"") is None


# --------------------------------------------------------------------------------------
# Scan manager
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_id",
    ["../etc/passwd", "..", "foo", "2026-09-24_143001_ZZZZ", "2026-09-24_143001_ab12/../x", ""],
)
def test_invalid_scan_ids_are_refused(tmp_path, bad_id):
    manager = ScanManager(tmp_path / "scans")
    with pytest.raises(InvalidScanIdError):
        manager.path_for(bad_id)


def test_scan_workspace_layout(tmp_path):
    manager = ScanManager(tmp_path / "scans")
    session = manager.create()
    for directory in (session.images_dir, session.sparse_dir, session.result_dir, session.logs_dir):
        assert directory.is_dir()
    assert session.manifest_path.is_file()
    assert json.loads(session.manifest_path.read_text())["status"] == "capturing"


def test_scan_reloads_from_disk(tmp_path):
    manager = ScanManager(tmp_path / "scans")
    session = manager.create()
    session.stats.accepted = 7
    session.set_status(ScanStatus.COMPLETE)

    fresh = ScanManager(tmp_path / "scans").get(session.id)
    assert fresh.stats.accepted == 7
    assert fresh.status is ScanStatus.COMPLETE


def test_interrupted_scan_reloads_as_cancelled(tmp_path):
    """A scan still 'mapping' on disk means the server died mid-run; it cannot be resumed."""
    manager = ScanManager(tmp_path / "scans")
    session = manager.create()
    session.set_status(ScanStatus.MAPPING)

    fresh = ScanManager(tmp_path / "scans").get(session.id)
    assert fresh.status is ScanStatus.CANCELLED


# --------------------------------------------------------------------------------------
# COLMAP command construction
# --------------------------------------------------------------------------------------


#: Option names as COLMAP 3.x advertised them.
OPTIONS_V3 = {
    "database_path",
    "image_path",
    "output_path",
    "output_type",
    "input_path",
    "ImageReader.single_camera",
    "ImageReader.camera_model",
    "SiftExtraction.max_image_size",
    "SiftExtraction.max_num_features",
    "SiftExtraction.use_gpu",
    "SequentialMatching.overlap",
    "SiftMatching.use_gpu",
    "Mapper.ba_global_images_ratio",
}

#: Option names as COLMAP 4.2 advertises them (verified against the installed build).
OPTIONS_V4 = {
    "database_path",
    "image_path",
    "output_path",
    "output_type",
    "input_path",
    "ImageReader.single_camera",
    "ImageReader.camera_model",
    "FeatureExtraction.max_image_size",
    "FeatureExtraction.use_gpu",
    "SiftExtraction.max_num_features",
    "SequentialMatching.overlap",
    "FeatureMatching.use_gpu",
    "Mapper.ba_global_frames_ratio",
}


class FakeRunner(ColmapRunner):
    """A ColmapRunner that advertises a fixed option set and records what was run."""

    def __init__(
        self,
        options: set[str] | None = None,
        gpu: str = "unknown",
        gpu_build: bool = True,
    ) -> None:
        super().__init__()
        self._fake_options = OPTIONS_V4 if options is None else options
        self._info = ColmapInfo(
            available=True, path="colmap", version="4.2.0", gpu=gpu, gpu_build=gpu_build
        )
        self._executable = Path("colmap")
        self.calls: list[tuple[str, list[str]]] = []

    def supported_options(self, command: str) -> set[str]:
        return self._fake_options

    async def run(self, command, args, log_sink=None, line_callback=None, timeout=0):  # type: ignore[override]
        self.calls.append((command, list(args)))
        return ""


def _pipeline(tmp_path, runner: ColmapRunner) -> ReconstructionPipeline:
    manager = ScanManager(tmp_path / "scans")
    return ReconstructionPipeline(manager.create(preset="fast"), runner=runner)


def value_of(args: list[str], name: str) -> str:
    assert f"--{name}" in args, f"{name} missing from {args}"
    return args[args.index(f"--{name}") + 1]


def test_feature_extractor_shares_one_camera(tmp_path):
    args = _pipeline(tmp_path, FakeRunner()).feature_extractor_args()
    assert value_of(args, "ImageReader.single_camera") == "1"
    assert value_of(args, "ImageReader.camera_model") == "SIMPLE_RADIAL"


@pytest.mark.parametrize(
    ("options", "image_size_option", "ratio_option", "matcher_gpu_option"),
    [
        (OPTIONS_V3, "SiftExtraction.max_image_size", "Mapper.ba_global_images_ratio",
         "SiftMatching.use_gpu"),
        (OPTIONS_V4, "FeatureExtraction.max_image_size", "Mapper.ba_global_frames_ratio",
         "FeatureMatching.use_gpu"),
    ],
    ids=["colmap-3.x", "colmap-4.x"],
)
def test_renamed_options_resolve_per_version(
    tmp_path, options, image_size_option, ratio_option, matcher_gpu_option
):
    """COLMAP 4 renamed several options; the tuning must survive on both generations."""
    pipeline = _pipeline(tmp_path, FakeRunner(options=options))

    extractor = pipeline.feature_extractor_args()
    assert value_of(extractor, image_size_option) == "1024"  # fast preset
    assert value_of(extractor, "SiftExtraction.max_num_features") == "4096"

    matcher = pipeline.sequential_matcher_args()
    assert value_of(matcher, "SequentialMatching.overlap") == "5"
    assert value_of(matcher, matcher_gpu_option) == "0"

    assert value_of(pipeline.mapper_args(), ratio_option) == "1.4"


def test_only_one_spelling_of_an_option_is_passed(tmp_path):
    """Passing both the old and new name would make COLMAP reject the command line."""
    args = _pipeline(tmp_path, FakeRunner(options=OPTIONS_V3 | OPTIONS_V4)).feature_extractor_args()
    assert args.count("--FeatureExtraction.max_image_size") +         args.count("--SiftExtraction.max_image_size") == 1


def test_gpu_flags_follow_detection(tmp_path):
    cpu = _pipeline(tmp_path, FakeRunner(gpu="unknown")).feature_extractor_args()
    assert value_of(cpu, "FeatureExtraction.use_gpu") == "0"

    gpu = _pipeline(tmp_path, FakeRunner(gpu="NVIDIA RTX 4070")).feature_extractor_args()
    assert value_of(gpu, "FeatureExtraction.use_gpu") == "1"


def test_cpu_only_build_never_asks_for_the_gpu(tmp_path):
    """The official no-CUDA Windows build cannot do SIFT on the GPU even with an NVIDIA card."""
    runner = FakeRunner(gpu="NVIDIA RTX 4070", gpu_build=False)
    args = _pipeline(tmp_path, runner).feature_extractor_args()
    assert value_of(args, "FeatureExtraction.use_gpu") == "0"


def test_unsupported_options_are_dropped(tmp_path):
    """An older COLMAP that does not advertise an option must not receive it."""
    runner = FakeRunner(options={"database_path", "image_path"})
    args = _pipeline(tmp_path, runner).feature_extractor_args()
    assert "--SiftExtraction.max_num_features" not in args
    assert "--database_path" in args and "--image_path" in args


def test_options_are_kept_when_help_is_unparseable(tmp_path):
    """Empty introspection means 'unknown', not 'unsupported' - keep the documented flags."""
    args = _pipeline(tmp_path, FakeRunner(options=set())).feature_extractor_args()
    assert "--FeatureExtraction.max_image_size" in args


def test_model_converter_targets_the_result_ply(tmp_path):
    pipeline = _pipeline(tmp_path, FakeRunner())
    args = pipeline.model_converter_args(Path("sparse/0"))
    assert value_of(args, "output_type") == "PLY"
    assert value_of(args, "output_path").endswith("map.ply")


def test_command_uses_argument_array_not_a_shell_string(tmp_path):
    runner = FakeRunner()
    argv = runner.build_command("mapper", ["--database_path", r"C:\a b\database.db"])
    assert isinstance(argv, list) and argv[1] == "mapper"


# --------------------------------------------------------------------------------------
# COLMAP model readers
# --------------------------------------------------------------------------------------


def write_images_bin(path: Path, poses: list[tuple[tuple[float, ...], tuple[float, ...], str]]):
    """Write a minimal COLMAP images.bin containing the given (quat, translation, name)."""
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(poses)))
        for index, (quat, translation, name) in enumerate(poses, start=1):
            handle.write(struct.pack("<I7dI", index, *quat, *translation, 1))
            handle.write(name.encode() + b"\x00")
            handle.write(struct.pack("<Q", 0))


def test_model_stats_reads_binary_headers(tmp_path):
    model = tmp_path / "0"
    model.mkdir()
    write_images_bin(model / "images.bin", [((1, 0, 0, 0), (0, 0, 0), "a.jpg")] * 3)
    (model / "points3D.bin").write_bytes(struct.pack("<Q", 512) + b"\x00" * 16)
    assert model_stats(model) == (3, 512)


def test_best_model_is_the_largest_component(tmp_path):
    sparse = tmp_path / "sparse"
    for index, (images, points) in enumerate([(4, 100), (11, 900), (2, 30)]):
        model = sparse / str(index)
        model.mkdir(parents=True)
        write_images_bin(model / "images.bin", [((1, 0, 0, 0), (0, 0, 0), "a.jpg")] * images)
        (model / "points3D.bin").write_bytes(struct.pack("<Q", points))
    chosen, images, points = select_best_model(sparse)
    assert chosen is not None and chosen.name == "1" and images == 11 and points == 900


def test_camera_centre_inverts_the_world_to_camera_pose(tmp_path):
    """Identity rotation with t = (0,0,-5) means the camera sits at (0,0,5)."""
    model = tmp_path / "0"
    model.mkdir()
    write_images_bin(model / "images.bin", [((1, 0, 0, 0), (0.0, 0.0, -5.0), "frame_000001.jpg")])
    cameras = read_camera_centers(model)
    assert len(cameras) == 1
    assert cameras[0]["position"] == pytest.approx([0.0, 0.0, 5.0])
    assert cameras[0]["name"] == "frame_000001.jpg"


def test_missing_model_files_are_handled(tmp_path):
    assert select_best_model(tmp_path / "nope") == (None, 0, 0)
    assert read_camera_centers(tmp_path) == []


# --------------------------------------------------------------------------------------
# Error translation
# --------------------------------------------------------------------------------------


def test_mapper_failure_becomes_actionable_guidance():
    from backend.reconstruction import _translate_command_error

    error = _translate_command_error(ColmapCommandError("mapper", 1, "some output"))
    assert error.code == "mapping_failed"
    assert "exit code" not in error.message.lower()
    assert len(error.hints) >= 3


def test_a_crash_is_reported_as_a_crash():
    """COLMAP can hard-crash on textureless frames; blaming only technique would mislead."""
    from backend.reconstruction import _translate_command_error

    crash = _translate_command_error(
        ColmapCommandError("sequential_matcher", 0xC0000409, "Processing image [6/15]")
    )
    clean = _translate_command_error(ColmapCommandError("sequential_matcher", 1, "nope"))
    assert "stopped unexpectedly" in crash.hints[0]
    assert not any("stopped unexpectedly" in hint for hint in clean.hints)


def test_gpu_failure_is_recognised():
    from backend.reconstruction import _translate_command_error

    error = _translate_command_error(
        ColmapCommandError("feature_extractor", 1, "ERROR: SiftGPU not fully supported")
    )
    assert error.code == "gpu_error"


# --------------------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------------------


def test_system_endpoint_reports_readiness(client):
    payload = client.get("/api/system").json()
    assert "colmap" in payload and "available" in payload["colmap"]
    assert payload["capture"]["intervalMs"] == config.CAPTURE_INTERVAL_MS
    assert {preset["name"] for preset in payload["presets"]} == {"fast", "balanced"}


def test_frontend_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "3D Scanner" in response.text
    for asset in ("/static/app.js", "/static/viewer.js", "/static/styles.css",
                  "/static/vendor/three.module.min.js"):
        assert client.get(asset).status_code == 200, asset


def test_scan_lifecycle_and_frame_filtering(client):
    scan_id = client.post("/api/scans", json={"preset": "fast"}).json()["id"]

    accepted = 0
    for seed in range(6):
        response = client.post(
            f"/api/scans/{scan_id}/frames",
            files={"frame": ("f.jpg", jpeg_bytes(textured_frame(seed)), "image/jpeg")},
        )
        assert response.status_code == 200
        accepted = response.json()["acceptedFrames"]

    # A duplicate of the last frame must be rejected without changing the accepted count.
    duplicate = client.post(
        f"/api/scans/{scan_id}/frames",
        files={"frame": ("f.jpg", jpeg_bytes(textured_frame(5)), "image/jpeg")},
    ).json()
    assert duplicate["accepted"] is False and duplicate["acceptedFrames"] == accepted

    status = client.get(f"/api/scans/{scan_id}/status").json()
    assert status["status"] == "capturing"
    assert status["acceptedFrames"] == accepted == 6

    images = list((client.manager.get(scan_id).images_dir).glob("frame_*.jpg"))
    assert len(images) == accepted
    assert images[0].name == "frame_000001.jpg"


def test_frames_are_refused_after_finish(client):
    scan_id = client.post("/api/scans", json={}).json()["id"]
    client.post(
        f"/api/scans/{scan_id}/frames",
        files={"frame": ("f.jpg", jpeg_bytes(textured_frame(9)), "image/jpeg")},
    )
    client.post(f"/api/scans/{scan_id}/finish")
    response = client.post(
        f"/api/scans/{scan_id}/frames",
        files={"frame": ("f.jpg", jpeg_bytes(textured_frame(10)), "image/jpeg")},
    )
    assert response.status_code == 409


def test_too_few_frames_produces_guidance_not_a_crash(client):
    """With COLMAP absent this still has to fail cleanly, before any subprocess is attempted."""
    scan_id = client.post("/api/scans", json={}).json()["id"]
    client.post(
        f"/api/scans/{scan_id}/frames",
        files={"frame": ("f.jpg", jpeg_bytes(textured_frame(11)), "image/jpeg")},
    )
    client.post(f"/api/scans/{scan_id}/finish")

    for _ in range(80):
        status = client.get(f"/api/scans/{scan_id}/status").json()
        if status["status"] in ("failed", "complete"):
            break
    assert status["status"] == "failed"
    assert status["error"]["code"] in ("too_few_frames", "colmap_missing")
    assert status["error"]["message"]


def test_result_and_ply_endpoints_guard_incomplete_scans(client):
    scan_id = client.post("/api/scans", json={}).json()["id"]
    assert client.get(f"/api/scans/{scan_id}/result").status_code == 409
    assert client.get(f"/api/scans/{scan_id}/map.ply").status_code == 404


def test_ply_endpoint_serves_a_finished_scan(client):
    scan_id = client.post("/api/scans", json={}).json()["id"]
    session = client.manager.get(scan_id)
    session.ply_path.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
    session.set_status(ScanStatus.COMPLETE)

    response = client.get(f"/api/scans/{scan_id}/map.ply")
    assert response.status_code == 200
    assert response.content.startswith(b"ply")
    assert client.get(f"/api/scans/{scan_id}/result").json()["plyUrl"].endswith("map.ply")


def test_unknown_and_malformed_scan_ids(client):
    assert client.get("/api/scans/2026-01-01_000000_abcd/status").status_code == 404
    assert client.get("/api/scans/not-a-scan/status").status_code == 400


def test_delete_removes_the_workspace(client):
    scan_id = client.post("/api/scans", json={}).json()["id"]
    root = client.manager.get(scan_id).root
    assert root.is_dir()
    assert client.request("DELETE", f"/api/scans/{scan_id}").status_code == 200
    assert not root.exists()


def test_dev_import_rejects_a_bad_folder(client):
    response = client.post("/api/dev/import", json={"folder": "C:/definitely/not/here"})
    assert response.status_code == 400


# --------------------------------------------------------------------------------------
# Frontend asset integrity
# --------------------------------------------------------------------------------------


def test_every_module_import_resolves(client):
    """Catches a half-vendored library.

    `three.module.min.js` imports a sibling `three.core.min.js`; forgetting it breaks the whole
    viewer with nothing but a console error. Every relative import in every shipped script must
    resolve to a file the server actually serves.
    """
    import re

    frontend = Path(__file__).resolve().parent.parent / "frontend"
    checked = 0

    for script in sorted(frontend.rglob("*.js")):
        source = script.read_text(encoding="utf-8", errors="replace")
        for specifier in re.findall(r"""(?:from|import)\s*['"](\.[^'"]+)['"]""", source):
            target = (script.parent / specifier).resolve()
            assert target.is_file(), f"{script.name} imports missing {specifier}"
            url = "/static/" + target.relative_to(frontend).as_posix()
            assert client.get(url).status_code == 200, url
            checked += 1

    assert checked, "no relative imports found - the scan regex is probably wrong"


def test_importmap_targets_are_served(client):
    """The bare specifiers `index.html` maps must point at files that exist."""
    import json
    import re

    html = (Path(__file__).resolve().parent.parent / "frontend" / "index.html").read_text(
        encoding="utf-8"
    )
    importmap = json.loads(
        re.search(r'<script type="importmap">(.*?)</script>', html, re.S).group(1)
    )
    for specifier, url in importmap["imports"].items():
        if url.endswith("/"):
            continue  # a prefix mapping; the file check above covers its targets
        assert client.get(url).status_code == 200, f"{specifier} -> {url}"


def test_html_references_existing_assets(client):
    import re

    html = (Path(__file__).resolve().parent.parent / "frontend" / "index.html").read_text(
        encoding="utf-8"
    )
    for url in re.findall(r'(?:src|href)="(/static/[^"]+)"', html):
        assert client.get(url).status_code == 200, url
