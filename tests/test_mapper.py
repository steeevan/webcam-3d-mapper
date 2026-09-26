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
from backend.tracking import TrackingEstimator  # noqa: E402
from backend.models import ScanStatus  # noqa: E402
from backend.reconstruction import (  # noqa: E402
    ReconstructionPipeline,
    model_stats,
    read_camera_centers,
    select_best_model,
)
from backend.scan_manager import InvalidScanIdError, ScanManager, ScanSession  # noqa: E402


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
    # Unit tests never launch a real COLMAP preview; preview tests inject their own.
    monkeypatch.setattr(application, "_start_preview", lambda session: None)
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


def test_overexposed_frame_is_reported_as_bright():
    decision = FrameQualityFilter().evaluate(np.full((480, 640, 3), 254, dtype=np.uint8))
    assert not decision.accepted and decision.reason == "bright", decision


def test_decode_rejects_garbage():
    assert decode_jpeg(b"not an image") is None
    assert decode_jpeg(b"") is None


# --------------------------------------------------------------------------------------
# Live tracking
# --------------------------------------------------------------------------------------


def gray_480(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (854, 480), interpolation=cv2.INTER_AREA)


def two_layer_scene(step: int) -> np.ndarray:
    """A background and a nearer foreground panel that slide at different speeds.

    Different image motion at different depths is parallax - exactly what a single homography
    cannot explain and what a real sideways camera move produces.
    """
    background = np.roll(textured_frame(20), -8 * step, axis=1)
    foreground = np.roll(textured_frame(21, size=(420, 1280)), -26 * step, axis=1)
    frame = background.copy()
    frame[150:570, 380:900] = foreground[:, 380:900]
    return frame


def run_tracker(frames: list[np.ndarray]) -> list:
    tracker = TrackingEstimator()
    return [tracker.observe(gray_480(frame), accepted=True) for frame in frames]


def test_featureless_view_is_flagged_as_low_texture():
    sample = TrackingEstimator().observe(np.full((480, 854), 128, dtype=np.uint8), accepted=True)
    assert sample.state == "low_texture"
    assert sample.features < 50


def test_turning_in_place_is_flagged_as_no_depth():
    """A pure image shift is fully explained by a homography: no parallax, no 3D."""
    base = textured_frame(22)
    samples = run_tracker([np.roll(base, -10 * i, axis=1) for i in range(14)])
    assert samples[-1].homography_ratio is not None and samples[-1].homography_ratio > 0.96
    assert samples[-1].state == "rotating", [s.state for s in samples]


def test_parallax_from_a_sideways_move_is_good_tracking():
    samples = run_tracker([two_layer_scene(i) for i in range(14)])
    assert samples[-1].homography_ratio is not None and samples[-1].homography_ratio < 0.9
    assert samples[-1].state == "good", [(s.state, s.homography_ratio) for s in samples]
    assert samples[-1].score > 60


def scattered_discs(seed: int) -> np.ndarray:
    """A scene with no structure in common with `textured_frame`.

    Two `textured_frame`s share the same block grid, so their corners genuinely match; a
    "different view" test needs a different scene, not just different colours.
    """
    rng = np.random.default_rng(seed)
    image = np.full((720, 1280, 3), 90, np.uint8)
    for _ in range(260):
        colour = tuple(int(v) for v in rng.integers(0, 255, 3))
        centre = tuple(int(v) for v in rng.integers(0, [1280, 720]))
        cv2.circle(image, centre, int(rng.integers(6, 40)), colour, -1)
    return image


def test_a_jump_to_an_unrelated_view_is_flagged_as_lost():
    samples = run_tracker([textured_frame(23), textured_frame(23), scattered_discs(24)])
    assert samples[-1].state == "lost"
    assert samples[-1].inliers < 12


def test_rejected_frames_do_not_become_the_reference():
    """Only accepted frames may anchor continuity; a blurry frame must not reset it."""
    tracker = TrackingEstimator()
    tracker.observe(gray_480(textured_frame(25)), accepted=True)
    tracker.observe(gray_480(textured_frame(26)), accepted=False)
    again = tracker.observe(gray_480(textured_frame(25)), accepted=True)
    assert again.inliers > 100


def test_overlay_keypoints_are_normalised_and_capped():
    sample = TrackingEstimator().observe(gray_480(textured_frame(27)), accepted=True)
    assert 0 < len(sample.keypoints) <= config.TRACKING_OVERLAY_POINTS
    assert all(0 <= x <= 1 and 0 <= y <= 1 for x, y in sample.keypoints)


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


def test_frames_are_published_whole(tmp_path):
    """COLMAP may be reading the folder while capture writes to it; no partial frame may show."""
    session = ScanManager(tmp_path / "scans").create()
    path = session.write_frame(b"jpeg bytes")
    assert path.name == "frame_000001.jpg" and path.read_bytes() == b"jpeg bytes"
    assert session.frame_names == ["frame_000001.jpg"]
    assert not list(session.images_dir.glob("*.tmp"))


def test_a_failed_frame_write_leaves_nothing_behind(tmp_path, monkeypatch):
    import backend.scan_manager as scan_manager

    session = ScanManager(tmp_path / "scans").create()

    def fail(*_):
        raise OSError("disk full")

    monkeypatch.setattr(scan_manager.os, "replace", fail)
    with pytest.raises(OSError):
        session.write_frame(b"jpeg bytes")
    assert session.frame_names == []
    assert list(session.images_dir.iterdir()) == []


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


def test_gpu_failure_reruns_the_stage_on_the_cpu(tmp_path):
    """A CUDA build that fails on this machine must not cost the scan."""

    class GpuFailsRunner(FakeRunner):
        async def run(self, command, args, log_sink=None, line_callback=None, timeout=0):  # type: ignore[override]
            await super().run(command, args)
            if value_of(list(args), "FeatureExtraction.use_gpu") == "1":
                raise ColmapCommandError(command, 1, "CUDA error: no kernel image is available")
            return ""

    runner = GpuFailsRunner(gpu="NVIDIA RTX 4070")
    pipeline = _pipeline(tmp_path, runner)
    pipeline.session.database_path.write_bytes(b"half-written")

    run_async(pipeline._run_sift_stage("feature_extractor", pipeline.feature_extractor_args, print))

    flags = [value_of(args, "FeatureExtraction.use_gpu") for _, args in runner.calls]
    assert flags == ["1", "0"]
    assert runner.gpu_failed and pipeline.use_gpu == "0"
    assert not pipeline.session.database_path.exists()
    assert "running it again on the CPU" in pipeline.session.read_log()


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


def test_detailed_preset_adds_cpu_refinements(tmp_path):
    manager = ScanManager(tmp_path / "scans")
    options = OPTIONS_V4 | {
        "SiftExtraction.estimate_affine_shape",
        "SiftExtraction.domain_size_pooling",
        "FeatureMatching.guided_matching",
    }
    detailed = ReconstructionPipeline(manager.create(preset="detailed"), runner=FakeRunner(options))
    assert value_of(detailed.feature_extractor_args(), "SiftExtraction.domain_size_pooling") == "1"
    assert value_of(detailed.feature_extractor_args(), "SiftExtraction.estimate_affine_shape") == "1"
    assert value_of(detailed.sequential_matcher_args(), "FeatureMatching.guided_matching") == "1"

    fast = ReconstructionPipeline(manager.create(preset="fast"), runner=FakeRunner(options))
    assert "--SiftExtraction.domain_size_pooling" not in fast.feature_extractor_args()
    assert "--FeatureMatching.guided_matching" not in fast.sequential_matcher_args()


def test_model_converter_targets_the_result_ply(tmp_path):
    pipeline = _pipeline(tmp_path, FakeRunner())
    args = pipeline.model_converter_args(Path("sparse/0"))
    assert value_of(args, "output_type") == "PLY"
    assert value_of(args, "output_path").endswith("map.ply")


def test_command_uses_argument_array_not_a_shell_string(tmp_path):
    runner = FakeRunner()
    argv = runner.build_command("mapper", ["--database_path", r"C:\a b\database.db"])
    assert isinstance(argv, list) and argv[1] == "mapper"


def _process_alive(pid: int) -> bool:
    import subprocess

    listing = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True
    ).stdout
    return str(pid) in listing


@pytest.mark.skipif(sys.platform != "win32", reason="COLMAP.bat wrapper is Windows-only")
def test_cancelling_a_run_kills_the_child_behind_the_batch_file(tmp_path):
    """COLMAP.bat starts colmap.exe as a child; killing only cmd.exe left it running ~20 s."""
    import asyncio

    marker = tmp_path / "child.pid"
    child = tmp_path / "child.py"
    child.write_text(
        "import os, pathlib, sys, time\n"
        f"pathlib.Path(r'{marker}').write_text(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    script = tmp_path / "wrapper.bat"
    script.write_text(f'@echo off\r\n"{sys.executable}" "{child}"\r\n')

    class BatchRunner(FakeRunner):
        def build_command(self, command, args):
            return ["cmd.exe", "/c", str(script)]

    async def scenario():
        task = asyncio.create_task(ColmapRunner.run(BatchRunner(), "mapper", []))
        for _ in range(100):
            await asyncio.sleep(0.1)
            if marker.exists() and marker.read_text():
                break
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=15)
        # Checked right away: an orphan would still be sleeping here.
        pid = int(marker.read_text())
        alive = _process_alive(pid)
        if alive:  # don't leave it holding the pipe for the rest of the run
            import subprocess

            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
        return alive

    assert asyncio.run(scenario()) is False


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
    # COLMAP's image y axis points down, so an upright identity camera has world up = -Y.
    assert cameras[0]["up"] == pytest.approx([0.0, -1.0, 0.0])


def test_missing_model_files_are_handled(tmp_path):
    assert select_best_model(tmp_path / "nope") == (None, 0, 0)
    assert read_camera_centers(tmp_path) == []


def write_points3d_bin(path: Path, points: list[tuple[float, int]]) -> None:
    """A COLMAP points3D.bin with the given (reprojection error, track length) per point."""
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(points)))
        for index, (error, track_length) in enumerate(points, start=1):
            handle.write(struct.pack("<Q3d3BdQ", index, 0.1 * index, 0.0, 1.0, 200, 100, 50,
                                     error, track_length))
            for observation in range(track_length):
                handle.write(struct.pack("<II", observation + 1, index * 10 + observation))


def test_point_stats_average_track_length_and_error(tmp_path):
    from backend.reconstruction import read_point_stats

    write_points3d_bin(tmp_path / "points3D.bin", [(0.5, 2), (1.0, 3), (1.5, 7)])
    stats = read_point_stats(tmp_path)
    assert stats["points"] == 3
    assert stats["mean_track_length"] == pytest.approx(4.0)
    assert stats["mean_reprojection_error"] == pytest.approx(1.0)


def test_point_stats_survive_a_truncated_file(tmp_path):
    """A model still being written must not crash the comparison view."""
    from backend.reconstruction import read_point_stats

    write_points3d_bin(tmp_path / "points3D.bin", [(0.5, 2), (1.0, 3), (1.5, 7)])
    data = (tmp_path / "points3D.bin").read_bytes()
    (tmp_path / "points3D.bin").write_bytes(data[:-20])  # cut into the last point's track
    stats = read_point_stats(tmp_path)
    assert stats["points"] == 2 and stats["mean_track_length"] == pytest.approx(2.5)
    assert read_point_stats(tmp_path / "missing") is None


def test_point_stats_of_an_empty_model(tmp_path):
    from backend.reconstruction import read_point_stats

    write_points3d_bin(tmp_path / "points3D.bin", [])
    assert read_point_stats(tmp_path) == {
        "points": 0, "mean_track_length": 0.0, "mean_reprojection_error": 0.0
    }


def test_read_model_accepts_a_continued_model_in_place(tmp_path):
    """`mapper --input_path` writes the model straight into output_path, not output_path/0."""
    from backend.reconstruction import read_model

    flat = tmp_path / "flat"
    flat.mkdir()
    write_images_bin(flat / "images.bin", [((1, 0, 0, 0), (0, 0, 0), "a.jpg")] * 5)
    (flat / "points3D.bin").write_bytes(struct.pack("<Q", 40))
    assert read_model(flat) == (flat, 5, 40)

    numbered = tmp_path / "numbered"
    (numbered / "0").mkdir(parents=True)
    write_images_bin(numbered / "0" / "images.bin", [((1, 0, 0, 0), (0, 0, 0), "a.jpg")] * 3)
    (numbered / "0" / "points3D.bin").write_bytes(struct.pack("<Q", 12))
    assert read_model(numbered) == (numbered / "0", 3, 12)
    assert read_model(tmp_path / "empty") == (None, 0, 0)


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
    assert {preset["name"] for preset in payload["presets"]} == {"fast", "balanced", "detailed"}


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

    assert duplicate["tracking"]["features"] > 0
    assert duplicate["tracking"]["state"] in ("starting", "good", "rotating", "lost", "low_texture")

    status = client.get(f"/api/scans/{scan_id}/status").json()
    assert status["status"] == "capturing"
    assert status["acceptedFrames"] == accepted == 6
    assert len(status["timeline"]) == accepted

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


def test_thumbnail_waits_for_capture_to_end(client):
    scan_id = client.post("/api/scans", json={}).json()["id"]
    for seed in range(3):
        client.post(
            f"/api/scans/{scan_id}/frames",
            files={"frame": ("f.jpg", jpeg_bytes(textured_frame(seed)), "image/jpeg")},
        )
    assert client.get(f"/api/scans/{scan_id}/thumbnail.jpg").status_code == 409

    client.manager.get(scan_id).set_status(ScanStatus.CANCELLED)
    response = client.get(f"/api/scans/{scan_id}/thumbnail.jpg")
    assert response.status_code == 200
    thumb = decode_jpeg(response.content)
    assert thumb is not None and thumb.shape[1] == config.LIBRARY_THUMBNAIL_WIDTH


def test_old_camera_files_gain_an_up_vector(client):
    """Scans made before orientation was exported still load upright."""
    from backend.models import ReconstructionResult

    scan_id = client.post("/api/scans", json={}).json()["id"]
    session = client.manager.get(scan_id)
    model = session.sparse_dir / "0"
    model.mkdir(parents=True)
    write_images_bin(model / "images.bin", [((1, 0, 0, 0), (0.0, 0.0, -5.0), "frame_000001.jpg")])
    session.cameras_path.write_text(
        json.dumps({"cameras": [{"name": "frame_000001.jpg", "position": [0, 0, 5], "forward": [0, 0, 1]}]})
    )
    session.result = ReconstructionResult(model_index=0)
    session.set_status(ScanStatus.COMPLETE)

    cameras = client.get(f"/api/scans/{scan_id}/cameras.json").json()["cameras"]
    assert cameras[0]["up"] == pytest.approx([0.0, -1.0, 0.0])


def test_dev_import_rejects_a_bad_folder(client):
    response = client.post("/api/dev/import", json={"folder": "C:/definitely/not/here"})
    assert response.status_code == 400


# --------------------------------------------------------------------------------------
# Live preview
# --------------------------------------------------------------------------------------


class FakePreviewColmap(FakeRunner):
    """Stands in for COLMAP in preview rounds by writing the files each stage would produce.

    ``placed(frames)`` decides how many frames the mapper registers; ``fail`` names a stage that
    raises; ``gate`` holds every stage until it is set.
    """

    def __init__(self, placed=lambda frames: frames, fail: str | None = None, gate=None):
        super().__init__(options=set())  # empty introspection keeps every first spelling
        self.placed = placed
        self.fail = fail
        self.gate = gate
        self.cancelled = False
        self.frames = 0

    async def run(self, command, args, log_sink=None, line_callback=None, timeout=0):  # type: ignore[override]
        import asyncio
        import sqlite3

        self.calls.append((command, list(args)))
        try:
            if self.gate is not None:
                await self.gate.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if command == self.fail:
            raise ColmapCommandError(command, 1, "No good initial image pair found")

        if command == "feature_extractor":
            self.frames = len(Path(value_of(args, "image_list_path")).read_text().split())
            connection = sqlite3.connect(value_of(args, "database_path"))
            connection.execute("CREATE TABLE IF NOT EXISTS cameras (camera_id INTEGER)")
            if not connection.execute("SELECT COUNT(*) FROM cameras").fetchone()[0]:
                connection.execute("INSERT INTO cameras VALUES (1)")
            connection.commit()
            connection.close()
        elif command == "mapper":
            output = Path(value_of(args, "output_path"))
            model = output if "--input_path" in args else output / "0"
            model.mkdir(parents=True, exist_ok=True)
            write_ring_model(model, self.placed(self.frames))
        elif command == "model_converter":
            Path(value_of(args, "output_path")).write_bytes(b"ply\nformat ascii 1.0\nend_header\n")
        return ""


def quaternion(r: np.ndarray) -> tuple[float, float, float, float]:
    """(w, x, y, z) of a rotation matrix, branching on the largest diagonal term so that
    half-turn rotations (trace -1) stay well conditioned."""
    trace = np.trace(r)
    if trace > 0:
        s = 2 * np.sqrt(1 + trace)
        return (s / 4, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
    i = int(np.argmax(np.diag(r)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2 * np.sqrt(1 + r[i, i] - r[j, j] - r[k, k])
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (r[k, j] - r[j, k]) / s
    q[1 + i] = s / 4
    q[1 + j] = (r[j, i] + r[i, j]) / s
    q[1 + k] = (r[k, i] + r[i, k]) / s
    return tuple(q)


def write_ring_model(model: Path, count: int) -> None:
    """images.bin + points3D.bin for ``count`` cameras on a half circle looking at the origin."""
    poses = []
    for index in range(count):
        angle = np.radians(180.0 * index / max(count - 1, 1))
        centre = np.array([5 * np.cos(angle), 0.0, 5 * np.sin(angle)])
        forward = -centre / np.linalg.norm(centre)
        down = np.array([0.0, 1.0, 0.0])  # COLMAP image y points down
        right = np.cross(down, forward)
        rotation = np.stack([right, down, forward])  # rows: camera axes in world coordinates
        translation = -rotation @ centre
        poses.append((quaternion(rotation), tuple(translation), f"frame_{index + 1:06d}.jpg"))
    write_images_bin(model / "images.bin", poses)
    write_points3d_bin(model / "points3D.bin", [(0.8, 3)] * (count * 10))


def capturing_session(tmp_path, frames: int = 12):
    session = ScanManager(tmp_path / "scans").create()
    for _ in range(frames):
        session.write_frame(b"jpeg")
    return session


def run_async(coroutine):
    import asyncio

    return asyncio.run(coroutine)


def test_preview_round_publishes_cloud_counts_and_coverage(tmp_path):
    from backend.preview import LivePreview

    session = capturing_session(tmp_path)
    preview = LivePreview(session, runner=FakePreviewColmap())
    run_async(preview.run_round(list(session.frame_names)))

    payload = preview.to_dict()
    assert payload["state"] == "ready" and not payload["lastRoundFailed"]
    assert (payload["placed"], payload["total"], payload["round"]) == (12, 12, 1)
    assert payload["plyUrl"] == f"/api/scans/{session.id}/preview/1/map.ply"
    assert (preview.round_dir(1) / "map.ply").is_file()
    assert json.loads((preview.round_dir(1) / "cameras.json").read_text())["cameras"]
    assert payload["coverage"]["covered"] == 7  # the fake model is a half circle
    assert payload["history"][0][4] == "rebuilt"


def test_preview_stays_in_its_own_workspace_and_lists_only_finished_frames(tmp_path):
    from backend.preview import LivePreview

    session = capturing_session(tmp_path)
    (session.images_dir / "frame_000013.jpg.tmp").write_bytes(b"half written")
    runner = FakePreviewColmap()
    run_async(LivePreview(session, runner=runner).run_round(list(session.frame_names)))

    calls = dict(runner.calls)
    extractor = calls["feature_extractor"]
    listed = Path(value_of(extractor, "image_list_path")).read_text().split()
    assert listed == session.frame_names and "frame_000013.jpg.tmp" not in listed
    assert value_of(extractor, "FeatureExtraction.max_image_size") == str(config.PREVIEW_MAX_IMAGE_SIZE)
    assert value_of(extractor, "SiftExtraction.max_num_features") == str(config.PREVIEW_MAX_FEATURES)
    assert value_of(extractor, "FeatureExtraction.num_threads") == str(config.PREVIEW_NUM_THREADS)
    assert value_of(calls["sequential_matcher"], "FeatureMatching.num_threads") == str(
        config.PREVIEW_NUM_THREADS
    )
    assert value_of(calls["mapper"], "Mapper.num_threads") == str(config.PREVIEW_NUM_THREADS)

    preview_root = session.preview_dir.resolve()
    for command, args in runner.calls:
        for option in ("database_path", "output_path"):
            if f"--{option}" in args:
                target = Path(value_of(args, option)).resolve()
                assert preview_root in target.parents, (command, option, target)
    assert not session.database_path.exists()
    assert list(session.sparse_dir.iterdir()) == []


def test_next_round_reuses_the_camera_and_continues_the_model(tmp_path):
    """Without existing_camera_id every round added a camera and the model fell apart."""
    from backend.preview import LivePreview

    session = capturing_session(tmp_path)
    runner = FakePreviewColmap()
    preview = LivePreview(session, runner=runner)
    run_async(preview.run_round(list(session.frame_names)))
    first_extract = runner.calls[0][1]
    assert "--ImageReader.existing_camera_id" not in first_extract

    for _ in range(10):
        session.write_frame(b"jpeg")
    runner.calls.clear()
    run_async(preview.run_round(list(session.frame_names)))
    calls = dict(runner.calls)
    assert value_of(calls["feature_extractor"], "ImageReader.existing_camera_id") == "1"
    assert Path(value_of(calls["mapper"], "input_path")) == session.preview_dir / "sparse" / "1" / "0"
    assert preview.result.placed == 22 and preview.result.continued


def test_a_poor_model_is_rebuilt_rather_than_continued(tmp_path):
    from backend.preview import LivePreview

    session = capturing_session(tmp_path)
    runner = FakePreviewColmap(placed=lambda frames: 3)
    preview = LivePreview(session, runner=runner)
    run_async(preview.run_round(list(session.frame_names)))
    runner.calls.clear()
    session.write_frame(b"jpeg")
    run_async(preview.run_round(list(session.frame_names)))
    assert "--input_path" not in dict(runner.calls)["mapper"]
    assert preview.history[-1][4] == "rebuilt"


def test_a_failed_round_means_waiting_not_an_error(tmp_path):
    from backend.preview import LivePreview

    session = capturing_session(tmp_path)
    preview = LivePreview(session, runner=FakePreviewColmap(fail="mapper"))
    run_async(preview.run_round(list(session.frame_names)))
    payload = preview.to_dict()
    assert payload["state"] == "waiting" and payload["lastRoundFailed"]
    assert "plyUrl" not in payload


def test_a_failed_round_keeps_showing_the_last_good_one(tmp_path):
    from backend.preview import LivePreview

    session = capturing_session(tmp_path)
    runner = FakePreviewColmap()
    preview = LivePreview(session, runner=runner)
    run_async(preview.run_round(list(session.frame_names)))
    runner.fail = "sequential_matcher"
    session.write_frame(b"jpeg")
    run_async(preview.run_round(list(session.frame_names)))
    payload = preview.to_dict()
    assert payload["state"] == "ready" and payload["round"] == 1 and payload["lastRoundFailed"]


def test_preview_waits_for_enough_frames(tmp_path):
    from backend.preview import LivePreview

    session = capturing_session(tmp_path, frames=config.PREVIEW_MIN_FRAMES - 1)
    preview = LivePreview(session, runner=FakePreviewColmap())

    async def scenario():
        return preview.tick()

    assert run_async(scenario()) is False
    assert preview.to_dict()["state"] == "collecting" and preview.rounds == 0


def test_a_busy_round_skips_the_tick_instead_of_queueing(tmp_path):
    import asyncio

    from backend.preview import LivePreview

    session = capturing_session(tmp_path)

    async def scenario():
        gate = asyncio.Event()
        preview = LivePreview(session, runner=FakePreviewColmap(gate=gate))
        assert preview.tick() is True
        await asyncio.sleep(0)
        session.write_frame(b"jpeg")
        assert preview.tick() is False and preview.tick() is False
        assert preview.skipped == 2 and preview.rounds == 1 and preview.running
        gate.set()
        await preview._round_task
        # Nothing new since the last round: no pointless re-run either.
        assert preview.tick() is True  # the frame added mid-round is new
        await preview._round_task
        assert preview.tick() is False
        return preview

    preview = run_async(scenario())
    assert preview.rounds == 2 and preview.result.total == 13


def test_preview_ticks_faster_until_a_round_is_trustworthy(tmp_path):
    from backend.preview import LivePreview

    session = capturing_session(tmp_path)
    preview = LivePreview(session, runner=FakePreviewColmap())
    assert preview.next_delay() == config.PREVIEW_EARLY_INTERVAL_S < config.PREVIEW_INTERVAL_S
    run_async(preview.run_round(list(session.frame_names)))  # 12 frames: still early
    assert preview.next_delay() == config.PREVIEW_EARLY_INTERVAL_S
    for _ in range(config.PREVIEW_TRUSTED_FRAMES):
        session.write_frame(b"jpeg")
    run_async(preview.run_round(list(session.frame_names)))
    assert preview.next_delay() == config.PREVIEW_INTERVAL_S


def test_stopping_the_preview_cancels_the_running_round(tmp_path):
    import asyncio

    from backend.preview import LivePreview

    session = capturing_session(tmp_path)
    runner = FakePreviewColmap(gate=asyncio.Event())  # never released

    async def scenario():
        preview = LivePreview(session, runner=runner, interval=60)
        preview.start()
        for _ in range(50):
            await asyncio.sleep(0.02)
            if preview.running:
                break
        assert preview.running
        await preview.stop()
        return preview

    preview = run_async(scenario())
    assert runner.cancelled and not preview.running and preview.result is None


def test_old_preview_rounds_are_pruned(tmp_path):
    from backend.preview import LivePreview

    session = capturing_session(tmp_path)
    preview = LivePreview(session, runner=FakePreviewColmap())
    for _ in range(4):
        session.write_frame(b"jpeg")
        run_async(preview.run_round(list(session.frame_names)))
    assert sorted(p.name for p in (session.preview_dir / "rounds").iterdir()) == ["3", "4"]
    assert [p.name for p in (session.preview_dir / "sparse").iterdir()] == ["4"]

    preview.discard_workspace()
    assert not (session.preview_dir / "sparse").exists()
    assert (preview.round_dir(4) / "map.ply").is_file()


def test_preview_endpoint_serves_the_latest_round(client):
    from backend.preview import LivePreview

    scan_id = client.post("/api/scans", json={}).json()["id"]
    assert client.get(f"/api/scans/{scan_id}/preview").json() == {"state": "off"}

    session = client.manager.get(scan_id)
    for _ in range(12):
        session.write_frame(b"jpeg")
    preview = LivePreview(session, runner=FakePreviewColmap())
    run_async(preview.run_round(list(session.frame_names)))
    application._previews[scan_id] = preview
    try:
        payload = client.get(f"/api/scans/{scan_id}/preview").json()
        assert payload["state"] == "ready" and payload["placed"] == 12
        assert client.get(payload["plyUrl"]).content.startswith(b"ply")
        assert client.get(payload["camerasUrl"]).json()["cameras"]
        assert client.get(f"/api/scans/{scan_id}/preview/1/database.db").status_code == 404
        assert client.get(f"/api/scans/{scan_id}/preview/7/map.ply").status_code == 404
        assert client.get(f"/api/scans/{scan_id}/preview/..%2F..%2Fscan.json/map.ply").status_code in (404, 422)
    finally:
        application._previews.pop(scan_id, None)


def test_finish_cancels_the_preview_before_reconstructing(client):
    scan_id = client.post("/api/scans", json={}).json()["id"]
    events = []

    class Recorder:
        async def stop(self):
            events.append("stopped")

        def discard_workspace(self):
            events.append("discarded")

    application._previews[scan_id] = Recorder()
    client.post(f"/api/scans/{scan_id}/finish")
    assert events == ["stopped", "discarded"]
    assert scan_id not in application._previews


def test_deleting_a_capturing_scan_stops_its_preview(client):
    scan_id = client.post("/api/scans", json={}).json()["id"]
    events = []

    class Recorder:
        async def stop(self):
            events.append("stopped")

    application._previews[scan_id] = Recorder()
    assert client.request("DELETE", f"/api/scans/{scan_id}").status_code == 200
    assert events == ["stopped"] and scan_id not in application._previews


def test_scans_start_a_preview_only_when_colmap_is_there(tmp_path, monkeypatch):
    manager = ScanManager(tmp_path / "scans")
    started = []

    class FakePreview:
        def __init__(self, session):
            self.session = session

        def start(self):
            started.append(self.session.id)

    monkeypatch.setattr(application, "LivePreview", FakePreview)
    session = manager.create()
    monkeypatch.setattr(application.colmap, "detect", lambda force=False: ColmapInfo(available=False))
    application._start_preview(session)
    assert started == []
    monkeypatch.setattr(application.colmap, "detect", lambda force=False: ColmapInfo(available=True))
    application._start_preview(session)
    assert started == [session.id]
    application._previews.pop(session.id, None)


# --------------------------------------------------------------------------------------
# Scan comparison
# --------------------------------------------------------------------------------------


def test_metrics_include_model_quality_from_points3d(client):
    from backend.models import ReconstructionResult

    scan_id = client.post("/api/scans", json={}).json()["id"]
    session = client.manager.get(scan_id)
    model = session.sparse_dir / "0"
    model.mkdir(parents=True)
    write_points3d_bin(model / "points3D.bin", [(0.4, 4), (0.6, 6)])
    session.result = ReconstructionResult(points=2, registered_images=9, model_index=0,
                                          duration_seconds=12.5)
    session.stats.accepted, session.stats.captured = 10, 12
    session.set_status(ScanStatus.COMPLETE)

    metrics = client.get(f"/api/scans/{scan_id}/metrics").json()
    assert metrics["meanTrackLength"] == pytest.approx(5.0)
    assert metrics["meanReprojectionError"] == pytest.approx(0.5)
    assert (metrics["placed"], metrics["accepted"], metrics["captured"]) == (9, 10, 12)
    assert metrics["durationSeconds"] == 12.5 and metrics["preset"] == "fast"
    # No timeline means the scan predates live tracking: "not measured", not zero.
    assert metrics["weakLinks"] is None and metrics["rotationFrames"] is None

    session.timeline = [[80, "good"], [20, "lost"]]
    session.stats.weak_links = 1
    metrics = client.get(f"/api/scans/{scan_id}/metrics").json()
    assert metrics["weakLinks"] == 1 and metrics["rotationFrames"] == 0


def test_metrics_of_an_unfinished_scan(client):
    scan_id = client.post("/api/scans", json={}).json()["id"]
    metrics = client.get(f"/api/scans/{scan_id}/metrics").json()
    assert metrics["meanTrackLength"] is None and metrics["points"] == 0


# --------------------------------------------------------------------------------------
# Coverage guide
# --------------------------------------------------------------------------------------


def ring_cameras(
    azimuths: list[float],
    centre=(3.0, 1.0, -2.0),
    radius: float = 5.0,
    height: float = 0.0,
    up=(0.0, 1.0, 0.0),
    aim_offset=(0.0, 0.0, 0.0),
) -> list[dict]:
    """Cameras on a circle around ``centre``, each looking at it, in the order given.

    Azimuth 0 is along +X and positive azimuth is towards ``up x e1`` - the scanner's right when
    facing the subject - which the direction tests below verify independently.
    """
    centre = np.array(centre, float)
    up = np.array(up, float) / np.linalg.norm(up)
    e1 = np.cross(np.cross(up, [1.0, 0.0, 0.0]), up)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    cameras = []
    for index, azimuth in enumerate(azimuths):
        a = np.radians(azimuth)
        position = centre + radius * (np.cos(a) * e1 + np.sin(a) * e2) + height * up
        forward = centre + np.array(aim_offset) - position
        forward /= np.linalg.norm(forward)
        cam_up = up - forward * (up @ forward)
        cameras.append({
            "name": f"frame_{index + 1:06d}.jpg",
            "position": position.tolist(),
            "forward": forward.tolist(),
            "up": (cam_up / np.linalg.norm(cam_up)).tolist(),
        })
    return cameras


def test_subject_is_where_the_view_rays_meet():
    from backend.coverage import look_at_point

    cameras = ring_cameras(list(range(0, 120, 10)), height=2.5)
    centre = look_at_point(
        np.array([c["position"] for c in cameras]), np.array([c["forward"] for c in cameras])
    )
    assert centre == pytest.approx([3.0, 1.0, -2.0], abs=1e-6)


def test_full_circle_is_complete():
    from backend.coverage import compute_coverage

    coverage, reason = compute_coverage(ring_cameras(list(range(0, 360, 15))))
    assert reason is None
    assert coverage.complete and coverage.covered == 12
    assert coverage.target is None and coverage.turn is None
    assert coverage.centre == pytest.approx([3.0, 1.0, -2.0], abs=1e-6)


def test_half_circle_points_past_the_end_you_are_walking_to():
    from backend.coverage import compute_coverage

    coverage, _ = compute_coverage(ring_cameras(list(range(0, 181, 10))))
    assert coverage.covered == 7  # sectors centred on 0, 30, ... 180 degrees
    assert coverage.current == pytest.approx(180.0, abs=1e-6)
    # Standing at 180 having walked right from 0: keep walking right into 210.
    assert coverage.turn == "right" and coverage.target == 7


def test_walking_the_other_way_points_the_other_way():
    from backend.coverage import compute_coverage

    coverage, _ = compute_coverage(ring_cameras([0, -10, -20, -30, -40, -50, -60, -70, -80, -90]))
    assert coverage.turn == "left"
    assert coverage.target == (coverage.current_sector - 1) % 12


def test_one_sided_arc_points_to_the_nearer_open_end():
    """Scanned 0..90 degrees, now standing back near the start: the gap behind is closer."""
    from backend.coverage import compute_coverage

    coverage, _ = compute_coverage(ring_cameras([0, 20, 40, 60, 80, 90, 60, 30, 10]))
    assert coverage.covered == 4  # 0, 30, 60, 90
    assert coverage.current_sector == 0
    assert coverage.target == 11 and coverage.turn == "left"


def test_right_means_the_scanners_right_hand():
    """Independent of the ring helper: step along forward x up and the guide must say right."""
    from backend.coverage import compute_coverage

    start = ring_cameras([0, 5])
    forward = np.array(start[-1]["forward"])
    up = np.array(start[-1]["up"])
    right_hand = np.cross(forward, up)
    position = np.array(start[-1]["position"])
    cameras = list(start)
    for step in range(1, 6):
        moved = position + 0.4 * step * right_hand
        look = np.array([3.0, 1.0, -2.0]) - moved
        cameras.append({**start[-1], "position": moved.tolist(),
                        "forward": (look / np.linalg.norm(look)).tolist()})
    coverage, _ = compute_coverage(cameras)
    assert coverage.turn == "right"


def test_coverage_ignores_how_colmap_happened_to_orient_the_world():
    """COLMAP's frame is arbitrary (y points down); only the cameras' own up matters."""
    from backend.coverage import compute_coverage

    upright, _ = compute_coverage(ring_cameras(list(range(0, 181, 10))))
    tilted, _ = compute_coverage(ring_cameras(list(range(0, 181, 10)), up=(0.3, -1.0, 0.4)))
    assert tilted.sectors == upright.sectors
    assert (tilted.target, tilted.turn) == (upright.target, upright.turn)


def test_raised_cameras_looking_down_still_guide_the_same_way():
    """Looking down at a subject tilts every camera's up towards it, so on a partial ring the
    averaged "up" leans (~10 degrees here). Sector counts and guidance must survive that."""
    from backend.coverage import compute_coverage

    level, _ = compute_coverage(ring_cameras(list(range(0, 181, 10))))
    raised, _ = compute_coverage(ring_cameras(list(range(0, 181, 10)), height=1.5))
    assert raised.covered == level.covered
    assert (raised.target, raised.turn) == (level.target, level.turn)


def test_parallel_views_have_no_subject():
    """Panning along a wall: the view rays never meet, so there is nothing to walk around."""
    from backend.coverage import compute_coverage

    cameras = [
        {"name": f"f{i}", "position": [i * 0.3, 0.0, 0.0], "forward": [0.0, 0.0, 1.0],
         "up": [0.0, -1.0, 0.0]}
        for i in range(10)
    ]
    assert compute_coverage(cameras) == (None, "no_common_subject")


def test_cameras_looking_outwards_have_no_subject():
    from backend.coverage import compute_coverage

    cameras = ring_cameras(list(range(0, 360, 30)))
    for camera in cameras:
        camera["forward"] = [-v for v in camera["forward"]]
    assert compute_coverage(cameras) == (None, "no_common_subject")


def test_coverage_needs_a_few_cameras():
    from backend.coverage import compute_coverage

    assert compute_coverage(ring_cameras([0, 10])) == (None, "too_few_cameras")
    assert compute_coverage([]) == (None, "too_few_cameras")


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


# --------------------------------------------------------------------------------------
# Find record
# --------------------------------------------------------------------------------------


def full_record(**overrides) -> dict:
    record = {
        "findNumber": "BK-2026-0142",
        "siteCode": "BK26",
        "context": "1004",
        "material": "ceramic",
        "materialOther": "",
        "dateFound": "2026-09-20",
        "recorder": "A. Recorder",
        "notes": "Rim sherd.\nSooting on the exterior.",
    }
    record.update(overrides)
    return record


def test_a_valid_record_is_kept_as_typed():
    from backend.find_record import validate_record

    assert validate_record(full_record()) == full_record()


def test_record_text_is_cleaned_of_invisible_and_control_characters():
    from backend.find_record import validate_record

    record = validate_record(full_record(
        findNumber="  BK\u202e-01\x00\u200b  ",       # bidi override, NUL, zero-width space
        siteCode="Site\tNorth\r\nTrench",             # tabs and line breaks become spaces
        notes="line one\r\n\x07line two  \n",         # notes keep line breaks, lose the bell
        recorder="Jos\u0065\u0301",                    # decomposed accent -> NFC
    ))
    assert record["findNumber"] == "BK-01"
    assert record["siteCode"] == "Site North Trench"
    assert record["notes"] == "line one\nline two"
    assert record["recorder"] == "Jos\u00e9"


def test_a_lone_surrogate_never_reaches_the_file():
    """JSON allows "\\ud800"; UTF-8 cannot encode it, so saving would crash."""
    from backend.find_record import validate_record

    record = validate_record(json.loads('{"findNumber": "A\\ud800B"}'))
    assert record["findNumber"] == "AB"
    record["findNumber"].encode("utf-8")


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        (full_record(findNumber="   "), "findNumber"),
        ({"findNumber": "A", "colour": "red"}, "colour"),
        (full_record(findNumber=123), "findNumber"),
        (full_record(context="x" * 41), "context"),
        (full_record(notes="x" * 2001), "notes"),
        (full_record(material="plastic"), "material"),
        (full_record(dateFound="20260920"), "dateFound"),
        (full_record(dateFound="2026-W39-4"), "dateFound"),
        (full_record(dateFound="2026-02-30"), "dateFound"),
        (full_record(dateFound="2099-01-01"), "dateFound"),
    ],
    ids=["empty-number", "unknown-field", "not-text", "too-long", "notes-too-long",
         "bad-material", "basic-date", "week-date", "impossible-date", "future-date"],
)
def test_bad_records_are_refused_with_the_field_named(payload, field):
    from backend.find_record import RecordError, validate_record

    with pytest.raises(RecordError) as caught:
        validate_record(payload)
    assert caught.value.field == field


def test_a_record_must_be_an_object():
    from backend.find_record import RecordError, validate_record

    for payload in (["BK-1"], "BK-1", None):
        with pytest.raises(RecordError):
            validate_record(payload)


def test_other_material_keeps_its_description_and_nothing_else_does():
    from backend.find_record import material_label, validate_record

    other = validate_record(full_record(material="other", materialOther="Amber bead"))
    assert other["materialOther"] == "Amber bead" and material_label(other) == "Amber bead"
    stone = validate_record(full_record(material="lithic", materialOther="leftover text"))
    assert stone["materialOther"] == "" and material_label(stone) == "Stone / lithic"


def test_a_hand_edited_record_on_disk_is_read_leniently():
    from backend.find_record import load_record

    record = load_record({"findNumber": 7, "siteCode": "S\x00" + "x" * 90, "junk": {"a": 1}})
    assert record["findNumber"] == "" and record["siteCode"] == "S" + "x" * 39
    assert load_record(None) is None and load_record({}) is None


def test_search_needs_every_term_somewhere():
    from backend.find_record import record_matches

    record = full_record(material="lithic")
    assert record_matches(record, "bk-2026")
    assert record_matches(record, "BK26 1004")
    assert record_matches(record, "stone")          # the label, not just the value
    assert not record_matches(record, "bk26 1005")
    assert not record_matches(record, "sooting")    # notes are not searched
    assert record_matches(None, "") and not record_matches(None, "bk")


def test_record_api_round_trip_and_persistence(client):
    scan_id = client.post("/api/scans", json={}).json()["id"]
    assert client.get(f"/api/scans/{scan_id}/record").json()["record"] is None

    saved = client.put(f"/api/scans/{scan_id}/record", json=full_record()).json()["record"]
    assert saved == full_record()
    on_disk = json.loads(client.manager.get(scan_id).manifest_path.read_text(encoding="utf-8"))
    assert on_disk["record"] == full_record()

    fresh = ScanManager(client.manager.scans_dir).get(scan_id)
    assert fresh.record == full_record()


def test_record_api_refuses_unknown_fields_and_reports_the_field(client):
    scan_id = client.post("/api/scans", json={}).json()["id"]
    response = client.put(f"/api/scans/{scan_id}/record", json={**full_record(), "owner": "x"})
    assert response.status_code == 422
    assert response.json()["detail"]["field"] == "owner"
    assert client.put(f"/api/scans/{scan_id}/record", json=["BK-1"]).status_code == 422
    assert client.get(f"/api/scans/{scan_id}/record").json()["record"] is None


def test_record_is_editable_after_reconstruction(client):
    scan_id = client.post("/api/scans", json={"record": full_record()}).json()["id"]
    client.manager.get(scan_id).set_status(ScanStatus.COMPLETE)
    response = client.put(f"/api/scans/{scan_id}/record", json=full_record(context="1005"))
    assert response.status_code == 200 and response.json()["record"]["context"] == "1005"


def test_a_scan_can_start_with_a_record_and_camera_label(client):
    payload = {"record": full_record(), "camera": "HD Webcam\x00 (046d:0825)", "markerSizeMm": 29.6}
    scan = client.post("/api/scans", json=payload).json()
    assert scan["record"]["findNumber"] == "BK-2026-0142"
    assert scan["capture"]["camera"] == "HD Webcam (046d:0825)"
    assert scan["markerSizeMm"] == 29.6
    assert client.post("/api/scans", json={"record": {"siteCode": "no number"}}).status_code == 422
    assert client.post("/api/scans", json={"markerSizeMm": 300}).status_code == 422
    assert client.post("/api/scans", json={"markerSizeMm": "30"}).status_code == 422


def test_scans_without_a_record_still_load(tmp_path):
    """Scans from before find records existed have no "record", "capture" or "scale" keys."""
    root = tmp_path / "scans" / "2026-09-24_183412_0f63"
    root.mkdir(parents=True)
    (root / "scan.json").write_text(json.dumps({
        "id": root.name, "status": "complete", "preset": "balanced",
        "stats": {"captured": 59, "accepted": 56}, "createdAt": 1790300052.5,
    }))
    session = ScanManager(tmp_path / "scans").get(root.name)
    assert session.record is None and session.capture == {} and session.scale is None
    assert session.to_dict()["record"] is None


def test_library_search_covers_every_scan_not_just_recent_ones(client):
    first = client.post("/api/scans", json={"record": full_record(findNumber="OLD-001")}).json()["id"]
    for index in range(22):
        client.manager.create(record=full_record(findNumber=f"NEW-{index:03d}", siteCode="XX"))
    assert len(client.get("/api/scans").json()["scans"]) == 20  # unfiltered: recent 20 only

    found = client.get("/api/scans", params={"q": "old-001"}).json()["scans"]
    assert [scan["id"] for scan in found] == [first]
    assert len(client.get("/api/scans", params={"q": "ceramic xx"}).json()["scans"]) == 22
    assert client.get("/api/scans", params={"q": "nothing-like-this"}).json()["scans"] == []


def test_system_lists_materials_and_board(client):
    payload = client.get("/api/system").json()
    assert {"value": "other", "label": "Other"} in payload["record"]["materials"]
    assert payload["record"]["limits"]["findNumber"] == 40
    assert payload["markerBoard"]["markerMm"] == 30.0 and payload["version"] == "1.1.0"


# --------------------------------------------------------------------------------------
# Capture log
# --------------------------------------------------------------------------------------


def test_capture_log_records_what_was_measured(client):
    scan_id = client.post("/api/scans", json={"camera": "USB Camera"}).json()["id"]
    client.post(f"/api/scans/{scan_id}/frames",
                files={"frame": ("f.jpg", jpeg_bytes(textured_frame(1, (480, 640))), "image/jpeg")})
    log = client.get(f"/api/scans/{scan_id}/capture-log").json()
    assert log["camera"] == "USB Camera" and log["resolution"] == [640, 480]
    assert log["framesAccepted"] == 1 and log["framesPlaced"] is None
    assert log["scaleStatus"] == "not_measured" and log["mmPerUnit"] is None


def test_capture_log_of_an_old_scan_reads_version_and_size_from_its_files(client):
    scan_id = client.post("/api/scans", json={}).json()["id"]
    session = client.manager.get(scan_id)
    cv2.imwrite(str(session.images_dir / "frame_000001.jpg"), textured_frame(2, (720, 1280)))
    session.log_path.write_text("=== Scan x ===\nCOLMAP: C:\\x\\COLMAP.bat (version 4.2.0)\n")
    log = client.get(f"/api/scans/{scan_id}/capture-log").json()
    assert log["colmapVersion"] == "4.2.0" and log["resolution"] == [1280, 720]
    assert log["camera"] is None


# --------------------------------------------------------------------------------------
# COLMAP camera reader
# --------------------------------------------------------------------------------------


def write_cameras_bin(path: Path, cameras: list[tuple[int, int, int, int, tuple[float, ...]]]):
    """(camera_id, model_id, width, height, params) per camera, COLMAP's binary layout."""
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(cameras)))
        for camera_id, model_id, width, height, params in cameras:
            handle.write(struct.pack("<IiQQ", camera_id, model_id, width, height))
            handle.write(struct.pack(f"<{len(params)}d", *params))


def test_cameras_bin_is_read_per_model(tmp_path):
    from backend.reconstruction import read_cameras

    write_cameras_bin(tmp_path / "cameras.bin", [
        (1, 2, 1280, 720, (695.96, 640.0, 360.0, -0.0294)),   # SIMPLE_RADIAL
        (3, 1, 640, 480, (500.0, 510.0, 320.0, 240.0)),      # PINHOLE
    ])
    cameras = read_cameras(tmp_path)
    assert cameras[1] == {"model": "SIMPLE_RADIAL", "width": 1280, "height": 720,
                          "params": pytest.approx([695.96, 640.0, 360.0, -0.0294])}
    assert cameras[3]["model"] == "PINHOLE" and cameras[3]["params"][1] == 510.0
    assert read_cameras(tmp_path / "missing") == {}


def test_cameras_bin_matches_the_real_colmap_layout():
    """The header of a real COLMAP 4.2 cameras.bin: one SIMPLE_RADIAL camera in 64 bytes."""
    from backend.reconstruction import _CAMERA_HEADER

    assert 8 + _CAMERA_HEADER.size + 4 * 8 == 64


def test_a_truncated_cameras_bin_is_not_fatal(tmp_path):
    from backend.reconstruction import read_cameras

    write_cameras_bin(tmp_path / "cameras.bin", [(1, 2, 1280, 720, (700.0, 640.0, 360.0, 0.0))])
    (tmp_path / "cameras.bin").write_bytes((tmp_path / "cameras.bin").read_bytes()[:40])
    assert read_cameras(tmp_path) == {}


def test_image_poses_and_points_are_read(tmp_path):
    from backend.reconstruction import read_image_poses, read_points3d

    write_images_bin(tmp_path / "images.bin", [
        ((1, 0, 0, 0), (0.0, 0.0, -5.0), "frame_000002.jpg"),
        ((0, 1, 0, 0), (1.0, 2.0, 3.0), "frame_000001.jpg"),
    ])
    poses = read_image_poses(tmp_path)
    assert [pose["name"] for pose in poses] == ["frame_000001.jpg", "frame_000002.jpg"]
    assert np.allclose(poses[0]["rotation"], np.diag([1.0, -1.0, -1.0]))  # 180 deg about x
    assert poses[1]["translation"] == (0.0, 0.0, -5.0)

    write_points3d_bin(tmp_path / "points3D.bin", [(0.5, 2), (1.0, 3)])
    xyz, rgb = read_points3d(tmp_path)
    assert xyz.shape == (2, 3) and np.allclose(xyz[1], [0.2, 0.0, 1.0])
    assert rgb.tolist() == [[200, 100, 50], [200, 100, 50]]
    assert read_points3d(tmp_path / "missing")[0].shape == (0, 3)


# --------------------------------------------------------------------------------------
# Marker board
# --------------------------------------------------------------------------------------


def raster_board(px_per_mm: int = 6) -> tuple[np.ndarray, float, float]:
    """The printable board rasterised from the same layout as the PDF."""
    from backend import markers

    layout = markers._page_layout("a4")
    page = np.full((round(layout["height"] * px_per_mm), round(layout["width"] * px_per_mm)), 255, np.uint8)
    for x, y, w, h in markers._cell_rects(layout["ring_x"], layout["ring_y"]):
        page[round(y * px_per_mm):round((y + h) * px_per_mm), round(x * px_per_mm):round((x + w) * px_per_mm)] = 0
    return page, layout["ring_x"], layout["ring_y"]


def test_the_printed_layout_is_what_the_detector_expects():
    """Every marker on the page is found, with its id, at the corners the layout promises."""
    from backend import markers

    page, ring_x, ring_y = raster_board()
    found = markers.detect(page)
    assert sorted(found) == list(range(14))
    expected = markers.corner_positions()
    for (marker_id, k), (x, y) in expected.items():
        # COLMAP convention: pixel edges sit on whole numbers, so mm * 6 maps directly.
        assert found[marker_id][k] == pytest.approx([(ring_x + x) * 6, (ring_y + y) * 6], abs=0.35)
    assert markers.count(page) == 14


def test_detection_reports_colmap_pixel_centres():
    """OpenCV's pixel centre is (0, 0), COLMAP's (0.5, 0.5): corners must come back shifted."""
    from backend import markers

    image = np.full((200, 200), 255, np.uint8)
    image[40:160, 40:160] = cv2.aruco.generateImageMarker(markers._dictionary(), 3, 120, borderBits=1)
    corners = markers.detect(image)[3]
    assert corners[0] == pytest.approx([40.0, 40.0], abs=0.1)     # an edge, not a centre
    assert corners[2] == pytest.approx([160.0, 160.0], abs=0.1)


def test_markers_off_the_board_and_duplicates_are_ignored():
    from backend import markers

    image = np.full((200, 420), 255, np.uint8)
    dictionary = markers._dictionary()
    image[40:160, 30:150] = cv2.aruco.generateImageMarker(dictionary, 40, 120, borderBits=1)
    assert markers.detect(image) == {}                           # id 40 is not on the board
    image[40:160, 30:150] = cv2.aruco.generateImageMarker(dictionary, 5, 120, borderBits=1)
    image[40:160, 260:380] = cv2.aruco.generateImageMarker(dictionary, 5, 120, borderBits=1)
    assert markers.detect(image) == {}                           # the same id twice


def test_board_pdf_is_one_exact_size_page():
    from backend import markers

    for paper, (width, height) in markers.PAPERS.items():
        pdf = markers.board_pdf(paper)
        assert pdf.startswith(b"%PDF-1.4") and pdf.count(b"/Type /Page ") == 1
        box = f"/MediaBox [0 0 {width * 72 / 25.4:.3f}".rstrip("0").rstrip(".")
        assert box.encode() in pdf
        assert b"must measure exactly 100 mm" in pdf


def test_board_endpoint(client):
    response = client.get("/api/marker-board.pdf", params={"paper": "letter"})
    assert response.status_code == 200 and response.headers["content-type"] == "application/pdf"
    assert client.get("/api/marker-board.pdf", params={"paper": "a3"}).status_code == 400


def test_frame_upload_reports_the_board(client):
    scan_id = client.post("/api/scans", json={}).json()["id"]
    page, _, _ = raster_board(4)
    frame = cv2.cvtColor(cv2.resize(page, (840, 1188))[:720], cv2.COLOR_GRAY2BGR)
    frame = cv2.copyMakeBorder(frame, 0, 0, 220, 220, cv2.BORDER_CONSTANT, value=(90, 90, 90))
    response = client.post(f"/api/scans/{scan_id}/frames",
                           files={"frame": ("f.jpg", jpeg_bytes(frame), "image/jpeg")}).json()
    assert response["board"] >= 8
    plain = client.post(f"/api/scans/{scan_id}/frames",
                        files={"frame": ("f.jpg", jpeg_bytes(textured_frame(3)), "image/jpeg")}).json()
    assert plain["board"] == 0


# --------------------------------------------------------------------------------------
# Scale
# --------------------------------------------------------------------------------------


SIMPLE_RADIAL = {"model": "SIMPLE_RADIAL", "width": 1280, "height": 720,
                 "params": [696.0, 640.0, 360.0, -0.0294]}


def test_undistortion_inverts_colmaps_radial_model():
    from backend.scale import distort, undistort

    normalized = np.array([[0.0, 0.0], [0.5, -0.3], [-0.9, 0.5], [0.8, 0.45]])
    pixels = distort(normalized, SIMPLE_RADIAL)
    assert np.allclose(undistort(pixels, SIMPLE_RADIAL), normalized, atol=1e-10)
    # Distortion really is applied: the corner moves by more than a pixel.
    straight = distort(normalized, {**SIMPLE_RADIAL, "params": [696.0, 640.0, 360.0, 0.0]})
    assert np.linalg.norm(pixels[2] - straight[2]) > 5


def test_unsupported_camera_models_are_refused():
    from backend.scale import UnsupportedCameraError, undistort

    with pytest.raises(UnsupportedCameraError):
        undistort(np.zeros((1, 2)), {"model": "OPENCV_FISHEYE", "params": [0.0] * 8})


def ring_views(count: int = 8, radius: float = 4.0):
    from backend.scale import View

    views = []
    for index in range(count):
        angle = 2 * np.pi * index / count
        centre = np.array([radius * np.cos(angle), -1.5, radius * np.sin(angle)])
        forward = -centre / np.linalg.norm(centre)
        right = np.cross([0.0, 1.0, 0.0], forward)
        right /= np.linalg.norm(right)
        rotation = np.stack([right, np.cross(forward, right), forward])
        views.append(View(f"frame_{index:06d}.jpg", rotation, -rotation @ centre, SIMPLE_RADIAL))
    return views


def test_triangulation_recovers_a_point_through_distortion():
    from backend.scale import triangulate_robust

    views = ring_views()
    point = np.array([0.3, 0.2, -0.1])
    pixels = np.array([view.project(point)[0] for view in views])
    solved, errors, used = triangulate_robust(views, pixels)
    assert np.allclose(solved, point, atol=1e-9) and max(errors) < 1e-6 and used == 8


def test_a_bad_observation_is_dropped_not_averaged_in():
    from backend.scale import triangulate_robust

    views = ring_views()
    point = np.array([0.3, 0.2, -0.1])
    pixels = np.array([view.project(point)[0] for view in views])
    pixels[3] += [25.0, -12.0]  # a mis-detected corner
    solved, errors, used = triangulate_robust(views, pixels)
    assert used == 7 and np.allclose(solved, point, atol=1e-9)


def test_too_few_views_or_too_little_angle_gives_no_point():
    from backend.scale import triangulate_robust

    views = ring_views()
    point = np.zeros(3)
    pixels = np.array([view.project(point)[0] for view in views])
    assert triangulate_robust(views[:2], pixels[:2]) is None
    narrow = ring_views(count=400)[:3]  # 0.9 degrees apart: no depth
    assert triangulate_robust(narrow, np.array([v.project(point)[0] for v in narrow])) is None


def test_similarity_fit_recovers_scale_from_a_flat_sheet():
    from backend.scale import fit_similarity

    rng = np.random.default_rng(3)
    board = np.column_stack([rng.random((20, 2)) * 150, np.zeros(20)])
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.linalg.det(q))
    model = 0.037 * board @ q.T + [4.0, -2.0, 1.0]
    scale, rotation, translation = fit_similarity(board, model)
    assert scale == pytest.approx(0.037) and np.linalg.det(rotation) == pytest.approx(1.0)
    assert np.allclose(scale * board @ rotation.T + translation, model)


def test_spacing_cancels_the_inward_corner_bias_that_edges_suffer():
    """Corner refinement pulls a blurred square's corners inwards. Distances between the same
    corner of two markers do not change; a marker's own edges do."""
    from backend import markers
    from backend.scale import edge_estimate, spacing_estimates

    units_per_mm = 0.02
    corners: dict[int, list[np.ndarray]] = {}
    for marker_id in range(14):
        points = [np.array([*markers.corner_positions()[(marker_id, k)], 0.0]) for k in range(4)]
        middle = np.mean(points, axis=0)
        shrunk = [middle + (p - middle) * 0.99 for p in points]  # 1% inward bias
        corners[marker_id] = [p * units_per_mm for p in shrunk]
    spacing = spacing_estimates(corners, 30.0)
    assert np.allclose(list(spacing.values()), 1 / units_per_mm, rtol=1e-9)
    assert edge_estimate(corners[0], 30.0) == pytest.approx(1 / units_per_mm / 0.99)


def test_a_non_square_marker_is_not_used():
    from backend.scale import edge_estimate

    square = [np.array(p, float) for p in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))]
    skewed = [np.array(p, float) for p in ((0, 0, 0), (1.3, 0, 0), (1.3, 1, 0), (0, 1, 0))]
    assert edge_estimate(square, 30.0) == pytest.approx(30.0)
    assert edge_estimate(skewed, 30.0) is None


def test_robust_spread_ignores_one_outlier():
    from backend.scale import robust_spread

    values = np.array([10.0, 10.1, 9.9, 10.05, 9.95, 50.0])
    assert robust_spread(values) < 0.2


@pytest.fixture(scope="module")
def find_scene(tmp_path_factory):
    """A small rendered find scene on the A4 board, with its ground-truth COLMAP model."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from make_test_scene import write_scene

    folder = tmp_path_factory.mktemp("find_scene")
    truth = write_scene(folder, scene="find", frames=12, model=True)
    return folder, truth


def test_scale_from_rendered_frames_matches_the_ground_truth(find_scene):
    """End to end on pixels: detection, undistortion, triangulation and the estimator."""
    from backend.scale import estimate_scale

    folder, truth = find_scene
    result = estimate_scale(folder / "model", folder, 30.0)
    true_mm_per_unit = 1.0 / truth["model"]["unitsPerScene"]
    assert result["status"] == "scaled"
    assert result["markersUsed"] >= 12 and result["framesUsed"] == 12
    assert result["mmPerUnit"] == pytest.approx(true_mm_per_unit, rel=5e-4)
    assert result["spreadPct"] < 0.1 and result["bootstrapPct"] is not None
    assert result["warnings"] == [] and result["boardResidualMm"] < 0.5
    # The board pose maps sheet millimetres onto the model at the same scale.
    assert result["boardPose"]["scale"] == pytest.approx(truth["model"]["unitsPerScene"], rel=5e-3)


def test_a_different_marker_size_scales_the_result(find_scene):
    from backend.scale import estimate_scale

    folder, _ = find_scene
    nominal = estimate_scale(folder / "model", folder, 30.0)["mmPerUnit"]
    printed_small = estimate_scale(folder / "model", folder, 29.4)["mmPerUnit"]
    assert printed_small / nominal == pytest.approx(29.4 / 30.0, rel=1e-9)


def test_no_board_means_no_scale(tmp_path, find_scene):
    from backend.scale import estimate_scale

    folder, _ = find_scene
    for image in sorted(folder.glob("frame_*.jpg")):
        cv2.imwrite(str(tmp_path / image.name), textured_frame(int(image.stem[-2:])))
    result = estimate_scale(folder / "model", tmp_path, 30.0)
    assert result["status"] == "no_markers" and result["framesChecked"] == 12
    assert "mmPerUnit" not in result


def test_a_broken_model_is_an_error_status_not_an_exception(tmp_path):
    from backend.scale import estimate_scale

    assert estimate_scale(tmp_path, tmp_path)["status"] == "error"
    write_cameras_bin(tmp_path / "cameras.bin", [(1, 5, 1280, 720, (1.0,) * 8)])  # fisheye
    write_images_bin(tmp_path / "images.bin", [((1, 0, 0, 0), (0, 0, 0), "a.jpg")])
    result = estimate_scale(tmp_path, tmp_path)
    assert result["status"] == "error" and "OPENCV_FISHEYE" in result["message"]


def completed_find_scan(manager: ScanManager, scene: Path, record: dict | None = None,
                        measure: bool = True) -> ScanSession:
    """A complete scan whose model is the ground-truth model of a rendered find scene."""
    import shutil

    from backend.models import ReconstructionResult
    from backend.reconstruction import read_points3d
    from backend.scale import measure_session

    session = manager.create(record=record, source="import")
    frames = sorted(scene.glob("frame_*.jpg"))
    for frame in frames:
        shutil.copyfile(frame, session.images_dir / frame.name)
    model = session.sparse_dir / "0"
    shutil.copytree(scene / "model", model)
    session.ply_path.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
    session.cameras_path.write_text(json.dumps({"cameras": read_camera_centers(model)}))
    session.stats.captured = session.stats.accepted = len(frames)
    session.result = ReconstructionResult(points=len(read_points3d(model)[0]),
                                          registered_images=len(frames), input_images=len(frames),
                                          model_index=0, duration_seconds=1.0)
    if measure:
        session.scale = measure_session(session)
    session.set_status(ScanStatus.COMPLETE)
    return session


def test_scale_endpoints_remeasure_with_a_new_marker_size(client, find_scene):
    folder, truth = find_scene
    session = completed_find_scan(client.manager, folder, measure=False)
    assert client.get(f"/api/scans/{session.id}/scale").json()["status"] == "not_measured"

    measured = client.post(f"/api/scans/{session.id}/scale", json={"markerSizeMm": 29.4}).json()
    assert measured["status"] == "scaled" and measured["markerSizeMm"] == 29.4
    assert measured["uncertaintyPct"] == max(measured["spreadPct"], measured["bootstrapPct"])
    assert measured["mmPerUnit"] == pytest.approx(29.4 / 30 / truth["model"]["unitsPerScene"], rel=5e-4)
    stored = json.loads(session.manifest_path.read_text())
    assert stored["scale"]["mmPerUnit"] == measured["mmPerUnit"] and stored["markerSizeMm"] == 29.4
    assert client.post(f"/api/scans/{session.id}/scale", json={"markerSizeMm": 2}).status_code == 422

    capturing = client.post("/api/scans", json={}).json()["id"]
    assert client.post(f"/api/scans/{capturing}/scale", json={}).status_code == 409


def test_the_pipeline_measures_scale_before_completing(tmp_path, find_scene):
    """A successful run ends with a scale (here from the ground-truth model the fake mapper
    'produces'); a failing scale never fails the reconstruction."""
    import asyncio
    import shutil

    folder, truth = find_scene

    class ModelRunner(FakeRunner):
        async def run(self, command, args, log_sink=None, line_callback=None, timeout=0):
            self.calls.append((command, list(args)))
            if command == "mapper":
                shutil.copytree(folder / "model", Path(value_of(args, "output_path")) / "0")
            elif command == "model_converter":
                Path(value_of(args, "output_path")).write_bytes(b"ply\n")
            return ""

    manager = ScanManager(tmp_path / "scans")
    session = manager.create()
    for frame in sorted(folder.glob("frame_*.jpg")):
        shutil.copyfile(frame, session.images_dir / frame.name)
    asyncio.run(ReconstructionPipeline(session, runner=ModelRunner()).run())
    assert session.status is ScanStatus.COMPLETE
    assert session.colmap_version == "4.2.0"
    assert session.scale["status"] == "scaled" and session.scale["modelIndex"] == 0
    assert session.scale["mmPerUnit"] == pytest.approx(1 / truth["model"]["unitsPerScene"], rel=5e-4)
    assert "Scale:" in session.read_log()


# --------------------------------------------------------------------------------------
# Orthographic views
# --------------------------------------------------------------------------------------


def cube_points(size: float, per_face: int = 400, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = []
    for axis in range(3):
        for side in (0.0, size):
            face = rng.random((per_face, 3)) * size
            face[:, axis] = side
            points.append(face)
    return np.vstack(points) - size / 2


def test_a_cube_is_drawn_at_its_true_size():
    """A 40 mm cube at 1:1 and 8 px per page mm must span 320 px (plus the splat radius)."""
    from backend.ortho import Layout, choose_layout, render_view

    points = cube_points(40.0)
    colours = np.full((len(points), 3), 60, np.uint8)
    layout = choose_layout(points, (87.0, 88.0), scaled=True)
    assert layout.ratio_label == "3:2"   # 60 mm fits 90% of an 87 mm box; 2:1 (80 mm) would not
    for ratio, expected_px in ((1.0, 320), (layout.ratio, 480)):
        fixed = Layout(ratio, None, np.zeros(3))
        for view in ("top", "front", "side", "bottom"):
            image = render_view(points, colours, view, fixed, (800, 800), 8.0, radius_px=0)
            ys, xs = np.nonzero(image[:, :, 0] < 255)
            assert xs.max() - xs.min() + 1 == pytest.approx(expected_px, abs=1), view
            assert ys.max() - ys.min() + 1 == pytest.approx(expected_px, abs=1), view


def test_the_nearest_point_wins_each_pixel():
    from backend.ortho import Layout, render_view

    points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 5.0], [0.0, 0.0, 3.0]])  # front view: +z is near
    colours = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], np.uint8)
    layout = Layout(1.0, None, np.zeros(3))
    front = render_view(points, colours, "front", layout, (21, 21), 1.0)
    assert front[10, 10, 1] > 0 and front[10, 10, 0] == 0 and front[10, 10, 2] == 0
    # From the top all three are at the same depth, one below the other on the page (+z is
    # down); where their discs overlap, the one listed first wins.
    top = render_view(points, colours, "top", layout, (21, 21), 1.0)
    rgb = lambda row: tuple(int(c > 0) for c in top[row, 10])  # noqa: E731
    assert rgb(11) == (1, 0, 0)          # red alone (z = 1)
    assert rgb(13) == (1, 0, 0)          # red, green and blue overlap: red is listed first
    assert rgb(14) == (0, 1, 0)          # green and blue overlap: green is listed before blue
    assert rgb(18) == (1, 1, 1)          # background


def test_rendering_is_deterministic():
    from backend.ortho import choose_layout, render_view

    points = cube_points(30.0, seed=5)
    colours = (np.random.default_rng(1).random((len(points), 3)) * 255).astype(np.uint8)
    layout = choose_layout(points, (80.0, 80.0), scaled=False)
    first = render_view(points, colours, "side", layout, (400, 400), 5.0)
    assert (first == render_view(points, colours, "side", layout, (400, 400), 5.0)).all()


def test_upright_rotation_matches_the_viewer():
    from backend.ortho import upright_rotation

    ups = np.array([[0.1, -1.0, 0.2], [0.0, -0.9, 0.1]])
    rotation = upright_rotation(ups)
    mean = ups.sum(axis=0) / np.linalg.norm(ups.sum(axis=0))
    assert rotation @ mean == pytest.approx([0.0, 1.0, 0.0])
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    # No cameras: COLMAP's y-down convention, flipped like three.js does for opposite vectors.
    assert upright_rotation(np.zeros((0, 3))) @ np.array([0, -1.0, 0]) == pytest.approx([0, 1.0, 0])
    assert upright_rotation([[0.0, 1.0, 0.0]]) == pytest.approx(np.eye(3))


def test_yaw_alignment_lays_the_long_axis_left_to_right():
    from backend.ortho import yaw_alignment

    rng = np.random.default_rng(2)
    angle = np.radians(35)
    long_axis = np.array([np.cos(angle), 0.0, np.sin(angle)])
    points = np.outer(rng.normal(size=500) * 30, long_axis) + rng.normal(size=(500, 3)) * [1, 5, 1]
    aligned = points @ yaw_alignment(points).T
    spread = aligned.std(axis=0)
    assert spread[0] > 5 * spread[2]
    assert aligned[:, 1] == pytest.approx(points[:, 1])  # height untouched


def test_nice_scale_bar_lengths():
    from backend.ortho import nice_length

    assert [nice_length(v) for v in (40, 19.9, 7, 1.2, 0.35)] == [20, 10, 5, 1, 0.2]


def test_board_pose_isolates_the_find(find_scene):
    from backend.ortho import find_points
    from backend.reconstruction import read_points3d
    from backend.scale import estimate_scale

    folder, truth = find_scene
    xyz, _ = read_points3d(folder / "model")
    scale = estimate_scale(folder / "model", folder, 30.0)
    chosen, how = find_points(xyz, scale)
    assert "on the sheet" in how
    # In the ground-truth frame, what is kept is the box: above the sheet, within its footprint.
    frame = truth["model"]
    world = (xyz[chosen] - frame["translation"]) @ np.array(frame["rotation"]) / frame["unitsPerScene"]
    centre = np.array(truth["find"]["centre"])
    assert (world[:, 2] > 0.5).all() and (world[:, 2] < 28.5).all()
    assert (np.linalg.norm(world[:, :2] - centre[:2], axis=1) < 40).all()
    assert len(chosen) > 4000  # nearly all of the 4,500 points sampled on the box


# --------------------------------------------------------------------------------------
# PDF writer and find report
# --------------------------------------------------------------------------------------


def pdf_pages(pdf: bytes) -> int:
    import re

    return len(re.findall(rb"/Type /Page\b(?!s)", pdf))


def test_pdf_cross_reference_table_points_at_every_object():
    import re

    from backend.pdf import PdfDocument

    doc = PdfDocument(title="T (1)", producer="test")
    page = doc.add_page()
    page.text(10, 10, "Find (A\\B) caf\u00e9 \u4e2d")
    doc.image_rgb(page, np.zeros((4, 5, 3), np.uint8), 10, 20, 50, 40)
    pdf = doc.to_bytes()
    assert b"(Find \\(A\\\\B\\) caf\xe9 ?) Tj" in pdf  # escaped, WinAnsi, unknown -> "?"
    xref = int(re.search(rb"startxref\n(\d+)", pdf).group(1))
    assert pdf[xref:xref + 4] == b"xref"
    offsets = [int(m) for m in re.findall(rb"(\d{10}) 00000 n", pdf)]
    for number, offset in enumerate(offsets, start=1):
        assert pdf[offset:].startswith(f"{number} 0 obj".encode())
    assert pdf_pages(pdf) == 1


def test_wrapping_keeps_to_the_width():
    from backend.pdf import text_width, wrap

    text = "A find report line that is definitely longer than forty millimetres " + "x" * 80
    lines = wrap(text, 40.0, 9)
    assert len(lines) > 3 and all(text_width(line, 9) <= 40.0 for line in lines)
    assert wrap("one\ntwo", 100, 9) == ["one", "two"]


def test_the_report_of_a_scaled_find(tmp_path, find_scene):
    from datetime import datetime

    from backend.report import build_report

    folder, _ = find_scene
    session = completed_find_scan(ScanManager(tmp_path / "scans"), folder,
                                  record=full_record(findNumber="BK-2026-0142"))
    pdf = build_report(session, "1.1.0", datetime(2026, 9, 25, 17, 0))
    assert pdf.startswith(b"%PDF") and pdf_pages(pdf) == 2
    assert pdf.count(b"(BK-2026-0142) Tj") >= 3          # both headers and the record
    assert b"Drawing scale 1:1" in pdf and pdf.count(b"20 mm) Tj") == 4
    assert b"not to scale) Tj" not in pdf
    assert b"Generated 2026-09-25 17:00" in pdf and b"1.1.0" in pdf
    assert pdf == build_report(session, "1.1.0", datetime(2026, 9, 25, 17, 0))  # reproducible


def test_the_report_of_an_unscaled_scan_says_so(tmp_path, find_scene):
    from backend.report import build_report

    folder, _ = find_scene
    session = completed_find_scan(ScanManager(tmp_path / "scans"), folder, measure=False)
    pdf = build_report(session, "1.1.0")
    assert pdf_pages(pdf) == 2
    assert pdf.count(b"(not to scale) Tj") == 4 and b"20 mm) Tj" not in pdf
    assert b"No find number recorded" in pdf and b"not recorded" in pdf


def test_report_endpoint(client, find_scene):
    folder, _ = find_scene
    session = completed_find_scan(client.manager, folder, record=full_record(findNumber="BK/7 a"))
    response = client.get(f"/api/scans/{session.id}/report.pdf")
    assert response.status_code == 200 and response.headers["content-type"] == "application/pdf"
    assert 'filename="find-report-BK_7_a.pdf"' in response.headers["content-disposition"]
    assert pdf_pages(response.content) == 2
    before = sorted(p.name for p in session.root.rglob("*"))
    client.get(f"/api/scans/{session.id}/report.pdf")
    assert sorted(p.name for p in session.root.rglob("*")) == before  # nothing written

    capturing = client.post("/api/scans", json={}).json()["id"]
    assert client.get(f"/api/scans/{capturing}/report.pdf").status_code == 409


def test_a_blocked_colmap_is_reported_as_blocked_not_missing(tmp_path, monkeypatch):
    """Smart App Control stops COLMAP with 0xC0E90002; "not found" would send people looking
    for an install that is already there."""
    import backend.colmap_runner as colmap_runner

    exe = tmp_path / "COLMAP.bat"
    exe.write_text("@echo off")
    monkeypatch.setattr(colmap_runner, "_candidate_paths", lambda: [(exe, "test")])
    monkeypatch.setattr(colmap_runner, "_probe", lambda path: (False, "", -1058471934))
    monkeypatch.setattr(colmap_runner, "_detect_gpu", lambda: "unknown")
    info = ColmapRunner().detect()
    assert not info.available
    assert "could not start" in info.error and "Smart App Control" in info.error
    assert "0xC0E90002" in info.error

    monkeypatch.setattr(colmap_runner, "_probe", lambda path: (False, "not colmap", 0))
    assert ColmapRunner().detect().error.startswith("No COLMAP executable found")
    assert colmap_runner.describe_start_failure(1) is None
