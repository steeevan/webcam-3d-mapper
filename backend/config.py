"""Application configuration.

Everything is local: paths live under the project directory, the server binds to loopback,
and the only external dependency is the COLMAP executable (discovered at runtime).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --- Paths -------------------------------------------------------------------

APP_DIR: Path = Path(__file__).resolve().parent.parent
FRONTEND_DIR: Path = APP_DIR / "frontend"
#: ``MAPPER_SCANS_DIR`` points a second server (tests, a demo) at its own workspace.
SCANS_DIR: Path = Path(os.environ.get("MAPPER_SCANS_DIR") or APP_DIR / "scans")
SAMPLES_DIR: Path = APP_DIR / "samples"

# Persisted override for the COLMAP executable, written by the UI's settings panel.
COLMAP_PATH_FILE: Path = APP_DIR / "colmap_path.txt"

# --- Server ------------------------------------------------------------------

HOST: str = os.environ.get("MAPPER_HOST", "127.0.0.1")
PORT: int = int(os.environ.get("MAPPER_PORT", "8765"))

# --- Capture -----------------------------------------------------------------

CAPTURE_INTERVAL_MS: int = 400          # ~2.5 candidate frames per second
MAX_ACCEPTED_FRAMES: int = 180          # hard cap so COLMAP stays fast
MIN_FRAMES_FOR_RECONSTRUCTION: int = 12 # below this we refuse and explain why
RECOMMENDED_FRAMES: int = 25            # below this we warn but still try
MAX_UPLOAD_BYTES: int = 8 * 1024 * 1024

# --- Frame quality thresholds -------------------------------------------------
# Deliberately permissive: overlap matters more to photogrammetry than per-frame
# sharpness, so we only drop frames that are *clearly* unusable.

BLUR_THRESHOLD: float = 28.0        # variance of Laplacian, below == too blurry
DARK_THRESHOLD: float = 22.0        # mean grayscale 0-255, below == too dark
BRIGHT_THRESHOLD: float = 248.0     # blown out
SIMILARITY_THRESHOLD: float = 2.2   # mean abs diff on a 64x36 thumbnail
THUMBNAIL_SIZE: tuple[int, int] = (64, 36)

# --- Live tracking (see tracking.py) -------------------------------------------
# Calibrated against real webcam scans: a scan that registered 55/56 frames versus one that
# registered 8/30 despite equally sharp frames.

TRACKING_MAX_FEATURES: int = 800
TRACKING_OVERLAY_POINTS: int = 140      # keypoints sent back for the live overlay
TRACKING_REFERENCE_GAP: int = 4         # compare against the accepted frame this many back
TRACKING_MIN_FEATURES: int = 120        # below this the scene has too little texture
TRACKING_GOOD_FEATURES: int = 450
TRACKING_MIN_INLIERS: int = 12          # below this the views no longer overlap
TRACKING_GOOD_INLIERS: int = 120
TRACKING_MIN_INLIERS_FOR_PARALLAX: int = 30
TRACKING_PARALLAX_WINDOW: int = 8       # rolling window of homography/fundamental ratios
TRACKING_ROTATION_RATIO: float = 0.96   # rolling H/F at or above this == turning on the spot

# --- Live 3D preview while scanning (see preview.py) ------------------------------
# Measured on the real 56-frame scan, CPU-only COLMAP 4.2, 16 logical cores. 1024 px / 4096
# features took 37 s for a 56-frame round; 640 px / 2048 features keeps a round at 2-13 s.

PREVIEW_INTERVAL_S: float = 15.0        # rounds start on this grid; a busy round skips a tick
PREVIEW_MIN_FRAMES: int = 12            # the first round fires as soon as this many exist
#: Tick at half the interval until a round has seen PREVIEW_TRUSTED_FRAMES. The two real scans
#: lasted 11 s and 24 s: on the plain grid the second one only ever showed its 12-frame round.
PREVIEW_EARLY_INTERVAL_S: float = PREVIEW_INTERVAL_S / 2
PREVIEW_MAX_IMAGE_SIZE: int = 640
PREVIEW_MAX_FEATURES: int = 2048
PREVIEW_SEQUENTIAL_OVERLAP: int = 6     # quadratic in COLMAP 4: neighbours +1, +2, +4 ... +32
#: Half the features of the Fast preset means roughly half the inliers per image, so COLMAP's
#: default of 30 rejected most frames of the good scan at 15 frames (2/15 placed vs 15/15).
PREVIEW_ABS_POSE_MIN_INLIERS: int = 15
#: Continue from the previous round's model while it placed at least this share of its frames;
#: below that, rebuild from scratch so a bad first initialisation is not locked in.
PREVIEW_CONTINUE_RATIO: float = 0.5
#: Fewer placed-of-total readings below this are too noisy to warn about (see README).
PREVIEW_TRUSTED_FRAMES: int = 20
PREVIEW_STAGE_TIMEOUT_S: float = 120.0
#: Four threads was as fast as eight here; the rest of the machine stays free for uploads,
#: the quality check and the browser.
PREVIEW_NUM_THREADS: int = max(2, min(4, (os.cpu_count() or 4) // 2))

# --- Coverage guide (see coverage.py) ----------------------------------------------

COVERAGE_SECTORS: int = 12              # 30 degrees each
COVERAGE_MIN_CAMERAS: int = 3

# --- Scale board (see markers.py, scale.py) -------------------------------------------

#: Accepted range for a measured marker size. The board is designed at 30 mm; anything far
#: outside this is a typo, not a printer's rescaling.
MARKER_SIZE_RANGE_MM: tuple[float, float] = (10.0, 60.0)

# --- Library thumbnails ---------------------------------------------------------

LIBRARY_THUMBNAIL_WIDTH: int = 360


@dataclass(frozen=True)
class QualityPreset:
    """Reconstruction speed/quality trade-off exposed to the user as Fast / Balanced / Detailed."""

    name: str
    label: str
    max_image_size: int
    max_num_features: int
    sequential_overlap: int
    mapper_ba_global_images_ratio: float
    #: CPU-only SIFT refinements: DSP-SIFT pooling, affine-adapted features, guided matching.
    #: Together they roughly doubled the point count on a real 56-frame webcam scan
    #: (2,441 -> 4,606) for ~2.2x the time. Dense stereo would do far more but needs CUDA.
    refined_features: bool = False
    hint: str = ""

    @property
    def description(self) -> str:
        return f"{self.label} · images downscaled to {self.max_image_size}px · {self.hint}"


PRESETS: dict[str, QualityPreset] = {
    "fast": QualityPreset(
        name="fast",
        label="Fast",
        max_image_size=1024,
        max_num_features=4096,
        sequential_overlap=5,
        mapper_ba_global_images_ratio=1.4,
        hint="quickest preview",
    ),
    "balanced": QualityPreset(
        name="balanced",
        label="Balanced",
        max_image_size=1600,
        max_num_features=8192,
        sequential_overlap=10,
        mapper_ba_global_images_ratio=1.1,
        hint="about 2x slower",
    ),
    "detailed": QualityPreset(
        name="detailed",
        label="Detailed",
        max_image_size=1600,
        max_num_features=12000,
        sequential_overlap=12,
        mapper_ba_global_images_ratio=1.1,
        refined_features=True,
        hint="densest cloud, about 3x slower",
    ),
}
DEFAULT_PRESET: str = "fast"

# --- COLMAP ------------------------------------------------------------------

CAMERA_MODEL: str = "SIMPLE_RADIAL"

#: Directories searched for a COLMAP install, in addition to PATH.
COLMAP_SEARCH_DIRS: tuple[str, ...] = (
    r"C:\Program Files\COLMAP",
    r"C:\Program Files\colmap",
    r"C:\COLMAP",
    r"C:\colmap",
    r"C:\tools\colmap",
    str(Path.home() / "COLMAP"),
    str(Path.home() / "colmap"),
    str(Path.home() / "Downloads" / "COLMAP"),
    str(APP_DIR / "vendor" / "colmap"),
    "/usr/local/bin",
    "/usr/bin",
    "/opt/homebrew/bin",
    "/opt/colmap/bin",
)

#: Executable names to try, most specific first.
COLMAP_EXE_NAMES: tuple[str, ...] = ("COLMAP.bat", "colmap.exe", "colmap")

#: A COLMAP process that produces no output for this long is considered hung.
COLMAP_STAGE_TIMEOUT_S: float = 60 * 60
