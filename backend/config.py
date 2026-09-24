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
SCANS_DIR: Path = APP_DIR / "scans"
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


@dataclass(frozen=True)
class QualityPreset:
    """Reconstruction speed/quality trade-off exposed to the user as Fast / Balanced."""

    name: str
    label: str
    max_image_size: int
    max_num_features: int
    sequential_overlap: int
    mapper_ba_global_images_ratio: float

    @property
    def description(self) -> str:
        return f"{self.label} · images downscaled to {self.max_image_size}px"


PRESETS: dict[str, QualityPreset] = {
    "fast": QualityPreset(
        name="fast",
        label="Fast",
        max_image_size=1024,
        max_num_features=4096,
        sequential_overlap=5,
        mapper_ba_global_images_ratio=1.4,
    ),
    "balanced": QualityPreset(
        name="balanced",
        label="Balanced",
        max_image_size=1600,
        max_num_features=8192,
        sequential_overlap=10,
        mapper_ba_global_images_ratio=1.1,
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
