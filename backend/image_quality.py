"""Lightweight frame quality filtering.

Three cheap OpenCV checks decide whether a captured frame is worth handing to COLMAP:

* **blur**      - variance of the Laplacian (low variance == few sharp edges == soft frame)
* **exposure**  - mean grayscale intensity (a black or blown-out frame carries no features)
* **novelty**   - mean absolute difference against the previous *accepted* frame's thumbnail

The bias is strongly towards keeping frames. Overlap is what makes structure-from-motion work,
so a merely mediocre frame is more useful than a gap in the sequence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from . import config
from .models import FrameDecision

logger = logging.getLogger(__name__)


@dataclass
class _Metrics:
    blur: float
    brightness: float
    thumbnail: np.ndarray


def _measure(image_bgr: np.ndarray) -> _Metrics:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Measuring blur on a fixed-height version keeps the threshold resolution-independent.
    height = 480
    if gray.shape[0] > height:
        scale = height / gray.shape[0]
        gray_small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        gray_small = gray

    blur = float(cv2.Laplacian(gray_small, cv2.CV_64F).var())
    brightness = float(gray.mean())
    thumbnail = cv2.resize(gray, config.THUMBNAIL_SIZE, interpolation=cv2.INTER_AREA).astype(
        np.float32
    )
    return _Metrics(blur=blur, brightness=brightness, thumbnail=thumbnail)


def decode_jpeg(data: bytes) -> np.ndarray | None:
    """Decode JPEG bytes to a BGR array, or ``None`` if the payload is not a usable image."""
    buffer = np.frombuffer(data, dtype=np.uint8)
    if buffer.size == 0:
        return None
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return image


class FrameQualityFilter:
    """Stateful per-scan filter. Holds only the last accepted thumbnail, not whole frames."""

    def __init__(
        self,
        blur_threshold: float = config.BLUR_THRESHOLD,
        dark_threshold: float = config.DARK_THRESHOLD,
        bright_threshold: float = config.BRIGHT_THRESHOLD,
        similarity_threshold: float = config.SIMILARITY_THRESHOLD,
    ) -> None:
        self.blur_threshold = blur_threshold
        self.dark_threshold = dark_threshold
        self.bright_threshold = bright_threshold
        self.similarity_threshold = similarity_threshold
        self._previous_thumbnail: np.ndarray | None = None
        self._accepted = 0

    def evaluate(self, image_bgr: np.ndarray) -> FrameDecision:
        """Score one frame and, if accepted, remember it as the new reference."""
        metrics = _measure(image_bgr)

        if metrics.brightness < self.dark_threshold:
            return FrameDecision(False, "dark", metrics.blur, metrics.brightness)
        if metrics.brightness > self.bright_threshold:
            return FrameDecision(False, "dark", metrics.blur, metrics.brightness)

        # Blur is checked before novelty so a soft frame is reported as blurry rather than
        # being swallowed by the duplicate test (a blurred frame resembles its neighbour).
        if metrics.blur < self.blur_threshold:
            return FrameDecision(False, "blurry", metrics.blur, metrics.brightness)

        # The very first frame sets the reference; never reject it for being a duplicate.
        difference = float("inf")
        if self._previous_thumbnail is not None:
            difference = float(np.abs(metrics.thumbnail - self._previous_thumbnail).mean())
            if difference < self.similarity_threshold:
                return FrameDecision(
                    False, "duplicate", metrics.blur, metrics.brightness, difference
                )

        self._previous_thumbnail = metrics.thumbnail
        self._accepted += 1
        return FrameDecision(
            True,
            "accepted",
            metrics.blur,
            metrics.brightness,
            0.0 if difference == float("inf") else difference,
        )

    def reset(self) -> None:
        self._previous_thumbnail = None
        self._accepted = 0
