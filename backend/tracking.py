"""Live tracking quality: will COLMAP be able to use what the camera is seeing?

The quality filter answers "is this frame sharp and new?". That is not enough to predict a
reconstruction: a scan can be full of crisp, distinct frames and still fail because

* the scene has nothing to lock onto          -> **texture**   (ORB keypoint count)
* consecutive views stopped overlapping        -> **continuity** (verified matches to a
                                                  frame a few steps back)
* the camera turned on the spot                -> **parallax**  (homography vs fundamental fit)

The last one is the subtle killer. When the camera only rotates, every match is explained by a
single homography and there is no depth to recover - the scan *looks* healthy frame to frame.
Comparing homography inliers with fundamental-matrix inliers is the same test COLMAP uses to
choose its initial pair, applied here in real time so the person scanning gets told to step
sideways before they have spent a minute on a scan that cannot work.

All measurements run on the ~480 px greyscale frame the quality filter already produced, and
cost roughly 10-20 ms per frame on a laptop CPU.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import median
from typing import Any

import cv2
import numpy as np

from . import config


@dataclass
class TrackingSample:
    """Tracking measurements for one frame, plus the rolling verdict."""

    features: int = 0
    matches: int = 0
    inliers: int = 0
    homography_ratio: float | None = None
    state: str = "starting"
    score: int = 0
    keypoints: list[list[float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": self.features,
            "matches": self.matches,
            "inliers": self.inliers,
            "homographyRatio": (
                None if self.homography_ratio is None else round(self.homography_ratio, 3)
            ),
            "state": self.state,
            "score": self.score,
            "keypoints": self.keypoints,
        }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class TrackingEstimator:
    """Stateful per-scan tracker. Keeps descriptors for the last few *accepted* frames only."""

    def __init__(self) -> None:
        self._orb = cv2.ORB_create(nfeatures=config.TRACKING_MAX_FEATURES, fastThreshold=12)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        # Comparing against a frame a few accepted steps back rather than the immediate
        # neighbour accumulates baseline, which is what makes the parallax test discriminate.
        self._references: deque[tuple[np.ndarray, np.ndarray]] = deque(
            maxlen=config.TRACKING_REFERENCE_GAP
        )
        self._ratios: deque[float] = deque(maxlen=config.TRACKING_PARALLAX_WINDOW)

    def observe(self, gray: np.ndarray, accepted: bool) -> TrackingSample:
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)
        points = (
            np.float32([kp.pt for kp in keypoints]) if keypoints else np.empty((0, 2), np.float32)
        )
        sample = TrackingSample(features=len(keypoints), keypoints=_overlay_sample(points, gray))

        if descriptors is not None and len(descriptors) >= 2 and self._references:
            ref_points, ref_descriptors = self._references[0]
            self._compare(sample, ref_points, ref_descriptors, points, descriptors)

        sample.state, sample.score = self._verdict(sample)

        if accepted and descriptors is not None and len(descriptors) >= 2:
            self._references.append((points, descriptors))
            if sample.homography_ratio is not None:
                self._ratios.append(sample.homography_ratio)
        return sample

    def _compare(
        self,
        sample: TrackingSample,
        ref_points: np.ndarray,
        ref_descriptors: np.ndarray,
        points: np.ndarray,
        descriptors: np.ndarray,
    ) -> None:
        pairs = self._matcher.knnMatch(ref_descriptors, descriptors, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        sample.matches = len(good)
        if len(good) < 15:
            return

        src = np.float32([ref_points[m.queryIdx] for m in good])
        dst = np.float32([points[m.trainIdx] for m in good])
        _, f_mask = cv2.findFundamentalMat(src, dst, cv2.USAC_MAGSAC, 1.5, 0.999)
        _, h_mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 3.0)
        f_inliers = int(f_mask.sum()) if f_mask is not None else 0
        h_inliers = int(h_mask.sum()) if h_mask is not None else 0
        sample.inliers = f_inliers
        # Only trust the ratio when there is enough geometry behind it.
        if f_inliers >= config.TRACKING_MIN_INLIERS_FOR_PARALLAX:
            sample.homography_ratio = h_inliers / f_inliers

    def _verdict(self, sample: TrackingSample) -> tuple[str, int]:
        texture = _clamp01(sample.features / config.TRACKING_GOOD_FEATURES)
        if not self._references:
            return ("low_texture" if sample.features < config.TRACKING_MIN_FEATURES else "starting",
                    round(100 * texture))

        continuity = _clamp01(sample.inliers / config.TRACKING_GOOD_INLIERS)
        ratios = list(self._ratios)
        if sample.homography_ratio is not None:
            ratios.append(sample.homography_ratio)
        # H/F near 1.0 means "a homography explains everything" -> rotation only, no depth.
        rolling = median(ratios[-config.TRACKING_PARALLAX_WINDOW :]) if ratios else None
        parallax = 1.0 if rolling is None else _clamp01((0.99 - rolling) / 0.12)
        score = round(100 * (0.25 * texture + 0.45 * continuity + 0.30 * parallax))

        if sample.features < config.TRACKING_MIN_FEATURES:
            return "low_texture", score
        if sample.inliers < config.TRACKING_MIN_INLIERS:
            return "lost", score
        if (
            rolling is not None
            and len(ratios) >= config.TRACKING_PARALLAX_WINDOW // 2
            and rolling >= config.TRACKING_ROTATION_RATIO
        ):
            return "rotating", score
        return "good", score

    def reset(self) -> None:
        self._references.clear()
        self._ratios.clear()


def _overlay_sample(points: np.ndarray, gray: np.ndarray) -> list[list[float]]:
    """A thinned, normalised subset of keypoints for the live overlay (x, y in 0..1)."""
    if len(points) == 0:
        return []
    step = max(1, len(points) // config.TRACKING_OVERLAY_POINTS)
    height, width = gray.shape[:2]
    chosen = points[::step][: config.TRACKING_OVERLAY_POINTS]
    return [[round(float(x) / width, 4), round(float(y) / height, 4)] for x, y in chosen]
