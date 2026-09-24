"""Render a synthetic multi-view image sequence for testing the reconstruction pipeline.

Developer tool. It builds a textured "room corner plus a box" scene and renders it from a
camera moving along a smooth arc, which is what a careful webcam scan looks like. The output
folder can be fed straight into the app's developer import
(Settings -> Developer -> import images), giving a genuine COLMAP reconstruction without
needing a webcam or a physical subject.

    python scripts/make_test_scene.py --out scans/_testscene --frames 40

The renderer is a painter's-algorithm homography warp: each planar face is warped into the
image with the homography implied by its four projected corners. That is enough to produce
real parallax and real, matchable texture.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def make_texture(seed: int, size: int = 512) -> np.ndarray:
    """Multi-scale blob texture. SIFT needs blobs at several scales, not flat noise."""
    rng = np.random.default_rng(seed)
    accumulator = np.zeros((size, size, 3), np.float32)
    for octave in (3, 6, 12, 24, 48, 96):
        noise = rng.random((octave, octave, 3)).astype(np.float32)
        accumulator += cv2.resize(noise, (size, size), interpolation=cv2.INTER_CUBIC) / math.sqrt(
            octave
        )
    accumulator -= accumulator.min()
    accumulator /= max(accumulator.max(), 1e-6)
    return (25 + accumulator * 205).astype(np.uint8)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """World-to-camera rotation for OpenCV conventions (+x right, +y down, +z forward)."""
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward])


def quad(origin, edge_u, edge_v) -> np.ndarray:
    """Four corners of a planar face, counter-clockwise from `origin`."""
    origin, edge_u, edge_v = map(np.asarray, (origin, edge_u, edge_v), [float] * 3)
    return np.stack([origin, origin + edge_u, origin + edge_u + edge_v, origin + edge_v])


def tessellate(origin, edge_u, edge_v, texture: np.ndarray, tiles: int):
    """Split a planar face into `tiles` x `tiles` sub-quads with matching texture crops.

    A single large quad is useless here: as soon as one of its four corners falls behind the
    camera the whole face has to be dropped. Tiling means only the off-screen tiles disappear,
    and each tile's homography stays well conditioned.
    """
    origin, edge_u, edge_v = map(np.asarray, (origin, edge_u, edge_v), [float] * 3)
    height, width = texture.shape[:2]
    faces = []
    for i in range(tiles):
        for j in range(tiles):
            u0, u1 = i / tiles, (i + 1) / tiles
            v0, v1 = j / tiles, (j + 1) / tiles
            corners = np.stack(
                [
                    origin + edge_u * u0 + edge_v * v0,
                    origin + edge_u * u1 + edge_v * v0,
                    origin + edge_u * u1 + edge_v * v1,
                    origin + edge_u * u0 + edge_v * v1,
                ]
            )
            crop = texture[
                int(v0 * height) : max(int(v1 * height), int(v0 * height) + 2),
                int(u0 * width) : max(int(u1 * width), int(u0 * width) + 2),
            ]
            faces.append((corners, crop))
    return faces


def build_scene() -> list[tuple[np.ndarray, np.ndarray]]:
    """A room corner plus a box on the floor: enough depth variation for real parallax."""
    planes = [
        ((-5.0, -5.0, 0.0), (10.0, 0.0, 0.0), (0.0, 10.0, 0.0), 6),  # floor
        ((-5.0, 2.6, 0.0), (10.0, 0.0, 0.0), (0.0, 0.0, 4.0), 5),    # back wall
        ((-2.6, -5.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 4.0), 5),   # side wall
    ]
    # A 0.9 m box standing slightly off-centre.
    bx, by, bz, s = 0.25, 0.15, 0.0, 0.9
    planes += [
        ((bx, by, bz + s), (s, 0, 0), (0, s, 0), 1),   # top
        ((bx, by, bz), (s, 0, 0), (0, 0, s), 1),       # front
        ((bx, by + s, bz), (s, 0, 0), (0, 0, s), 1),   # back
        ((bx, by, bz), (0, s, 0), (0, 0, s), 1),       # left
        ((bx + s, by, bz), (0, s, 0), (0, 0, s), 1),   # right
    ]

    faces: list[tuple[np.ndarray, np.ndarray]] = []
    for index, (origin, edge_u, edge_v, tiles) in enumerate(planes):
        texture = make_texture(index * 17 + 3, size=1024 if tiles > 1 else 512)
        faces += tessellate(origin, edge_u, edge_v, texture, tiles)
    return faces


def render(
    faces: list[tuple[np.ndarray, np.ndarray]],
    rotation: np.ndarray,
    eye: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    frame = np.full((height, width, 3), 18, np.uint8)

    # Painter's algorithm: draw the furthest face first.
    order = sorted(faces, key=lambda item: -np.linalg.norm(item[0].mean(axis=0) - eye))

    for corners, texture in order:
        camera_space = (rotation @ (corners - eye).T).T
        if np.any(camera_space[:, 2] < 0.25):
            continue  # at least one corner is behind (or too near) the camera

        projected = (intrinsics @ camera_space.T).T
        projected = projected[:, :2] / projected[:, 2:3]
        if not np.isfinite(projected).all():
            continue

        th, tw = texture.shape[:2]
        source = np.float32([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]])
        homography = cv2.getPerspectiveTransform(source, projected.astype(np.float32))

        warped = cv2.warpPerspective(texture, homography, (width, height))
        mask = cv2.warpPerspective(
            np.full((th, tw), 255, np.uint8), homography, (width, height)
        )
        frame[mask > 127] = warped[mask > 127]

    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="scans/_testscene", help="output folder")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    focal = args.width * 0.8
    intrinsics = np.float32(
        [[focal, 0, args.width / 2], [0, focal, args.height / 2], [0, 0, 1]]
    )
    faces = build_scene()
    target = np.array([0.1, 0.1, 0.65])
    up = np.array([0.0, 0.0, 1.0])  # world +z is up; look_at derives the camera's down axis

    for index in range(args.frames):
        t = index / max(args.frames - 1, 1)
        angle = math.radians(-125 + 80 * t)           # an 80 degree arc around the corner
        radius = 3.6 - 0.45 * math.sin(math.pi * t)   # gentle in-and-out, like a hand-held scan
        eye = np.array(
            [
                target[0] + radius * math.cos(angle),
                target[1] + radius * math.sin(angle),
                1.3 + 0.3 * math.sin(2 * math.pi * t),
            ]
        )
        rotation = look_at(eye, target, up)
        frame = render(faces, rotation, eye, intrinsics, args.width, args.height)
        cv2.imwrite(str(out / f"frame_{index + 1:06d}.jpg"), frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    print(f"Wrote {args.frames} frames to {out.resolve()}")


if __name__ == "__main__":
    main()
