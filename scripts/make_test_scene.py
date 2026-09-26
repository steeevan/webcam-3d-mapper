"""Render a synthetic multi-view image sequence for testing the reconstruction pipeline.

Developer tool with two scenes, both written as numbered JPEGs plus ``scene.json`` (the exact
camera poses and object sizes, i.e. the ground truth):

* ``--scene room`` (default): a textured room corner plus a box, seen from a camera moving
  along an 80 degree arc - what a careful webcam scan of a room looks like.
* ``--scene find``: the use case of the marker board. A textured box of known size in
  millimetres (64 x 42 x 28) lies on the printed A4 scale board (``backend/markers.py``) on a
  textured table, and the camera circles it once at webcam distance. Units are millimetres, so
  a scale recovered from the markers can be checked against the true camera positions.

The output folder can be fed straight into the app's developer import
(Settings -> Developer -> import images), giving a genuine COLMAP reconstruction without
needing a webcam or a physical subject.

    python scripts/make_test_scene.py --out scans/_testscene --frames 40
    python scripts/make_test_scene.py --scene find --out .cache/findscene --frames 60

The renderer is a painter's-algorithm homography warp: each planar face is warped into the
image with the homography implied by its four projected corners. That is enough to produce
real parallax and real, matchable texture. Faces are drawn in layers (table, then sheet, then
object) and back to front within a layer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import markers  # noqa: E402


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


def build_scene() -> list[tuple[np.ndarray, np.ndarray, int]]:
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

    faces: list[tuple[np.ndarray, np.ndarray, int]] = []
    for index, (origin, edge_u, edge_v, tiles) in enumerate(planes):
        texture = make_texture(index * 17 + 3, size=1024 if tiles > 1 else 512)
        faces += [(c, t, 0) for c, t in tessellate(origin, edge_u, edge_v, texture, tiles)]
    return faces


#: The find: a box this size in millimetres (length, width, height), turned this far about
#: the vertical, centred in the free area of the board.
FIND_SIZE_MM = (64.0, 42.0, 28.0)
FIND_YAW_DEG = 25.0
#: Texture resolution of the printed sheet. Six pixels per millimetre puts every marker edge
#: of the A4 layout exactly on a pixel boundary (22.5, 42.5 and 5 mm are all whole pixels).
BOARD_PX_PER_MM = 6


def board_texture(paper: str = "a4") -> tuple[np.ndarray, float, float]:
    """The printable sheet as an image, drawn from the same layout as the PDF."""
    layout = markers._page_layout(paper)
    k = BOARD_PX_PER_MM
    page = np.full((round(layout["height"] * k), round(layout["width"] * k), 3), 246, np.uint8)
    for x, y, w, h in markers._cell_rects(layout["ring_x"], layout["ring_y"]):
        page[round(y * k) : round((y + h) * k), round(x * k) : round((x + w) * k)] = 12
    return page, layout["width"], layout["height"]


def page_to_world(u: float, v: float, width: float, height: float) -> np.ndarray:
    """Sheet millimetres (x right, y down as printed) -> world, sheet centre at the origin.

    Seen from above (+z) with +y up, the sheet reads the right way round, not mirrored.
    """
    return np.array([u - width / 2, -(v - height / 2), 0.0])


def build_find_scene(paper: str = "a4") -> tuple[list[tuple[np.ndarray, np.ndarray, int]], dict]:
    """Table, printed board and a textured box, in millimetres with +z up."""
    faces: list[tuple[np.ndarray, np.ndarray, int]] = []
    table = make_texture(101, size=1024)
    faces += [(c, t, 0) for c, t in tessellate((-400, -300, 0), (800, 0, 0), (0, 600, 0), table, 8)]

    sheet, width, height = board_texture(paper)
    corners = np.stack(
        [page_to_world(u, v, width, height) for u, v in ((0, 0), (width, 0), (width, height), (0, height))]
    )
    faces.append((corners, sheet, 1))

    layout = markers._page_layout(paper)
    x0, y0, x1, y1 = markers.INNER_AREA
    centre = page_to_world(
        layout["ring_x"] + (x0 + x1) / 2, layout["ring_y"] + (y0 + y1) / 2, width, height
    )
    length, breadth, tall = FIND_SIZE_MM
    yaw = math.radians(FIND_YAW_DEG)
    ax = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    ay = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    az = np.array([0.0, 0.0, 1.0])
    base = centre - ax * length / 2 - ay * breadth / 2
    box = [
        (base + az * tall, ax * length, ay * breadth),       # top
        (base, ax * length, az * tall),                      # front
        (base + ay * breadth, ax * length, az * tall),       # back
        (base, ay * breadth, az * tall),                     # left
        (base + ax * length, ay * breadth, az * tall),       # right
    ]
    for index, (origin, edge_u, edge_v) in enumerate(box):
        texture = make_texture(200 + index * 7, size=512)
        faces += [(c, t, 2) for c, t in tessellate(origin, edge_u, edge_v, texture, 1)]

    truth = {
        "units": "mm",
        "board": {"paper": paper, "markerMm": markers.MARKER_MM, "name": markers.BOARD_NAME},
        "find": {
            "sizeMm": list(FIND_SIZE_MM),
            "yawDeg": FIND_YAW_DEG,
            "centre": (centre + az * tall / 2).tolist(),
        },
    }
    return faces, truth


def render(
    faces: list[tuple[np.ndarray, np.ndarray, int]],
    rotation: np.ndarray,
    eye: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    near: float = 0.25,
) -> np.ndarray:
    """One view. ``intrinsics`` follow COLMAP's pixel convention (the centre of the top-left
    pixel is (0.5, 0.5)); OpenCV's is half a pixel off, which the warp accounts for."""
    frame = np.full((height, width, 3), 18, np.uint8)

    # Painter's algorithm: lower layers first, then the furthest face first within a layer.
    order = sorted(faces, key=lambda item: (item[2], -np.linalg.norm(item[0].mean(axis=0) - eye)))

    for corners, texture, _layer in order:
        camera_space = (rotation @ (corners - eye).T).T
        if np.any(camera_space[:, 2] < near):
            continue  # at least one corner is behind (or too near) the camera

        projected = (intrinsics @ camera_space.T).T
        projected = projected[:, :2] / projected[:, 2:3] - 0.5
        if not np.isfinite(projected).all():
            continue

        # Only warp into the face's bounding box: a full-frame warp per face is the slow part.
        x0, y0 = np.floor(projected.min(axis=0)).astype(int)
        x1, y1 = np.ceil(projected.max(axis=0)).astype(int) + 1
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
        if x0 >= x1 or y0 >= y1:
            continue

        th, tw = texture.shape[:2]
        # Texture pixel edges, not centres, map onto the face's corners.
        source = np.float32([[-0.5, -0.5], [tw - 0.5, -0.5], [tw - 0.5, th - 0.5], [-0.5, th - 0.5]])
        homography = cv2.getPerspectiveTransform(source, (projected - [x0, y0]).astype(np.float32))
        size = (int(x1 - x0), int(y1 - y0))
        warped = cv2.warpPerspective(texture, homography, size, borderMode=cv2.BORDER_REPLICATE)
        mask = cv2.warpPerspective(np.full((th, tw), 255, np.uint8), homography, size)
        region = frame[y0:y1, x0:x1]
        region[mask > 127] = warped[mask > 127]

    return frame


def find_orbit(index: int, frames: int, target: np.ndarray) -> np.ndarray:
    """One full circle around the find at webcam distance (~43 cm), rising and falling a little."""
    t = index / frames
    angle = 2 * math.pi * t - math.pi / 2
    radius = 340.0 + 25.0 * math.sin(4 * math.pi * t)
    return np.array(
        [
            target[0] + radius * math.cos(angle),
            target[1] + radius * math.sin(angle),
            260.0 + 30.0 * math.sin(6 * math.pi * t),
        ]
    )


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def quaternion_wxyz(r: np.ndarray) -> tuple[float, float, float, float]:
    """(w, x, y, z) of a rotation matrix, stable for any angle."""
    trace = np.trace(r)
    if trace > 0:
        s = 2 * math.sqrt(1 + trace)
        return (s / 4, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
    i = int(np.argmax(np.diag(r)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2 * math.sqrt(1 + r[i, i] - r[j, j] - r[k, k])
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (r[k, j] - r[j, k]) / s
    q[1 + i] = s / 4
    q[1 + j] = (r[j, i] + r[i, j]) / s
    q[1 + k] = (r[k, i] + r[i, k]) / s
    return tuple(q)


def sample_surfaces(faces, per_face: dict[int, int], rng: np.random.Generator):
    """Points on the scene's faces with their texture colour: a stand-in for a sparse cloud."""
    points, colours = [], []
    for corners, texture, layer in faces:
        count = per_face.get(layer, 0)
        if not count:
            continue
        uv = rng.random((count, 2))
        a, b, d = corners[0], corners[1], corners[3]
        points.append(a + np.outer(uv[:, 0], b - a) + np.outer(uv[:, 1], d - a))
        h, w = texture.shape[:2]
        bgr = texture[(uv[:, 1] * (h - 1)).astype(int), (uv[:, 0] * (w - 1)).astype(int)]
        colours.append(bgr[:, ::-1])
    return np.vstack(points), np.vstack(colours).astype(np.uint8)


def write_colmap_model(
    model_dir: Path,
    truth: dict,
    points: np.ndarray,
    colours: np.ndarray,
    seed: int = 0,
    position_noise: float = 0.0,
    rotation_noise_deg: float = 0.0,
    focal_scale: float = 1.0,
) -> dict:
    """A COLMAP binary model (cameras/images/points3D.bin) made from the true poses.

    It lives in a random similarity frame - arbitrary scale, rotation and origin, like a real
    reconstruction - so code that reads it cannot accidentally rely on scene units. Optional
    noise on camera positions (scene units), orientations and focal length imitates an
    imperfect reconstruction. Returns the frame: ``unitsPerScene`` is the true scale.
    """
    rng = np.random.default_rng(seed)
    scale = float(10 ** rng.uniform(-2.5, -1.0)) if truth.get("units") == "mm" else float(rng.uniform(0.5, 2.0))
    rotation = random_rotation(rng)
    translation = rng.normal(size=3) * 3.0
    model_dir.mkdir(parents=True, exist_ok=True)

    import struct

    focal = truth["focal"] * focal_scale
    cx, cy = truth["principalPoint"]
    with (model_dir / "cameras.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<IiQQ", 1, 2, truth["width"], truth["height"]))
        handle.write(struct.pack("<4d", focal, cx, cy, 0.0))  # SIMPLE_RADIAL, k = 0

    with (model_dir / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(truth["frames"])))
        for index, frame in enumerate(truth["frames"], start=1):
            r_world = np.array(frame["rotation"])
            eye = np.array(frame["eye"]) + rng.normal(size=3) * position_noise
            if rotation_noise_deg:
                axis = rng.normal(size=3)
                axis /= np.linalg.norm(axis)
                angle = math.radians(rng.normal() * rotation_noise_deg)
                skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                r_world = (np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * skew @ skew) @ r_world
            # model = s R world + T  =>  camera = R_w R^T (model - T) - s R_w eye
            r_model = r_world @ rotation.T
            t_model = -r_model @ translation - scale * r_world @ eye
            handle.write(struct.pack("<I7dI", index, *quaternion_wxyz(r_model), *t_model, 1))
            handle.write(frame["name"].encode() + b"\x00")
            handle.write(struct.pack("<Q", 0))

    model_points = scale * (rotation @ points.T).T + translation
    with (model_dir / "points3D.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(model_points)))
        for index, (xyz, rgb) in enumerate(zip(model_points, colours), start=1):
            handle.write(struct.pack("<Q3d3BdQ", index, *xyz, *(int(c) for c in rgb), 0.5, 2))
            handle.write(struct.pack("<IIII", 1, 0, 2, 0))

    frame = {"unitsPerScene": scale, "rotation": rotation.tolist(), "translation": translation.tolist()}
    (model_dir / "truth.json").write_text(json.dumps(frame, indent=1), encoding="utf-8")
    return frame


def write_scene(
    out: Path,
    scene: str = "room",
    frames: int = 40,
    width: int = 1280,
    height: int = 720,
    supersample: int = 0,
    model: bool = False,
) -> dict:
    """Render ``frames`` views into ``out`` and write ``scene.json``; returns the truth.

    With ``model``, also writes the ground-truth COLMAP model to ``out/model``.
    """
    out.mkdir(parents=True, exist_ok=True)
    focal = width * 0.8
    up = np.array([0.0, 0.0, 1.0])  # world +z is up; look_at derives the camera's down axis
    if scene == "find":
        faces, truth = build_find_scene()
        target = np.array(truth["find"]["centre"])
        near = 5.0
        k = supersample or 2
    else:
        faces = build_scene()
        truth = {"units": "scene"}
        target = np.array([0.1, 0.1, 0.65])
        near = 0.25
        k = supersample or 1

    intrinsics = np.float64([[focal * k, 0, width * k / 2], [0, focal * k, height * k / 2], [0, 0, 1]])
    poses = []
    for index in range(frames):
        if scene == "find":
            eye = find_orbit(index, frames, target)
        else:
            t = index / max(frames - 1, 1)
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
        frame = render(faces, rotation, eye, intrinsics, width * k, height * k, near)
        if k > 1:
            # Averaging k x k blocks anti-aliases edges and keeps COLMAP's pixel convention.
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        name = f"frame_{index + 1:06d}.jpg"
        cv2.imwrite(str(out / name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        poses.append({"name": name, "eye": eye.tolist(), "rotation": rotation.tolist()})

    truth.update(
        {
            "scene": scene,
            "width": width,
            "height": height,
            "focal": focal,
            "principalPoint": [width / 2, height / 2],
            "frames": poses,
        }
    )
    (out / "scene.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")

    if model:
        # Dense near the find and on the sheet, sparse on the wide table: like a real cloud.
        per_face = {0: 40, 1: 2500, 2: 900} if scene == "find" else {0: 60}
        points, colours = sample_surfaces(faces, per_face, np.random.default_rng(7))
        truth["model"] = write_colmap_model(out / "model", truth, points, colours)
        truth["model"]["points"] = len(points)
    return truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="scans/_testscene", help="output folder")
    parser.add_argument("--scene", choices=("room", "find"), default="room")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--supersample",
        type=int,
        default=0,
        help="render at N x resolution and downsample (default: 2 for find, 1 for room)",
    )
    parser.add_argument(
        "--model",
        action="store_true",
        help="also write the ground-truth COLMAP model to <out>/model (tests scale and report "
        "code without running COLMAP)",
    )
    args = parser.parse_args()
    out = Path(args.out)
    truth = write_scene(out, args.scene, args.frames, args.width, args.height, args.supersample, args.model)
    print(f"Wrote {args.frames} frames to {out.resolve()}")
    if "model" in truth:
        print(f"Wrote ground-truth model ({truth['model']['points']} points, "
              f"{truth['model']['unitsPerScene']:.6g} model units per {truth['units']}) to {out / 'model'}")


if __name__ == "__main__":
    main()
