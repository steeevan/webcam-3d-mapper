"""The printed marker board that gives a scan real-world scale.

A single webcam cannot measure distance: structure-from-motion recovers shape only up to an
unknown scale. Putting the find on a sheet of markers of known size fixes that. After
reconstruction the markers are found in the frames, triangulated with COLMAP's camera poses
(``scale.py``), and their known edge length converts model units to millimetres.

The board is a ring of 14 ArUco markers (4x4 bits, ``DICT_4X4_50``, ids 0-13) around an empty
area where the find goes. The choices were measured, not guessed (see README):

* **30 mm markers.** In a synthetic sweep with the real webcam's intrinsics (f = 696 px at
  1280x720, blur sigma 1.6 px, JPEG quality 90), 30 mm markers were found in 100% of views up to
  45 cm away and 60 degrees from face-on; 20 mm markers dropped to 25-50% at 45 cm.
* **4x4 bits**, the coarsest grid, so each cell stays as large as possible in the image.
* **Only board ids count.** 0 of the 334 frames in the real scans produced any detection.

Layout coordinates are nominal millimetres from the top-left corner of marker 0, x to the
right and y down, as printed. A board printed at a different size scales uniformly, so the
measured marker size is all that is needed to correct it.
"""

from __future__ import annotations

import functools

import cv2
import numpy as np

from .pdf import A4_MM, LETTER_MM, PdfDocument

BOARD_NAME = "ring14-v1"
DICTIONARY_ID = cv2.aruco.DICT_4X4_50
DICTIONARY_NAME = "DICT_4X4_50"
#: Edge of the black square, as designed. The user can enter what their print measures.
MARKER_MM = 30.0
MARKER_BITS = 4

#: Marker left edges and top edges on the grid the ring is cut from.
_COLUMNS = (0.0, 45.0, 90.0, 135.0)
_ROWS = (0.0, 42.5, 85.0, 127.5, 170.0)
RING_WIDTH = _COLUMNS[-1] + MARKER_MM   # 165 mm
RING_HEIGHT = _ROWS[-1] + MARKER_MM     # 200 mm
#: The free area inside the ring: x0, y0, x1, y1.
INNER_AREA = (MARKER_MM, MARKER_MM, _COLUMNS[-1], _ROWS[-1])


def _ring() -> dict[int, tuple[float, float]]:
    """Marker id -> top-left corner, numbered clockwise from the top-left marker."""
    last_col, last_row = len(_COLUMNS) - 1, len(_ROWS) - 1
    cells = [(c, 0) for c in range(len(_COLUMNS))]                        # top, left to right
    cells += [(last_col, r) for r in range(1, last_row)]                  # right, downwards
    cells += [(c, last_row) for c in range(last_col, -1, -1)]             # bottom, right to left
    cells += [(0, r) for r in range(last_row - 1, 0, -1)]                 # left, upwards
    return {marker_id: (_COLUMNS[c], _ROWS[r]) for marker_id, (c, r) in enumerate(cells)}


MARKERS: dict[int, tuple[float, float]] = _ring()
MARKER_IDS = frozenset(MARKERS)

#: ArUco reports corners clockwise from the marker's own top-left.
_CORNER_OFFSETS = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))


def corner_positions(marker_mm: float = MARKER_MM) -> dict[tuple[int, int], tuple[float, float]]:
    """(marker id, corner index) -> position in millimetres on a board whose markers measure
    ``marker_mm``. Everything scales with the marker because a printer scales the whole page."""
    factor = marker_mm / MARKER_MM
    return {
        (marker_id, k): ((x + dx * MARKER_MM) * factor, (y + dy * MARKER_MM) * factor)
        for marker_id, (x, y) in MARKERS.items()
        for k, (dx, dy) in enumerate(_CORNER_OFFSETS)
    }


@functools.cache
def _dictionary() -> cv2.aruco.Dictionary:
    return cv2.aruco.getPredefinedDictionary(DICTIONARY_ID)


def marker_cells(marker_id: int) -> np.ndarray:
    """``(6, 6)`` bools, True where the marker is black: the 4x4 code plus its black border."""
    image = cv2.aruco.generateImageMarker(_dictionary(), marker_id, MARKER_BITS + 2, borderBits=1)
    return image < 128


@functools.cache
def _detector() -> cv2.aruco.ArucoDetector:
    parameters = cv2.aruco.DetectorParameters()
    # Sub-pixel corners: the scale is only as good as these positions.
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(_dictionary(), parameters)


@functools.cache
def _quick_detector() -> cv2.aruco.ArucoDetector:
    return cv2.aruco.ArucoDetector(_dictionary(), cv2.aruco.DetectorParameters())


def count(gray: np.ndarray) -> int:
    """How many board markers an image shows, without corner refinement: for live feedback
    while capturing, where only presence matters."""
    _, ids, _ = _quick_detector().detectMarkers(gray)
    return 0 if ids is None else len(set(ids.ravel().tolist()) & MARKER_IDS)


def detect(gray: np.ndarray) -> dict[int, np.ndarray]:
    """Board markers in a greyscale image: id -> ``(4, 2)`` corner pixels.

    Returned in **COLMAP's pixel convention**, where the centre of the top-left pixel is
    (0.5, 0.5); OpenCV puts it at (0, 0). Mixing the two would shift every corner by half a
    pixel against the intrinsics. Ids that are not on the board are ignored, and so is an id
    seen twice in one image (a reflection, or two boards), since neither can be trusted.
    """
    corners, ids, _ = _detector().detectMarkers(gray)
    if ids is None:
        return {}
    found: dict[int, np.ndarray] = {}
    seen: set[int] = set()
    for marker_corners, marker_id in zip(corners, ids.ravel().tolist()):
        if marker_id in seen:
            found.pop(marker_id, None)
            continue
        seen.add(marker_id)
        if marker_id in MARKER_IDS:
            found[marker_id] = marker_corners.reshape(4, 2).astype(np.float64) + 0.5
    return found


# --------------------------------------------------------------------------------------
# Printable board
# --------------------------------------------------------------------------------------

PAPERS: dict[str, tuple[float, float]] = {"a4": A4_MM, "letter": LETTER_MM}

INSTRUCTIONS = (
    "Print at 100% (\"Actual size\"). Turn off \"Fit to page\" and any scaling.",
    "Check the bar below with a ruler: it must be exactly 100 mm. If it is not, measure the edge "
    "of one black marker square and enter that size in Settings > Marker board.",
    "Lay the sheet flat (tape the corners) and place the find inside the dashed area.",
    "Circle the find slowly with the camera, keeping several markers in view. Do not move the "
    "sheet or the find during the scan.",
)


def _page_layout(paper: str) -> dict[str, float]:
    width, height = PAPERS[paper]
    return {
        "width": width,
        "height": height,
        "ring_x": (width - RING_WIDTH) / 2,
        "ring_y": 14.0,
        "bar_x": (width - 100.0) / 2,
    }


def _cell_rects(ring_x: float, ring_y: float) -> list[tuple[float, float, float, float]]:
    """Black cells of every marker as page rectangles, merging horizontal runs."""
    cell = MARKER_MM / (MARKER_BITS + 2)
    rects = []
    for marker_id, (mx, my) in MARKERS.items():
        cells = marker_cells(marker_id)
        for row in range(cells.shape[0]):
            col = 0
            while col < cells.shape[1]:
                if not cells[row, col]:
                    col += 1
                    continue
                start = col
                while col < cells.shape[1] and cells[row, col]:
                    col += 1
                rects.append(
                    (ring_x + mx + start * cell, ring_y + my + row * cell, (col - start) * cell, cell)
                )
    return rects


def board_pdf(paper: str = "a4") -> bytes:
    """The printable board as a one-page PDF at exact physical size (vector, no raster)."""
    layout = _page_layout(paper)
    doc = PdfDocument(title=f"Scale board {BOARD_NAME} ({paper.upper()})", producer="Webcam 3D Mapper")
    page = doc.add_page((layout["width"], layout["height"]))
    rx, ry = layout["ring_x"], layout["ring_y"]
    grey = (0.55, 0.55, 0.55)

    page.filled_rects(_cell_rects(rx, ry))
    x0, y0, x1, y1 = INNER_AREA
    page.rect(rx + x0 + 6, ry + y0 + 6, x1 - x0 - 12, y1 - y0 - 12, stroke=grey, width=0.3, dash=2.0)
    page.text(rx + RING_WIDTH / 2, ry + y0 + 12, "Place the find here", 8, color=grey, align="center")

    y = ry + RING_HEIGHT + 9
    page.text(rx, y, "Scale board", 11, bold=True)
    page.text(rx + RING_WIDTH, y, f"markers {MARKER_MM:.0f} mm, {BOARD_NAME}", 8, color=grey, align="right")
    y += 6
    for number, line in enumerate(INSTRUCTIONS, start=1):
        page.text(rx, y, f"{number}.", 8.5)
        y = page.paragraph(rx + 5, y, line, RING_WIDTH - 5, size=8.5) + 0.8

    # 100 mm verification bar with 10 mm ticks: printers rescale pages without saying so.
    bx, by = layout["bar_x"], y + 4
    page.rect(bx, by, 100.0, 1.2, fill=(0, 0, 0))
    for tick in range(0, 101, 10):
        page.line(bx + tick, by - (2.5 if tick % 50 == 0 else 1.5), bx + tick, by, width=0.25)
    page.text(bx, by + 5, "0", 8, align="center")
    page.text(bx + 100, by + 5, "100 mm", 8, align="center")
    page.text(bx + 50, by + 5, "must measure exactly 100 mm", 7.5, color=grey, align="center")
    page.text(
        layout["width"] / 2,
        layout["height"] - 8,
        f"ArUco {DICTIONARY_NAME} ids 0-{len(MARKERS) - 1} - {paper.upper()} - Webcam 3D Mapper",
        6.5,
        color=grey,
        align="center",
    )
    return doc.to_bytes()
