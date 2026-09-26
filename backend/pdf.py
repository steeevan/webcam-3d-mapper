"""A deliberately small PDF writer: text, lines, rectangles and images on fixed-size pages.

Why not a library: none is installed, and the two documents this app prints (the find report
and the marker board) need only these primitives. Writing them directly keeps the output
byte-for-byte deterministic for a given input, adds no dependency, and every drawing command
lands in the file uncompressed, so tests can look for text in it.

Coordinates are millimetres from the **top-left** corner of the page, like the layouts that use
them; the conversion to PDF's bottom-left points happens here and nowhere else. Text uses the
standard Helvetica fonts with WinAnsi encoding: Western European characters print as typed,
anything outside that set prints as "?" (embedding a Unicode font is out of scope).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

import numpy as np

PT_PER_MM = 72.0 / 25.4

A4_MM = (210.0, 297.0)
LETTER_MM = (215.9, 279.4)

# Advance widths (1/1000 em) of Helvetica and Helvetica-Bold for ASCII 32..126, from the
# standard Adobe font metrics. Anything else is measured as a digit-width glyph.
_HELVETICA = [
    278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278,
    556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556,
    1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778,
    667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556,
    333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556,
    556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584,
]
_HELVETICA_BOLD = [
    278, 333, 474, 556, 556, 889, 722, 238, 333, 333, 389, 584, 278, 333, 278, 278,
    556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 333, 333, 584, 584, 584, 611,
    975, 722, 722, 722, 722, 667, 611, 778, 722, 278, 556, 722, 611, 833, 722, 778,
    667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 333, 278, 333, 584, 556,
    333, 556, 611, 556, 611, 556, 333, 611, 611, 278, 278, 556, 278, 889, 611, 611,
    611, 611, 389, 556, 333, 611, 556, 778, 556, 556, 500, 389, 280, 389, 584,
]

Color = tuple[float, float, float]
BLACK: Color = (0.0, 0.0, 0.0)


def text_width(text: str, size: float, bold: bool = False) -> float:
    """Width of ``text`` in millimetres at ``size`` points."""
    table = _HELVETICA_BOLD if bold else _HELVETICA
    units = sum(table[ord(c) - 32] if 32 <= ord(c) <= 126 else 556 for c in text)
    return units / 1000.0 * size / PT_PER_MM


def wrap(text: str, width_mm: float, size: float, bold: bool = False) -> list[str]:
    """Greedy word wrap to ``width_mm``. Existing line breaks are kept; a single word longer
    than the line is split by character rather than overflowing the margin."""
    lines: list[str] = []
    for paragraph in text.split("\n"):
        current = ""
        for word in paragraph.split(" "):
            candidate = f"{current} {word}" if current else word
            if text_width(candidate, size, bold) <= width_mm:
                current = candidate
                continue
            if current:
                lines.append(current)
            current = ""
            while text_width(word, size, bold) > width_mm and len(word) > 1:
                cut = len(word)
                while cut > 1 and text_width(word[:cut], size, bold) > width_mm:
                    cut -= 1
                lines.append(word[:cut])
                word = word[cut:]
            current = word
        lines.append(current)
    return lines


def _escape(text: str) -> bytes:
    encoded = text.encode("cp1252", errors="replace")
    return encoded.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def _num(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".") or "0"


@dataclass
class _Image:
    width: int
    height: int
    data: bytes
    filter: str
    colorspace: str


@dataclass
class Page:
    """One page's drawing commands. Create through ``PdfDocument.add_page``."""

    width_mm: float
    height_mm: float
    _ops: list[bytes] = field(default_factory=list)
    _images: list[int] = field(default_factory=list)

    def _x(self, x: float) -> str:
        return _num(x * PT_PER_MM)

    def _y(self, y: float) -> str:
        return _num((self.height_mm - y) * PT_PER_MM)

    def text(
        self,
        x: float,
        y: float,
        text: str,
        size: float = 10,
        bold: bool = False,
        color: Color = BLACK,
        align: str = "left",
    ) -> None:
        """Draw one line of text with its baseline at ``y``."""
        if align != "left":
            width = text_width(text, size, bold)
            x -= width if align == "right" else width / 2
        font = "F2" if bold else "F1"
        self._ops.append(
            b"BT /" + font.encode() + f" {_num(size)} Tf {_rgb(color)} rg "
            f"{self._x(x)} {self._y(y)} Td (".encode() + _escape(text) + b") Tj ET"
        )

    def paragraph(
        self,
        x: float,
        y: float,
        text: str,
        width: float,
        size: float = 9,
        leading: float = 1.3,
        bold: bool = False,
        color: Color = BLACK,
    ) -> float:
        """Wrapped text starting with its first baseline at ``y``. Returns the next free y."""
        step = size * leading / PT_PER_MM
        for line in wrap(text, width, size, bold):
            self.text(x, y, line, size, bold, color)
            y += step
        return y

    def line(
        self, x1: float, y1: float, x2: float, y2: float, width: float = 0.25, color: Color = BLACK
    ) -> None:
        """A straight line; ``width`` in millimetres."""
        self._ops.append(
            f"{_num(width * PT_PER_MM)} w {_rgb(color)} RG "
            f"{self._x(x1)} {self._y(y1)} m {self._x(x2)} {self._y(y2)} l S".encode()
        )

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: Color | None = None,
        stroke: Color | None = None,
        width: float = 0.25,
        dash: float | None = None,
    ) -> None:
        """An axis-aligned rectangle with its top-left corner at (x, y)."""
        ops = [f"{self._x(x)} {self._y(y + h)} {_num(w * PT_PER_MM)} {_num(h * PT_PER_MM)} re"]
        if fill is not None:
            ops.insert(0, f"{_rgb(fill)} rg")
        if stroke is not None:
            ops.insert(0, f"{_num(width * PT_PER_MM)} w {_rgb(stroke)} RG")
            if dash:
                ops.insert(0, f"[{_num(dash * PT_PER_MM)}] 0 d")
        ops.append("B" if fill is not None and stroke is not None else "f" if fill is not None else "S")
        # q/Q keeps the dash pattern and colours from leaking into later drawing.
        self._ops.append(("q " + " ".join(ops) + " Q").encode())

    def filled_rects(self, rects: Iterable[tuple[float, float, float, float]], color: Color = BLACK) -> None:
        """Many filled rectangles in one path - for the marker cells, which must be exact."""
        parts = [f"{_rgb(color)} rg"]
        for x, y, w, h in rects:
            parts.append(f"{self._x(x)} {self._y(y + h)} {_num(w * PT_PER_MM)} {_num(h * PT_PER_MM)} re")
        parts.append("f")
        self._ops.append(" ".join(parts).encode())

    def _place(self, index: int, x: float, y: float, w: float, h: float) -> None:
        self._images.append(index)
        self._ops.append(
            f"q {_num(w * PT_PER_MM)} 0 0 {_num(h * PT_PER_MM)} {self._x(x)} {self._y(y + h)} cm "
            f"/Im{index} Do Q".encode()
        )


class PdfDocument:
    """Collects pages and images, then serialises them with a correct cross-reference table."""

    def __init__(self, title: str = "", producer: str = "", created: datetime | None = None) -> None:
        self.title = title
        self.producer = producer
        self.created = created or datetime.now()
        self.pages: list[Page] = []
        self._images: list[_Image] = []

    def add_page(self, size_mm: tuple[float, float] = A4_MM) -> Page:
        page = Page(*size_mm)
        self.pages.append(page)
        return page

    def image_jpeg(self, page: Page, jpeg: bytes, width: int, height: int, gray: bool,
                   x: float, y: float, w: float, h: float) -> None:
        """Embed JPEG bytes as they are (DCTDecode) - no re-encoding of the photograph."""
        self._images.append(_Image(width, height, jpeg, "DCTDecode", "DeviceGray" if gray else "DeviceRGB"))
        page._place(len(self._images) - 1, x, y, w, h)

    def image_rgb(self, page: Page, pixels: np.ndarray, x: float, y: float, w: float, h: float) -> None:
        """Embed an ``(H, W, 3)`` uint8 RGB array losslessly (FlateDecode)."""
        pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
        height, width = pixels.shape[:2]
        self._images.append(
            _Image(width, height, zlib.compress(pixels.tobytes(), 9), "FlateDecode", "DeviceRGB")
        )
        page._place(len(self._images) - 1, x, y, w, h)

    def to_bytes(self) -> bytes:
        objects: list[bytes] = []

        def add(body: bytes) -> int:
            objects.append(body)
            return len(objects)  # object numbers start at 1

        catalog = add(b"")  # patched below, once the page tree number is known
        pages_ref = add(b"")
        font_regular = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
        font_bold = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")

        image_refs = []
        for image in self._images:
            header = (
                f"<< /Type /XObject /Subtype /Image /Width {image.width} /Height {image.height} "
                f"/ColorSpace /{image.colorspace} /BitsPerComponent 8 /Filter /{image.filter} "
                f"/Length {len(image.data)} >>\nstream\n"
            ).encode()
            image_refs.append(add(header + image.data + b"\nendstream"))

        page_refs = []
        for page in self.pages:
            content = b"\n".join(page._ops)
            content_ref = add(f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream")
            xobjects = " ".join(f"/Im{i} {image_refs[i]} 0 R" for i in sorted(set(page._images)))
            page_refs.append(
                add(
                    (
                        f"<< /Type /Page /Parent {pages_ref} 0 R "
                        f"/MediaBox [0 0 {_num(page.width_mm * PT_PER_MM)} {_num(page.height_mm * PT_PER_MM)}] "
                        f"/Resources << /Font << /F1 {font_regular} 0 R /F2 {font_bold} 0 R >> "
                        f"/XObject << {xobjects} >> >> /Contents {content_ref} 0 R >>"
                    ).encode()
                )
            )

        objects[catalog - 1] = f"<< /Type /Catalog /Pages {pages_ref} 0 R >>".encode()
        kids = " ".join(f"{ref} 0 R" for ref in page_refs)
        objects[pages_ref - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_refs)} >>".encode()
        stamp = self.created.strftime("D:%Y%m%d%H%M%S")
        info = add(
            b"<< /Title (" + _escape(self.title) + b") /Producer (" + _escape(self.producer)
            + f") /CreationDate ({stamp}) >>".encode()
        )

        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = []
        for number, body in enumerate(objects, start=1):
            offsets.append(len(out))
            out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
        xref = len(out)
        out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
        for offset in offsets:
            out += f"{offset:010d} 00000 n \n".encode()
        out += (
            f"trailer\n<< /Size {len(objects) + 1} /Root {catalog} 0 R /Info {info} 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode()
        return bytes(out)


def _rgb(color: Color) -> str:
    return " ".join(_num(max(0.0, min(1.0, c))) for c in color)
