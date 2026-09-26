"""The find record: what would be written on the finds bag, attached to a scan.

Stored in ``scan.json`` under ``"record"``. Everything here is plain text typed by a person, so
the server is strict about it before it is written to disk or printed in a report:

* only the known fields are accepted; anything else is refused, not silently dropped
* control and invisible formatting characters are removed (a right-to-left override can make a
  find number read differently from what is stored), and lone surrogates, which JSON allows but
  UTF-8 cannot encode, never reach the file
* each field has a length limit; an over-long value is refused rather than truncated, so what
  is stored is always exactly what the person saw in the form

The browser shows these values with ``textContent`` only.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, timedelta
from typing import Any

#: Material classes offered in the form. "other" unlocks a free-text description.
MATERIALS: tuple[tuple[str, str], ...] = (
    ("ceramic", "Ceramic"),
    ("lithic", "Stone / lithic"),
    ("bone", "Bone, antler, ivory"),
    ("shell", "Shell"),
    ("metal", "Metal"),
    ("glass", "Glass"),
    ("wood", "Wood"),
    ("other", "Other"),
)
MATERIAL_VALUES = frozenset(value for value, _ in MATERIALS)

#: Field -> maximum length in characters, after cleaning. The form uses the same limits.
FIELD_LIMITS: dict[str, int] = {
    "findNumber": 40,
    "siteCode": 40,
    "context": 40,
    "material": 20,
    "materialOther": 60,
    "dateFound": 10,
    "recorder": 80,
    "notes": 2000,
}
FIELDS: tuple[str, ...] = tuple(FIELD_LIMITS)
MULTILINE_FIELDS = frozenset({"notes"})
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

#: Fields a library search looks in.
SEARCH_FIELDS: tuple[str, ...] = ("findNumber", "siteCode", "context", "material", "materialOther")


class RecordError(ValueError):
    """A record the server refuses to store. ``field`` names the offending input, if one."""

    def __init__(self, message: str, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


def clean_text(value: str, multiline: bool = False) -> str:
    """Normalise to NFC and drop characters that are invisible or break a printed record.

    Removed: control characters (Unicode category Cc; newlines and tabs survive in multi-line
    fields), formatting characters (Cf: bidi overrides, zero-width joiners, soft hyphens) and
    surrogates (Cs). Single-line fields have their whitespace runs collapsed to one space.
    """
    text = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    kept: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if char in "\n\t":
            kept.append(char if multiline else " ")
        elif category in ("Cc", "Cf", "Cs"):
            continue
        else:
            kept.append(char)
    text = "".join(kept)
    if multiline:
        return "\n".join(line.rstrip() for line in text.strip().split("\n"))
    return " ".join(text.split())


def validate_record(payload: Any, today: date | None = None) -> dict[str, str]:
    """Check and normalise a record submitted by the browser. Raises ``RecordError``.

    Every field is optional except the find number. Missing fields are stored as empty strings,
    so a stored record always has the same keys.
    """
    if not isinstance(payload, dict):
        raise RecordError("The record must be a JSON object.")
    unknown = sorted(str(key) for key in payload if key not in FIELD_LIMITS)
    if unknown:
        raise RecordError(f"Unknown field(s): {', '.join(unknown)}", unknown[0])

    record: dict[str, str] = {}
    for field in FIELDS:
        value = payload.get(field)
        if value is None:
            value = ""
        if not isinstance(value, str):
            raise RecordError(f"{field} must be text.", field)
        value = clean_text(value, multiline=field in MULTILINE_FIELDS)
        if len(value) > FIELD_LIMITS[field]:
            raise RecordError(
                f"{field} is too long ({len(value)} characters, at most {FIELD_LIMITS[field]}).",
                field,
            )
        record[field] = value

    if not record["findNumber"]:
        raise RecordError("A find number is required.", "findNumber")

    if record["material"] and record["material"] not in MATERIAL_VALUES:
        raise RecordError(f"Unknown material: {record['material']!r}.", "material")
    if record["material"] != "other":
        record["materialOther"] = ""

    if record["dateFound"]:
        # fromisoformat alone would also take "20260924" and week dates like "2026-W39-4".
        if not _DATE_RE.match(record["dateFound"]):
            raise RecordError("Date found must be a date (YYYY-MM-DD).", "dateFound")
        try:
            found = date.fromisoformat(record["dateFound"])
        except ValueError:
            raise RecordError("Date found is not a real calendar date.", "dateFound") from None
        # One day of slack for a browser in a timezone ahead of this machine.
        if found > (today or date.today()) + timedelta(days=1):
            raise RecordError("Date found is in the future.", "dateFound")
    return record


def load_record(raw: Any) -> dict[str, str] | None:
    """Best-effort reading of a record from ``scan.json``, which a person may have edited.

    Known text fields are cleaned and cut to their limits; anything else is ignored. A record
    without a find number is kept (nothing on disk is thrown away), and the UI shows it as such.
    """
    if not isinstance(raw, dict):
        return None
    record = {}
    for field in FIELDS:
        value = raw.get(field)
        text = value if isinstance(value, str) else ""
        record[field] = clean_text(text, multiline=field in MULTILINE_FIELDS)[: FIELD_LIMITS[field]]
    return record if any(record.values()) else None


def material_label(record: dict[str, str]) -> str:
    """Human-readable material, including the free-text description for "other"."""
    value = record.get("material", "")
    if value == "other":
        return record.get("materialOther") or "Other"
    return dict(MATERIALS).get(value, value)


def record_matches(record: dict[str, str] | None, query: str) -> bool:
    """Library search: every whitespace-separated term must occur in some searchable field.

    Case-insensitive substring matching, so "BK-12" finds "bk-12a" and "ker 3" needs both terms.
    Material matches its value and its label ("lithic" and "stone" both find stone tools).
    """
    terms = clean_text(query).casefold().split()
    if not terms:
        return True
    if not record:
        return False
    haystack = [record.get(field, "").casefold() for field in SEARCH_FIELDS]
    haystack.append(material_label(record).casefold())
    return all(any(term in text for text in haystack) for term in terms)
