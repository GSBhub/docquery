"""Structured-region query API: parse kind-tagged documents back into records.

Detectors (:mod:`docquery._bitgrid`, :mod:`docquery._tables`) store their
recoveries as machine-parseable text lines inside kind-tagged Documents.
:func:`get_structures` scans the store for those documents and parses the
lines back into structured records, so callers can work with bit fields and
table rows programmatically — no LLM involved. Pure metadata/content scan,
like :mod:`docquery._coverage`.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from docquery.config import ENTITY_PREFIX

if TYPE_CHECKING:
    from docquery.config import Settings

logger = logging.getLogger(__name__)

# Bit ranges in table cells: "31:16", "[7:0]", "3", "15-8". Single number →
# hi == lo.
_BIT_RANGE_RE = re.compile(r"^\[?(\d{1,2})(?:\s*[:\-–]\s*(\d{1,2}))?\]?,?$")


def _annotate_bit_ranges(record: dict[str, Any]) -> dict[str, Any]:
    """Add ``{"hi", "lo"}`` ints when a table row's cell holds a bit range.

    The raw cell string is kept; the parsed range lands under ``"bits"``. This
    is what makes register-field rows structurally usable without a bespoke
    bit-table detector.
    """
    for value in record.values():
        if not isinstance(value, str):
            continue
        if m := _BIT_RANGE_RE.match(value.strip()):
            hi, lo = int(m[1]), int(m[2] if m[2] is not None else m[1])
            if lo > hi:
                hi, lo = lo, hi
            record["bits"] = {"hi": hi, "lo": lo}
            break
    return record


def _parse_lines(lines: list[str]) -> list[dict[str, Any]]:
    """Parse a structure doc's machine lines into records by line shape.

    ``ENCODING`` lines each become one ``{"width", "segments"}`` record;
    ``TABLE``/``ROW`` blocks become one ``{"kind", "columns", "rows"}`` record
    per table, with bit-range cells annotated. Unparseable lines are skipped.
    """
    from docquery._bitgrid import parse_encoding_line
    from docquery._tables import parse_table_lines

    records: list[dict[str, Any]] = []
    table_lines: list[str] = []
    for line in lines:
        if enc := parse_encoding_line(line):
            records.append(enc)
        elif line.strip().startswith(("TABLE ", "ROW ")):
            table_lines.append(line)
    for table in parse_table_lines(table_lines):
        table["rows"] = [_annotate_bit_ranges(r) for r in table["rows"]]
        records.append(table)
    return records


def get_structures(
    settings: "Settings",
    kind: str | None = None,
    collection_name: str = "db_knowledge",
) -> list[dict[str, Any]]:
    """Return parsed structured regions from the store, ordered by source/page.

    Each item is ``{"kind", "source", "page", "section",
    "entities": {entity_type: [names]}, "lines": [str], "records": [dict]}``
    — ``lines`` holds the raw machine lines (headings/intro excluded),
    ``records`` their parsed form. Returns ``[]`` for an empty/unreachable
    store or when no document carries the requested kind.
    """
    from docquery._ingest import _build_chroma
    vs = _build_chroma(settings, collection_name)
    try:
        collection = vs._collection  # type: ignore[attr-defined]
        kwargs = {"where": {"kind": kind}} if kind else {}
        raw = collection.get(include=["documents", "metadatas"], **kwargs)
    except Exception:
        logger.warning("get_structures: collection %r not accessible", collection_name)
        return []

    results: list[dict[str, Any]] = []
    for content, meta in zip(raw.get("documents") or [], raw.get("metadatas") or [],
                             strict=False):
        meta = meta or {}
        if not content or not meta.get("kind"):
            continue
        lines = [
            ln for ln in content.splitlines()
            if ln.strip().startswith(("ENCODING ", "TABLE ", "ROW "))
        ]
        entities = {
            k[len(ENTITY_PREFIX):]: [n.strip() for n in str(v).split(";") if n.strip()]
            for k, v in meta.items()
            if k.startswith(ENTITY_PREFIX) and v
        }
        results.append({
            "kind": meta.get("kind"),
            "source": meta.get("source"),
            "page": meta.get("page"),
            "section": meta.get("section"),
            "entities": entities,
            "lines": lines,
            "records": _parse_lines(lines),
        })

    results.sort(key=lambda r: (str(r["source"]),
                                r["page"] if isinstance(r["page"], int) else 0))
    return results
