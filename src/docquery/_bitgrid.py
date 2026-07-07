"""Deterministic instruction-encoding bit-grid extraction from PDF geometry.

ISA reference manuals draw instruction encodings as a row of bit numbers
(31..16 / 15..0) above a row of boxed cell values (fixed bits like ``0``/``1``/
``(0)`` and field names like ``Rn``/``imm3``). Text-stream loaders flatten those
boxes into a jumbled vertical token soup, destroying the bit↔value association
— which is exactly the information an ISA extraction pipeline needs most.

This module recovers the grids from word coordinates (via pymupdf): find bit-
number header rows, snap the value row's tokens to the nearest bit column, and
render each encoding as one deterministic, machine-parseable text line::

    ENCODING 32-bit: bits[31:25]=0001100 S[24:24] Rn[19:16] ?[15:12] Rm[3:0]

``bits[hi:lo]=…`` are fixed opcode bits, ``name[hi:lo]`` are named fields, and
``?[hi:lo]`` marks columns whose label could not be read. Downstream consumers
can parse these lines back into bit fields without any LLM involvement.

Like ``_pdf.py``, this module imports only pymupdf so it stays cheap to import,
and every public function degrades to "no encodings found" on any failure.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# A word is (x0, y0, x1, y1, text, ...) — the first five slots of a pymupdf
# ``page.get_text("words")`` tuple. Extra trailing slots are ignored.
Word = Sequence[Any]

# Cell values that mean "this bit is fixed". ``(0)``/``(1)`` are ARM's
# should-be-zero/one notation — still a decode constraint.
_FIXED_TOKENS = {"0": "0", "(0)": "0", "1": "1", "(1)": "1"}

_MAX_BIT = 63          # highest bit number recognised in a header row
_MIN_HEADER_RUN = 6    # a header row needs at least this many bit numbers
_VALUE_ROW_GAP = 16.0  # max pts below a header row to look for the value row

_ENCODING_LINE_RE = re.compile(r"^ENCODING (\d+)-bit: ", re.MULTILINE)


def _rows_by_y(words: "list[Word]") -> "dict[int, list[Word]]":
    """Group words into visual rows keyed by rounded top-y, sorted by x."""
    rows: dict[int, list[Word]] = {}
    for w in words:
        rows.setdefault(round(float(w[1])), []).append(w)
    return {y: sorted(ws, key=lambda w: float(w[0])) for y, ws in rows.items()}


def _header_runs(row: "list[Word]") -> "list[list[tuple[float, int]]]":
    """Return the bit-number header runs in *row* as ``[[(x_center, bit), …]]``.

    A header run is a strictly descending left-to-right sequence of at least
    ``_MIN_HEADER_RUN`` integers ≤ ``_MAX_BIT`` (e.g. ``31 30 … 16``). One row
    can hold several runs: ARMv7 draws a 32-bit encoding as two side-by-side
    16-bit grids, i.e. ``15 14 … 0 15 14 … 0`` on a single line.
    """
    cells = [
        ((float(w[0]) + float(w[2])) / 2, int(str(w[4])))
        for w in row
        if str(w[4]).isdigit() and int(str(w[4])) <= _MAX_BIT
    ]
    runs: list[list[tuple[float, int]]] = []
    i = 0
    while i < len(cells):
        j = i + 1
        while j < len(cells) and cells[j][1] < cells[j - 1][1]:
            j += 1
        if j - i >= _MIN_HEADER_RUN:
            runs.append(cells[i:j])
        i = j
    return runs


def _interpolate_cells(cells: "list[tuple[float, int]]") -> "list[tuple[float, int]]":
    """Fill bit-number gaps in a header run by linear x interpolation.

    Some manuals (ARMv8) label only field-boundary bits (``31 30 24 23 … 5 4 0``);
    the boxes for the unlabeled bits still exist, evenly spaced between their
    labeled neighbours. Dense headers pass through unchanged.
    """
    out: list[tuple[float, int]] = []
    for (x1, b1), (x2, b2) in zip(cells, cells[1:]):
        out.append((x1, b1))
        span = b1 - b2
        for step in range(1, span):
            out.append((x1 + (x2 - x1) * step / span, b1 - step))
    if cells:
        out.append(cells[-1])
    return out


def _decode_value_row(
    header: "list[tuple[float, int]]",
    row: "list[Word]",
) -> "list[dict[str, Any]]":
    """Assign the value row's tokens to bit columns; return ordered segments.

    Each segment is ``{"name": str|None, "hi": int, "lo": int, "value":
    str|None}`` in hi→lo order: ``value`` set for fixed-bit runs, ``name`` for
    labelled fields, neither for unreadable columns.
    """
    centers = {bit: xc for xc, bit in header}
    bits_desc = sorted(centers, reverse=True)

    # Typical column width → tolerance for label-to-column matching.
    xs = sorted(centers.values())
    gaps = [b - a for a, b in zip(xs, xs[1:])]
    tol = (sorted(gaps)[len(gaps) // 2] / 2 + 2) if gaps else 8.0

    fixed: dict[int, str] = {}
    labels: list[tuple[float, str]] = []
    for w in row:
        text = str(w[4]).strip()
        if not text:
            continue
        xc = (float(w[0]) + float(w[2])) / 2
        if text in _FIXED_TOKENS:
            bit = min(bits_desc, key=lambda b: abs(centers[b] - xc))
            fixed[bit] = _FIXED_TOKENS[text]
        else:
            labels.append((xc, text))

    segments: list[dict[str, Any]] = []
    i = 0
    while i < len(bits_desc):
        # Contiguous run sharing fixed-ness, split further by labels below.
        is_fixed = bits_desc[i] in fixed
        j = i
        while (
            j < len(bits_desc)
            and (bits_desc[j] in fixed) == is_fixed
            and (j == i or bits_desc[j] == bits_desc[j - 1] - 1)
        ):
            j += 1
        run = bits_desc[i:j]

        if is_fixed:
            segments.append({
                "name": None, "hi": run[0], "lo": run[-1],
                "value": "".join(fixed[b] for b in run),
            })
        else:
            left = min(centers[run[0]], centers[run[-1]]) - tol
            right = max(centers[run[0]], centers[run[-1]]) + tol
            in_run = [(xc, t) for xc, t in labels if left <= xc <= right]
            if not in_run:
                segments.append({"name": None, "hi": run[0], "lo": run[-1], "value": None})
            elif len(in_run) == 1:
                segments.append({"name": in_run[0][1], "hi": run[0], "lo": run[-1], "value": None})
            else:
                # Several labels share the run: give each bit the nearest label,
                # then split the run wherever the label changes.
                assigned = [
                    min(in_run, key=lambda lt: abs(lt[0] - centers[b]))[1]
                    for b in run
                ]
                k = 0
                while k < len(run):
                    m = k
                    while m < len(run) and assigned[m] == assigned[k]:
                        m += 1
                    segments.append({"name": assigned[k], "hi": run[k], "lo": run[m - 1], "value": None})
                    k = m
        i = j
    return segments


def _shift_segments(segments: "list[dict[str, Any]]", by: int) -> "list[dict[str, Any]]":
    return [{**s, "hi": s["hi"] + by, "lo": s["lo"] + by} for s in segments]


def encodings_from_words(words: "list[Word]") -> "list[dict[str, Any]]":
    """Extract encodings from one page's words.

    Returns ``[{"width": int, "segments": [segment, …]}]``. Two drawing styles
    for wide encodings are recombined into one encoding:

    - stacked halves: header ``31..16`` above header ``15..0`` (ARMv8 style)
    - side-by-side halves: ``15..0  15..0`` on one row, left half = upper bits
      (ARMv7 Thumb-2 style)
    """
    rows = _rows_by_y(words)
    ys = sorted(rows)

    # (y, x_left, cells) per header run; a row can contribute several runs.
    headers: list[tuple[int, float, list[tuple[float, int]]]] = []
    header_ys: set[int] = set()
    for y in ys:
        for cells in _header_runs(rows[y]):
            headers.append((y, cells[0][0], cells))
            header_ys.add(y)

    halves: list[dict[str, Any]] = []
    for y, x_left, cells in headers:
        candidates = [
            vy for vy in ys
            if y < vy <= y + _VALUE_ROW_GAP and vy not in header_ys
        ]
        if not candidates:
            continue
        cells = _interpolate_cells(cells)
        # Restrict the value row to this run's x-span so side-by-side grids
        # don't claim each other's tokens.
        xs = [xc for xc, _ in cells]
        gaps = sorted(b - a for a, b in zip(sorted(xs), sorted(xs)[1:]))
        tol = (gaps[len(gaps) // 2] / 2 + 2) if gaps else 8.0
        row_words = [
            w for w in rows[candidates[0]]
            if min(xs) - tol <= (float(w[0]) + float(w[2])) / 2 <= max(xs) + tol
        ]
        segments = _decode_value_row(cells, row_words)
        bits = [b for _, b in cells]
        halves.append({
            "y": y, "x": x_left,
            "hi": max(bits), "lo": min(bits), "segments": segments,
        })

    halves.sort(key=lambda h: (h["y"], h["x"]))

    encodings: list[dict[str, Any]] = []
    k = 0
    while k < len(halves):
        h = halves[k]
        nxt = halves[k + 1] if k + 1 < len(halves) else None
        if (
            nxt and nxt["y"] == h["y"] and h["lo"] == 0 and nxt["lo"] == 0
            and h["hi"] == nxt["hi"]
        ):
            # Side-by-side halves on one row: left half is the upper bits.
            width = 2 * (h["hi"] + 1)
            encodings.append({
                "width": width,
                "segments": _shift_segments(h["segments"], h["hi"] + 1) + nxt["segments"],
            })
            k += 2
        elif h["lo"] > 0 and nxt and nxt["hi"] == h["lo"] - 1 and nxt["lo"] == 0:
            # Stacked halves: header bit numbers are already absolute.
            encodings.append({
                "width": h["hi"] + 1,
                "segments": h["segments"] + nxt["segments"],
            })
            k += 2
        elif h["lo"] == 0:
            encodings.append({"width": h["hi"] + 1, "segments": h["segments"]})
            k += 1
        else:
            k += 1  # dangling upper half — incomplete grid, drop it
    return encodings


def render_encoding_lines(encodings: "list[dict[str, Any]]") -> "list[str]":
    """Render encodings as deterministic one-line text blocks (see module doc)."""
    lines: list[str] = []
    for enc in encodings:
        parts: list[str] = []
        for seg in enc["segments"]:
            span = f"[{seg['hi']}:{seg['lo']}]"
            if seg.get("value"):
                parts.append(f"bits{span}={seg['value']}")
            elif seg.get("name"):
                parts.append(f"{seg['name']}{span}")
            else:
                parts.append(f"?{span}")
        lines.append(f"ENCODING {enc['width']}-bit: " + " ".join(parts))
    return lines


def extract_page_encodings(page: Any) -> "list[dict[str, Any]]":
    """Extract encodings from a pymupdf page object."""
    return encodings_from_words(page.get_text("words"))


def extract_document_encodings(pdf_path: "str | Path") -> "dict[int, list[str]]":
    """Return ``{1-based page: [rendered ENCODING lines]}`` for the whole PDF.

    Pages without a decodable bit grid are omitted. Returns ``{}`` when the
    file cannot be opened or pymupdf is unavailable — callers just skip the
    encoding pass.
    """
    try:
        import fitz  # pymupdf
    except ImportError:  # pragma: no cover - dependency is declared
        logger.warning("pymupdf not available; skipping bit-grid extraction")
        return {}

    result: dict[int, list[str]] = {}
    try:
        with fitz.open(str(pdf_path)) as doc:
            for i, page in enumerate(doc, start=1):
                try:
                    encodings = extract_page_encodings(page)
                except Exception as exc:  # noqa: BLE001 - one bad page ≠ no grids
                    logger.debug("bit-grid extraction failed on page %d of %s: %s", i, pdf_path, exc)
                    continue
                # Only keep grids that pin at least one bit — a grid with no
                # fixed bits carries no decode information.
                encodings = [
                    e for e in encodings
                    if any(seg.get("value") for seg in e["segments"])
                ]
                if encodings:
                    result[i] = render_encoding_lines(encodings)
    except Exception as exc:  # noqa: BLE001 - any pymupdf failure → no grids
        logger.warning("extract_document_encodings failed for %s: %s", pdf_path, exc)
        return {}
    if result:
        total = sum(len(v) for v in result.values())
        logger.info("bit-grid extraction: %d encodings on %d pages in %s", total, len(result), pdf_path)
    return result
