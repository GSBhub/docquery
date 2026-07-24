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
_WIDTH_ROW_GAP = 26.0  # max pts below the value row to look for a field-width row
_WIDTH_MATCH_TOL = 16.0  # max pts between a field label and its width number

_ENCODING_LINE_RE = re.compile(r"^ENCODING (\d+)-bit: ", re.MULTILINE)

# Segment tokens within an ENCODING line, in the order the renderer emits them.
_SEG_FIXED_RE = re.compile(r"bits\[(\d+):(\d+)\]=([01]+)")
_SEG_UNREAD_RE = re.compile(r"\?\[(\d+):(\d+)\]")
_SEG_FIELD_RE = re.compile(r"(\S+)\[(\d+):(\d+)\]")


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


def _width_row_words(
    value_y: int, ys: "list[int]", rows: "dict[int, list[Word]]",
    x_lo: float, x_hi: float, header_ys: "set[int]",
) -> "list[Word]":
    """The explicit field-width row below the value row, or ``[]``.

    TI/DSP manuals print a row of field widths (``3 1 5 5 5 1 7 …``) under the
    value row, each number centred on its field. Returns those integer words
    when the nearest row below the value row is mostly small integers.
    """
    cands = [wy for wy in ys if value_y < wy <= value_y + _WIDTH_ROW_GAP and wy not in header_ys]
    if not cands:
        return []
    words = [
        w for w in rows[cands[0]]
        if x_lo <= (float(w[0]) + float(w[2])) / 2 <= x_hi
    ]
    ints = [w for w in words if str(w[4]).strip().isdigit()]
    if len(ints) >= 2 and len(ints) >= 0.6 * len(words):
        return ints
    return []


def _decode_with_widths(
    max_bit: int, value_row: "list[Word]", width_row: "list[Word]",
) -> "list[dict[str, Any]] | None":
    """Assign field boundaries from an explicit width row (TI/DSP style).

    Value-row tokens are walked MSB→LSB; each named field takes its width from
    the nearest width-row integer, fixed ``0``/``1`` tokens are width 1. This is
    exact — no x-interpolation guesswork — but only used when the widths tile
    ``[max_bit:0]`` precisely; otherwise the caller falls back to x-snapping.
    """
    widths = [
        ((float(w[0]) + float(w[2])) / 2, int(str(w[4]).strip()))
        for w in width_row
        if str(w[4]).strip().isdigit() and 1 <= int(str(w[4]).strip()) <= max_bit + 1
    ]
    if len(widths) < 2:
        return None

    toks = [
        ((float(w[0]) + float(w[2])) / 2, str(w[4]).strip())
        for w in value_row if str(w[4]).strip()
    ]
    toks.sort(key=lambda t: t[0])

    seq: list[tuple[str | None, str | None, int]] = []  # (name, value, width)
    for xc, text in toks:
        if text in _FIXED_TOKENS:
            seq.append((None, _FIXED_TOKENS[text], 1))
            continue
        near = min(widths, key=lambda wv: abs(wv[0] - xc))
        if abs(near[0] - xc) > _WIDTH_MATCH_TOL:
            return None  # a named field with no width number → can't be exact
        seq.append((text, None, near[1]))

    if sum(w for _, _, w in seq) != max_bit + 1:
        return None

    segments: list[dict[str, Any]] = []
    hi = max_bit
    for name, value, w in seq:
        lo = hi - w + 1
        if value is not None and segments and segments[-1].get("value") is not None:
            segments[-1]["value"] += value  # merge an adjacent fixed-bit run
            segments[-1]["lo"] = lo
        else:
            segments.append({"name": name, "hi": hi, "lo": lo, "value": value})
        hi = lo - 1
    return segments


def _shift_segments(segments: "list[dict[str, Any]]", by: int) -> "list[dict[str, Any]]":
    return [{**s, "hi": s["hi"] + by, "lo": s["lo"] + by} for s in segments]


def _opfield_values(words: "list[Word]") -> "list[str]":
    """Binary values from a TI ``Opfield`` opcode-map column, or ``[]``.

    C6x instructions share one unit diagram whose ``op`` field is a *variable*;
    the actual opcode value per operand-type is listed in the ``Opfield`` column
    of the "Opcode map field used…" table below the diagram (e.g. ``000 0011``,
    split across two words). Returns one concatenated bit-string per table row.
    """
    hdrs = [w for w in words if str(w[4]).strip().lower() == "opfield"]
    if not hdrs:
        return []
    h = hdrs[0]
    hxc = (float(h[0]) + float(h[2])) / 2
    hy = float(h[1])
    by_row: dict[int, list[tuple[float, str]]] = {}
    for w in words:
        t = str(w[4]).strip()
        if not re.fullmatch(r"[01]+", t):
            continue
        xc = (float(w[0]) + float(w[2])) / 2
        if float(w[1]) > hy and abs(xc - hxc) <= 30.0:
            by_row.setdefault(round(float(w[1])), []).append((xc, t))
    values: list[str] = []
    for y in sorted(by_row):
        parts = sorted(by_row[y], key=lambda p: p[0])
        values.append("".join(t for _, t in parts))
    return values


def _apply_opfields(
    encodings: "list[dict[str, Any]]", opvals: "list[str]",
) -> "list[dict[str, Any]]":
    """Expand each encoding's variable ``op`` field into one encoding per Opfield
    value (constraining ``op`` to that value). Encodings without an identifiable
    ``op`` field pass through unchanged."""
    if not opvals:
        return encodings
    val_widths = {len(v) for v in opvals}
    out: list[dict[str, Any]] = []
    for enc in encodings:
        segs = enc["segments"]
        idx = next(
            (i for i, s in enumerate(segs)
             if s.get("value") is None and (s.get("name") or "").lower() in ("op", "opcode")),
            None,
        )
        if idx is None:  # fall back: a unique variable field whose width matches
            cand = [
                i for i, s in enumerate(segs)
                if s.get("value") is None and s.get("name")
                and (s["hi"] - s["lo"] + 1) in val_widths
            ]
            idx = cand[0] if len(cand) == 1 else None
        if idx is None:
            out.append(enc)
            continue
        width = segs[idx]["hi"] - segs[idx]["lo"] + 1
        matching = [v for v in opvals if len(v) == width]
        if not matching:
            out.append(enc)
            continue
        for v in matching:
            new_segs = [dict(s) for s in segs]
            new_segs[idx] = {"name": None, "hi": segs[idx]["hi"], "lo": segs[idx]["lo"], "value": v}
            out.append({"width": enc["width"], "segments": new_segs})
    return out


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
        value_y = candidates[0]
        raw_bits = [b for _, b in cells]
        max_bit, min_bit = max(raw_bits), min(raw_bits)
        cells = _interpolate_cells(cells)
        # Restrict the value row to this run's x-span so side-by-side grids
        # don't claim each other's tokens.
        xs = [xc for xc, _ in cells]
        gaps = sorted(b - a for a, b in zip(sorted(xs), sorted(xs)[1:]))
        tol = (gaps[len(gaps) // 2] / 2 + 2) if gaps else 8.0
        x_lo, x_hi = min(xs) - tol, max(xs) + tol
        row_words = [
            w for w in rows[value_y]
            if x_lo <= (float(w[0]) + float(w[2])) / 2 <= x_hi
        ]
        # Prefer an explicit field-width row (exact) over x-interpolation.
        segments = None
        if min_bit == 0:
            width_words = _width_row_words(value_y, ys, rows, x_lo, x_hi, header_ys)
            if width_words:
                segments = _decode_with_widths(max_bit, row_words, width_words)
        if segments is None:
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

    # Constrain a shared diagram's variable `op` field from the page's Opfield
    # opcode-map table, yielding one encoding per opcode value (TI C6x).
    return _apply_opfields(encodings, _opfield_values(words))


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


def parse_encoding_line(line: str) -> "dict[str, Any] | None":
    """Parse one rendered ENCODING line back into ``{"width", "segments"}``.

    Exact inverse of :func:`render_encoding_lines` for a single encoding;
    returns ``None`` for anything that is not a well-formed ENCODING line.
    """
    line = line.strip()
    m = _ENCODING_LINE_RE.match(line)
    if not m:
        return None
    segments: list[dict[str, Any]] = []
    for token in line[m.end():].split():
        if fixed := _SEG_FIXED_RE.fullmatch(token):
            hi, lo, value = int(fixed[1]), int(fixed[2]), fixed[3]
            segments.append({"name": None, "hi": hi, "lo": lo, "value": value})
        elif unread := _SEG_UNREAD_RE.fullmatch(token):
            segments.append({"name": None, "hi": int(unread[1]), "lo": int(unread[2]), "value": None})
        elif field := _SEG_FIELD_RE.fullmatch(token):
            segments.append({"name": field[1], "hi": int(field[2]), "lo": int(field[3]), "value": None})
        else:
            return None
    if not segments:
        return None
    return {"width": int(m[1]), "segments": segments}


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
