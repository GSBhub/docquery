"""Generic rule-driven table extraction from PDF geometry.

Information-dense documents (ISA manuals, datasheets, peripheral references)
carry their most queryable facts in column-aligned tables: register bit-field
maps, interrupt vector tables, pin/function maps, and whatever else a document
defines. Text-stream loaders shred those tables across chunk boundaries and
lose the column structure entirely.

This module recovers such tables from word coordinates (via pymupdf) and
classifies them with configurable :class:`~docquery.config.StructureRule`\\ s —
domain knowledge lives in the rules, not in this code. Each recovered table is
rendered as deterministic, machine-parseable text lines::

    TABLE register_fields: bits | field | description
    ROW 31:16 | RES0 | Reserved
    ROW 7:0 | EN | Enable bits for channels 0-7

The ``TABLE`` line carries the rule's kind and the canonical column keys (the
matched synonym groups' first synonyms), so consumers see stable keys no matter
how the document worded its headers. :func:`parse_table_lines` is the exact
inverse of :func:`render_table_lines`.

Like ``_bitgrid.py``, this module imports only pymupdf (lazily) so it stays
cheap to import, and every public function degrades to "no tables found" on
any failure.
"""

from __future__ import annotations

import logging
import re
from bisect import bisect
from pathlib import Path
from typing import Any, Sequence

# Word-geometry row grouping is generic; shared with the bit-grid detector.
from docquery._bitgrid import _rows_by_y
from docquery.config import StructureRule

logger = logging.getLogger(__name__)

# A word is (x0, y0, x1, y1, text, ...) — pymupdf ``page.get_text("words")``.
Word = Sequence[Any]

_MAX_ROW_GAP_FACTOR = 3.0  # stop after a vertical gap > this many row heights
_MAX_STRIKES = 2           # consecutive non-table rows before the table ends
# A single-row table is real only when the row is dense (a one-register
# field table); sparse one-row candidates are usually prose misreads.
_MIN_SINGLE_ROW_FILLED = 4

_TABLE_LINE_RE = re.compile(r"^TABLE (\S+): (.+)$")

# Footnote/list markers ("a.", "12.") that land in the key column and would
# otherwise pass the single-token key check.
_MARKER_KEY_RE = re.compile(r"^(?:[a-z]|\d{1,2})\.$")

# A "Bit"/"Bits" label glued to the range in the key cell ("Bit 31:16",
# "Bit 2y+1:2y") — the one multi-word key shape that is data, not prose.
_BIT_LABEL_RE = re.compile(r"^bits?$", re.IGNORECASE)


def _is_key_cell(text: str) -> bool:
    """True when *text* is shaped like a data row's key cell.

    One token ("31:16", "PA0", "5") — multi-word key cells are how prose
    sneaks in as rows — or exactly two tokens whose first is a literal
    Bit/Bits label.
    """
    words = text.split()
    if len(words) <= 1:
        return True
    return len(words) == 2 and bool(_BIT_LABEL_RE.match(words[0]))


def _merge_row_cells(row: "list[Word]") -> "list[list[Any]]":
    """Merge a row's adjacent words into visual cells ``[x0, x1, text]``.

    Words separated by less than one word-height are one cell (spaces within a
    header like "Alternate Function"); column gaps are wider. This also folds
    prose lines into few long cells, which keeps them from matching multi-group
    header rules.
    """
    cells: list[list[Any]] = []
    for w in row:
        x0, y0, x1, y1, text = float(w[0]), float(w[1]), float(w[2]), float(w[3]), str(w[4])
        height = (y1 - y0) or 8.0
        if cells and x0 - cells[-1][1] <= height:
            cells[-1][1] = x1
            cells[-1][2] = f"{cells[-1][2]} {text}"
        else:
            cells.append([x0, x1, text])
    return cells


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _cell_matches(text: str, group: "list[str]") -> bool:
    """True when any word of *text* starts with any synonym, case-insensitively."""
    words = re.split(r"\W+", text.lower())
    return any(w.startswith(syn) for w in words for syn in group)


def _match_header(cells: "list[list[Any]]", rules: "list[StructureRule]"):
    """Match a merged header row against the rules.

    Returns ``(rule, columns)`` where ``columns`` is ``[(x0, x1, key)]`` for
    every header cell left-to-right — matched cells keyed by their group's
    first synonym, the rest by their own slugified text — or ``(None, None)``.
    Every synonym group must be matched by a distinct cell (leftmost wins).
    """
    if len(cells) < 2:
        return None, None
    for rule in rules:
        claimed: dict[int, str] = {}
        for group in rule.headers:
            found = next(
                (i for i, (_, _, text) in enumerate(cells)
                 if i not in claimed and _cell_matches(text, group)),
                None,
            )
            if found is None:
                break
            claimed[found] = group[0]
        else:
            columns: list[tuple[float, float, str]] = []
            seen: set[str] = set()
            for i, (x0, x1, text) in enumerate(cells):
                key = claimed.get(i) or _slug(text) or f"col{i}"
                while key in seen:
                    key += "_"
                seen.add(key)
                columns.append((x0, x1, key))
            return rule, columns
    return None, None


def _band_columns(band_rows: "list[list[Word]]") -> "list[list[Any]]":
    """Cluster a multi-line header band's words into columns by x-overlap.

    Wrapped column headers ("Exception" over "number") overlap horizontally
    across lines while distinct columns stay separated, so clustering on x
    recovers the full column set that no single line shows. Cell text joins
    each cluster's words in reading order.
    """
    clusters: list[list[Any]] = []  # [x0, x1, [words]]
    words = sorted((w for row in band_rows for w in row), key=lambda w: float(w[0]))
    for w in words:
        height = (float(w[3]) - float(w[1])) or 8.0
        if clusters and float(w[0]) <= clusters[-1][1] + height:
            clusters[-1][1] = max(clusters[-1][1], float(w[2]))
            clusters[-1][2].append(w)
        else:
            clusters.append([float(w[0]), float(w[2]), [w]])
    return [
        [x0, x1, " ".join(
            str(w[4]) for w in sorted(ws, key=lambda w: (round(float(w[1])), float(w[0])))
        )]
        for x0, x1, ws in clusters
    ]


def _assign_row(row: "list[Word]", boundaries: "list[float]", n: int) -> "list[str]":
    """Distribute a row's words into ``n`` columns split at *boundaries*."""
    values = [""] * n
    for w in row:
        xc = (float(w[0]) + float(w[2])) / 2
        i = bisect(boundaries, xc)
        values[i] = f"{values[i]} {w[4]}".strip()
    return values


def tables_from_words(words: "list[Word]", rules: "list[StructureRule]") -> "list[dict[str, Any]]":
    """Extract rule-matching tables from one page's words.

    Returns ``[{"kind": str, "columns": [str], "rows": [{col: str}],
    "names": [str]}]``. ``names`` holds the rule's name-column values (empty
    when the rule has no ``name_column``). A candidate table is accepted only
    if it has at least two data rows and at least half of them fill the name
    (or first) column — false negatives are cheap, false positives pollute
    structured retrieval.
    """
    if not rules:
        return []
    rows = _rows_by_y(words)
    ys = sorted(rows)
    tables: list[dict[str, Any]] = []

    i = 0
    while i < len(ys):
        row_height = max(
            (float(w[3]) - float(w[1]) for w in rows[ys[i]]), default=8.0
        )

        # Wrapped column headers span several sub-line-height-spaced lines;
        # data rows sit at least a full line further down. Collect that header
        # band first and match its x-clustered columns — a single line of a
        # wrapped header shows too few columns and misplaces every boundary.
        # Fall back to matching row i alone (plain single-line headers).
        band_gap = 0.8 * max(row_height, 8.0)
        hi = i
        while hi + 1 < len(ys) and ys[hi + 1] - ys[hi] <= band_gap:
            hi += 1
        rule = columns = None
        if hi > i:
            rule, columns = _match_header(
                _band_columns([rows[ys[k]] for k in range(i, hi + 1)]), rules,
            )
        if rule is None:
            rule, columns = _match_header(_merge_row_cells(rows[ys[i]]), rules)
            hi = i
        if rule is None:
            i += 1
            continue

        keys = [key for _, _, key in columns]
        boundaries = [
            (columns[k][1] + columns[k + 1][0]) / 2 for k in range(len(columns) - 1)
        ]
        max_gap = _MAX_ROW_GAP_FACTOR * max(row_height, 8.0)
        name_key = rule.name_column if rule.name_column in keys else keys[0]

        data_rows: list[dict[str, str]] = []
        strikes = 0
        last_y = ys[hi]
        j = hi + 1
        while j < len(ys):
            if ys[j] - last_y > max_gap:
                break
            if _match_header(_merge_row_cells(rows[ys[j]]), rules)[0] is not None:
                break  # next table's header; outer loop will revisit it
            values = _assign_row(rows[ys[j]], boundaries, len(keys))
            filled = [k for k, v in zip(keys, values) if v]
            key_ok = (
                bool(values[0])
                and _is_key_cell(values[0])
                and not _MARKER_KEY_RE.match(values[0])
            )
            # Merged/spanning key cells (a memory map's "Bus" column) leave
            # the key empty on continuation rows; rules opt in via
            # allow_empty_key, gated on the naming cell being present.
            empty_key_ok = (
                rule.allow_empty_key
                and not values[0]
                and bool(values[keys.index(name_key)])
            )
            if len(filled) >= 2 and (key_ok or empty_key_ok):
                data_rows.append(dict(zip(keys, values)))
                strikes = 0
                last_y = ys[j]
            elif len(filled) == 1 and data_rows and filled[0] != keys[0]:
                # Wrapped cell text: continue the previous row's cell.
                prev = data_rows[-1]
                prev[filled[0]] = f"{prev[filled[0]]} {values[keys.index(filled[0])]}".strip()
                last_y = ys[j]
            elif len(filled) == 1 and not data_rows and filled[0] != keys[0]:
                # Wrapped cell text rendered above the first data row (some
                # generators emit a row's tall description cell first): skip
                # without striking, keeping the gap window alive.
                last_y = ys[j]
            else:
                strikes += 1
                if strikes >= _MAX_STRIKES:
                    break
            j += 1

        named = [r for r in data_rows if r.get(name_key)]
        dense_single = (
            len(data_rows) == 1
            and named
            and sum(1 for v in data_rows[0].values() if v) >= _MIN_SINGLE_ROW_FILLED
        )
        if (len(data_rows) >= 2 and 2 * len(named) >= len(data_rows)) or dense_single:
            tables.append({
                "kind": rule.kind,
                "columns": keys,
                "rows": data_rows,
                "names": [r[name_key] for r in named] if rule.name_column else [],
                # y of the table's header row — the anchor used to attribute the
                # table to the entity heading above it (see docquery._owners).
                "y": ys[i],
            })
            i = j
        else:
            i += 1
    return tables


def render_table_lines(tables: "list[dict[str, Any]]") -> "list[str]":
    """Render tables as deterministic TABLE/ROW text lines (see module doc).

    Literal ``|`` in cell text becomes ``/`` so the pipe separator stays
    unambiguous (documented, lossy, rare).
    """
    lines: list[str] = []
    for t in tables:
        lines.append(f"TABLE {t['kind']}: " + " | ".join(t["columns"]))
        for row in t["rows"]:
            lines.append("ROW " + " | ".join(
                str(row.get(c, "")).replace("|", "/").strip() for c in t["columns"]
            ))
    return lines


def parse_table_lines(lines: "list[str]") -> "list[dict[str, Any]]":
    """Parse TABLE/ROW lines back into ``[{"kind", "columns", "rows"}]``.

    Exact inverse of :func:`render_table_lines`; lines that are neither TABLE
    nor ROW (or ROW lines before any TABLE line) are skipped.
    """
    tables: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in lines:
        line = line.strip()
        if m := _TABLE_LINE_RE.match(line):
            current = {
                "kind": m[1],
                "columns": [c.strip() for c in m[2].split(" | ")],
                "rows": [],
            }
            tables.append(current)
        elif line.startswith("ROW ") and current is not None:
            values = [v.strip() for v in line[4:].split(" | ")]
            cols = current["columns"]
            values += [""] * (len(cols) - len(values))
            current["rows"].append(dict(zip(cols, values)))
    return tables


def split_lines_by_kind(lines_by_page: "dict[int, list[str]]") -> "dict[str, dict[int, list[str]]]":
    """Regroup ``{page: [TABLE/ROW lines]}`` into ``{kind: {page: [lines]}}``.

    A page can hold tables of several kinds; each TABLE marker line starts a
    block that owns the ROW lines after it.
    """
    result: dict[str, dict[int, list[str]]] = {}
    for page, lines in lines_by_page.items():
        kind = None
        for line in lines:
            if m := _TABLE_LINE_RE.match(line.strip()):
                kind = m[1]
            if kind is not None:
                result.setdefault(kind, {}).setdefault(page, []).append(line)
    return result


def table_names(lines: "list[str]", name_column: str) -> "list[str]":
    """Distinct non-empty *name_column* values across the tables in *lines*."""
    names: list[str] = []
    for t in parse_table_lines(lines):
        for row in t["rows"]:
            name = row.get(name_column, "").strip()
            if name and name not in names:
                names.append(name)
    return names


def extract_page_tables(page: Any, rules: "list[StructureRule]") -> "list[dict[str, Any]]":
    """Extract rule-matching tables from a pymupdf page object."""
    return tables_from_words(page.get_text("words"), rules)


def extract_document_tables_owned(
    pdf_path: "str | Path",
    rules: "list[StructureRule]",
    page_entities: "dict[int, dict[str, list[tuple[str, str]]]]",
) -> "dict[int, list[tuple[str, list[str], list[str], str | None]]]":
    """``{page: [(kind, lines, names, owner_or_None)]}`` for the whole PDF.

    Same extraction as :func:`extract_document_tables`, but each table is
    attributed to the entity that owns it using reading-order geometry
    (:mod:`docquery._owners`) instead of being lumped in with everything else on
    its page. Manuals put several registers/peripherals on a page and continue
    them past page breaks, and the page's first text line — the usual heading
    fallback — is often just a running header, so page-level grouping mislabels
    the block.
    """
    if not rules:
        return {}
    try:
        import fitz  # pymupdf
    except ImportError:  # pragma: no cover - dependency is declared
        logger.warning("pymupdf not available; skipping table extraction")
        return {}

    from docquery._owners import assign_owners, heading_positions

    blocks_by_page: dict[int, list[tuple[int, Any]]] = {}
    headings_by_page: dict[int, list[tuple[int, str]]] = {}
    try:
        with fitz.open(str(pdf_path)) as doc:
            for i, page in enumerate(doc, start=1):
                words = page.get_text("words")
                try:
                    tables = tables_from_words(words, rules)
                except Exception as exc:  # noqa: BLE001 - one bad page ≠ no tables
                    logger.debug("table extraction failed on page %d of %s: %s", i, pdf_path, exc)
                    continue
                if tables:
                    blocks_by_page[i] = [
                        (
                            int(t.get("y") or 0),
                            (t["kind"], render_table_lines([t]), list(t.get("names") or [])),
                        )
                        for t in tables
                    ]
                entries = [
                    pair
                    for per_rule in (page_entities.get(i) or {}).values()
                    for pair in per_rule
                ]
                if entries:
                    headings_by_page[i] = heading_positions(words, entries)
    except Exception as exc:  # noqa: BLE001 - any pymupdf failure → no tables
        logger.warning("extract_document_tables_owned failed for %s: %s", pdf_path, exc)
        return {}

    owned = assign_owners(blocks_by_page, headings_by_page)
    result = {
        pg: [(kind, lines, names, owner) for (kind, lines, names), owner in items]
        for pg, items in owned.items()
        if items
    }
    if result:
        total = sum(len(v) for v in result.values())
        attributed = sum(1 for v in result.values() for item in v if item[3])
        logger.info(
            "table extraction: %d tables on %d pages (%d attributed to an entity) in %s",
            total, len(result), attributed, pdf_path,
        )
    return result


def extract_document_tables(
    pdf_path: "str | Path", rules: "list[StructureRule]",
) -> "dict[int, list[str]]":
    """Return ``{1-based page: [rendered TABLE/ROW lines]}`` for the whole PDF.

    Pages without a rule-matching table are omitted. Returns ``{}`` when the
    file cannot be opened or pymupdf is unavailable — callers just skip the
    table pass.
    """
    if not rules:
        return {}
    try:
        import fitz  # pymupdf
    except ImportError:  # pragma: no cover - dependency is declared
        logger.warning("pymupdf not available; skipping table extraction")
        return {}

    result: dict[int, list[str]] = {}
    try:
        with fitz.open(str(pdf_path)) as doc:
            for i, page in enumerate(doc, start=1):
                try:
                    tables = extract_page_tables(page, rules)
                except Exception as exc:  # noqa: BLE001 - one bad page ≠ no tables
                    logger.debug("table extraction failed on page %d of %s: %s", i, pdf_path, exc)
                    continue
                if tables:
                    result[i] = render_table_lines(tables)
    except Exception as exc:  # noqa: BLE001 - any pymupdf failure → no tables
        logger.warning("extract_document_tables failed for %s: %s", pdf_path, exc)
        return {}
    if result:
        total = sum(len(v) for v in result.values())
        logger.info("table extraction: %d lines on %d pages in %s", total, len(result), pdf_path)
    return result
