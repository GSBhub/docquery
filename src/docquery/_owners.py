"""Associate recovered structure blocks with the entity that owns them.

Structure recovery (:mod:`docquery._bitgrid`, :mod:`docquery._tables`) finds
blocks — instruction encoding grids, register-field tables, pin/interrupt tables
— but a *page* is too coarse an owner: reference manuals pack several entities
onto one page and routinely split an entity's description across a page break.
Attributing a block to the wrong entity is worse than not attributing it, since
downstream consumers treat it as ground truth (e.g. a bit diagram becomes a
decode pattern for the wrong instruction).

This module resolves ownership from **reading-order geometry**: a block belongs
to the nearest entity heading *above* it, carrying the owner across page breaks
when a page begins mid-description. That rule is generic — it works the same for
"which instruction owns this encoding grid" and "which peripheral/register owns
this field table" — so every structure kind can share it.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

logger = logging.getLogger(__name__)

Word = Sequence[Any]


_ROW_TOL = 3.0  # pts of baseline wobble tolerated within one visual row


def _rows(words: "list[Word]", tol: float = _ROW_TOL) -> "dict[int, list[str]]":
    """Word texts grouped into visual rows keyed by the row's top-y.

    Grouping is tolerance-based, not exact-rounded: a heading's label and its
    value frequently sit a point apart (e.g. ``ADD`` at y=103 with its ``Syntax``
    anchor at y=104), and exact keying would split them into different rows and
    lose the pairing.
    """
    rows: dict[int, list[str]] = {}
    start: float | None = None
    bucket: list[str] = []
    for w in sorted(words, key=lambda w: float(w[1])):
        y = float(w[1])
        if start is None:
            start = y
        elif y - start > tol:
            rows[round(start)] = bucket
            start, bucket = y, []
        bucket.append(str(w[4]).strip())
    if start is not None:
        rows[round(start)] = bucket
    return rows


def heading_positions(
    words: "list[Word]", entries: "Sequence[tuple[str, str]]",
) -> "list[tuple[int, str]]":
    """``[(y, entity_name)]`` for each heading found in *words*, sorted by y.

    *entries* are ``(entity_name, full_matched_line)`` pairs as produced by
    ``match_page_entities``. A row is the heading when it contains the entity
    name **and** (when the matched line has other tokens, e.g. a ``Syntax``
    anchor or a section number) at least one of those tokens — that pairing
    avoids latching onto a running page header that merely repeats the name.
    """
    rows = _rows(words)
    out: list[tuple[int, str]] = []
    for name, line in entries:
        if not name:
            continue
        others = {t for t in str(line).split() if t and t != name}
        best: int | None = None
        for y, texts in rows.items():
            if name not in texts:
                continue
            if others and not (others & set(texts)):
                continue
            if best is None or y < best:
                best = y
        if best is not None:
            out.append((best, name))
    return sorted(out)


def assign_owners(
    blocks_by_page: "dict[int, list[tuple[int, Any]]]",
    headings_by_page: "dict[int, list[tuple[int, str]]]",
) -> "dict[int, list[tuple[Any, str | None]]]":
    """Attribute each positioned block to its owning entity.

    *blocks_by_page* is ``{page: [(y, block)]}`` and *headings_by_page* is
    ``{page: [(y, name)]}`` (see :func:`heading_positions`). Returns
    ``{page: [(block, owner_or_None)]}``: a block's owner is the nearest heading
    at or above it on the same page, else the owner carried over from the
    previous page (an entity whose description continues past a page break).
    """
    owners: dict[int, list[tuple[Any, str | None]]] = {}
    carried: str | None = None
    for page in sorted(set(blocks_by_page) | set(headings_by_page)):
        heads = sorted(headings_by_page.get(page) or [])
        placed: list[tuple[Any, str | None]] = []
        for y, block in sorted(blocks_by_page.get(page) or []):
            above = [name for hy, name in heads if hy <= y]
            placed.append((block, above[-1] if above else carried))
        if placed:
            owners[page] = placed
        if heads:
            carried = heads[-1][1]
    return owners
