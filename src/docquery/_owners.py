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
from bisect import bisect_right
from typing import Any, Sequence

logger = logging.getLogger(__name__)

Word = Sequence[Any]


_ROW_TOL = 3.0  # pts of baseline wobble tolerated within one visual row
# Height of the running-header band below the topmost row on a page. Only used
# for anchorless patterns, where nothing else distinguishes a repeated header
# from the real heading.
_HEADER_BAND = 20.0
# Pages an owner may carry past its heading before expiring. Descriptions
# legitimately run long (a C6x instruction documents .L/.S/.D on consecutive
# pages, each with its own diagrams), so this is a runaway guard — it stops an
# owner leaking into an unrelated section, not a tight span limit.
_MAX_CARRY_PAGES = 8


def _norm(text: str) -> str:
    """Strip surrounding punctuation so a token matches its bare name.

    Headings routinely wrap the entity name in delimiters — ``(GPIOx_OMODE)``,
    ``ADD,`` , ``CTRL:`` — and the PDF keeps them attached to the word, so exact
    text comparison would never match the name.
    """
    return str(text).strip().strip("()[]{}<>,;:.\"'").strip()


def _rows(words: "list[Word]", tol: float = _ROW_TOL) -> "dict[int, list[str]]":
    """Word texts grouped into visual rows keyed by the row's top-y.

    Grouping is tolerance-based, not exact-rounded: a heading's label and its
    value frequently sit a point apart (e.g. ``ADD`` at y=103 with its ``Syntax``
    anchor at y=104), and exact keying would split them into different rows and
    lose the pairing.
    """
    # Cluster starts are >tol apart, so their rounded keys never collide.
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

    Patterns that capture only a bare name have no anchor to pair with, so for
    those the page's header band is skipped instead; the topmost match is used
    only when the name appears nowhere else on the page.
    """
    rows = {y: {_norm(t) for t in texts} for y, texts in _rows(words).items()}
    if not rows:
        return []
    # A running header repeats the entity name at the very top of the page. When
    # the matched line carries an anchor token the pairing rejects it, but a
    # pattern that captures only a bare name has no anchor to pair with — so for
    # those, skip the header band outright rather than latching onto the topmost
    # occurrence and placing the heading above every block on the page.
    top_y = min(rows)
    header_band = top_y + _HEADER_BAND

    out: list[tuple[int, str]] = []
    for name, line in entries:
        key = _norm(name)
        if not key:
            continue
        others = {_norm(t) for t in str(line).split()} - {key, ""}
        candidates = [
            y for y, texts in rows.items()
            if key in texts and (not others or (others & texts))
        ]
        if others:
            best = min(candidates, default=None)
        else:
            # Prefer the first occurrence below the header band; fall back to the
            # topmost only if the name appears nowhere else. (`or` would be wrong
            # here — a legitimate y of 0 is falsy.)
            below = [y for y in candidates if y > header_band]
            best = min(below) if below else min(candidates, default=None)
        if best is not None:
            out.append((best, name))
    return sorted(out)


def assign_owners(
    blocks_by_page: "dict[int, list[tuple[int, Any]]]",
    headings_by_page: "dict[int, list[tuple[int, str]]]",
    max_carry_pages: int = _MAX_CARRY_PAGES,
) -> "dict[int, list[tuple[Any, str | None]]]":
    """Attribute each positioned block to its owning entity.

    *blocks_by_page* is ``{page: [(y, block)]}`` and *headings_by_page* is
    ``{page: [(y, name)]}`` (see :func:`heading_positions`). Returns
    ``{page: [(block, owner_or_None)]}``: a block's owner is the nearest heading
    at or above it on the same page, else the owner carried over from a recent
    previous page (an entity whose description continues past a page break).

    The carry expires after *max_carry_pages* pages without a heading: a
    description rarely runs longer than that, so a stale owner propagating
    onward would silently mislabel unrelated blocks — worse than no owner, since
    consumers treat attribution as ground truth.
    """
    owners: dict[int, list[tuple[Any, str | None]]] = {}
    carried: str | None = None
    carried_page: int | None = None
    for page in sorted(set(blocks_by_page) | set(headings_by_page)):
        heads = sorted(headings_by_page.get(page) or [])
        if (
            carried is not None
            and carried_page is not None
            and page - carried_page > max_carry_pages
        ):
            carried = None
        head_ys = [hy for hy, _ in heads]
        placed: list[tuple[Any, str | None]] = []
        for y, block in sorted(blocks_by_page.get(page) or []):
            # Rightmost heading at or above the block, in reading order.
            i = bisect_right(head_ys, y)
            placed.append((block, heads[i - 1][1] if i else carried))
        if placed:
            owners[page] = placed
        if heads:
            carried = heads[-1][1]
            carried_page = page
    return owners
