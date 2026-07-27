"""Infer the entity rule that best explains a document's structure.

Entity tagging needs a regex whose capture group is an entity's name
(``Syntax  ADD``, ``19.4.3 … (GPIOx_OMODE)``). Authoring that per manual is
guesswork, but it does not have to be: the document itself says which pattern is
right, because the *correct* rule is the one whose headings actually own the
recovered structure blocks — encoding diagrams, register-field tables, pin maps.

So candidates can be scored, and scoring is cheap: matching, geometry and
attribution are pure text/coordinate work with no LLM and no embeddings.

The objective needs both halves:

``recall``
    fraction of structure blocks that found an owner.
``precision``
    fraction of tagged entities that actually own a block.

Recall alone is actively misleading — a pattern like ``\\n([A-Z]+)\\n`` tags every
capitalised line, so *every* block finds some owner (recall 1.0) while the
"entities" are mostly noise. On a TI C674x manual that scores 440/440 attributed
from 2017 bogus entities, beating the correct ``Syntax`` rule's 345/440 from 238.
Precision separates them (0.11 vs 0.89), so the score is their product.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# Heading shapes seen across ISA and MCU reference manuals. The capture group is
# always the entity name.
DEFAULT_CANDIDATES: "dict[str, str]" = {
    # "Syntax   ADD (.unit) src1, src2" — TI DSP instruction pages
    "syntax-anchor": r"Syntax\s+([A-Z][A-Z0-9_]*)",
    # "3.10.4  ADD" — numbered section ending in the mnemonic
    "section-inline": r"\n\s*[A-Z]?\d+(?:\.\d+)+\s+([A-Z][A-Z0-9_]{1,11})\s*\n",
    # "A7.7.12\nADC" — section number and name on consecutive lines (ARM)
    "section-newline": r"\n\s*[A-Z]?\d+(?:\.\d+)+\s*\n\s*([A-Z][A-Z0-9_]{1,11})\s*\n",
    # "19.4.3 GPIO output mode register (GPIOx_OMODE)" — MCU register headings
    "parenthesized-name": r"\n\s*[A-Z]?\d+(?:\.\d+)+\s+.{0,80}?\((\w+)\)",
    # "ADD   Add Two Signed Integers" — name-first page heading
    "name-first": r"\n([A-Z][A-Z0-9_]{1,11})\s{2,}[A-Z][a-z]",
}

_OUTLINE_LEAF_RE = re.compile(r"^([A-Z]?[\d.]*\d)\s+([A-Z][A-Z0-9_]{1,11})\s*$")
_MIN_OUTLINE_SUPPORT = 8  # leaf titles sharing a prefix shape before we trust it


def outline_candidates(outline: "Sequence[dict[str, Any]]") -> "dict[str, str]":
    """Candidate patterns synthesised from a PDF outline, if it enumerates entities.

    Many manuals bookmark each entity ("A7.7.7 ADR"), which pins both the section
    numbering and the name shape — far stronger evidence than a generic guess.
    Returns ``{}`` when the outline is absent or does not enumerate (some manuals
    ship a single root bookmark).
    """
    shapes: dict[str, int] = {}
    for entry in outline or []:
        title = str(entry.get("title", "")).replace(" ", " ").replace("\xa0", " ").strip()
        m = _OUTLINE_LEAF_RE.match(title)
        if not m:
            continue
        # "A7.7.12" -> "A7.7.<n>": the varying leaf number becomes \d+
        prefix = m.group(1)
        shape = re.sub(r"\d+$", r"\\d+", re.escape(prefix).replace(r"\.", r"\."))
        shapes[shape] = shapes.get(shape, 0) + 1

    out: dict[str, str] = {}
    for shape, count in sorted(shapes.items(), key=lambda kv: -kv[1]):
        if count < _MIN_OUTLINE_SUPPORT:
            continue
        out["outline-inline"] = rf"\n\s*{shape}\s+([A-Z][A-Z0-9_]{{1,11}})\s*\n"
        out["outline-newline"] = rf"\n\s*{shape}\s*\n\s*([A-Z][A-Z0-9_]{{1,11}})\s*\n"
        break
    return out


def _page_blocks(pdf_path: "str | Path", rules: "list[Any] | None"):
    """``(words_by_page, blocks_by_page, n_blocks)`` — one geometry pass.

    Blocks are every recovered structure item (encoding diagrams *and* tables) so
    the score reflects whatever the document actually contains.
    """
    import fitz  # pymupdf

    from docquery._bitgrid import encodings_from_words
    from docquery._tables import tables_from_words

    words_by_page: dict[int, Any] = {}
    blocks: dict[int, list[tuple[int, str]]] = {}
    with fitz.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc, start=1):
            words = page.get_text("words")
            words_by_page[i] = words
            items: list[tuple[int, str]] = []
            try:
                items += [
                    (int(e.get("y") or 0), "encoding")
                    for e in encodings_from_words(words)
                    if any(s.get("value") for s in e["segments"])
                ]
            except Exception:  # noqa: BLE001 - a bad page must not sink inference
                pass
            if rules:
                try:
                    items += [(int(t.get("y") or 0), t["kind"]) for t in tables_from_words(words, rules)]
                except Exception:  # noqa: BLE001
                    pass
            if items:
                blocks[i] = items
    return words_by_page, blocks, sum(len(v) for v in blocks.values())


def score_patterns(
    pdf_path: "str | Path",
    candidates: "dict[str, str] | None" = None,
    entity_name: str = "instruction",
    structure_rules: "list[Any] | None" = None,
) -> "list[dict[str, Any]]":
    """Score entity-rule candidates against *pdf_path*, best first.

    Each result is ``{"name", "pattern", "entities", "attributed", "blocks",
    "owners", "precision", "recall", "score"}``. Uses no LLM and no embeddings,
    so it is cheap enough to run before committing to an ingest.
    """
    from docquery._ingest import match_page_entities
    from docquery._owners import assign_owners, heading_positions
    from docquery._pdf import extract_outline, extract_page_text
    from docquery.config import EntityRule, Settings

    if structure_rules is None:
        try:
            structure_rules = list(Settings().structure_rules)
        except Exception:  # noqa: BLE001 - defaults are a convenience, not a requirement
            structure_rules = []

    page_text = extract_page_text(pdf_path)
    words_by_page, blocks, n_blocks = _page_blocks(pdf_path, structure_rules)
    if not n_blocks:
        logger.warning("no structure blocks recovered from %s; cannot score patterns", pdf_path)
        return []

    pool = dict(candidates or DEFAULT_CANDIDATES)
    if candidates is None:
        pool.update(outline_candidates(extract_outline(pdf_path) or []))

    results: list[dict[str, Any]] = []
    for name, pattern in pool.items():
        try:
            per_page = match_page_entities(page_text, [EntityRule(name=entity_name, pattern=pattern)])
        except re.error as exc:
            logger.debug("candidate %s is not a valid regex: %s", name, exc)
            continue
        n_entities = sum(len(v) for pr in per_page.values() for v in pr.values())
        heads = {}
        for page, per_rule in per_page.items():
            entries = [pair for pairs in per_rule.values() for pair in pairs]
            if pos := heading_positions(words_by_page.get(page, []), entries):
                heads[page] = pos
        owned = assign_owners(blocks, heads)
        attributed = sum(1 for v in owned.values() for item in v if item[1])
        owners = len({item[1] for v in owned.values() for item in v if item[1]})
        precision = owners / n_entities if n_entities else 0.0
        recall = attributed / n_blocks
        results.append({
            "name": name, "pattern": pattern, "entities": n_entities,
            "attributed": attributed, "blocks": n_blocks, "owners": owners,
            "precision": round(precision, 3), "recall": round(recall, 3),
            "score": round(precision * recall, 4),
        })
    results.sort(key=lambda r: -r["score"])
    return results
