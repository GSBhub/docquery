"""Deterministic grounding checks: is a claim literally present in the source?

The document store is the sole source of truth. LLMs — even with correct tool
output in context — re-type values when synthesizing answers, and models that
know a domain from pretraining will happily fill gaps with plausible values
that are not in the document. Neither failure is reliably caught by prompting,
and LLM-as-judge graders can themselves hallucinate.

This module checks the *verifiable* subset of an output mechanically: hex
literals, bit ranges, binary field values, and quoted ENCODING/ROW machine
lines must appear (after normalization) in the source text they claim to come
from. Prose is out of scope — descriptions are legitimately paraphrased.

Normalization makes numerically equal spellings match: ``0x4``, ``0x0004`` and
``0x00000004`` are the same claim; ``[31:16]`` and ``31:16`` are the same
range. No LLM, no embedding — pure string/number work, unit-testable.
"""

from __future__ import annotations

import re
from functools import lru_cache

# Verifiable claim shapes. Bare decimals are deliberately NOT extracted from
# free text: they are too noisy (list positions, counts, page references) and
# would flood answers with false mismatches.
_HEX_RE = re.compile(r"0[xX][0-9a-fA-F_]+")
# Hex literals typeset with grouped digits ("0xA800 0000", "0x0000 0280"):
# continuation groups must be exactly 4 hex digits so unrelated adjacent
# numbers ("0x08 or 15") are never merged into one claim.
_SPACED_HEX_RE = re.compile(r"0[xX][0-9a-fA-F]{1,4}(?: [0-9a-fA-F]{4})+")
_BIT_RANGE_RE = re.compile(r"\[?(\d{1,2})\s*[:\-–]\s*(\d{1,2})\]?")
_MACHINE_LINE_RE = re.compile(r"^\s*(ENCODING \d+-bit:.*|ROW .*|TABLE \S+:.*)$", re.MULTILINE)
_BINARY_FIELD_RE = re.compile(r"=([01]{2,})\b")

GROUNDING_MODES = ("strict", "warn", "off")


def _norm_hex(token: str) -> str:
    return f"0x{int(token.replace('_', '').replace(' ', ''), 16):x}"


def _hex_claims(text: str) -> set[str]:
    """Normalized hex values present in *text*, including digit-grouped ones."""
    claims = {_norm_hex(m.group()) for m in _HEX_RE.finditer(text)}
    claims.update(_norm_hex(m.group()) for m in _SPACED_HEX_RE.finditer(text))
    return claims


def _norm_range(hi: str, lo: str) -> str:
    return f"{hi}:{lo}"


def _norm_line(line: str) -> str:
    return " ".join(line.split())


def extract_claims(text: str) -> set[str]:
    """Normalized verifiable claims made by *text*.

    Hex literals normalize by numeric value, bit ranges to ``hi:lo``, quoted
    machine lines (ENCODING/ROW/TABLE) to whitespace-collapsed form, and
    ``=0101``-style binary field values to ``=<bits>``.
    """
    claims: set[str] = _hex_claims(text)
    for m in _BIT_RANGE_RE.finditer(text):
        claims.add(_norm_range(m[1], m[2]))
    for m in _MACHINE_LINE_RE.finditer(text):
        claims.add(_norm_line(m[1]))
    for m in _BINARY_FIELD_RE.finditer(text):
        claims.add(f"={m[1]}")
    return claims


def unsupported_claims(answer: str, sources: str) -> list[str]:
    """Claims in *answer* whose normalized form does not appear in *sources*.

    ``sources`` is the concatenated raw text the answer was derived from (tool
    outputs for chat, retrieved context for extraction). Returns a sorted list
    for stable messages; empty means fully grounded (of the verifiable subset).
    """
    if not answer:
        return []
    supported = extract_claims(sources)
    return sorted(extract_claims(answer) - supported)


def _is_verifiable_scalar(value: object) -> bool:
    """True for leaf values precise enough to demand verbatim presence.

    Numbers always; strings only when they look like identifiers or literals
    (short, no sentence structure) — prose fields are legitimately paraphrased
    by the model and must not be enforced.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        v = value.strip()
        return bool(v) and len(v) <= 32 and " " not in v
    return False


def _scalar_in_text(value: object, text: str) -> bool:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # word-bounded, and not part of a longer decimal ("42" ≠ "42.5"/"3.42"),
        # but a sentence-final "42." must still count
        if re.search(rf"(?<!\w)(?<!\d\.){re.escape(str(value))}(?!\w|\.\d)", text):
            return True
        if isinstance(value, int):
            # the document may spell the same number in hex
            return bool(re.search(rf"0[xX]0*{value:x}\b", text, re.IGNORECASE))
        return False
    v = str(value).strip()
    if _HEX_RE.fullmatch(v):
        # compare hex by numeric value against whole hex tokens, never as a
        # substring ("0x00" is not in "0x00000010"); this also covers other
        # spellings of the same value (0x4 vs 0x00000004, or digit-grouped:
        # 0xA800 0000)
        return _norm_hex(v) in _hex_claims(text)
    return v.lower() in text.lower()


def _record_scalars(record: dict) -> list[object]:
    """Direct verifiable scalar leaves of one record (no recursion)."""
    return [v for v in record.values() if _is_verifiable_scalar(v)]


def _model_items(instance: object) -> "dict | None":
    """A Pydantic model's field values minus grounding-exempt fields, else None.

    A schema opts a field out of grounding enforcement with
    ``Field(json_schema_extra={"grounding": "off"})`` — for values that are
    legitimately absent from the document text (derived, reworded, or supplied
    by the caller) but too identifier-like for the prose exemption to apply.
    """
    fields = getattr(type(instance), "model_fields", None)
    if not hasattr(instance, "model_dump") or fields is None:
        return None
    items: dict = {}
    for name, f in fields.items():
        extra = getattr(f, "json_schema_extra", None)
        if isinstance(extra, dict) and extra.get("grounding") == "off":
            continue
        items[name] = getattr(instance, name)
    return items


_MACHINE_PREFIXES = ("ENCODING ", "TABLE ", "ROW ")


# Unstructured text loaders often split one table row's cells onto separate
# lines (often separate blank-line blocks): a register name, then its offset
# (and reset value). A name-only line followed directly by hex-only lines is
# re-joined into one virtual row unit. Name first and adjacent only, so a
# name is never paired with a hex cell that comes before it or after another
# name.
_NAME_CELL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_\[\]]{0,31}")
_HEX_CELL_RE = re.compile(r"0[xX][0-9a-fA-F_]+(?: [0-9a-fA-F]{4})*")
_MAX_ROW_HEX_CELLS = 3


def _virtual_row_units(context: str) -> list[str]:
    """``name 0xOFFSET [0xRESET …]`` units re-joined from split table cells."""
    cells = [ln.strip() for ln in context.splitlines() if ln.strip()]
    rows: list[str] = []
    for i, cell in enumerate(cells):
        if not _NAME_CELL_RE.fullmatch(cell) or _HEX_CELL_RE.fullmatch(cell):
            continue
        hexes: list[str] = []
        for nxt in cells[i + 1:i + 1 + _MAX_ROW_HEX_CELLS]:
            if not _HEX_CELL_RE.fullmatch(nxt):
                break
            hexes.append(nxt)
        if hexes:
            rows.append(" ".join([cell, *hexes]))
    return rows


# A block without machine lines only counts as one unit when it is small
# enough to plausibly describe a single entity (a register section's
# "heading / Address offset: … / Reset value: …" stanza), not a table
# flattened into prose where merging would hide swapped associations.
_MAX_BLOCK_UNIT_LINES = 4
_MAX_BLOCK_UNIT_CHARS = 400


@lru_cache(maxsize=8)
def _grounding_units(context: str) -> tuple[str, ...]:
    """Units for record co-occurrence checks.

    Each non-empty line stands alone, except that machine lines of a structure
    block (a paragraph containing ENCODING/TABLE/ROW lines) are augmented with
    the block's heading lines — the text above its first machine line.
    Documents name a structure once in a heading (mnemonic, register name,
    table title) and put the values on machine lines below it, so a correctly
    paired record spans heading + ONE machine line. Two different data lines
    are never merged into a unit, which keeps swapped associations (NMI with
    HardFault's address) failing.

    Two block-scoped units are added on top of the per-line ones, because
    reference manuals also state one entity's attributes as a short labeled
    stanza ("6.3.1 … (GPIOx_CFGR)" / "Address offset: 0x00" / "Reset value:
    0xA8000000") rather than a table row: a block's heading region (the lines
    above its first machine line) is one unit, and a small machine-line-free
    block (≤ ``_MAX_BLOCK_UNIT_LINES`` lines and ``_MAX_BLOCK_UNIT_CHARS``
    chars) is one unit. Machine/data lines are still never merged with each
    other, and blocks never merge across blank lines — except for virtual
    row units (see :func:`_virtual_row_units`): a name cell directly followed
    by hex cells (offset, reset value) is re-joined into one row, because
    unstructured loaders split a table row's cells onto separate lines.
    """
    units: list[str] = _virtual_row_units(context)
    for block in re.split(r"\n\s*\n", context):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        first_machine = next(
            (i for i, ln in enumerate(lines)
             if ln.strip().startswith(_MACHINE_PREFIXES)),
            None,
        )
        heading = "\n".join(lines[:first_machine]) if first_machine else ""
        if heading:
            units.append(heading)
        elif first_machine is None and 1 < len(lines) <= _MAX_BLOCK_UNIT_LINES \
                and sum(len(ln) for ln in lines) <= _MAX_BLOCK_UNIT_CHARS:
            units.append("\n".join(lines))
        for ln in lines:
            if heading and ln.strip().startswith(_MACHINE_PREFIXES):
                units.append(f"{heading}\n{ln}")
            else:
                units.append(ln)
    return tuple(units)


def ungrounded_records(instance: object, context: str, _prefix: str = "") -> list[str]:
    """Records whose verifiable values never co-occur in one grounding unit.

    Presence alone cannot catch association errors: pairing NMI with
    HardFault's address passes a presence check because both tokens exist
    somewhere in the context. A record (dict/model with two or more
    verifiable scalar fields) is grounded only if some unit — a context line,
    or a structure block's headings plus one of its machine lines (see
    :func:`_grounding_units`) — contains all of its verifiable values, which
    the machine-parseable ROW/ENCODING lines guarantee for correctly-paired
    data while still keeping values from two different data lines apart.
    """
    misses: list[str] = []
    if (items := _model_items(instance)) is not None:
        instance = items
    if isinstance(instance, dict):
        scalars = _record_scalars(instance)
        if len(scalars) >= 2:
            units = _grounding_units(context)
            if not any(all(_scalar_in_text(v, u) for v in scalars) for u in units):
                pairing = ", ".join(repr(v) for v in scalars)
                misses.append(f"{_prefix.rstrip('.') or 'record'}: ({pairing})")
        for key, value in instance.items():
            if isinstance(value, (dict, list, tuple)) or hasattr(value, "model_dump"):
                misses.extend(ungrounded_records(value, context, f"{_prefix}{key}."))
    elif isinstance(instance, (list, tuple)):
        for i, value in enumerate(instance):
            misses.extend(ungrounded_records(value, context, f"{_prefix}{i}."))
    return misses


def ungrounded_fields(instance: object, context: str, _prefix: str = "") -> list[str]:
    """Verifiable leaves of a (nested) model/dict/list absent from *context*.

    Walks Pydantic models, dicts, and lists; returns ``["field=value", ...]``
    dotted paths for every verifiable scalar that cannot be found in the
    retrieved context. Empty list means the extraction is grounded.
    """
    misses: list[str] = []
    if (items := _model_items(instance)) is not None:
        instance = items
    if isinstance(instance, dict):
        for key, value in instance.items():
            misses.extend(ungrounded_fields(value, context, f"{_prefix}{key}."))
    elif isinstance(instance, (list, tuple)):
        for i, value in enumerate(instance):
            misses.extend(ungrounded_fields(value, context, f"{_prefix}{i}."))
    elif _is_verifiable_scalar(instance) and not _scalar_in_text(instance, context):
        misses.append(f"{_prefix.rstrip('.')}={instance!r}")
    return misses


def _entry_misses(entry: object, context: str) -> list[str]:
    """Presence and co-occurrence misses of one list entry (record or scalar)."""
    return ungrounded_fields(entry, context) + ungrounded_records(entry, context)


def _prune(value: object, context: str, prefix: str, dropped: list[str]) -> object:
    if isinstance(value, (list, tuple)):
        kept = []
        for i, entry in enumerate(value):
            entry = _prune(entry, context, f"{prefix}{i}.", dropped)
            if misses := _entry_misses(entry, context):
                dropped.append(f"{prefix}{i}: " + "; ".join(misses))
                continue
            kept.append(entry)
        return type(value)(kept)
    if (items := _model_items(value)) is not None:
        updates = {}
        for name, v in items.items():
            new = _prune(v, context, f"{prefix}{name}.", dropped)
            if new is not v:
                updates[name] = new
        return value.model_copy(update=updates) if updates else value  # type: ignore[union-attr]
    if isinstance(value, dict):
        return {k: _prune(v, context, f"{prefix}{k}.", dropped) for k, v in value.items()}
    return value


def prune_ungrounded(instance: object, context: str) -> tuple[object, list[str]]:
    """Drop ungrounded list entries; return ``(pruned_instance, dropped)``.

    One invented or mispaired entry should cost that entry, not the whole
    extraction: failing the record wholesale for a single bad register
    collapses an otherwise-grounded list to nothing after retries. Walks the
    instance bottom-up, so an entry's own ungrounded sub-entries are pruned
    before the entry itself is judged, and drops every list entry (record or
    scalar) with a presence or co-occurrence miss. ``dropped`` describes each
    removed entry by dotted path. Values outside any list are left in place —
    :func:`ungrounded_fields` / :func:`ungrounded_records` on the pruned
    instance report them.
    """
    dropped: list[str] = []
    return _prune(instance, context, "", dropped), dropped
