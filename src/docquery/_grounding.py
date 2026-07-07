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

# Verifiable claim shapes. Bare decimals are deliberately NOT extracted from
# free text: they are too noisy (list positions, counts, page references) and
# would flood answers with false mismatches.
_HEX_RE = re.compile(r"0[xX][0-9a-fA-F_]+")
_BIT_RANGE_RE = re.compile(r"\[?(\d{1,2})\s*[:\-–]\s*(\d{1,2})\]?")
_MACHINE_LINE_RE = re.compile(r"^\s*(ENCODING \d+-bit:.*|ROW .*|TABLE \S+:.*)$", re.MULTILINE)
_BINARY_FIELD_RE = re.compile(r"=([01]{2,})\b")

GROUNDING_MODES = ("strict", "warn", "off")


def _norm_hex(token: str) -> str:
    return f"0x{int(token.replace('_', ''), 16):x}"


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
    claims: set[str] = set()
    for m in _HEX_RE.finditer(text):
        claims.add(_norm_hex(m.group()))
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
            return bool(re.search(rf"0[xX]0*{value:x}\b", text))
        return False
    v = str(value).strip()
    if v.lower() in text.lower():
        return True
    if m := _HEX_RE.fullmatch(v):
        # hex spelled differently in the document (0x4 vs 0x00000004)
        return _norm_hex(v) in {_norm_hex(t.group()) for t in _HEX_RE.finditer(text)}
    return False


def ungrounded_fields(instance: object, context: str, _prefix: str = "") -> list[str]:
    """Verifiable leaves of a (nested) model/dict/list absent from *context*.

    Walks Pydantic models, dicts, and lists; returns ``["field=value", ...]``
    dotted paths for every verifiable scalar that cannot be found in the
    retrieved context. Empty list means the extraction is grounded.
    """
    misses: list[str] = []
    if hasattr(instance, "model_dump"):
        instance = instance.model_dump()
    if isinstance(instance, dict):
        for key, value in instance.items():
            misses.extend(ungrounded_fields(value, context, f"{_prefix}{key}."))
    elif isinstance(instance, (list, tuple)):
        for i, value in enumerate(instance):
            misses.extend(ungrounded_fields(value, context, f"{_prefix}{i}."))
    elif _is_verifiable_scalar(instance) and not _scalar_in_text(instance, context):
        misses.append(f"{_prefix.rstrip('.')}={instance!r}")
    return misses
