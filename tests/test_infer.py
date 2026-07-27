"""Tests for entity-rule inference (docquery._infer)."""

from __future__ import annotations

from docquery._infer import DEFAULT_CANDIDATES, outline_candidates


def _entry(title, page=1):
    return {"title": title, "page": page, "level": 3, "order": 0}


def test_outline_candidates_synthesise_pattern_from_bookmarks():
    """A manual that bookmarks each entity pins both the section numbering and
    the name shape — stronger evidence than any generic guess."""
    outline = [_entry(f"A7.7.{n} INSTR{n}", page=100 + n) for n in range(1, 12)]
    got = outline_candidates(outline)
    assert set(got) == {"outline-inline", "outline-newline"}
    assert r"A7\.7\.\d+" in got["outline-inline"]
    assert r"A7\.7\.\d+" in got["outline-newline"]


def test_outline_candidates_ignore_thin_support():
    """Two stray bookmarks are not a numbering scheme."""
    assert outline_candidates([_entry("A7.7.1 ADD"), _entry("A7.7.2 SUB")]) == {}


def test_outline_candidates_empty_for_unbookmarked_manual():
    # Some manuals ship a single root bookmark and nothing else.
    assert outline_candidates([_entry("ACME SERIES REFERENCE MANUAL")]) == {}
    assert outline_candidates([]) == {}


def test_outline_candidates_tolerate_typographic_spaces():
    """PDF outlines use en/non-breaking spaces between number and title."""
    outline = [_entry(f"3.10.{n} MNEM{n}") for n in range(1, 12)]
    assert outline_candidates(outline)


def test_default_candidates_are_valid_regexes_with_one_group():
    import re
    for name, pattern in DEFAULT_CANDIDATES.items():
        rx = re.compile(pattern, re.MULTILINE)   # raises if malformed
        assert rx.groups == 1, f"{name} must capture exactly the entity name"
