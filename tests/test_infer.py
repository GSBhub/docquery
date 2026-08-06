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


# --- score_patterns ---------------------------------------------------------

def _pdf_with(tmp_path, pages: list[str]):
    """A tiny real PDF whose pages contain the given text lines."""
    import fitz
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 100), text, fontsize=11)
    path = tmp_path / "doc.pdf"
    doc.save(str(path))
    doc.close()
    return path


def test_precision_counts_distinct_names_not_per_page_matches(tmp_path, monkeypatch):
    """A name recurring across pages must not inflate the denominator.

    match_page_entities dedupes within a page but not across, so counting raw
    matches would penalise a rule purely for tagging entities whose descriptions
    span several pages.
    """
    from docquery import _infer

    # One entity ("ADD") tagged on three pages, owning one block.
    per_candidate = {
        "c": {1: {"e": [("ADD", "Syntax ADD")]},
              2: {"e": [("ADD", "Syntax ADD")]},
              3: {"e": [("ADD", "Syntax ADD")]}},
    }
    monkeypatch.setattr(_infer, "_scan",
                        lambda *_a, **_k: ({1: [(200, "encoding")]},
                                           {"c": {1: [(100, "ADD")]}}, 1))
    monkeypatch.setattr("docquery._pdf.extract_page_text", lambda *_a, **_k: {1: "", 2: "", 3: ""})
    monkeypatch.setattr("docquery._ingest.match_page_entities",
                        lambda *_a, **_k: per_candidate["c"])

    res = _infer.score_patterns(tmp_path / "x.pdf", candidates={"c": r"([A-Z]+)"})
    assert res[0]["entities"] == 1, "distinct names, not 3 per-page matches"
    assert res[0]["precision"] == 1.0
    assert res[0]["recall"] == 1.0


def test_score_patterns_ranks_a_precise_rule_over_an_over_matching_one(tmp_path, monkeypatch):
    """Recall alone prefers the rule that tags everything; precision must win."""
    from docquery import _infer

    blocks = {1: [(200, "encoding")], 2: [(200, "encoding")]}
    matches = {
        # precise: two entities, each owning a block
        "precise": {1: {"e": [("ADD", "Syntax ADD")]}, 2: {"e": [("SUB", "Syntax SUB")]}},
        # greedy: tags a lot of noise, still owns everything
        "greedy": {1: {"e": [("ADD", "ADD"), ("N1", "N1"), ("N2", "N2")]},
                   2: {"e": [("SUB", "SUB"), ("N3", "N3"), ("N4", "N4")]}},
    }
    heads = {"precise": {1: [(100, "ADD")], 2: [(100, "SUB")]},
             "greedy": {1: [(100, "ADD")], 2: [(100, "SUB")]}}
    monkeypatch.setattr(_infer, "_scan", lambda *_a, **_k: (blocks, heads, 2))
    monkeypatch.setattr("docquery._pdf.extract_page_text", lambda *_a, **_k: {1: "", 2: ""})
    # key off the pattern: score_patterns reuses one entity_name for every
    # candidate, so the name cannot distinguish them.
    by_pattern = {"a": matches["precise"], "b": matches["greedy"]}
    monkeypatch.setattr("docquery._ingest.match_page_entities",
                        lambda _t, rules: by_pattern[rules[0].pattern])

    res = _infer.score_patterns(tmp_path / "x.pdf",
                                candidates={"precise": "a", "greedy": "b"})
    assert [r["name"] for r in res][0] == "precise"
    assert res[0]["recall"] == 1.0
