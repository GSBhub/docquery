"""Tests for pre-chunk entity tagging (D1) and page normalization."""

from langchain_core.documents import Document

from docquery.config import EntityRule
from docquery._ingest import (
    _normalize_pages,
    match_page_entities,
    propagate_page_entities,
    tag_entities,
)
from docquery._pdf import extract_page_text, extract_outline


# --- helpers --------------------------------------------------------------

def _write_pdf(path, pages, toc=None):
    """Build a tiny multi-page PDF with optional bookmarks via pymupdf."""
    import fitz
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    if toc:
        doc.set_toc(toc)
    doc.save(str(path))
    doc.close()


# --- match_page_entities --------------------------------------------------

def test_match_page_entities_anchored_heading():
    page_text = {1: "A7.7.1 ADC\nAdd with carry\nA7.7.2 ADD\nAdd", 2: "no heading here"}
    rule = EntityRule(name="instruction", pattern=r"^A7\.7\.\d+\s+([A-Z][A-Z0-9]+)")
    out = match_page_entities(page_text, [rule])
    assert [e for e, _ in out[1]["instruction"]] == ["ADC", "ADD"]
    assert 2 not in out


def test_match_page_entities_empty_without_rules():
    assert match_page_entities({1: "A7.7.1 ADC"}, []) == {}


# --- propagate_page_entities ----------------------------------------------

def test_propagate_tags_chunk_containing_heading():
    docs = [
        Document(page_content="A7.7.1 ADC\nAdd with carry", metadata={"source": "m.pdf", "page": 1}),
        Document(page_content="continuation prose", metadata={"source": "m.pdf", "page": 1}),
    ]
    page_entities = {1: {"instruction": [("ADC", "A7.7.1 ADC")]}}
    n = propagate_page_entities(docs, page_entities, "m.pdf")
    assert n == 1
    assert docs[0].metadata["entity_instruction"] == "ADC"
    assert "entity_instruction" not in docs[1].metadata


def test_propagate_split_heading_falls_back_to_first_page_chunk():
    # heading line is absent from every chunk (split across boundary) -> first chunk
    docs = [
        Document(page_content="...prose only, no heading text...", metadata={"source": "m.pdf", "page": 3}),
        Document(page_content="more prose", metadata={"source": "m.pdf", "page": 3}),
    ]
    page_entities = {3: {"instruction": [("VLDR", "A7.7.250 VLDR")]}}
    propagate_page_entities(docs, page_entities, "m.pdf")
    assert docs[0].metadata["entity_instruction"] == "VLDR"


def test_propagate_matches_source_by_basename():
    docs = [Document(page_content="A7.7.1 ADC", metadata={"source": "/abs/path/m.pdf", "page": 1})]
    propagate_page_entities(docs, {1: {"instruction": [("ADC", "A7.7.1 ADC")]}}, "m.pdf")
    assert docs[0].metadata["entity_instruction"] == "ADC"


# --- _normalize_pages -----------------------------------------------------

def test_normalize_pages_maps_page_number():
    docs = [Document(page_content="x", metadata={"page_number": 4})]
    _normalize_pages(docs)
    assert docs[0].metadata["page"] == 4


def test_normalize_pages_keeps_existing_page():
    docs = [Document(page_content="x", metadata={"page": 2, "page_number": 9})]
    _normalize_pages(docs)
    assert docs[0].metadata["page"] == 2


# --- chunk-level fallback still works -------------------------------------

def test_chunk_level_fallback_still_tags_preloaded_documents():
    # pre-loaded Document (no PDF) — only the chunk-level tag_entities pass runs
    docs = [Document(page_content="A7.7.1 ADC", metadata={"page": 1})]
    rule = EntityRule(name="instruction", pattern=r"^A7\.7\.\d+\s+([A-Z][A-Z0-9]+)")
    tag_entities(docs, [rule])
    assert docs[0].metadata["entity_instruction"] == "ADC"


# --- pymupdf round-trip ---------------------------------------------------

def test_extract_page_text_roundtrip(tmp_path):
    pdf = tmp_path / "doc.pdf"
    _write_pdf(pdf, ["A7.7.1 ADC heading", "second page body"])
    pages = extract_page_text(pdf)
    assert "ADC" in pages[1]
    assert "second page" in pages[2]


def test_extract_page_text_bad_file_returns_empty(tmp_path):
    bad = tmp_path / "not.pdf"
    bad.write_text("not a pdf")
    assert extract_page_text(bad) == {}


def test_extract_outline_empty_without_toc(tmp_path):
    pdf = tmp_path / "doc.pdf"
    _write_pdf(pdf, ["page one"])
    assert extract_outline(pdf) == []
