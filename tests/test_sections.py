"""Tests for outline-driven section assignment (D3)."""

from langchain_core.documents import Document

from docquery._ingest import assign_sections, _outline_paths


_OUTLINE = [
    {"title": "Chapter A", "page": 1, "level": 1, "order": 0},
    {"title": "Data Processing", "page": 3, "level": 2, "order": 1},
    {"title": "Memory", "page": 8, "level": 2, "order": 2},
]


def test_outline_paths_builds_nested_path():
    annotated = _outline_paths(_OUTLINE)
    assert annotated[1]["path"] == "Chapter A > Data Processing"
    assert annotated[2]["path"] == "Chapter A > Memory"


def test_assign_sections_deepest_enclosing():
    docs = [
        Document(page_content="x", metadata={"source": "m.pdf", "page": 5}),  # under Data Processing
        Document(page_content="y", metadata={"source": "m.pdf", "page": 9}),  # under Memory
    ]
    n = assign_sections(docs, _OUTLINE, "m.pdf")
    assert n == 2
    assert docs[0].metadata["section"] == "Data Processing"
    assert docs[0].metadata["section_order"] == 1
    assert docs[1].metadata["section"] == "Memory"


def test_assign_sections_builds_path():
    docs = [Document(page_content="x", metadata={"source": "m.pdf", "page": 4})]
    assign_sections(docs, _OUTLINE, "m.pdf")
    assert docs[0].metadata["section_path"] == "Chapter A > Data Processing"


def test_chunks_before_first_bookmark_unsectioned():
    docs = [Document(page_content="cover", metadata={"source": "m.pdf", "page": 1})]
    out = [{"title": "Body", "page": 2, "level": 1, "order": 0}]
    assign_sections(docs, out, "m.pdf")
    assert "section" not in docs[0].metadata


def test_assign_sections_skips_other_sources():
    docs = [Document(page_content="x", metadata={"source": "other.pdf", "page": 5})]
    assert assign_sections(docs, _OUTLINE, "m.pdf") == 0
    assert "section" not in docs[0].metadata


def test_assign_sections_empty_outline_noop():
    docs = [Document(page_content="x", metadata={"source": "m.pdf", "page": 5})]
    assert assign_sections(docs, [], "m.pdf") == 0
