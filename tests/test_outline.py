"""Tests for outline extraction, persistence, and the public outline() API."""

import chromadb

from docquery.config import Settings
from docquery._pdf import OUTLINE_COLLECTION, extract_outline, persist_outline, read_outline


def _fresh_client():
    """A client with no leftover outline collection.

    chromadb.EphemeralClient() returns a process-shared instance, so the sidecar
    db_outline collection persists across tests; drop it for isolation.
    """
    client = chromadb.EphemeralClient()
    try:
        client.delete_collection(OUTLINE_COLLECTION)
    except Exception:
        pass
    return client


def _write_pdf_with_toc(path, pages, toc):
    import fitz
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.set_toc(toc)
    doc.save(str(path))
    doc.close()


def test_extract_outline_from_pdf_toc(tmp_path):
    pdf = tmp_path / "doc.pdf"
    # [level, title, page]
    toc = [[1, "Chapter A", 1], [2, "Data Processing", 2], [2, "Memory", 3]]
    _write_pdf_with_toc(pdf, ["p1", "p2", "p3"], toc)
    outline = extract_outline(pdf)
    assert [e["title"] for e in outline] == ["Chapter A", "Data Processing", "Memory"]
    assert outline[1] == {"title": "Data Processing", "page": 2, "level": 2, "order": 1}


def test_persist_and_read_outline_roundtrip():
    s = Settings()
    s.db_client = _fresh_client()
    outline = [
        {"title": "Chapter A", "page": 1, "level": 1, "order": 0},
        {"title": "Memory", "page": 8, "level": 2, "order": 1},
    ]
    persist_outline(s, outline, "m.pdf")
    got = read_outline(s)
    assert [e["title"] for e in got] == ["Chapter A", "Memory"]
    assert got[0]["page"] == 1 and got[0]["level"] == 1


def test_read_outline_empty_when_never_ingested():
    s = Settings()
    s.db_client = _fresh_client()
    assert read_outline(s) == []


def test_persist_outline_noop_on_empty():
    s = Settings()
    s.db_client = _fresh_client()
    persist_outline(s, [], "m.pdf")
    assert read_outline(s) == []
