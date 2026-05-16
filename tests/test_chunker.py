from langchain_core.documents import Document

from docquery.config import Settings
from docquery.ingestion.chunker import chunk


def test_chunk_produces_output():
    docs = [Document(page_content="word " * 500, metadata={"source": "t.pdf", "page": 0})]
    settings = Settings()
    settings.chunk_size = 200
    settings.chunk_overlap = 20
    chunks = chunk(docs, settings)
    assert len(chunks) > 1


def test_chunk_metadata_preserved():
    docs = [Document(page_content="hello world " * 100, metadata={"source": "t.pdf", "page": 5})]
    settings = Settings()
    settings.chunk_size = 200
    settings.chunk_overlap = 0
    chunks = chunk(docs, settings)
    for c in chunks:
        assert c.metadata["source"] == "t.pdf"
        assert c.metadata["page"] == 5
        assert "chunk_index" in c.metadata


def test_chunk_size_bounded():
    docs = [Document(page_content="x " * 1000, metadata={"source": "t.pdf", "page": 0})]
    settings = Settings()
    settings.chunk_size = 100
    settings.chunk_overlap = 10
    chunks = chunk(docs, settings)
    for c in chunks:
        assert len(c.page_content) <= 150  # some tolerance for splitter behaviour
