import math
import sqlite3
import struct
from unittest.mock import MagicMock

import pytest


def test_add_and_count(populated_store, sample_chunks):
    assert populated_store.get_chunk_count() == len(sample_chunks)


def test_dedup(populated_store, sample_chunks, mock_embeddings):
    # Adding the same chunks again should be silently skipped
    count_before = populated_store.get_chunk_count()
    populated_store.add_chunks(sample_chunks, mock_embeddings, batch_size=4)
    assert populated_store.get_chunk_count() == count_before


def test_similarity_search_returns_results(populated_store, mock_embeddings):
    query_vec = mock_embeddings.embed_query("MOV")
    results = populated_store.similarity_search(query_vec, k=2)
    assert len(results) > 0
    assert "content" in results[0]


def test_keyword_search_finds_match(populated_store):
    results = populated_store.keyword_search("MOV")
    assert any("MOV" in r["content"] for r in results)


def test_keyword_search_no_match(populated_store):
    results = populated_store.keyword_search("NONEXISTENT_OPCODE_XYZ")
    assert results == []


def test_page_lookup(populated_store):
    results = populated_store.page_lookup(2)
    assert len(results) == 1
    assert "LDR" in results[0]["content"]


def test_page_lookup_empty(populated_store):
    results = populated_store.page_lookup(999)
    assert results == []


def test_large_float_values_roundtrip(tmp_path):
    """Values above float32 max (~3.4e38) are stored and retrieved without corruption."""
    from langchain_core.documents import Document

    from docquery.storage.vector_store import VectorStore

    large_val = 4.0e38  # overflows float32 → inf under old "f" format
    dim = 4
    vec = [large_val, 0.1, 0.2, 0.3]

    mock_emb = MagicMock()
    mock_emb.embed_documents.return_value = [vec]
    mock_emb.embed_query.return_value = vec

    vs = VectorStore(str(tmp_path / "large.db"), embedding_dim=dim)
    chunks = [Document(page_content="large value chunk",
                       metadata={"source": "t", "page": 1, "chunk_index": 0})]
    vs.add_chunks(chunks, mock_emb, batch_size=1)

    results = vs.similarity_search(vec, k=1)
    assert len(results) == 1
    assert results[0]["similarity"] == pytest.approx(1.0)
    vs.close()


def test_struct_format_is_double_precision(tmp_path):
    """BLOB stores 8 bytes per value (float64), not 4 bytes (float32)."""
    from langchain_core.documents import Document

    from docquery.storage.vector_store import VectorStore

    dim = 4
    vec = [1.0, 2.0, 3.0, 4.0]

    mock_emb = MagicMock()
    mock_emb.embed_documents.return_value = [vec]

    vs = VectorStore(str(tmp_path / "fmt.db"), embedding_dim=dim)
    chunks = [Document(page_content="format check",
                       metadata={"source": "t", "page": 1, "chunk_index": 0})]
    vs.add_chunks(chunks, mock_emb, batch_size=1)

    conn = sqlite3.connect(str(tmp_path / "fmt.db"))
    blob = conn.execute("SELECT embedding FROM chunk_embeddings LIMIT 1").fetchone()[0]
    conn.close()

    assert len(blob) == dim * 8, (
        f"Expected {dim * 8} bytes (float64), got {len(blob)} bytes. "
        f"Old float32 format would produce {dim * 4} bytes."
    )
    assert list(struct.unpack(f"{dim}d", blob)) == pytest.approx(vec)
    vs.close()


def test_large_float_does_not_produce_nan_similarity(tmp_path):
    """Cosine similarity is never nan or inf for large-valued embeddings."""
    from langchain_core.documents import Document

    from docquery.storage.vector_store import VectorStore

    dim = 4
    vec = [4.0e38, 1.0, 0.5, 0.1]

    mock_emb = MagicMock()
    mock_emb.embed_documents.return_value = [vec]
    mock_emb.embed_query.return_value = vec

    vs = VectorStore(str(tmp_path / "nan.db"), embedding_dim=dim)
    chunks = [Document(page_content="nan test chunk",
                       metadata={"source": "t", "page": 1, "chunk_index": 0})]
    vs.add_chunks(chunks, mock_emb, batch_size=1)

    results = vs.similarity_search(vec, k=1)
    assert len(results) == 1
    assert not math.isnan(results[0]["similarity"])
    assert not math.isinf(results[0]["similarity"])
    vs.close()
