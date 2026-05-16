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
