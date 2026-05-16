from unittest.mock import MagicMock

import pytest

from docquery.config import Settings


@pytest.fixture
def settings(tmp_path):
    s = Settings()
    s.db_path = str(tmp_path / "test.db")
    s.embed_batch_size = 4
    s.top_k = 3
    return s


@pytest.fixture
def mock_embeddings():
    """Returns fixed-length zero vectors (dim=8) for testing without an API."""
    emb = MagicMock()
    emb.embed_documents.side_effect = lambda texts: [[0.1 * i for i in range(8)] for _ in texts]
    emb.embed_query.return_value = [0.1 * i for i in range(8)]
    return emb


@pytest.fixture
def vector_store(settings, mock_embeddings):
    from docquery.storage.vector_store import VectorStore
    vs = VectorStore(settings.db_path, embedding_dim=8)
    yield vs
    vs.close()


@pytest.fixture
def sample_chunks():
    from langchain_core.documents import Document
    return [
        Document(page_content="The MOV instruction copies a value.", metadata={"source": "test.pdf", "page": 1, "chunk_index": 0}),
        Document(page_content="The LDR instruction loads from memory.", metadata={"source": "test.pdf", "page": 2, "chunk_index": 0}),
        Document(page_content="Opcode 0xfe 0xed is a special encoding.", metadata={"source": "test.pdf", "page": 3, "chunk_index": 0}),
    ]


@pytest.fixture
def populated_store(vector_store, sample_chunks, mock_embeddings):
    vector_store.add_chunks(sample_chunks, mock_embeddings, batch_size=4)
    return vector_store
