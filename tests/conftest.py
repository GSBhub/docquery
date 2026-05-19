from unittest.mock import MagicMock

import chromadb
import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document

from docquery.config import Settings


@pytest.fixture
def settings():
    s = Settings()
    s.top_k = 3
    s.max_retries = 3
    return s


@pytest.fixture
def mock_embeddings():
    """Returns fixed 8-dim vectors for testing without an API."""
    emb = MagicMock()
    emb.embed_documents.side_effect = lambda texts: [[0.1 * i for i in range(8)] for _ in texts]
    emb.embed_query.return_value = [0.1 * i for i in range(8)]
    return emb


@pytest.fixture
def vector_store(mock_embeddings):
    """Ephemeral in-memory Chroma store — no disk I/O, isolated per test."""
    client = chromadb.EphemeralClient()
    store = Chroma(
        client=client,
        collection_name="test_collection",
        embedding_function=mock_embeddings,
    )
    yield store


@pytest.fixture
def sample_documents():
    return [
        Document(page_content="The MOV instruction copies a value.",
                 metadata={"source": "test.pdf", "page": 1}),
        Document(page_content="The LDR instruction loads from memory.",
                 metadata={"source": "test.pdf", "page": 2}),
        Document(page_content="Opcode 0xfe 0xed is a special encoding.",
                 metadata={"source": "test.pdf", "page": 3}),
    ]


@pytest.fixture
def populated_store(vector_store, sample_documents):
    vector_store.add_documents(sample_documents)
    return vector_store
