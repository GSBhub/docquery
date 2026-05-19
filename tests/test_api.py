"""Smoke tests for the public docquery library API."""
from unittest.mock import MagicMock, patch

import chromadb
import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

import docquery
from docquery import ChatSession, Settings
from docquery._chat import ChatAgent


@pytest.fixture
def mock_settings(mock_embeddings):
    s = Settings()
    s.top_k = 3
    client = chromadb.EphemeralClient()
    s.db_client = client
    s.vs = Chroma(client=client, collection_name="api_test", embedding_function=mock_embeddings)
    return s


def test_ingest_accepts_documents(mock_settings):
    docs = [Document(page_content="Hello world.", metadata={"source": "t.txt", "page": 1})]
    count = docquery.ingest(docs, settings=mock_settings)
    assert count == 1


def test_ingest_reuses_existing_vs(mock_settings):
    docs = [
        Document(page_content="Doc one.", metadata={"source": "a.txt", "page": 1}),
        Document(page_content="Doc two.", metadata={"source": "b.txt", "page": 1}),
    ]
    count = docquery.ingest(docs, settings=mock_settings)
    assert count == 2


def test_chat_session_returns_chat_session(mock_settings):
    with patch("docquery._chat.get_llm") as mock_get_llm:
        base_llm = MagicMock()
        bound_llm = MagicMock()
        base_llm.bind_tools.return_value = bound_llm
        mock_get_llm.return_value = base_llm
        session = docquery.chat_session(settings=mock_settings)

    assert isinstance(session, ChatSession)
    assert hasattr(session, "chat")
    assert hasattr(session, "reset")


def test_chat_session_reset(mock_settings):
    with patch("docquery._chat.get_llm") as mock_get_llm:
        base_llm = MagicMock()
        bound_llm = MagicMock()
        base_llm.bind_tools.return_value = bound_llm
        mock_get_llm.return_value = base_llm
        session = docquery.chat_session(settings=mock_settings)

    session.reset()  # should not raise


def test_query_plain_returns_string(mock_settings):
    with patch("docquery._chat.get_llm") as mock_get_llm:
        base_llm = MagicMock()
        bound_llm = MagicMock()
        bound_llm.invoke.return_value = AIMessage(content="The answer is 42.")
        base_llm.bind_tools.return_value = bound_llm
        mock_get_llm.return_value = base_llm

        agent = MagicMock()
        agent.chat.return_value = "The answer is 42."

        with patch("docquery._chat.ChatAgent") as MockAgent:
            MockAgent.return_value = agent
            result = docquery.query("What is the answer?", settings=mock_settings)

    assert isinstance(result, str)
