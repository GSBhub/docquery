import pytest


def test_get_llm_ollama_timeout():
    from langchain_ollama import ChatOllama
    from docquery.config import Settings
    from docquery.embeddings.llm import get_llm

    s = Settings(llm_provider="ollama", llm_timeout=30, llm_num_predict=512)
    result = get_llm(s)
    assert isinstance(result, ChatOllama)
    assert result.num_predict == 512
    assert result.sync_client_kwargs == {"timeout": 30}


def test_get_llm_anthropic():
    langchain_anthropic = pytest.importorskip("langchain_anthropic")
    from docquery.config import Settings
    from docquery.embeddings.llm import get_llm

    s = Settings(llm_provider="anthropic", llm_model="claude-haiku-4-5-20251001", llm_api_key="test")
    result = get_llm(s)
    assert isinstance(result, langchain_anthropic.ChatAnthropic)
