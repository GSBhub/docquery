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


def test_get_llm_azure():
    from langchain_openai import AzureChatOpenAI
    from docquery.config import Settings
    from docquery.embeddings.llm import get_llm

    s = Settings(
        llm_provider="azure",
        llm_base_url="https://myresource.openai.azure.com",
        llm_model="my-deployment",
        llm_api_key="test",
        llm_api_version="2024-10-21",
    )
    result = get_llm(s)
    assert isinstance(result, AzureChatOpenAI)
    assert result.deployment_name == "my-deployment"


def test_get_structured_llm_uses_json_schema_for_ollama():
    from unittest.mock import MagicMock
    from pydantic import BaseModel
    from docquery.config import Settings
    from docquery.embeddings.llm import get_structured_llm

    class M(BaseModel):
        x: int

    llm = MagicMock()
    s = Settings(llm_provider="ollama")
    result = get_structured_llm(llm, M, s)
    assert result is llm.with_structured_output.return_value
    llm.with_structured_output.assert_called_once_with(M, method="json_schema", include_raw=True)


def test_get_structured_llm_returns_none_when_unsupported():
    from unittest.mock import MagicMock
    from pydantic import BaseModel
    from docquery.config import Settings
    from docquery.embeddings.llm import get_structured_llm

    class M(BaseModel):
        x: int

    llm = MagicMock()
    llm.with_structured_output.side_effect = NotImplementedError
    assert get_structured_llm(llm, M, Settings(llm_provider="openai")) is None


def test_get_llm_azure_without_endpoint_raises(monkeypatch):
    from docquery.config import Settings
    from docquery.embeddings.llm import get_llm

    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    s = Settings(llm_provider="azure", llm_api_key="test", llm_api_version="2024-10-21")
    with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
        get_llm(s)
