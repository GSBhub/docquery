import pytest

from docquery.config import Settings
from docquery.embeddings.provider import get_embeddings


def test_get_embeddings_ollama():
    from langchain_ollama import OllamaEmbeddings

    s = Settings(embed_provider="ollama", embed_model="nomic-embed-text")
    result = get_embeddings(s)
    assert isinstance(result, OllamaEmbeddings)


def test_get_embeddings_azure():
    from langchain_openai import AzureOpenAIEmbeddings

    s = Settings(
        embed_provider="azure",
        embed_base_url="https://myresource.openai.azure.com",
        embed_model="my-embed-deployment",
        embed_api_key="test",
        embed_api_version="2024-10-21",
    )
    result = get_embeddings(s)
    assert isinstance(result, AzureOpenAIEmbeddings)
    assert result.deployment == "my-embed-deployment"


def test_get_embeddings_azure_without_endpoint_raises(monkeypatch):
    monkeypatch.delenv("EMBED_BASE_URL", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    s = Settings(embed_provider="azure", embed_api_key="test", embed_api_version="2024-10-21")
    with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
        get_embeddings(s)
