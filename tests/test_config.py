import pytest

from docquery.config import Settings


def test_openai_provider_defaults_to_openai_url(monkeypatch):
    monkeypatch.delenv("EMBED_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("EMBED_PROVIDER", "openai")
    assert Settings().embed_base_url == "https://api.openai.com/v1"


def test_ollama_embed_url_defaults_to_localhost(monkeypatch):
    """Regression: was incorrectly defaulting to https://api.openai.com/v1."""
    monkeypatch.setenv("EMBED_PROVIDER", "ollama")
    monkeypatch.delenv("EMBED_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert Settings().embed_base_url == "http://localhost:11434"


def test_ollama_llm_url_defaults_to_localhost(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert Settings().llm_base_url == "http://localhost:11434"


def test_ollama_url_is_not_openai(monkeypatch):
    monkeypatch.setenv("EMBED_PROVIDER", "ollama")
    monkeypatch.delenv("EMBED_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert "openai.com" not in Settings().embed_base_url


def test_explicit_embed_base_url_takes_priority(monkeypatch):
    monkeypatch.setenv("EMBED_PROVIDER", "ollama")
    monkeypatch.setenv("EMBED_BASE_URL", "http://myserver:11434")
    assert Settings().embed_base_url == "http://myserver:11434"


def test_ollama_host_env_var_used_when_no_embed_base_url(monkeypatch):
    monkeypatch.setenv("EMBED_PROVIDER", "ollama")
    monkeypatch.delenv("EMBED_BASE_URL", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "http://myserver:11434")
    assert Settings().embed_base_url == "http://myserver:11434"


def test_openai_base_url_env_var_used_as_fallback(monkeypatch):
    monkeypatch.setenv("EMBED_PROVIDER", "openai")
    monkeypatch.delenv("EMBED_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://my-azure.openai.azure.com/")
    assert Settings().embed_base_url == "https://my-azure.openai.azure.com/"


def test_openai_api_key_fallback(monkeypatch):
    monkeypatch.delenv("EMBED_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mykey")
    assert Settings().embed_api_key == "sk-mykey"


def test_llm_api_key_fallback(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mykey")
    assert Settings().llm_api_key == "sk-mykey"


def test_docquery_specific_key_takes_priority_over_openai(monkeypatch):
    monkeypatch.setenv("EMBED_API_KEY", "sk-specific")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-generic")
    assert Settings().embed_api_key == "sk-specific"


def test_azure_base_url_from_azure_endpoint(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://myresource.openai.azure.com")
    s = Settings()
    assert s.llm_base_url == "https://myresource.openai.azure.com"
    assert "api.openai.com" not in s.llm_base_url


def test_azure_embed_base_url_from_azure_endpoint(monkeypatch):
    monkeypatch.setenv("EMBED_PROVIDER", "azure")
    monkeypatch.delenv("EMBED_BASE_URL", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://myresource.openai.azure.com")
    assert Settings().embed_base_url == "https://myresource.openai.azure.com"


def test_azure_api_version_fallback_to_openai_api_version(monkeypatch):
    monkeypatch.delenv("LLM_API_VERSION", raising=False)
    monkeypatch.delenv("EMBED_API_VERSION", raising=False)
    monkeypatch.setenv("OPENAI_API_VERSION", "2024-10-21")
    s = Settings()
    assert s.llm_api_version == "2024-10-21"
    assert s.embed_api_version == "2024-10-21"


def test_azure_api_key_beats_openai_key_for_azure(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "sk-azure")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-generic")
    assert Settings().llm_api_key == "sk-azure"


def test_explicit_llm_api_key_beats_azure_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("LLM_API_KEY", "sk-explicit")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "sk-azure")
    assert Settings().llm_api_key == "sk-explicit"


def test_azure_key_not_used_for_non_azure_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "sk-azure")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-generic")
    assert Settings().llm_api_key == "sk-generic"
