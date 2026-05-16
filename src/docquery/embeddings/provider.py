from langchain_core.embeddings import Embeddings

from docquery.config import Settings


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    if settings is None:
        settings = Settings()

    if settings.embed_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=settings.embed_model,
            base_url=settings.embed_base_url,
        )

    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=settings.embed_model,
        base_url=settings.embed_base_url,
        api_key=settings.embed_api_key or "sk-placeholder",
    )
