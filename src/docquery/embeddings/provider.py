from langchain_core.embeddings import Embeddings

from docquery.config import Settings


def _is_non_openai_base_url(base_url: str) -> bool:
    """ Heuristic: treat any non-empty base URL that does not
    point at openai as a non-openai provider.

    Apparently, openai has some weird ass optimization where you
    can just shoot token ID arrays at it instead of embedding strings.
    """

    if not base_url:
        return False

    lowered = base_url.lower()
    return "api.openai.com" not in lowered


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    if settings is None:
        settings = Settings()

    if settings.embed_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        # Proxied Ollama endpoints (e.g. open-webui's /ollama passthrough)
        # authenticate with a bearer token that plain Ollama ignores.
        ollama_kwargs: dict[str, object] = {}
        if settings.embed_api_key:
            headers = {"Authorization": f"Bearer {settings.embed_api_key}"}
            ollama_kwargs["client_kwargs"] = {"headers": headers}
        return OllamaEmbeddings(
            model=settings.embed_model,
            base_url=settings.embed_base_url,
            **ollama_kwargs,
        )

    if settings.embed_provider == "azure":
        from langchain_openai import AzureOpenAIEmbeddings
        if not settings.embed_base_url:
            raise ValueError(
                "EMBED_PROVIDER=azure requires an endpoint: "
                "set AZURE_OPENAI_ENDPOINT or EMBED_BASE_URL")
        # Azure speaks the real OpenAI embeddings protocol, so the
        # non-openai base-url heuristic below must not apply to it.
        return AzureOpenAIEmbeddings(
            azure_endpoint=settings.embed_base_url,
            azure_deployment=settings.embed_model,
            api_version=settings.embed_api_version or None,
            api_key=settings.embed_api_key or None,
        )

    from langchain_openai import OpenAIEmbeddings

    # Disable the embedding ctx + embeddings vector input type 
    # for not openai api endpoints (ie, self-hosted, ollama)
    extra_kwargs: dict[str, object] = {}
    if _is_non_openai_base_url(settings.embed_base_url):
        extra_kwargs["check_embedding_ctx_length"] = False

    return OpenAIEmbeddings(
        model=settings.embed_model,
        base_url=settings.embed_base_url,
        api_key=settings.embed_api_key or "sk-placeholder",
        **extra_kwargs,
    )
