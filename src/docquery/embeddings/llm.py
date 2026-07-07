import logging

from langchain_core.language_models import BaseChatModel

from docquery.config import Settings

logger = logging.getLogger(__name__)


def get_structured_llm(llm: BaseChatModel, schema, settings: Settings):
    """Bind *schema* to *llm* with the provider's structured-output support.

    For Ollama this selects ``method="json_schema"`` — grammar-constrained
    decoding, so schema compliance is guaranteed at the token level rather
    than prompted for. Returns a runnable yielding ``{"raw", "parsed",
    "parsing_error"}``, or ``None`` when the provider/model cannot bind the
    schema (callers fall back to prompt-based JSON).
    """
    try:
        if settings.llm_provider == "ollama":
            return llm.with_structured_output(schema, method="json_schema", include_raw=True)
        return llm.with_structured_output(schema, include_raw=True)
    except Exception as exc:  # noqa: BLE001 - fall back to prompt-based JSON
        logger.warning("Structured output unavailable for provider %r: %s",
                       settings.llm_provider, exc)
        return None


def get_llm(settings: Settings | None = None) -> BaseChatModel:
    if settings is None:
        settings = Settings()

    temp = getattr(settings, "temperature", 0)

    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            temperature=temp,
            num_predict=settings.llm_num_predict,
            sync_client_kwargs={"timeout": settings.llm_timeout},
            client_kwargs={"timeout": settings.llm_timeout},
        )

    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            temperature=temp,
        )

    if settings.llm_provider == "azure":
        from langchain_openai import AzureChatOpenAI
        if not settings.llm_base_url:
            raise ValueError(
                "LLM_PROVIDER=azure requires an endpoint: "
                "set AZURE_OPENAI_ENDPOINT or LLM_BASE_URL")
        return AzureChatOpenAI(
            azure_endpoint=settings.llm_base_url,
            azure_deployment=settings.llm_model,
            api_version=settings.llm_api_version or None,
            api_key=settings.llm_api_key or None,
            temperature=temp,
        )

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key or "sk-placeholder",
        temperature=temp,
    )
