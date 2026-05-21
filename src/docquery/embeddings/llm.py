from langchain_core.language_models import BaseChatModel

from docquery.config import Settings


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

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key or "sk-placeholder",
        temperature=temp,
    )
