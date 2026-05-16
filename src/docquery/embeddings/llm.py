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
        )

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key or "sk-placeholder",
        temperature=temp,
    )
