"""docquery — ingest files and query them via a LangGraph RAG pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from docquery.config import EntityRule, Settings
from docquery._chat import ChatSession

if TYPE_CHECKING:
    from langchain_core.documents import Document


def ingest(
    items: list,
    *,
    settings: Settings | None = None,
    collection_name: str = "db_knowledge",
) -> int:
    """Ingest files (str/Path) and/or pre-loaded LangChain Documents.

    Args:
        items: A list of file paths (str or Path) and/or LangChain Document objects.
        settings: Optional Settings instance. If None, constructed from env vars.
        collection_name: ChromaDB collection to store documents in.

    Returns:
        Number of documents added to the vector store.
    """
    from docquery._ingest import ingest_documents
    s = settings or Settings()
    return ingest_documents(items, s, collection_name)


def query(
    prompt: str,
    *,
    schema: type[BaseModel] | None = None,
    system_prompt: str | None = None,
    doc_language: str | None = None,
    settings: Settings | None = None,
) -> str | BaseModel:
    """One-shot RAG query against the vector store.

    Without schema: returns a plain str answer via the chat agent.
    With schema: returns a validated Pydantic model instance via the extraction pipeline.

    Args:
        prompt: The question or extraction instruction.
        schema: Optional Pydantic model class for structured extraction.
        system_prompt: Override the default system prompt.
        doc_language: Language the source document is written in (overrides
            the DOC_LANGUAGE env var / ``settings.doc_language``).
        settings: Optional Settings instance. If None, constructed from env vars.

    Returns:
        A str answer, or a validated Pydantic instance when schema is provided.
    """
    from docquery._ingest import _build_chroma
    s = settings or Settings()
    if doc_language is not None:
        s.doc_language = doc_language
    _build_chroma(s)

    if schema is not None:
        from docquery._extractor import ExtractionPipeline
        pipeline = ExtractionPipeline(
            output_model=schema,
            system_prompt=system_prompt or f"Extract structured data matching the {schema.__name__} schema.",
            settings=s,
        )
        return pipeline.run(prompt)

    from docquery._chat import ChatAgent
    from docquery.tools.registry import ToolRegistry
    registry = ToolRegistry(s.vs, s)
    agent = ChatAgent(registry, s, system_prompt=system_prompt)
    return agent.chat(prompt)


def chat_session(
    *,
    system_prompt: str | None = None,
    doc_language: str | None = None,
    settings: Settings | None = None,
) -> ChatSession:
    """Create a stateful ChatSession backed by the configured vector store.

    Args:
        system_prompt: Override the default system prompt.
        doc_language: Language the source document is written in (overrides
            the DOC_LANGUAGE env var / ``settings.doc_language``).
        settings: Optional Settings instance. If None, constructed from env vars.

    Returns:
        A ChatSession with .chat(message) and .reset() methods.
    """
    from docquery._ingest import _build_chroma
    from docquery._chat import ChatAgent
    from docquery.tools.registry import ToolRegistry
    s = settings or Settings()
    if doc_language is not None:
        s.doc_language = doc_language
    _build_chroma(s)
    registry = ToolRegistry(s.vs, s)
    agent = ChatAgent(registry, s, system_prompt=system_prompt)
    return ChatSession(agent)


def outline(db_path: str | None = None, *, settings: Settings | None = None) -> list[dict]:
    """Return the persisted document outline (PDF bookmarks) for a store.

    Args:
        db_path: Path to the ChromaDB store. Ignored if ``settings`` is given.
        settings: Optional Settings instance. If None, constructed from env vars.

    Returns:
        Ordered list of ``{"title", "page", "level"}``; ``[]`` if none was captured.
    """
    from docquery._pdf import read_outline
    s = settings or Settings()
    if db_path is not None:
        s.db_path = db_path
    return read_outline(s)


def coverage(db_path: str | None = None, *, settings: Settings | None = None) -> dict[str, dict]:
    """Return counts of distinct tagged entities per type (and per section).

    Args:
        db_path: Path to the ChromaDB store. Ignored if ``settings`` is given.
        settings: Optional Settings instance. If None, constructed from env vars.

    Returns:
        ``{entity_type: {"count": int, "by_section": {section_title: int}}}``.
    """
    from docquery._coverage import entity_coverage
    s = settings or Settings()
    if db_path is not None:
        s.db_path = db_path
    return entity_coverage(s)


__all__ = ["ingest", "query", "chat_session", "outline", "coverage",
           "ChatSession", "Settings", "EntityRule"]
