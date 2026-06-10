import logging

from langchain_core.documents import Document
from langchain_core.tools import BaseTool, tool

from langchain_chroma.vectorstores import Chroma

from docquery.config import Settings

logger = logging.getLogger(__name__)


def make_similarity_tool(vector_store: Chroma, settings: Settings | None = None) -> BaseTool:
    if settings is None:
        settings = Settings()
    k = settings.top_k

    @tool
    def similarity_search(query: str) -> str:
        """PRIMARY search tool. Use this FIRST for any question about the document — \
instruction behaviour, ISA overview, registers, encodings, or any other content. \
Accepts a natural-language query and returns the most relevant passages."""
        results = vector_store.similarity_search(query, k=k)
        logger.info("similarity_search: query=%r, returned %d results", query, len(results))
        if not results:
            return "No relevant content found."
        parts: list[str] = []
        for r in results:
            if isinstance(r, Document):
                meta = r.metadata or {}
                source = meta.get("source", "unknown")
                page = meta.get("page", "?")
                content = r.page_content
            else:
                try:
                    meta = r  # type: ignore[assignment]
                except Exception:
                    meta = {}
                source = meta.get("source", "unknown")
                page = meta.get("page", "?")
                content = meta.get("content", "")
            parts.append(f"[Source: {source}, Page {page}\n{content}")

        return "\n\n---\n\n".join(parts)

    return similarity_search
