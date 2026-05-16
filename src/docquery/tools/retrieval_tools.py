import logging

from langchain_core.embeddings import Embeddings
from langchain_core.tools import Tool

from docquery.config import Settings
from docquery.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


def make_similarity_tool(vector_store: VectorStore, embeddings: Embeddings, settings: Settings | None = None) -> Tool:
    if settings is None:
        settings = Settings()
    k = settings.top_k

    def _search(query: str) -> str:
        vec = embeddings.embed_query(query)
        logger.debug("similarity_search embedding norm=%.4f", sum(v**2 for v in vec) ** 0.5)
        results = vector_store.similarity_search(vec, k=k)
        logger.info("similarity_search: query=%r, returned %d results", query, len(results))
        if not results:
            return "No relevant content found."
        parts = []
        for r in results:
            parts.append(f"[Source: {r['source']}, Page: {r['page']}]\n{r['content']}")
        return "\n\n---\n\n".join(parts)

    return Tool(
        name="similarity_search",
        description=(
            "PRIMARY search tool. Use this FIRST for any question about the document — "
            "instruction behaviour, ISA overview, registers, encodings, or any other content. "
            "Accepts a natural-language query and returns the most relevant passages."
        ),
        func=_search,
    )
