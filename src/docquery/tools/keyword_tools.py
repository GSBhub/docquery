import logging

from langchain_core.tools import Tool

from docquery.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


def make_keyword_tool(vector_store: VectorStore) -> Tool:
    def _search(query: str) -> str:
        logger.debug("keyword_search FTS5 query=%r", query)
        results = vector_store.keyword_search(query)
        logger.info("keyword_search: query=%r, returned %d matches", query, len(results))
        if not results:
            return f"No exact matches found for: {query!r}"
        parts = []
        for r in results:
            parts.append(f"[Source: {r['source']}, Page: {r['page']}]\n{r['content']}")
        return "\n\n---\n\n".join(parts)

    return Tool(
        name="keyword_search",
        description=(
            "SECONDARY search tool for exact string matches. "
            "Use this only to supplement similarity_search when you need to find a specific hex value, "
            "opcode byte sequence, or known identifier (e.g. '0xfe 0xed', 'STMDB'). "
            "Do NOT use as the first tool — always call similarity_search first."
        ),
        func=_search,
    )
