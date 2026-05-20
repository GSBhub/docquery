import logging

from langchain_core.tools import Tool

from langchain_chroma.vectorstores import Chroma

logger = logging.getLogger(__name__)


def make_keyword_tool(vector_store: Chroma) -> Tool:
    def _search(query: str) -> str:
        results = _keyword_search(vector_store, query)
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


def _keyword_search(vector_store: Chroma, query: str) -> list[dict]:
    """Return docs whose content contains the query as a substring.

    Implemented on top of ChromaDB collection by scanning
    documents and filtering by substring.
    """

    try:
        collection = vector_store._collection # type: ignore[attr-defined]
    except Exception:
        logger.error("keyword_search: underlying Chromadb collection not available")
        return []

    raw = collection.get()
    docs = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []

    results: list[dict] = []
    for content, meta in zip(docs, metadatas, strict=False):
        if not content:
            continue
        if query not in content:
            continue
        meta = meta or {}
        results.append(
                {
                    "content": content,
                    "source": meta.get("source"),
                    "page": meta.get("page"),
                }
            )

    return results
