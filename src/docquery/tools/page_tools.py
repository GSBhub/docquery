import logging
import re

from langchain_core.tools import BaseTool, tool

from langchain_chroma.vectorstores import Chroma

logger = logging.getLogger(__name__)


def make_page_lookup_tool(vector_store: Chroma) -> BaseTool:
    @tool
    def page_lookup(query: str) -> str:
        """Retrieve all text from a specific page number of the document. \
Input should be a page number (e.g. '50' or 'page 50')."""
        # Accept "50", "page 50", "page_number: 50", etc.
        match = re.search(r"\d+", query)
        if not match:
            return "Could not parse a page number from the input. Provide a number."
        page_num = int(match.group())
        logger.debug("page_lookup: page=%d", page_num)
        results = _page_lookup(vector_store, page_num)
        logger.info("page_lookup: page=%d, returned %d chunks", page_num, len(results))
        if not results:
            return f"No content found for page {page_num}."
        return "\n\n".join(r["content"] for r in results)

    return page_lookup

def _page_lookup(vector_store: Chroma, page_num: int) -> list[dict]:
    """ Return all chunks wohse metadata.page equals the given page number."""

    try:
        collection = vector_store._collection # type: ignore[attr-defined]
    except Exception:
        logger.error("page_lookup: underlying ChromaDB collection not available")
        return []
    
    raw = collection.get(where={"page": page_num})
    docs = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []

    results: list[dict] = []
    for content, meta in zip(docs, metadatas, strict=False):
        if not content:
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
