import logging
import re

from langchain_core.tools import Tool

from docquery.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


def make_page_lookup_tool(vector_store: VectorStore) -> Tool:
    def _lookup(query: str) -> str:
        # Accept "50", "page 50", "page_number: 50", etc.
        match = re.search(r"\d+", query)
        if not match:
            return "Could not parse a page number from the input. Provide a number."
        page_num = int(match.group())
        logger.debug("page_lookup SQL: page=%d", page_num)
        results = vector_store.page_lookup(page_num)
        logger.info("page_lookup: page=%d, returned %d chunks", page_num, len(results))
        if not results:
            return f"No content found for page {page_num}."
        return "\n\n".join(r["content"] for r in results)

    return Tool(
        name="page_lookup",
        description=(
            "Retrieve all text from a specific page number of the document. "
            "Input should be a page number (e.g. '50' or 'page 50')."
        ),
        func=_lookup,
    )
