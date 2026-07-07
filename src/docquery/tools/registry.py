from langchain_core.tools import BaseTool
from langchain_chroma.vectorstores import Chroma

from docquery.config import Settings
from docquery.tools.cursor_tools import make_cursor_tools
from docquery.tools.keyword_tools import make_keyword_tool
from docquery.tools.page_tools import make_page_lookup_tool
from docquery.tools.retrieval_tools import make_similarity_tool
from docquery.tools.structure_tools import make_structure_lookup_tool


class ToolRegistry:
    def __init__(self, vector_store: Chroma, settings: Settings | None = None):
        if settings is None:
            settings = Settings()
        self._tools: list[BaseTool] = [
            make_similarity_tool(vector_store, settings),
            make_page_lookup_tool(vector_store),
            make_keyword_tool(vector_store),
            make_structure_lookup_tool(vector_store),
            *make_cursor_tools(vector_store, settings),
        ]

    def register(self, tool: BaseTool) -> None:
        """Add a user-defined tool (e.g. an external API or local skill)."""
        self._tools.append(tool)

    def get_tools(self) -> list[BaseTool]:
        return list(self._tools)
