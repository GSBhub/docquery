import logging

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from docquery.config import Settings
from docquery.embeddings.llm import get_llm
from docquery._nodes import make_extraction_nodes
from docquery._state import ExtractionState
from docquery.tools.retrieval_tools import make_similarity_tool

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    def __init__(
        self,
        output_model: type[BaseModel],
        system_prompt: str,
        settings: Settings | None = None,
    ):
        self._output_model = output_model
        self._settings = settings or Settings()

        similarity_tool = make_similarity_tool(self._settings.vs, self._settings)
        llm = get_llm(self._settings)

        retrieve, extract, validate, should_retry = make_extraction_nodes(
            llm, similarity_tool, output_model, system_prompt, self._settings
        )

        graph = StateGraph(ExtractionState)
        graph.add_node("retrieve", retrieve)
        graph.add_node("extract", extract)
        graph.add_node("validate", validate)

        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "extract")
        graph.add_edge("extract", "validate")
        graph.add_conditional_edges("validate", should_retry, {"extract": "extract", "__end__": END})

        self._graph = graph.compile()

    def run(self, query: str) -> BaseModel:
        initial: ExtractionState = {
            "query": query,
            "retrieved_context": "",
            "raw_response": "",
            "validated": None,
            "validation_errors": [],
            "retry_count": 0,
        }
        final = self._graph.invoke(initial)
        if final["validated"] is None:
            raise ValueError(
                f"Extraction failed after {self._settings.max_retries} retries. "
                f"Last errors: {final['validation_errors']}"
            )
        return final["validated"]

    def run_batch(self, queries: list[str]) -> list[BaseModel]:
        return [self.run(q) for q in queries]
