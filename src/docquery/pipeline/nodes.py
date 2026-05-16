import json
import logging
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from docquery.config import Settings
from docquery.pipeline.state import ExtractionState
from docquery.tools.retrieval_tools import make_similarity_tool

logger = logging.getLogger(__name__)


def make_extraction_nodes(
    llm: BaseChatModel,
    similarity_tool,
    output_model: type[BaseModel],
    system_prompt: str,
    settings: Settings,
):
    schema_json = json.dumps(output_model.model_json_schema(), indent=2)

    def retrieve(state: ExtractionState) -> ExtractionState:
        logger.info("Node: retrieve (query=%r)", state["query"])
        context = similarity_tool.invoke(state["query"])
        logger.debug("retrieve: context length=%d", len(context))
        return {**state, "retrieved_context": context}

    def extract(state: ExtractionState) -> ExtractionState:
        logger.info("Node: extract (retry=%d)", state["retry_count"])
        messages = [
            SystemMessage(content=(
                f"{system_prompt}\n\n"
                f"Output ONLY valid JSON matching this schema:\n{schema_json}\n"
                "Do not include markdown code fences or explanation."
            )),
            HumanMessage(content=(
                f"Context:\n{state['retrieved_context']}\n\n"
                f"Query: {state['query']}"
                + (
                    f"\n\nPrevious attempt had validation errors:\n" + "\n".join(state["validation_errors"])
                    if state.get("validation_errors")
                    else ""
                )
            )),
        ]
        response = llm.invoke(messages)
        raw = response.content.strip()
        logger.debug("extract: raw_response=%r", raw[:200])
        return {**state, "raw_response": raw, "validation_errors": []}

    def validate(state: ExtractionState) -> ExtractionState:
        logger.info("Node: validate")
        try:
            instance = output_model.model_validate_json(state["raw_response"])
            logger.info("Validation succeeded")
            return {**state, "validated": instance, "validation_errors": []}
        except Exception as exc:
            errors = [str(exc)]
            logger.info("Validation failed: %s", errors[0][:120])
            return {**state, "validated": None, "validation_errors": errors, "retry_count": state["retry_count"] + 1}

    def should_retry(state: ExtractionState) -> Literal["extract", "__end__"]:
        if state.get("validated") is not None:
            return "__end__"
        if state["retry_count"] >= settings.max_retries:
            logger.info("Max retries (%d) reached without valid output", settings.max_retries)
            return "__end__"
        logger.info("Retrying extraction (attempt %d/%d)", state["retry_count"], settings.max_retries)
        return "extract"

    return retrieve, extract, validate, should_retry
