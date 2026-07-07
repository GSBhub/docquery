import json
import logging
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from pydantic import BaseModel

from docquery.config import Settings, language_directive
from docquery.embeddings.llm import get_structured_llm
from docquery._grounding import ungrounded_fields
from docquery._state import ExtractionState

logger = logging.getLogger(__name__)


def make_extraction_nodes(
    llm: BaseChatModel,
    similarity_tool,
    output_model: type[BaseModel],
    system_prompt: str,
    settings: Settings,
):
    schema_json = json.dumps(output_model.model_json_schema(), indent=2)
    directive = language_directive(settings.doc_language)
    if directive:
        system_prompt = f"{system_prompt}\n\n{directive}"
    structured_llm = get_structured_llm(llm, output_model, settings)

    @traceable(name="retrieve")
    def retrieve(state: ExtractionState) -> ExtractionState:
        logger.info("Node: retrieve (query=%r)", state["query"])
        context = similarity_tool.invoke(state["query"])
        logger.debug("retrieve: context length=%d", len(context))
        return {**state, "retrieved_context": context}

    @traceable(name="extract")
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
        raw = None
        if structured_llm is not None:
            # Constrained decoding: the provider guarantees schema-shaped
            # output, so validate() only ever fails on grounding, not parsing.
            try:
                result = structured_llm.invoke(messages)
                parsed = result.get("parsed")
                if parsed is not None:
                    raw = parsed.model_dump_json()
                else:
                    raw = (result.get("raw") or "").content if result.get("raw") else ""
            except Exception as exc:  # noqa: BLE001 - fall back to prompt-based JSON
                logger.warning("structured extraction failed (%s); falling back", exc)
        if raw is None:
            response = llm.invoke(messages)
            raw = response.content.strip()
        logger.debug("extract: raw_response=%r", raw[:200])
        return {**state, "raw_response": raw, "validation_errors": []}

    @traceable(name="validate")
    def validate(state: ExtractionState) -> ExtractionState:
        logger.info("Node: validate")
        try:
            instance = output_model.model_validate_json(state["raw_response"])
        except Exception as exc:
            errors = [str(exc)]
            logger.info("Validation failed: %s", errors[0][:120])
            return {**state, "validated": None, "validation_errors": errors, "retry_count": state["retry_count"] + 1}

        # Schema-valid is not document-true: verifiable values must literally
        # appear in the retrieved context (the store is the source of truth).
        if settings.grounding != "off":
            misses = ungrounded_fields(instance, state["retrieved_context"])
            if misses and settings.grounding == "strict":
                errors = [
                    f"Value not found in the retrieved context: {m}. "
                    "Use only values that appear in the context; do not fill "
                    "gaps from prior knowledge."
                    for m in misses
                ]
                logger.info("Grounding failed: %d ungrounded field(s): %s",
                            len(misses), "; ".join(misses)[:200])
                return {**state, "validated": None, "validation_errors": errors,
                        "retry_count": state["retry_count"] + 1}
            if misses:
                logger.warning("Ungrounded extraction fields (GROUNDING=warn): %s",
                               "; ".join(misses))

        logger.info("Validation succeeded")
        return {**state, "validated": instance, "validation_errors": []}

    def should_retry(state: ExtractionState) -> Literal["extract", "__end__"]:
        if state.get("validated") is not None:
            return "__end__"
        if state["retry_count"] >= settings.max_retries:
            logger.info("Max retries (%d) reached without valid output", settings.max_retries)
            return "__end__"
        logger.info("Retrying extraction (attempt %d/%d)", state["retry_count"], settings.max_retries)
        return "extract"

    return retrieve, extract, validate, should_retry
