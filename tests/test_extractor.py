import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from docquery.config import Settings
from docquery.pipeline.nodes import make_extraction_nodes
from docquery.pipeline.state import ExtractionState


class SimpleModel(BaseModel):
    name: str
    value: int


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    return llm


@pytest.fixture
def mock_tool():
    tool = MagicMock()
    tool.invoke.return_value = "Context: the name is 'foo' and value is 42."
    return tool


@pytest.fixture
def settings():
    s = Settings()
    s.max_retries = 3
    return s


def _make_response(content: str):
    resp = MagicMock()
    resp.content = content
    return resp


def _run_nodes(state, retrieve, extract, validate):
    state = retrieve(state)
    state = extract(state)
    state = validate(state)
    return state


def _initial_state(query="test query") -> ExtractionState:
    return {
        "query": query,
        "retrieved_context": "",
        "raw_response": "",
        "validated": None,
        "validation_errors": [],
        "retry_count": 0,
    }


def test_valid_extraction(mock_llm, mock_tool, settings):
    mock_llm.invoke.return_value = _make_response(json.dumps({"name": "foo", "value": 42}))
    retrieve, extract, validate, should_retry = make_extraction_nodes(
        mock_llm, mock_tool, SimpleModel, "Extract data.", settings
    )
    state = _run_nodes(_initial_state(), retrieve, extract, validate)
    assert state["validated"] is not None
    assert state["validated"].name == "foo"
    assert state["validated"].value == 42
    assert should_retry(state) == "__end__"


def test_invalid_json_increments_retry(mock_llm, mock_tool, settings):
    mock_llm.invoke.return_value = _make_response("not json at all")
    retrieve, extract, validate, should_retry = make_extraction_nodes(
        mock_llm, mock_tool, SimpleModel, "Extract data.", settings
    )
    state = _run_nodes(_initial_state(), retrieve, extract, validate)
    assert state["validated"] is None
    assert state["retry_count"] == 1
    assert len(state["validation_errors"]) > 0
    assert should_retry(state) == "extract"


def test_max_retries_stops_loop(mock_llm, mock_tool, settings):
    mock_llm.invoke.return_value = _make_response("{}")  # missing required fields
    retrieve, extract, validate, should_retry = make_extraction_nodes(
        mock_llm, mock_tool, SimpleModel, "Extract data.", settings
    )
    state = _initial_state()
    state["retry_count"] = settings.max_retries  # already at limit
    state = retrieve(state)
    state = extract(state)
    state = validate(state)
    assert should_retry(state) == "__end__"


def test_retry_includes_error_feedback(mock_llm, mock_tool, settings):
    mock_llm.invoke.return_value = _make_response("bad")
    retrieve, extract, validate, should_retry = make_extraction_nodes(
        mock_llm, mock_tool, SimpleModel, "Extract data.", settings
    )
    state = _run_nodes(_initial_state(), retrieve, extract, validate)
    # On the next extract call, error feedback should be present in the messages
    extract(state)  # just verify it doesn't crash with errors present
    call_args = mock_llm.invoke.call_args_list[-1][0][0]
    human_msg = call_args[1].content
    assert "validation errors" in human_msg.lower() or "error" in human_msg.lower()
