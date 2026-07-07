import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from docquery.config import Settings
from docquery._nodes import make_extraction_nodes
from docquery._state import ExtractionState


class SimpleModel(BaseModel):
    name: str
    value: int


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    # exercise the prompt-based JSON fallback; structured-output tests build
    # their own llm with a working with_structured_output
    llm.with_structured_output.side_effect = NotImplementedError("no structured output")
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


def test_doc_language_in_extraction_system_message(mock_llm, mock_tool, settings):
    mock_llm.invoke.return_value = _make_response(json.dumps({"name": "foo", "value": 42}))
    settings.doc_language = "German"
    retrieve, extract, validate, should_retry = make_extraction_nodes(
        mock_llm, mock_tool, SimpleModel, "Extract data.", settings
    )
    _run_nodes(_initial_state(), retrieve, extract, validate)
    system_msg = mock_llm.invoke.call_args[0][0][0].content
    assert system_msg.startswith("Extract data.")
    assert "German" in system_msg
    assert "Output ONLY valid JSON" in system_msg


def test_no_doc_language_keeps_extraction_message_unchanged(mock_llm, mock_tool, settings):
    mock_llm.invoke.return_value = _make_response(json.dumps({"name": "foo", "value": 42}))
    settings.doc_language = ""
    retrieve, extract, validate, should_retry = make_extraction_nodes(
        mock_llm, mock_tool, SimpleModel, "Extract data.", settings
    )
    _run_nodes(_initial_state(), retrieve, extract, validate)
    system_msg = mock_llm.invoke.call_args[0][0][0].content
    assert system_msg.startswith("Extract data.\n\nOutput ONLY valid JSON")


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


# ---------------------------------------------------------------------------
# constrained structured output
# ---------------------------------------------------------------------------

def test_structured_output_path_bypasses_prompt_json(mock_tool, settings):
    llm = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = {
        "parsed": SimpleModel(name="foo", value=42), "raw": None, "parsing_error": None,
    }
    llm.with_structured_output.return_value = structured
    retrieve, extract, validate, _ = make_extraction_nodes(
        llm, mock_tool, SimpleModel, "Extract data.", settings
    )
    state = _run_nodes(_initial_state(), retrieve, extract, validate)
    assert state["validated"] == SimpleModel(name="foo", value=42)
    llm.invoke.assert_not_called()  # constrained path, no raw JSON prompting


def test_structured_invoke_failure_falls_back_to_prompt_json(mock_tool, settings):
    llm = MagicMock()
    structured = MagicMock()
    structured.invoke.side_effect = RuntimeError("server rejected schema")
    llm.with_structured_output.return_value = structured
    llm.invoke.return_value = _make_response(json.dumps({"name": "foo", "value": 42}))
    retrieve, extract, validate, _ = make_extraction_nodes(
        llm, mock_tool, SimpleModel, "Extract data.", settings
    )
    state = _run_nodes(_initial_state(), retrieve, extract, validate)
    assert state["validated"] is not None
    llm.invoke.assert_called_once()


# ---------------------------------------------------------------------------
# grounded validation (the retrieved context is the source of truth)
# ---------------------------------------------------------------------------

def test_strict_grounding_rejects_invented_value(mock_llm, mock_tool, settings):
    # context says value is 42; the model invents 999
    settings.grounding = "strict"
    mock_llm.invoke.return_value = _make_response(json.dumps({"name": "foo", "value": 999}))
    retrieve, extract, validate, should_retry = make_extraction_nodes(
        mock_llm, mock_tool, SimpleModel, "Extract data.", settings
    )
    state = _run_nodes(_initial_state(), retrieve, extract, validate)
    assert state["validated"] is None
    assert state["retry_count"] == 1
    assert any("not found in the retrieved context" in e for e in state["validation_errors"])
    assert any("value=999" in e for e in state["validation_errors"])
    assert should_retry(state) == "extract"


def test_warn_grounding_accepts_but_logs(mock_llm, mock_tool, settings, caplog):
    settings.grounding = "warn"
    mock_llm.invoke.return_value = _make_response(json.dumps({"name": "foo", "value": 999}))
    retrieve, extract, validate, _ = make_extraction_nodes(
        mock_llm, mock_tool, SimpleModel, "Extract data.", settings
    )
    with caplog.at_level("WARNING"):
        state = _run_nodes(_initial_state(), retrieve, extract, validate)
    assert state["validated"] is not None
    assert any("Ungrounded" in r.message for r in caplog.records)


def test_off_grounding_skips_check(mock_llm, mock_tool, settings):
    settings.grounding = "off"
    mock_llm.invoke.return_value = _make_response(json.dumps({"name": "foo", "value": 999}))
    retrieve, extract, validate, _ = make_extraction_nodes(
        mock_llm, mock_tool, SimpleModel, "Extract data.", settings
    )
    state = _run_nodes(_initial_state(), retrieve, extract, validate)
    assert state["validated"] is not None


def test_strict_grounding_rejects_mispaired_record(mock_llm, mock_tool, settings):
    # context has NMI→0x08 and HardFault→0x0C on separate rows; the model
    # pairs NMI with HardFault's address — presence passes, association fails
    class Vec(BaseModel):
        name: str
        address: str

    settings.grounding = "strict"
    mock_tool.invoke.return_value = (
        "ROW 2 | NMI | 0x00000008\nROW 3 | HardFault | 0x0000000C")
    mock_llm.invoke.return_value = _make_response(
        json.dumps({"name": "NMI", "address": "0x0C"}))
    retrieve, extract, validate, _ = make_extraction_nodes(
        mock_llm, mock_tool, Vec, "Extract data.", settings
    )
    state = _run_nodes(_initial_state(), retrieve, extract, validate)
    assert state["validated"] is None
    assert any("Wrongly associated" in e for e in state["validation_errors"])


def test_grounded_values_pass_strict(mock_llm, mock_tool, settings):
    settings.grounding = "strict"
    mock_llm.invoke.return_value = _make_response(json.dumps({"name": "foo", "value": 42}))
    retrieve, extract, validate, _ = make_extraction_nodes(
        mock_llm, mock_tool, SimpleModel, "Extract data.", settings
    )
    state = _run_nodes(_initial_state(), retrieve, extract, validate)
    assert state["validated"] == SimpleModel(name="foo", value=42)
