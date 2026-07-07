from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from docquery.config import Settings
from docquery._chat import ChatAgent
from docquery.tools.registry import ToolRegistry


@pytest.fixture
def mock_registry(populated_store, settings):
    return ToolRegistry(populated_store, settings)


def _make_agent(mock_registry, settings, responses: list[AIMessage]) -> ChatAgent:
    """Build a ChatAgent whose LLM returns preset AIMessage responses."""
    with patch("docquery._chat.get_llm") as mock_get_llm:
        base_llm = MagicMock()
        bound_llm = MagicMock()
        pending = list(responses)
        bound_llm.invoke.side_effect = lambda msgs, **kw: pending.pop(0)
        base_llm.bind_tools.return_value = bound_llm
        mock_get_llm.return_value = base_llm
        agent = ChatAgent(mock_registry, settings)

    def _fake_invoke(state, **kw):
        msgs = list(state["messages"])
        response = bound_llm.invoke(msgs)
        return {"messages": msgs + [response]}

    agent._graph = MagicMock()
    agent._graph.invoke.side_effect = _fake_invoke
    return agent


def test_chat_returns_string(mock_registry, settings):
    agent = _make_agent(
        mock_registry, settings,
        [AIMessage(content="The MOV instruction copies a register value.")]
    )
    response = agent.chat("What does MOV do?")
    assert isinstance(response, str)
    assert "MOV" in response


def test_chat_history_accumulates(mock_registry, settings):
    agent = _make_agent(
        mock_registry, settings,
        [
            AIMessage(content="MOV copies a value."),
            AIMessage(content="LDR loads from memory."),
        ],
    )
    agent.chat("What does MOV do?")
    agent.chat("What about LDR?")
    assert len(agent._history) == 4


def test_reset_clears_history(mock_registry, settings):
    agent = _make_agent(
        mock_registry, settings,
        [AIMessage(content="Answer.")]
    )
    agent.chat("Question?")
    agent.reset()
    assert agent._history == []


def test_doc_language_appended_to_default_prompt(mock_registry, settings):
    from docquery._chat import _SYSTEM_PROMPT

    settings.doc_language = "German"
    agent = _make_agent(mock_registry, settings, [AIMessage(content="ok")])
    assert agent._system_prompt.startswith(_SYSTEM_PROMPT)
    assert "German" in agent._system_prompt


def test_doc_language_appended_to_custom_prompt(mock_registry, settings):
    settings.doc_language = "Japanese"
    with patch("docquery._chat.get_llm") as mock_get_llm:
        mock_get_llm.return_value = MagicMock()
        agent = ChatAgent(mock_registry, settings, system_prompt="Custom prompt.")
    assert agent._system_prompt.startswith("Custom prompt.")
    assert "Japanese" in agent._system_prompt


def test_no_doc_language_leaves_prompt_unchanged(mock_registry, settings):
    from docquery._chat import _SYSTEM_PROMPT

    settings.doc_language = ""
    agent = _make_agent(mock_registry, settings, [AIMessage(content="ok")])
    assert agent._system_prompt == _SYSTEM_PROMPT
