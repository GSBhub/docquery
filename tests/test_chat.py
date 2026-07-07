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


# ---------------------------------------------------------------------------
# grounding verifier + structural enforcement (real graph, scripted LLM)
# ---------------------------------------------------------------------------

def _make_real_agent(mock_registry, settings, responses: list[AIMessage]) -> ChatAgent:
    """ChatAgent with the real LangGraph graph and a scripted LLM."""
    with patch("docquery._chat.get_llm") as mock_get_llm:
        base_llm = MagicMock()
        bound_llm = MagicMock()
        pending = list(responses)
        bound_llm.invoke.side_effect = lambda msgs, **kw: pending.pop(0)
        base_llm.bind_tools.return_value = bound_llm
        mock_get_llm.return_value = base_llm
        return ChatAgent(mock_registry, settings)


def _search_call(query: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[
        {"name": "keyword_search", "args": {"query": query}, "id": "tc1"}])


def test_answer_without_tools_is_bounced_then_accepted(mock_registry, settings):
    settings.grounding = "strict"
    agent = _make_real_agent(mock_registry, settings, [
        AIMessage(content="From memory: MOV copies a value."),   # no tools → bounce
        _search_call("MOV"),                                     # complies after nudge
        AIMessage(content="The MOV instruction copies a value."),
    ])
    answer = agent.chat("What does MOV do?")
    assert "grounding warning" not in answer
    nudges = [m for m in agent._history if "docquery-guardrail" in str(m.content)]
    assert len(nudges) == 1 and "without consulting" in str(nudges[0].content)


def test_persistent_tool_refusal_is_annotated(mock_registry, settings):
    settings.grounding = "strict"
    agent = _make_real_agent(mock_registry, settings, [
        AIMessage(content="From memory."),
        AIMessage(content="Still from memory."),  # refuses again after nudge
    ])
    answer = agent.chat("What does MOV do?")
    assert "produced without consulting the document" in answer


def test_ungrounded_hex_is_bounced_with_specific_values(mock_registry, settings):
    settings.grounding = "strict"
    agent = _make_real_agent(mock_registry, settings, [
        _search_call("0xfe"),                                    # store has "0xfe 0xed"
        AIMessage(content="The opcode is 0xDEAD."),              # invented → bounce
        AIMessage(content="The opcode bytes are 0xfe 0xed."),    # grounded retry
    ])
    answer = agent.chat("What is the special opcode?")
    assert answer == "The opcode bytes are 0xfe 0xed."
    nudge = next(m for m in agent._history
                 if "not present in the document" in str(m.content))
    assert "0xdead" in str(nudge.content)


def test_still_ungrounded_after_bounce_is_annotated(mock_registry, settings):
    settings.grounding = "strict"
    agent = _make_real_agent(mock_registry, settings, [
        _search_call("0xfe"),
        AIMessage(content="The opcode is 0xDEAD."),
        AIMessage(content="I insist: it is 0xDEAD."),
    ])
    answer = agent.chat("What is the special opcode?")
    assert "grounding warning" in answer and "0xdead" in answer


def test_warn_mode_annotates_without_bouncing(mock_registry, settings):
    settings.grounding = "warn"
    agent = _make_real_agent(mock_registry, settings, [
        _search_call("0xfe"),
        AIMessage(content="The opcode is 0xDEAD."),
    ])
    answer = agent.chat("What is the special opcode?")
    assert "grounding warning" in answer and "0xdead" in answer
    assert not any("docquery-guardrail" in str(m.content) for m in agent._history)


def test_off_mode_skips_verification(mock_registry, settings):
    settings.grounding = "off"
    agent = _make_real_agent(mock_registry, settings, [
        AIMessage(content="From memory: the opcode is 0xDEAD."),
    ])
    answer = agent.chat("What is the special opcode?")
    assert answer == "From memory: the opcode is 0xDEAD."


def test_grounded_answer_passes_untouched(mock_registry, settings):
    settings.grounding = "strict"
    agent = _make_real_agent(mock_registry, settings, [
        _search_call("0xfe"),
        AIMessage(content="The special opcode is 0xfe followed by 0xed."),
    ])
    answer = agent.chat("What is the special opcode?")
    assert "grounding warning" not in answer
    assert not any("docquery-guardrail" in str(m.content) for m in agent._history)
