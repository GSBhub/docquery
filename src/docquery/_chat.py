import logging

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from docquery.config import Settings, language_directive
from docquery.embeddings.llm import get_llm
from docquery._grounding import unsupported_claims
from docquery._state import ChatState
from docquery.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Prefix for guardrail messages the verifier injects into the conversation.
# They are invisible to the caller (chat() returns only the final answer) but
# let the verifier distinguish its own nudges from real user messages.
_GUARDRAIL_MARK = "[docquery-guardrail]"

_NO_TOOLS_NUDGE = (
    f"{_GUARDRAIL_MARK} You answered without consulting the document. Call at "
    "least one tool (similarity_search, structure_lookup, keyword_search, ...) "
    "and answer ONLY from its output."
)
_UNGROUNDED_NUDGE = (
    f"{_GUARDRAIL_MARK} These values in your answer are not present in the "
    "document excerpts you retrieved: {misses}. Correct or remove them — quote "
    "values exactly as the tools returned them, and say so if the document "
    "does not contain the answer."
)
_UNGROUNDED_NOTE = (
    "\n\n[grounding warning] These values could not be verified against the "
    "document: {misses}"
)
_NO_TOOLS_NOTE = (
    "\n\n[grounding warning] This answer was produced without consulting the "
    "document."
)


def _turn_messages(messages: list) -> list:
    """Messages after the last real (non-guardrail) user message."""
    start = 0
    for i, m in enumerate(messages):
        if isinstance(m, HumanMessage) and not str(m.content).startswith(_GUARDRAIL_MARK):
            start = i
    return messages[start + 1:]

_SYSTEM_PROMPT = """\
You are a document analysis assistant. \
You answer questions by searching an indexed document database.

SEARCH STRATEGY — follow this order every time:
1. Call similarity_search FIRST for any question about the document.
2. If similarity_search does not give enough detail, call it again with a more specific query, \
or call keyword_search for exact identifiers (e.g. specific terms, codes, or mnemonics).
3. Use page_lookup only when the user mentions a specific page number.
4. ALWAYS call structure_lookup when the question involves bit positions or layouts, \
instruction/register encodings, register fields, interrupt or vector tables, or pin maps \
— do not answer such questions from prose alone. It returns deterministic \
machine-parseable extractions, far more reliable than prose search. Call \
structure_lookup() with no arguments to see which kinds this document has. Lines look like:
  ENCODING 32-bit: bits[31:25]=0001100 S[24:24] Rn[19:16] ?[15:12] Rm[3:0]
  (bits[hi:lo]=… fixed bits, name[hi:lo] named fields, ?[hi:lo] unreadable)
  TABLE register_fields: bit | field | description
  ROW 7:0 | EN | Enable
5. To walk EVERY instance of a tagged entity type (e.g. every instruction, register, \
interrupt, or pin — table rows are tagged too), call cursor_enumerate(entity_type) ONCE — \
it is deterministic and complete. For a fuzzy "most relevant matches" list, call \
cursor_count(criteria) instead. Either way, then call cursor_current and cursor_next to \
walk the result one item at a time. Do not use similarity_search for exhaustive enumeration.

RULES:
- NEVER reply that you cannot find information without first calling similarity_search.
- Always attempt at least one tool call before answering.
- When results are returned, synthesise them into a clear answer and cite the page number(s).
- The document is the ONLY source of truth. State values (addresses, bit positions, \
encodings, numbers) exactly as they appear in tool output — never from prior knowledge, \
and never "corrected". If the document does not contain a value, say so.
- When answering from ENCODING or TABLE/ROW lines, reproduce the relevant lines VERBATIM \
in a code block and explain around them; do not re-type their values into prose.
- If you genuinely cannot find relevant content after searching, say so and suggest rephrasing.\
"""


class ChatAgent:
    def __init__(self, tool_registry: ToolRegistry, settings: Settings | None = None,
                 system_prompt: str | None = None):
        self._settings = settings or Settings()
        self._system_prompt = system_prompt or _SYSTEM_PROMPT
        directive = language_directive(self._settings.doc_language)
        if directive:
            self._system_prompt = f"{self._system_prompt}\n\n{directive}"
        self._tools = tool_registry.get_tools()
        llm = get_llm(self._settings)
        self._llm_with_tools = llm.bind_tools(self._tools)
        self._tool_map = {t.name: t for t in self._tools}
        self._history: list = []
        self._graph = self._build_graph()

    def _build_graph(self):
        system_msg = self._system_prompt

        @traceable(name="agent")
        def agent(state: ChatState) -> ChatState:
            from langchain_core.messages import SystemMessage
            messages = [SystemMessage(content=system_msg)] + list(state["messages"])
            logger.debug("ChatAgent/agent: %d messages", len(messages))
            response = self._llm_with_tools.invoke(messages)
            if getattr(response, "tool_calls", None):
                for tc in response.tool_calls:
                    logger.info("Tool call: %s(%s)", tc["name"], str(tc.get("args", {}))[:120])
            return {"messages": state["messages"] + [response]}

        @traceable(name="execute_tools")
        def execute_tools(state: ChatState) -> ChatState:
            last = state["messages"][-1]
            new_messages = list(state["messages"])
            for tc in last.tool_calls:
                tool = self._tool_map.get(tc["name"])
                if tool is None:
                    result = f"Unknown tool: {tc['name']}"
                else:
                    result = tool.invoke(tc.get("args", {}))
                new_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            return {"messages": new_messages}

        def should_call_tools(state: ChatState):
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                return "execute_tools"
            return "verify"

        @traceable(name="verify")
        def verify(state: ChatState) -> ChatState:
            """Deterministic grounding gate on the turn's final answer.

            Structural rule: an answer produced with zero tool calls is bounced
            back once. Grounding rule: verifiable claims in the answer (hex,
            bit ranges, machine lines) must appear in this turn's tool output;
            in strict mode unsupported claims bounce the answer back once with
            the exact offending values, after which (or in warn mode) the
            answer is annotated instead. No LLM judging — pure string checks.
            """
            messages = list(state["messages"])
            mode = self._settings.grounding
            if mode == "off":
                return {"messages": messages}

            turn = _turn_messages(messages)
            answer = messages[-1]
            if not isinstance(answer, AIMessage):
                return {"messages": messages}
            nudges = [m.content for m in turn
                      if isinstance(m, HumanMessage) and str(m.content).startswith(_GUARDRAIL_MARK)]
            tool_text = "\n".join(str(m.content) for m in turn if isinstance(m, ToolMessage))

            if not tool_text:
                if mode == "strict" and not any("without consulting" in n for n in nudges):
                    logger.info("verify: no tool calls this turn — bouncing answer")
                    return {"messages": messages + [HumanMessage(content=_NO_TOOLS_NUDGE)]}
                logger.warning("verify: answer produced without tool output")
                messages[-1] = AIMessage(content=str(answer.content) + _NO_TOOLS_NOTE)
                return {"messages": messages}

            misses = unsupported_claims(str(answer.content), tool_text)
            if not misses:
                return {"messages": messages}
            if mode == "strict" and not any("not present in the document" in n for n in nudges):
                logger.info("verify: ungrounded claims %s — bouncing answer", misses)
                nudge = _UNGROUNDED_NUDGE.format(misses=", ".join(misses))
                return {"messages": messages + [HumanMessage(content=nudge)]}
            logger.warning("verify: ungrounded claims in final answer: %s", misses)
            messages[-1] = AIMessage(
                content=str(answer.content) + _UNGROUNDED_NOTE.format(misses=", ".join(misses)))
            return {"messages": messages}

        def after_verify(state: ChatState):
            last = state["messages"][-1]
            if isinstance(last, HumanMessage) and str(last.content).startswith(_GUARDRAIL_MARK):
                return "agent"
            return END

        graph = StateGraph(ChatState)
        graph.add_node("agent", agent)
        graph.add_node("execute_tools", execute_tools)
        graph.add_node("verify", verify)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", should_call_tools,
                                    {"execute_tools": "execute_tools", "verify": "verify"})
        graph.add_edge("execute_tools", "agent")
        graph.add_conditional_edges("verify", after_verify, {"agent": "agent", END: END})
        return graph.compile()

    @traceable(name="chat")
    def chat(self, user_message: str) -> str:
        self._history.append(HumanMessage(content=user_message))
        final = self._graph.invoke({"messages": self._history})
        self._history = final["messages"]
        last = self._history[-1]
        return last.content if hasattr(last, "content") else str(last)

    def reset(self) -> None:
        self._history = []


class ChatSession:
    """Public-facing stateful chat session returned by docquery.chat_session()."""

    def __init__(self, agent: ChatAgent) -> None:
        self._agent = agent

    def chat(self, message: str) -> str:
        return self._agent.chat(message)

    def reset(self) -> None:
        self._agent.reset()
