import logging

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from docquery.config import Settings
from docquery.embeddings.llm import get_llm
from docquery._state import ChatState
from docquery.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a document analysis assistant. \
You answer questions by searching an indexed document database.

SEARCH STRATEGY — follow this order every time:
1. Call similarity_search FIRST for any question about the document.
2. If similarity_search does not give enough detail, call it again with a more specific query, \
or call keyword_search for exact identifiers (e.g. specific terms, codes, or mnemonics).
3. Use page_lookup only when the user mentions a specific page number.

RULES:
- NEVER reply that you cannot find information without first calling similarity_search.
- Always attempt at least one tool call before answering.
- When results are returned, synthesise them into a clear answer and cite the page number(s).
- If you genuinely cannot find relevant content after searching, say so and suggest rephrasing.\
"""


class ChatAgent:
    def __init__(self, tool_registry: ToolRegistry, settings: Settings | None = None,
                 system_prompt: str | None = None):
        self._settings = settings or Settings()
        self._system_prompt = system_prompt or _SYSTEM_PROMPT
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
            return END

        graph = StateGraph(ChatState)
        graph.add_node("agent", agent)
        graph.add_node("execute_tools", execute_tools)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", should_call_tools,
                                    {"execute_tools": "execute_tools", END: END})
        graph.add_edge("execute_tools", "agent")
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
