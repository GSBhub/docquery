from typing import Any

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class ExtractionState(TypedDict):
    query: str
    retrieved_context: str
    raw_response: str
    validated: Any | None
    validation_errors: list[str]
    retry_count: int


class ChatState(TypedDict):
    messages: list[BaseMessage]
