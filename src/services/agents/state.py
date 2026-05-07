"""
LangGraph runtime state.

The shape threaded through every node in [graph.py](graph.py). Output
schemas used for `with_structured_output` live next to this in
[models.py](models.py).
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    needs_clarification: bool
    clarification_question: str | None
    research_brief: str | None
    sub_questions: list[dict]       # supervisor's plan (for tracing)
    retrieved_chunks: list[dict]
