from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    needs_clarification: bool
    clarification_question: str | None
    research_brief: str | None
    retrieved_chunks: list[dict]


class ClarifyWithUser(BaseModel):
    need_clarification: bool
    question: str = ""       # clarification question when need_clarification=True
    verification: str = ""   # one-line confirmation of what will be looked up


class ResearchQuestion(BaseModel):
    research_brief: str      # focused query for retrieval
