from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    needs_clarification: bool
    clarification_question: str | None
    research_brief: str | None
    sub_questions: list[dict]       # supervisor's plan (for tracing)
    retrieved_chunks: list[dict]


class ClarifyWithUser(BaseModel):
    need_clarification: bool
    question: str = ""       # clarification question when need_clarification=True
    verification: str = ""   # one-line confirmation of what will be looked up


class ResearchQuestion(BaseModel):
    research_brief: str      # focused query for retrieval


class SubQuestion(BaseModel):
    topic: str = Field(
        description="3-7 word English description of the single legal topic."
    )
    polish_query: str = Field(
        description=(
            "6-10 word Polish keyword query for searching Polish statutes. "
            "Use specific legal terms (procedure names, document types, "
            "article subjects). No verbs, no fillers."
        )
    )


class SubQuestionPlan(BaseModel):
    """Supervisor's decomposition of a research brief into focused sub-questions."""
    sub_questions: list[SubQuestion] = Field(
        description=(
            "Between 1 and 5 sub-questions. Most briefs need 1-2; only multi-part "
            "questions covering distinct legal topics need 3+. Each sub-question "
            "must address ONE topic — never bundle two topics in one entry."
        )
    )
