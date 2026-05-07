"""
Clarification node — decides whether to ask the user a follow-up question
before retrieval. Currently NOT wired into graph.py; kept for future use.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langgraph.types import Command

from src.services.agents.nodes._shared import _ainvoke_structured, _get_llm
from src.services.agents.prompts import clarification_instructions, get_today_str
from src.services.agents.models import ClarifyWithUser
from src.services.agents.state import AgentState


async def clarify_with_user(
    state: AgentState,
) -> Command[Literal["write_research_brief", "__end__"]]:
    result = await _ainvoke_structured(
        _get_llm(),
        ClarifyWithUser,
        [
            HumanMessage(
                content=clarification_instructions.format(
                    messages=get_buffer_string(state.get("messages", [])),
                    date=get_today_str(),
                )
            )
        ],
    )

    if result.need_clarification:
        return Command(
            goto="__end__",
            update={
                "messages": [AIMessage(content=result.question)],
                "needs_clarification": True,
                "clarification_question": result.question,
            },
        )

    return Command(
        goto="write_research_brief",
        update={
            "messages": [AIMessage(content=result.verification)],
            "needs_clarification": False,
            "clarification_question": None,
        },
    )
