"""
Research-brief node — converts the user's informal narrative into a focused
Polish-legal-vocabulary search query (the `research_brief` field of state).
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, get_buffer_string

from src.services.agents.nodes._shared import _ainvoke_structured, _get_llm
from src.services.agents.prompts import (
    get_today_str,
    transform_messages_into_research_topic_prompt,
)
from src.services.agents.models import ResearchQuestion
from src.services.agents.state import AgentState


async def write_research_brief(state: AgentState) -> dict:
    result = await _ainvoke_structured(
        _get_llm(),
        ResearchQuestion,
        [
            HumanMessage(
                content=transform_messages_into_research_topic_prompt.format(
                    messages=get_buffer_string(state.get("messages", [])),
                    date=get_today_str(),
                )
            )
        ],
    )
    return {"research_brief": result.research_brief}
