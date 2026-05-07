"""
Input guardrail — both the policy (length + LLM topical-fit classifier)
and the LangGraph node that wraps it.

  - check(query)        -> GuardrailResult       (pure policy)
  - guardian(state)     -> Command(goto=...)     (graph entry node)

Skipped on follow-up turns — topic was validated on the first message.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.types import Command

from src.services.agents.prompts import GUARDRAIL_CLASSIFIER_PROMPT
from src.services.agents.state import AgentState
from src.services.llm import get_llm_client

MAX_WORDS = 200


@dataclass
class GuardrailResult:
    allowed: bool
    rejection_message: str = ""


_classifier_chain = None


def _get_chain():
    global _classifier_chain
    if _classifier_chain is None:
        model = get_llm_client().build_chat_model(streaming=False, request_timeout=10)
        _classifier_chain = GUARDRAIL_CLASSIFIER_PROMPT | model | StrOutputParser()
    return _classifier_chain


def check(query: str) -> GuardrailResult:
    """Run all guardrail checks. Returns GuardrailResult with allowed=False and a
    user-facing message if the query should be blocked."""
    words = query.split()
    if len(words) > MAX_WORDS:
        return GuardrailResult(
            allowed=False,
            rejection_message=(
                f"Your query is too long ({len(words)} words). "
                f"Please shorten it to {MAX_WORDS} words or fewer."
            ),
        )

    verdict = _get_chain().invoke({"query": query}).strip().upper()
    # Default to ALLOW. Only block on an unambiguous NO — the assistant itself
    # returns "no source found" when the corpus has nothing relevant, so we
    # don't need the guardrail to second-guess topical fit.
    if verdict.startswith("NO"):
        return GuardrailResult(
            allowed=False,
            rejection_message=(
                "I can only help with questions related to Polish law and legal acts. "
                "Please ask about Polish legislation, regulations, or legal procedures."
            ),
        )

    return GuardrailResult(allowed=True)


async def guardian(
    state: AgentState,
) -> Command[Literal["write_research_brief", "__end__"]]:
    """Graph entry node — wraps `check()` and produces a LangGraph Command."""
    # Skip guardrail on follow-up turns — topic was validated on the first message.
    prior_human = [m for m in state["messages"][:-1] if isinstance(m, HumanMessage)]
    if prior_human:
        return Command(goto="write_research_brief")

    last = state["messages"][-1]
    result = await asyncio.to_thread(check, last.content)
    if not result.allowed:
        return Command(
            goto="__end__",
            update={"messages": [AIMessage(content=result.rejection_message)]},
        )
    return Command(goto="write_research_brief")
