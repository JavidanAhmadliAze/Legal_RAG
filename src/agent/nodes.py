from __future__ import annotations

import asyncio
import os
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from src.agent.prompts import (
    clarification_instructions,
    get_today_str,
    transform_messages_into_research_topic_prompt,
)
from src.agent.state import AgentState, ClarifyWithUser, ResearchQuestion
from src.rag import guardrail
from src.rag.chain import (
    _PROMPT,
    _TRANSLATE_PROMPT,
    _format_context,
    _OUT_OF_SCOPE,
)
from src.rag.retriever import retrieve


def _get_model() -> ChatOpenAI:
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        temperature=0,
    )


_translate_chain: object = None


def _get_translate_chain():
    global _translate_chain
    if _translate_chain is None:
        _translate_chain = (
            _TRANSLATE_PROMPT
            | ChatOpenAI(
                model="deepseek-chat",
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com",
                streaming=False,
                temperature=0,
            )
            | StrOutputParser()
        )
    return _translate_chain


async def _ainvoke_structured(model: ChatOpenAI, schema, messages: list):
    # DeepSeek doesn't support json_schema response_format; use tool calling instead
    return await model.with_structured_output(schema, method="function_calling").ainvoke(messages)


# ---------------------------------------------------------------------------
# Node: guardian
# ---------------------------------------------------------------------------


async def guardian(
    state: AgentState,
) -> Command[Literal["write_research_brief", "__end__"]]:
    # Skip guardrail on follow-up turns — topic was validated on the first message.
    prior_human = [m for m in state["messages"][:-1] if isinstance(m, HumanMessage)]
    if prior_human:
        return Command(goto="write_research_brief")

    last = state["messages"][-1]
    result = await asyncio.to_thread(guardrail.check, last.content)
    if not result.allowed:
        return Command(
            goto="__end__",
            update={"messages": [AIMessage(content=result.rejection_message)]},
        )
    return Command(goto="write_research_brief")


# ---------------------------------------------------------------------------
# Node: clarify_with_user
# ---------------------------------------------------------------------------


async def clarify_with_user(
    state: AgentState,
) -> Command[Literal["write_research_brief", "__end__"]]:
    result = await _ainvoke_structured(
        _get_model(),
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


# ---------------------------------------------------------------------------
# Node: write_research_brief
# ---------------------------------------------------------------------------


async def write_research_brief(state: AgentState) -> dict:
    result = await _ainvoke_structured(
        _get_model(),
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


# ---------------------------------------------------------------------------
# Node: retriever
# ---------------------------------------------------------------------------


async def _translate(unit: str, preserve_hint: str) -> str:
    return await _get_translate_chain().ainvoke(
        {"question": unit, "preserve_hint": preserve_hint}
    )


async def retriever(state: AgentState) -> dict:
    brief = state.get("research_brief") or ""
    # The brief is already a focused keyword query produced by the LLM.
    # parse_query is designed for raw user messages (extracts embedded citations
    # like "Art. 106", "Dz.U. 2025 poz. 1794") — applying it to the brief adds
    # irrelevant employer-side Polish terms that pollute the search.
    # Only extract year/act_type filters; skip term augmentation.
    from src.rag.query_parser import _extract_filters
    filters = _extract_filters(brief)
    polish_query = await _translate(brief, "none")

    chunks = await asyncio.to_thread(
        retrieve,
        polish_query,
        top_k=10,
        rerank_query=polish_query,
        filters=filters or None,
    )
    return {"retrieved_chunks": chunks}


# ---------------------------------------------------------------------------
# Node: generator
# ---------------------------------------------------------------------------


async def generator(state: AgentState) -> dict:
    chunks = state.get("retrieved_chunks") or []
    human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    question = human_messages[-1].content if human_messages else ""

    # Guardian already validated the topic; trust what the retriever returned.
    context = _format_context(chunks) if chunks else _OUT_OF_SCOPE

    llm = _get_model()
    response = await (_PROMPT | llm | StrOutputParser()).ainvoke(
        {"context": context, "question": question}
    )
    return {"messages": [AIMessage(content=response)]}
