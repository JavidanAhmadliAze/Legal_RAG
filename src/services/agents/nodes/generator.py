"""
Generator node — formats retrieved chunks into context, asks the LLM to
answer using only that context, then strips any sentence whose article or
Dz.U./M.P. citation is not present in the retrieved chunks.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from src.services.agents._common import (
    _OUT_OF_SCOPE,
    _PROMPT,
    _format_context,
    _verify_citations,
)
from src.services.agents.nodes._shared import _get_llm
from src.services.agents.state import AgentState


async def generator(state: AgentState) -> dict:
    chunks = state.get("retrieved_chunks") or []
    human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    question = human_messages[-1].content if human_messages else ""

    context = _format_context(chunks) if chunks else _OUT_OF_SCOPE

    llm = _get_llm()
    response = await (_PROMPT | llm | StrOutputParser()).ainvoke(
        {"context": context, "question": question}
    )
    response = _verify_citations(response, context)
    return {"messages": [AIMessage(content=response)]}
