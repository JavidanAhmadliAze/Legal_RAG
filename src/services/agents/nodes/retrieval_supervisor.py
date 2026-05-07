"""
Retrieval supervisor node — decomposes the research brief into 1-5 distinct
sub-questions, fans them out as parallel retrieval coroutines, then merges
and ranks the results.
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from src.services.agents._common import _RELEVANCE_THRESHOLD
from src.services.agents.nodes._shared import (
    _ainvoke_structured,
    _get_llm,
    _get_translate_chain,
)
from src.services.agents.prompts import HYDE_PROMPT, SUPERVISOR_PROMPT, get_today_str
from src.services.agents.nodes.query_parser import _extract_filters, parse_query
from src.services.agents.nodes.retriever import aretrieve
from src.services.agents.models import SubQuestionPlan
from src.services.agents.state import AgentState

# Per-sub-question budget. Final merged context is capped at _TOTAL_TOP_K.
_PER_SUB_TOP_K = 6
_PER_SUB_FETCH_K = 30
_TOTAL_TOP_K = 15


async def _translate(unit: str, preserve_hint: str) -> str:
    return await _get_translate_chain().ainvoke(
        {"question": unit, "preserve_hint": preserve_hint}
    )


async def _retrieve_for_sub(polish_query: str, brief: str) -> list[dict]:
    """One sub-agent — fully async.

    Uses ``aretrieve`` (native AsyncOpenSearch) so BM25 and k-NN run as true
    concurrent I/O when N sub-agents are awaited together via ``asyncio.gather``.

    Pipeline per sub-agent:
      1. parse_query(polish_query)  — strip noise, extract preserved terms,
         detect domain, inject domain-specific Polish boost terms into the
         retrieval query so BM25 + HyDE focus on legal vocabulary only.
      2. HyDE on the *focused* query → embedding for k-NN.
      3. Merge filters from brief + sub-question.
      4. Run BM25 + k-NN concurrently against the focused query.
    """
    parsed = parse_query(polish_query)
    focused_query = parsed.augment(parsed.cleaned)

    hyde_passage = await (HYDE_PROMPT | _get_llm() | StrOutputParser()).ainvoke(
        {"polish_query": focused_query}
    )

    filters = _extract_filters(brief) or {}
    sub_filters = parsed.filters or {}
    for key in ("year", "act_type", "law_domain"):
        if key not in filters and key in sub_filters:
            filters[key] = sub_filters[key]

    chunks = await aretrieve(
        focused_query,
        top_k=_PER_SUB_TOP_K,
        fetch_k=_PER_SUB_FETCH_K,
        rerank_query=focused_query,
        filters=filters or None,
        hyde_passage=hyde_passage,
    )
    return [c for c in chunks if c.get("_rerank_score", 0) >= _RELEVANCE_THRESHOLD]


def _dedupe_chunks(chunks: list[dict]) -> list[dict]:
    seen: set[tuple[str, int]] = set()
    out: list[dict] = []
    for c in chunks:
        key = (c.get("document_id", ""), int(c.get("chunk_index", -1)))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


async def retrieval_supervisor(state: AgentState) -> dict:
    """
    Supervisor → N sub-agent retrievers (in parallel) → merge.

    1. Decompose research brief into 1-5 focused sub-questions, each with its
       own Polish keyword query.
    2. Spawn one retrieval sub-agent per sub-question (asyncio.gather).
       Each runs the full BM25 + k-NN + filter + rerank pipeline against its
       own focused query — so the cross-encoder isn't asked to score chunks
       against a polluted multi-topic dump.
    3. Merge results, dedupe by (document_id, chunk_index), sort by rerank
       score, cap at _TOTAL_TOP_K.
    """
    brief = state.get("research_brief") or ""
    if not brief:
        return {"retrieved_chunks": [], "sub_questions": []}

    plan: SubQuestionPlan = await _ainvoke_structured(
        _get_llm(),
        SubQuestionPlan,
        [HumanMessage(content=SUPERVISOR_PROMPT.format(
            brief=brief, date=get_today_str(),
        ))],
    )

    sub_questions = plan.sub_questions or []
    if not sub_questions:
        # Fallback: single-shot retrieval against a translated brief.
        polish = await _translate(brief, "none")
        chunks = await _retrieve_for_sub(polish, brief)
        return {
            "retrieved_chunks": chunks[:_TOTAL_TOP_K],
            "sub_questions": [{"topic": "brief", "polish_query": polish}],
        }

    # Drop near-duplicate sub-queries: if two queries share more than 2
    # keywords, only keep the first. Prevents fan-out to redundant searches
    # that just rerun the same retrieval against trivially-different queries.
    deduped: list = []
    for sq in sub_questions:
        words = set(sq.polish_query.lower().split())
        is_dup = any(
            len(words & set(d.polish_query.lower().split())) > 2
            for d in deduped
        )
        if not is_dup:
            deduped.append(sq)
    sub_questions = deduped

    # Fan out: spawn each sub-agent as an explicit task so they begin
    # immediately and run concurrently. asyncio.gather then awaits all of
    # them in parallel — OpenSearch I/O overlaps across sub-agents.
    tasks = [
        asyncio.create_task(
            _retrieve_for_sub(sq.polish_query, brief),
            name=f"sub_agent[{i}]:{sq.topic[:30]}",
        )
        for i, sq in enumerate(sub_questions)
    ]
    results = await asyncio.gather(*tasks)

    merged = [c for batch in results for c in batch]
    merged = _dedupe_chunks(merged)
    merged.sort(key=lambda c: c.get("_rerank_score", 0), reverse=True)
    merged = merged[:_TOTAL_TOP_K]

    return {
        "retrieved_chunks": merged,
        "sub_questions": [
            {"topic": sq.topic, "polish_query": sq.polish_query}
            for sq in sub_questions
        ],
    }
