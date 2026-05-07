"""
Legacy synchronous LangChain RAG runner used by the eval CLIs
([src/eval/cli/evaluate.py](../../eval/cli/evaluate.py),
[src/eval/cli/run_full_eval.py](../../eval/cli/run_full_eval.py)).

The live FastAPI path uses the LangGraph nodes in [nodes.py](nodes.py)
instead. Shared utilities (context formatter, citation verifier, relevance
threshold, prompt aliases) live in [_common.py](_common.py).
"""

from __future__ import annotations

import asyncio

from langchain_core.output_parsers import StrOutputParser

from src.services.agents._common import (
    _OUT_OF_SCOPE,
    _PROMPT,
    _RELEVANCE_THRESHOLD,
    _TRANSLATE_PROMPT,
    _format_context,
    _split_into_subquestions,
    _verify_citations,
)
from src.services.agents.prompts import HYDE_PROMPT
from src.services.agents.nodes.query_parser import parse_query
from src.services.agents.nodes.retriever import aretrieve
from src.services.llm import get_llm_client


def _dedupe_chunks(chunks: list[dict]) -> list[dict]:
    seen: set[tuple[str, int]] = set()
    out: list[dict] = []
    for chunk in chunks:
        key = (
            chunk.get("document_id", ""),
            int(chunk.get("chunk_index", chunk.get("page_num", -1))),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(chunk)
    return out


def build_chain():
    llm_client = get_llm_client()
    llm = llm_client.get_chat_model()
    translator = llm_client.get_translator_model()
    translate_chain = _TRANSLATE_PROMPT | translator | StrOutputParser()
    hyde_chain = HYDE_PROMPT | translator | StrOutputParser()

    def retriever_step(question: str) -> str:
        units = _split_into_subquestions(question)
        search_units = list(units)
        if len(units) > 1:
            search_units.append(question)

        collected: list[dict] = []
        seen_queries: set[str] = set()
        for unit in search_units:
            parsed = parse_query(unit)
            polish_query = translate_chain.invoke({
                "question": parsed.cleaned,
                "preserve_hint": parsed.preserved_hint() or "none",
            })
            final_query = parsed.augment(polish_query)
            if final_query in seen_queries:
                continue
            seen_queries.add(final_query)

            hyde_passage = hyde_chain.invoke({"polish_query": final_query})

            per_query_top_k = 15 if len(search_units) > 1 else 20
            collected.extend(
                asyncio.run(aretrieve(
                    final_query,
                    top_k=per_query_top_k,
                    fetch_k=20,
                    rerank_query=final_query,
                    filters=parsed.filters or None,
                    hyde_passage=hyde_passage,
                ))
            )

        chunks = _dedupe_chunks(collected)
        # Per-unit retrieval already applied Polish-Polish reranking; sort the merged
        # pool by those scores and take the best 15.  A second English-query global
        # rerank is omitted because the cross-encoder degrades badly on English→Polish
        # pairs and inverts the ranking for queries without embedded Polish terms.
        chunks.sort(key=lambda c: c.get("_rerank_score", -99), reverse=True)
        chunks = chunks[:15]

        relevant = [c for c in chunks if c.get("_rerank_score", -99) >= _RELEVANCE_THRESHOLD]
        if not relevant:
            return _OUT_OF_SCOPE
        return _format_context(relevant)

    def run(question: str) -> str:
        context = retriever_step(question)
        messages = _PROMPT.format_messages(context=context, question=question)
        raw = llm.invoke(messages).content
        return _verify_citations(raw, context)

    return run
