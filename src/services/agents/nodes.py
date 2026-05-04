from __future__ import annotations

import asyncio
import os
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from src.services.agents.prompts import (
    clarification_instructions,
    get_today_str,
    transform_messages_into_research_topic_prompt,
)
from src.services.agents.state import (
    AgentState,
    ClarifyWithUser,
    ResearchQuestion,
    SubQuestionPlan,
)
from src.services.agents import guardrail
from src.services.agents.chain import (
    _PROMPT,
    _TRANSLATE_PROMPT,
    _RELEVANCE_THRESHOLD,
    _format_context,
    _OUT_OF_SCOPE,
    _verify_citations,
)
from src.services.agents.retriever import aretrieve
from src.services.llm.prompts import HYDE_PROMPT

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
                request_timeout=30,
            )
            | StrOutputParser()
        )
    return _translate_chain


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        temperature=0,
        request_timeout=90,
    )


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


# ---------------------------------------------------------------------------
# Node: write_research_brief
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Node: retrieval_supervisor
# ---------------------------------------------------------------------------

# Per-sub-question budget. Final merged context is capped at TOTAL_TOP_K.
_PER_SUB_TOP_K  = 6
_PER_SUB_FETCH_K = 30
_TOTAL_TOP_K = 15

_SUPERVISOR_PROMPT = """You are a Polish legal research supervisor. Your job \
is to decompose the user's research brief into focused sub-questions, each \
addressing ONE distinct legal topic, so a separate retriever can search the \
corpus of Polish legal acts (Ustawy, Rozporządzenia, Obwieszczenia, Umowy \
międzynarodowe published in Dz.U. and M.P.).

# Decomposition rules

1. Produce between 1 and 5 sub-questions.
2. A single-topic question → ONE sub-question.
3. A multi-part question → ONE sub-question PER DISTINCT LEGAL TOPIC.
4. Drop sub-topics that are clearly off-topic (weather, jokes, math, recipes).

# Diversity rules — VIOLATION MAKES THE PLAN USELESS

5. NEVER produce two sub-questions that target the same article range or \
the same legal subject. Each sub-question must aim at a DIFFERENT corner of \
Polish law.
6. If two parts of the brief share a topic (e.g. "permit deadline" and \
"permit renewal procedure" are both about permits), MERGE them into ONE \
sub-question.
7. The polish_query of each sub-question MUST share NO MORE THAN 2 keywords \
with any other sub-question's polish_query in the same plan. If you find \
yourself repeating "zezwolenie pobyt" in three queries, you are doing it \
wrong — pivot to the OTHER topics in the brief.
8. Anchor each polish_query on the topic's most-specific Polish legal term \
(e.g. "rezydent długoterminowy UE" for EU long-term, "e-doręczenia" for \
electronic delivery, "próg dochodu zatrudnienie pracowników" for the \
business-employment threshold, "MOS portal wniosek elektroniczny" for the \
digital portal, "powrót wjazd cudzoziemiec stempel" for re-entry rights).

# Named-entity rule — EVERY explicit statute/act/legal-concept name in the \
brief MUST produce its own sub-question

9. Scan the brief for explicitly named statutes, acts, legal concepts, or \
keyword identifiers — anything the user typed by name. Examples of such \
named entities: "Kodeks pracy", "Ordynacja podatkowa", "ustawa o VAT", \
"wygaszenie act", "KPA", "PESEL UKR", "Karta Polaka", "Karta Pobytu", \
"Konstytucja", "rękojmia", "umowa międzynarodowa", any "ustawa o X". For \
EACH named entity, you MUST produce a dedicated sub-question whose \
polish_query is anchored on that exact Polish term — DO NOT fold it into \
a broader procedural query (e.g., do not merge "wygaszenie" into a \
generic "long-term residence" sub-question; do not merge "Kodeks pracy" \
into a generic "employment" sub-question). Named entities point to \
specific acts in the corpus that vocabulary-overlapping queries will miss.

# Output format per sub-question

- topic: 3-7 word English description naming the SPECIFIC legal subject \
(not a paraphrase of the user's question — the legal area itself).
- polish_query: 6-10 Polish keywords. Use SPECIFIC procedural terms, document \
types, subject matter (próg dochodu, rezydent długoterminowy, e-doręczenia, \
fikcja doręczenia, MOS, stempel, karta pobytu, wniosek elektroniczny, \
powrotu cudzoziemiec). No verbs. No fillers. Lowercase.

# Worked example

Brief: "Salon owner asks about (1) two-employee threshold for business permit, \
(2) MOS 2.0 portal launching tomorrow, (3) CUKR card and 5-year EU long-term \
clock, (4) unopened e-Doręczenia notification."

Correct plan (4 distinct sub-questions, no keyword overlap):
- topic: "business permit employment threshold"; \
  polish_query: "zezwolenie działalność gospodarcza próg dochodu zatrudnienie pracowników"
- topic: "MOS 2.0 digital portal application"; \
  polish_query: "MOS wniosek elektroniczny portal cudzoziemiec system teleinformatyczny"
- topic: "EU long-term resident 5-year clock"; \
  polish_query: "rezydent długoterminowy UE pięć lat nieprzerwany pobyt"
- topic: "electronic delivery fiction"; \
  polish_query: "e-doręczenia fikcja doręczenia korespondencja elektroniczna"

Today's date: {date}

Research brief:
{brief}
"""


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
    from langchain_core.output_parsers import StrOutputParser
    from src.services.agents.query_parser import _extract_filters, parse_query

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
        [HumanMessage(content=_SUPERVISOR_PROMPT.format(
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


# ---------------------------------------------------------------------------
# Node: generator
# ---------------------------------------------------------------------------


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
