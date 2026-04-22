"""
LangChain RAG chain using DeepSeek via the OpenAI-compatible API.
"""

from __future__ import annotations

import os
import re

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.rag.query_parser import parse_query
from src.rag.retriever import retrieve

_SYSTEM = """You are a dedicated Polish legal assistant covering immigration, residence permits, \
foreigners law, border control, Ukrainian temporary protection, and related administrative \
procedures in Poland.

Rules:
1. Answer directly and authoritatively. Do not mention "excerpts", "texts", "documents I have", \
or any internal retrieval mechanics — speak as a legal expert, not a search engine.
2. Cite specific provisions when available (e.g. "Art. 106c, Dz.U. 2025 poz. 1794").
3. If context is marked OUT_OF_SCOPE, the question falls outside your coverage area. \
Tell the user clearly that this topic (e.g. general employment law, tax, civil law) is outside \
your specialisation and recommend consulting a labour law or civil law specialist. \
Do NOT attempt to answer from unrelated sources.
4. Never invent legal provisions or deadlines.
5. Always answer in English."""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", "Context:\n{context}\n\n---\nQuestion: {question}"),
])

_TRANSLATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a Polish legal search query builder. "
        "Convert the user's question into a short Polish keyword query (5-12 words) "
        "suitable for searching a corpus of Polish legal acts. "
        "Focus on legal concepts, article subjects, and procedural terms. "
        "If a list of terms is marked PRESERVE, include them verbatim in the output. "
        "Output ONLY the query — no explanation, no punctuation beyond what appears in legal citations."
    )),
    ("human", "{question}\n\nPRESERVE (include verbatim if relevant): {preserve_hint}"),
])

_GREETING_RE = re.compile(r"^(hi|hello|hey|thanks|thank you)\b", re.IGNORECASE)
_QUESTION_START_RE = re.compile(
    r"^(is|are|can|could|do|does|did|should|would|will|what|when|where|which|who|why|how)\b",
    re.IGNORECASE,
)
_LEGAL_CUE_RE = re.compile(
    r"\b("
    r"report|reporting|deadline|notify|notification|rule|require|required|must|deadline|"
    r"fine|penalt|grzywn|PIP|inspector|inspection|B2B|contract|employment|employee|employer|"
    r"work permit|residence card|foreigner|foreign worker|Ukraine|temporary protection|"
    r"PESEL|UKR|MOS|portal|oświadczen|zezwoleni|powiadom|kara|umowa|stosunek pracy"
    r")\b",
    re.IGNORECASE,
)
_BACKGROUND_RE = re.compile(
    r"\b(i just started|i'?m hiring|i am hiring|my first employee|next week|consulting business)\b",
    re.IGNORECASE,
)


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        doc_id = c.get("document_id", "")
        journal = "M.P." if doc_id.startswith("WMP") else "Dz.U."
        header = f"[{c['title']} | {journal} {c['year']} poz. {c['pos']} | strona {c['page_num']}]"
        if c.get("anchor"):
            header += f" ({c['anchor']})"
        parts.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def _split_into_subquestions(question: str) -> list[str]:
    """Break a mixed user prompt into a few focused retrieval units."""
    text = " ".join(question.split())
    if not text:
        return []

    parts = [p.strip() for p in re.findall(r"[^?.!]+[?.!]?", text) if p.strip()]
    if len(parts) <= 1:
        return [text]

    merged: list[str] = []
    for part in parts:
        if (
            merged
            and len(part.split()) <= 8
            and _QUESTION_START_RE.match(part)
        ):
            merged[-1] = f"{merged[-1]} {part}"
            continue
        merged.append(part)

    scored: list[tuple[int, int, str]] = []
    for idx, part in enumerate(merged):
        score = 0
        if "?" in part:
            score += 3
        if _LEGAL_CUE_RE.search(part):
            score += 3
        if _QUESTION_START_RE.match(part):
            score += 1
        if _GREETING_RE.match(part):
            score -= 5
        if _BACKGROUND_RE.search(part) and not _LEGAL_CUE_RE.search(part):
            score -= 2
        if len(part.split()) <= 2:
            score -= 2
        scored.append((score, idx, part))

    substantive = [
        part
        for score, _, part in sorted(scored, key=lambda item: (-item[0], item[1]))
        if score > 0
    ]
    if not substantive:
        return [p for p in merged if not _GREETING_RE.match(p)][:3] or [text]

    chosen = set(substantive[:3])
    ordered = [part for part in merged if part in chosen]
    return ordered[:3]


# Polish-Polish cross-encoder threshold (per-unit reranking uses the Polish query).
# Scores above 1.5 are meaningful; below that is noise for Polish legal text.
_RELEVANCE_THRESHOLD = 1.5

_OUT_OF_SCOPE = (
    "OUT_OF_SCOPE — This question does not match Polish immigration or foreigners law. "
    "Apply rule 3 from the system prompt."
)


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


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        streaming=True,
        temperature=0,
    )


def _get_translator() -> ChatOpenAI:
    # Non-streaming, low-temperature for deterministic translation
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        streaming=False,
        temperature=0,
    )


def build_chain():
    llm = _get_llm()
    translator = _get_translator()
    translate_chain = _TRANSLATE_PROMPT | translator | StrOutputParser()

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
                "question": unit,
                "preserve_hint": parsed.preserved_hint() or "none",
            })
            final_query = parsed.augment(polish_query)
            if final_query in seen_queries:
                continue
            seen_queries.add(final_query)
            per_query_top_k = 6 if len(search_units) > 1 else 10
            collected.extend(
                retrieve(
                    final_query,
                    top_k=per_query_top_k,
                    rerank_query=final_query,
                    filters=parsed.filters or None,
                )
            )

        chunks = _dedupe_chunks(collected)
        # Per-unit retrieval already applied Polish-Polish reranking; sort the merged
        # pool by those scores and take the best 10.  A second English-query global
        # rerank is omitted because the cross-encoder degrades badly on English→Polish
        # pairs and inverts the ranking for queries without embedded Polish terms.
        chunks.sort(key=lambda c: c.get("_rerank_score", -99), reverse=True)
        chunks = chunks[:10]

        relevant = [c for c in chunks if c.get("_rerank_score", -99) >= _RELEVANCE_THRESHOLD]
        if not relevant:
            return _OUT_OF_SCOPE
        return _format_context(relevant)

    chain = (
        {"context": retriever_step, "question": RunnablePassthrough()}
        | _PROMPT
        | llm
        | StrOutputParser()
    )
    return chain
