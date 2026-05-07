"""
Shared agent utilities reused by the legacy LangChain runner ([chain.py](chain.py))
and the live LangGraph nodes ([nodes.py](nodes.py)):

  - prompt aliases pulled from the LLM module
  - the relevance threshold used to drop low-confidence rerank chunks
  - the OUT_OF_SCOPE sentinel returned when no chunk survives the threshold
  - the chunk → context formatter that builds the bracketed source headers
  - the regex-based question splitter (used by the legacy chain and unit tests)
  - the citation verifier that strips sentences citing articles or
    Dz.U./M.P. entries that are not present in the retrieved context
"""

from __future__ import annotations

import re

from src.services.agents.prompts import ANSWER_PROMPT, TRANSLATE_PROMPT


# ---------------------------------------------------------------------------
# Sub-question splitter (regex heuristic — used by the legacy chain runner
# and exercised by tests/unit/test_chain.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Relevance threshold + out-of-scope sentinel
# ---------------------------------------------------------------------------

# sdadas/polish-reranker-bge-v2 returns sigmoid-scaled scores in [0, 1].
# Scores below 0.3 are typically only-keyword-overlap noise; above 0.3 there is
# genuine semantic relevance.
_RELEVANCE_THRESHOLD = 0.15

_OUT_OF_SCOPE = (
    "OUT_OF_SCOPE — This question does not match Polish immigration or foreigners law. "
    "Apply rule 3 from the system prompt."
)


# ---------------------------------------------------------------------------
# Context formatter — chunk → bracketed source header + body
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Citation verifier — strip sentences citing articles/acts not in the context
# ---------------------------------------------------------------------------

# Matches the *base* article (letter suffix allowed, sub-numbers ignored).
# "Art. 195(1)(2)" → captures "195"; "Art. 252a" → captures "252a".
_ART_RE = re.compile(r"\bArt\.\s*(\d+[a-z]?)", re.IGNORECASE)
_DZU_RE = re.compile(
    r"\b(?:Dz\.?\s*U\.?|Dziennik\s+Ustaw|M\.?\s*P\.?|Monitor\s+Polski)\s*"
    r"(\d{4})\s*poz\.\s*(\d+)",
    re.IGNORECASE,
)
# With a broad-coverage corpus, no act name is automatically out-of-coverage.
# Hallucination is caught by the article/Dz.U. checks below — references must
# appear verbatim in the retrieved context to survive verification.
_OUT_OF_COVERAGE_RE = None


# Polish/Latin legal abbreviations that end in "." — must not trigger a
# sentence boundary. We swap the period for \x00 before splitting and restore
# it after.
_ABBREV_RE = re.compile(
    r"\b(Art|art|ust|lit|pkt|poz|Dz\.\s*U|M\.\s*P|Dz\.\s*Urz|nr|r|z|w|o|itd|itp|np|tj|ok)\.",
    re.IGNORECASE,
)


def _split_sentences(text: str) -> list[str]:
    masked = _ABBREV_RE.sub(lambda m: m.group(0).replace(".", "\x00"), text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ])", masked)
    return [p.replace("\x00", ".") for p in parts]


def _verify_citations(answer: str, context: str) -> str:
    """Drop answer sentences that cite articles, Dz.U./M.P. items, or acts
    not present in the retrieved context."""
    if not answer or not context:
        return answer

    context_articles = {m.group(1).lower() for m in _ART_RE.finditer(context)}
    context_dzu = {(m.group(1), m.group(2)) for m in _DZU_RE.finditer(context)}

    sentences = _split_sentences(answer.strip())
    kept: list[str] = []
    dropped: list[str] = []

    for sent in sentences:
        reasons: list[str] = []

        for m in _ART_RE.finditer(sent):
            if m.group(1).lower() not in context_articles:
                reasons.append(m.group(0))

        for m in _DZU_RE.finditer(sent):
            if (m.group(1), m.group(2)) not in context_dzu:
                reasons.append(m.group(0))

        if _OUT_OF_COVERAGE_RE is not None:
            scope_hit = _OUT_OF_COVERAGE_RE.search(sent)
            if scope_hit:
                reasons.append(scope_hit.group(0))

        if reasons:
            dropped.append(sent)
        else:
            kept.append(sent)

    cleaned = " ".join(kept).strip()

    if not cleaned and dropped:
        return (
            "I can't answer this from my indexed sources (Polish foreigners law). "
            "Please consult a specialist for the relevant area of law."
        )
    if dropped:
        cleaned += (
            "\n\n_(Some content was removed because it cited articles or acts "
            "outside the retrieved sources.)_"
        )
    return cleaned


# ---------------------------------------------------------------------------
# Prompt aliases — kept here so the rest of the agent code can refer to them
# without reaching into src.services.llm directly.
# ---------------------------------------------------------------------------

_PROMPT = ANSWER_PROMPT
_TRANSLATE_PROMPT = TRANSLATE_PROMPT
