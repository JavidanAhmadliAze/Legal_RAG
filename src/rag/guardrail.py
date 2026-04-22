"""
Input guardrail for the Polish legal RAG assistant.

Two checks, in order:
1. Length — reject queries over MAX_WORDS words.
2. Relevance — LLM classifier rejects queries unrelated to Polish law.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

MAX_WORDS = 120

_CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a query classifier for a Polish legal information system. "
        "Decide if the user's query is related to Polish law, Polish legal acts, "
        "Polish regulations, Polish administrative procedures, or Polish legal institutions. "
        "Reply with exactly one word: YES or NO."
    )),
    ("human", "{query}"),
])


@dataclass
class GuardrailResult:
    allowed: bool
    rejection_message: str = ""


def _get_classifier() -> ChatOpenAI:
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        streaming=False,
        temperature=0,
        max_tokens=5,
    )


_classifier_chain = None


def _get_chain():
    global _classifier_chain
    if _classifier_chain is None:
        _classifier_chain = _CLASSIFIER_PROMPT | _get_classifier() | StrOutputParser()
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
    if not verdict.startswith("YES"):
        return GuardrailResult(
            allowed=False,
            rejection_message=(
                "I can only help with questions related to Polish law and legal acts. "
                "Please ask about Polish legislation, regulations, or legal procedures."
            ),
        )

    return GuardrailResult(allowed=True)
