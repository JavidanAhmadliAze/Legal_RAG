"""
Retrieval evaluator — process evaluation without ground truth.

Uses an LLM judge to score how relevant the retrieved context chunks are
to the query on a 1-5 scale. Pass when score >= PASS_THRESHOLD (3).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.eval._llm_judge import PASS_THRESHOLD, judge

_SYSTEM = """You are an expert evaluator for information retrieval systems.
Your task is to assess how relevant the retrieved text passages are for answering the given query.

Score on a 1-5 scale:
5 - All passages are highly relevant and directly address the query.
4 - Most passages are relevant; minor off-topic content.
3 - Passages partially address the query; some relevant, some not.
2 - Few passages are relevant; most are off-topic.
1 - Passages are not relevant to the query at all.

Respond with your score (1-5) on the first line, followed by a brief explanation."""


@dataclass
class RetrievalResult:
    score: int
    passed: bool
    reason: str


def evaluate_retrieval(query: str, chunks: list[dict]) -> RetrievalResult:
    """
    Args:
        query:  The user question.
        chunks: Retrieved chunks, each with at least a ``text`` field.
    """
    context = "\n\n---\n\n".join(
        f"[Chunk {i + 1}]\n{c['text']}" for i, c in enumerate(chunks)
    )
    user_prompt = f"Query:\n{query}\n\nRetrieved passages:\n{context}"
    score, reason = judge(_SYSTEM, user_prompt)
    return RetrievalResult(score=score, passed=score >= PASS_THRESHOLD, reason=reason)
