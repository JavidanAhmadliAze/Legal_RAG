"""
Relevance evaluator — system evaluation without ground truth.

Measures accuracy, completeness, and direct relevance of the response
to the user's query. Uses an LLM judge. Pass when score >= PASS_THRESHOLD (3).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.eval._llm_judge import PASS_THRESHOLD, judge

_SYSTEM = """You are an expert evaluator assessing the relevance of an AI-generated response.
Relevance measures how accurately and completely the response addresses the user's query.

Score on a 1-5 scale:
5 - The response fully and accurately answers the query with appropriate detail.
4 - The response mostly answers the query; minor gaps or slight inaccuracies.
3 - The response partially answers the query; some important aspects are missing.
2 - The response barely addresses the query; mostly off-topic or incomplete.
1 - The response does not address the query at all.

Respond with your score (1-5) on the first line, followed by a brief explanation."""


@dataclass
class RelevanceResult:
    score: int
    passed: bool
    reason: str


def evaluate_relevance(query: str, response: str) -> RelevanceResult:
    """
    Args:
        query:    The user question.
        response: The LLM-generated answer.
    """
    user_prompt = f"Query:\n{query}\n\nResponse:\n{response}"
    score, reason = judge(_SYSTEM, user_prompt)
    return RelevanceResult(score=score, passed=score >= PASS_THRESHOLD, reason=reason)
