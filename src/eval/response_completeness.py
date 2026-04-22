"""
Response Completeness evaluator — system evaluation with ground truth.

Measures how completely the generated response covers the expected answer,
i.e. the recall aspect: does the response miss critical information that
the ground-truth answer contains?
Uses an LLM judge. Pass when score >= PASS_THRESHOLD (3).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.eval._llm_judge import PASS_THRESHOLD, judge

_SYSTEM = """You are an expert evaluator assessing response completeness.
Completeness measures whether the generated response covers all the critical information
present in the reference (ground-truth) answer. Focus only on missing information —
do not penalise for extra correct detail.

Score on a 1-5 scale:
5 - The response covers all key information from the reference answer.
4 - The response covers most key information; one minor point is missing.
3 - The response covers roughly half of the key information.
2 - The response misses most key information from the reference answer.
1 - The response is missing almost all key information.

Respond with your score (1-5) on the first line, followed by a brief explanation
noting what, if anything, is missing."""


@dataclass
class ResponseCompletenessResult:
    score: int
    passed: bool
    reason: str


def evaluate_response_completeness(
    response: str,
    ground_truth: str,
) -> ResponseCompletenessResult:
    """
    Args:
        response:     The LLM-generated answer.
        ground_truth: The expected (reference) answer.
    """
    user_prompt = (
        f"Reference answer:\n{ground_truth}\n\n"
        f"Generated response:\n{response}"
    )
    score, reason = judge(_SYSTEM, user_prompt)
    return ResponseCompletenessResult(
        score=score, passed=score >= PASS_THRESHOLD, reason=reason
    )
