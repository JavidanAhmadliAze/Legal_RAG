"""
Groundedness evaluator — system evaluation without ground truth.

Measures how well the generated response is grounded in the provided context,
i.e. whether it fabricates information not present in the retrieved chunks.
Uses an LLM judge (precision aspect). Pass when score >= PASS_THRESHOLD (3).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.eval._llm_judge import PASS_THRESHOLD, judge

_SYSTEM = """You are an expert evaluator assessing the groundedness of an AI-generated response.
Groundedness measures whether every factual claim in the response is supported by the provided context.
A response that invents information not present in the context is NOT grounded.

Score on a 1-5 scale:
5 - Every claim in the response is explicitly supported by the context.
4 - Nearly all claims are supported; one minor unsupported detail.
3 - Most claims are supported but there are noticeable gaps or additions.
2 - Several claims lack support in the context; notable fabrication.
1 - The response largely fabricates information not found in the context.

Respond with your score (1-5) on the first line, followed by a brief explanation."""


@dataclass
class GroundednessResult:
    score: int
    passed: bool
    reason: str


def evaluate_groundedness(response: str, chunks: list[dict]) -> GroundednessResult:
    """
    Args:
        response: The LLM-generated answer.
        chunks:   Retrieved chunks that were passed as context to the LLM.
    """
    context = "\n\n---\n\n".join(
        f"[Chunk {i + 1}]\n{c['text']}" for i, c in enumerate(chunks)
    )
    user_prompt = (
        f"Context provided to the model:\n{context}\n\n"
        f"Model response:\n{response}"
    )
    score, reason = judge(_SYSTEM, user_prompt)
    return GroundednessResult(score=score, passed=score >= PASS_THRESHOLD, reason=reason)
