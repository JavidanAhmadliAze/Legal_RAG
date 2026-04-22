"""
Cross-encoder reranker for final scoring of candidate chunks.

Model: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
  - Multilingual (mMARCO), handles Polish legal text well.
  - Scores (query, passage) pairs; higher = more relevant.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model


def rerank(query: str, chunks: list[dict], *, top_k: int = 5) -> list[dict]:
    """Score every (query, chunk) pair and return the top_k by cross-encoder score.

    Each returned chunk gets a ``_rerank_score`` field attached so callers
    (and OTel spans) can log relevance without a separate data structure.
    """
    if not chunks:
        return []
    model = _get_model()
    pairs = [(query, c["text"]) for c in chunks]
    scores = model.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    result = []
    for score, chunk in ranked[:top_k]:
        chunk = dict(chunk)          # shallow copy — don't mutate caller's dict
        chunk["_rerank_score"] = float(score)
        result.append(chunk)
    return result
