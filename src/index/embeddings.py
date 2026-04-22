"""
Embedding wrapper using intfloat/multilingual-e5-base.

e5 models require a task prefix:
  - "passage: " for chunks being indexed
  - "query: "   for search queries at retrieval time
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-base"
EMBEDDING_DIM = 768

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_passages(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed document chunks for indexing."""
    prefixed = [f"passage: {t}" for t in texts]
    vecs = _get_model().encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return vecs.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a search query for retrieval."""
    vec = _get_model().encode(f"query: {text}", normalize_embeddings=True)
    return vec.tolist()
