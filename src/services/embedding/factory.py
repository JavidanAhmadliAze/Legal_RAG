from __future__ import annotations

import os

from src.services.contracts import EmbeddingProvider

from .client import EmbeddingClient

_DEFAULT_PROVIDER = "sentence_transformers"

_client: EmbeddingProvider | None = None


def _build(provider: str) -> EmbeddingProvider:
    if provider == "sentence_transformers":
        return EmbeddingClient()
    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER={provider!r}. "
        f"Supported: 'sentence_transformers'."
    )


def get_embedding_client() -> EmbeddingProvider:
    global _client
    if _client is None:
        provider = os.getenv("EMBEDDING_PROVIDER", _DEFAULT_PROVIDER).strip().lower()
        _client = _build(provider or _DEFAULT_PROVIDER)
    return _client
