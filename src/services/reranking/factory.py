from __future__ import annotations

import os

from src.services.contracts import RerankingProvider

from .client import RerankingClient

_DEFAULT_PROVIDER = "cross_encoder"

_client: RerankingProvider | None = None


def _build(provider: str) -> RerankingProvider:
    if provider == "cross_encoder":
        return RerankingClient()
    raise ValueError(
        f"Unknown RERANKING_PROVIDER={provider!r}. "
        f"Supported: 'cross_encoder'."
    )


def get_reranking_client() -> RerankingProvider:
    global _client
    if _client is None:
        provider = os.getenv("RERANKING_PROVIDER", _DEFAULT_PROVIDER).strip().lower()
        _client = _build(provider or _DEFAULT_PROVIDER)
    return _client
