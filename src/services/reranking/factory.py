from __future__ import annotations

from .client import RerankingClient

_client: RerankingClient | None = None


def get_reranking_client() -> RerankingClient:
    global _client
    if _client is None:
        _client = RerankingClient()
    return _client
