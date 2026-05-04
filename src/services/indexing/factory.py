from __future__ import annotations

from .client import IndexingClient

_client: IndexingClient | None = None


def get_indexing_client() -> IndexingClient:
    global _client
    if _client is None:
        _client = IndexingClient()
    return _client
