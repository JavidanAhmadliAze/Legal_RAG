from __future__ import annotations

from .client import OpenSearchClient

_client: OpenSearchClient | None = None


def get_opensearch_client() -> OpenSearchClient:
    global _client
    if _client is None:
        _client = OpenSearchClient()
    return _client
