from __future__ import annotations

from .client import FetcherClient

_client: FetcherClient | None = None


def get_fetcher_client() -> FetcherClient:
    global _client
    if _client is None:
        _client = FetcherClient()
    return _client
