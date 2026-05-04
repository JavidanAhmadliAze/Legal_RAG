from __future__ import annotations

from .client import CacheClient

_client: CacheClient | None = None


def get_cache_client() -> CacheClient:
    global _client
    if _client is None:
        _client = CacheClient()
    return _client
