from __future__ import annotations

from .client import ChunkingClient

_client: ChunkingClient | None = None


def get_chunking_client() -> ChunkingClient:
    global _client
    if _client is None:
        _client = ChunkingClient()
    return _client
