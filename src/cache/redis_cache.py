"""
Redis cache for retrieval results.

Key: SHA-256 of (search_query, rerank_query, top_k, fetch_k, filters)
serialised as canonical JSON.
TTL: 24 h — legal documents change infrequently.
If Redis is unreachable the cache is silently bypassed.
"""

from __future__ import annotations

import hashlib
import json
import os

import redis

_TTL = 86_400  # 24 h

_client: redis.Redis | None = None


def _get_client() -> redis.Redis:
    global _client
    if _client is None:
        url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        _client = redis.Redis.from_url(url, decode_responses=True)
    return _client


def _make_key(
    search_query: str,
    rerank_query: str,
    top_k: int,
    fetch_k: int,
    filters: dict,
) -> str:
    payload = json.dumps(
        {
            "search_q": search_query,
            "rerank_q": rerank_query,
            "top_k": top_k,
            "fetch_k": fetch_k,
            "filters": filters,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return "rag:retrieve:" + hashlib.sha256(payload.encode()).hexdigest()


def get(
    search_query: str,
    rerank_query: str,
    top_k: int,
    fetch_k: int,
    filters: dict,
) -> list[dict] | None:
    """Return cached chunks, or None on miss / Redis unavailable."""
    try:
        raw = _get_client().get(
            _make_key(search_query, rerank_query, top_k, fetch_k, filters)
        )
        return json.loads(raw) if raw is not None else None
    except Exception:
        return None


def set(
    search_query: str,
    rerank_query: str,
    top_k: int,
    fetch_k: int,
    filters: dict,
    chunks: list[dict],
) -> None:
    """Cache chunks. Silently swallows errors if Redis is unavailable."""
    try:
        _get_client().set(
            _make_key(search_query, rerank_query, top_k, fetch_k, filters),
            json.dumps(chunks, ensure_ascii=False),
            ex=_TTL,
        )
    except Exception:
        pass
