from __future__ import annotations

import hashlib
import json

import redis

from src.services.config import CacheSettings, get_cache_settings

DEFAULT_TTL_S = get_cache_settings().ttl_s


class CacheClient:
    def __init__(
        self,
        *,
        settings: CacheSettings | None = None,
        redis_url: str | None = None,
        ttl_s: int | None = None,
    ) -> None:
        self._settings = settings or get_cache_settings()
        self._redis_url = redis_url or self._settings.redis_url
        self._ttl_s = self._settings.ttl_s if ttl_s is None else ttl_s
        self._key_prefix = self._settings.key_prefix
        self._client: redis.Redis | None = None

    def _get_client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.Redis.from_url(
                self._redis_url,
                decode_responses=True,
            )
        return self._client

    def _make_key(
        self,
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
        return self._key_prefix + hashlib.sha256(payload.encode()).hexdigest()

    def get(
        self,
        search_query: str,
        rerank_query: str,
        top_k: int,
        fetch_k: int,
        filters: dict,
    ) -> list[dict] | None:
        try:
            raw = self._get_client().get(
                self._make_key(
                    search_query,
                    rerank_query,
                    top_k,
                    fetch_k,
                    filters,
                )
            )
            return json.loads(raw) if raw is not None else None
        except Exception:
            return None

    def set(
        self,
        search_query: str,
        rerank_query: str,
        top_k: int,
        fetch_k: int,
        filters: dict,
        chunks: list[dict],
    ) -> None:
        try:
            self._get_client().set(
                self._make_key(
                    search_query,
                    rerank_query,
                    top_k,
                    fetch_k,
                    filters,
                ),
                json.dumps(chunks, ensure_ascii=False),
                ex=self._ttl_s,
            )
        except Exception:
            pass
