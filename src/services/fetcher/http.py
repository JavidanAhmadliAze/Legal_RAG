from __future__ import annotations

import time

import httpx

from src.services.config import get_fetcher_settings

USER_AGENT = get_fetcher_settings().user_agent


def fetch(
    url: str,
    *,
    retries: int | None = None,
    backoff: float | None = None,
) -> httpx.Response:
    settings = get_fetcher_settings()
    retry_count = max(1, settings.retries if retries is None else retries)
    backoff_factor = max(0.0, settings.backoff if backoff is None else backoff)
    headers = {"User-Agent": settings.user_agent}
    last_exc: Exception | None = None
    for attempt in range(retry_count):
        try:
            with httpx.Client(
                headers=headers,
                follow_redirects=settings.follow_redirects,
                timeout=settings.timeout_s,
            ) as client:
                response = client.get(url)
                response.raise_for_status()
                return response
        except httpx.HTTPError as exc:
            last_exc = exc
            if attempt < retry_count - 1:
                time.sleep(backoff_factor**attempt)
    raise last_exc  # type: ignore[misc]
