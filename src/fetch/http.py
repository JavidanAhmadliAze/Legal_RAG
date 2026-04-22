import time

import httpx

USER_AGENT = "LegalRAG/0.1 (research ingestion pipeline; non-commercial)"

_HEADERS = {"User-Agent": USER_AGENT}


def fetch(url: str, *, retries: int = 3, backoff: float = 2.0) -> httpx.Response:
    """Download *url*, retrying on transient errors with exponential backoff."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            with httpx.Client(headers=_HEADERS, follow_redirects=True, timeout=30) as client:
                resp = client.get(url)
                resp.raise_for_status()
                return resp
        except httpx.HTTPError as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(backoff**attempt)
    raise last_exc  # type: ignore[misc]
