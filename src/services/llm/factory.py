from __future__ import annotations

import os

from src.services.contracts import LLMProvider

from .client import LLMClient

_DEFAULT_PROVIDER = "deepseek"

_client: LLMProvider | None = None


def _build(provider: str) -> LLMProvider:
    if provider == "deepseek":
        return LLMClient()
    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}. "
        f"Supported: 'deepseek'."
    )


def get_llm_client() -> LLMProvider:
    global _client
    if _client is None:
        provider = os.getenv("LLM_PROVIDER", _DEFAULT_PROVIDER).strip().lower()
        _client = _build(provider or _DEFAULT_PROVIDER)
    return _client
