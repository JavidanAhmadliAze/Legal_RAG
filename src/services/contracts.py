from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


@runtime_checkable
class EmbeddingProvider(Protocol):
    def embed_passages(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
    ) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class LLMProvider(Protocol):
    def get_chat_model(self) -> "BaseChatModel": ...

    def get_translator_model(self) -> "BaseChatModel": ...

    def build_chat_model(
        self,
        *,
        streaming: bool,
        request_timeout: float,
    ) -> "BaseChatModel": ...


@runtime_checkable
class RerankingProvider(Protocol):
    def rerank(
        self,
        query: str,
        chunks: list[dict],
        *,
        top_k: int = 5,
        max_per_doc: int = 2,
    ) -> list[dict]: ...
