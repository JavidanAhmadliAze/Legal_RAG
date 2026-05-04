from __future__ import annotations

from sentence_transformers import SentenceTransformer

from src.services.config import EmbeddingSettings, get_embedding_settings

_SETTINGS = get_embedding_settings()
MODEL_NAME = _SETTINGS.model_name
EMBEDDING_DIM = _SETTINGS.embedding_dim


class EmbeddingClient:
    def __init__(
        self,
        model_name: str | None = None,
        *,
        settings: EmbeddingSettings | None = None,
    ) -> None:
        self._settings = settings or get_embedding_settings()
        self._model_name = model_name or self._settings.model_name
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_passages(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
    ) -> list[list[float]]:
        selected_batch_size = (
            self._settings.batch_size if batch_size is None else batch_size
        )
        prefixed = [f"passage: {text}" for text in texts]
        vectors = self._get_model().encode(
            prefixed,
            batch_size=selected_batch_size,
            normalize_embeddings=self._settings.normalize_embeddings,
            show_progress_bar=self._settings.show_progress_bar,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self._get_model().encode(
            f"query: {text}",
            normalize_embeddings=self._settings.normalize_embeddings,
        )
        return vector.tolist()
