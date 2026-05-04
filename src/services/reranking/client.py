from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.services.config import RerankingSettings, get_reranking_settings

_SETTINGS = get_reranking_settings()
MODEL_NAME = _SETTINGS.model_name


class RerankingClient:
    def __init__(self, *, settings: RerankingSettings | None = None) -> None:
        self._settings = settings or get_reranking_settings()
        self._model: CrossEncoder | None = None

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(
                self._settings.model_name,
                max_length=self._settings.max_length,
            )
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        *,
        top_k: int = 5,
        max_per_doc: int = 2,
    ) -> list[dict]:
        """Score every (query, chunk) pair and return the top_k by score.

        Diversity: at most ``max_per_doc`` chunks per ``document_id`` are kept.
        Each returned chunk gets a ``_rerank_score`` field attached so callers
        can log relevance without a separate data structure.
        """
        if not chunks:
            return []

        model = self._get_model()
        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = model.predict(
            pairs,
            show_progress_bar=self._settings.show_progress_bar,
            batch_size=self._settings.batch_size,
        )
        ranked = sorted(zip(scores, chunks), key=lambda item: item[0], reverse=True)

        result: list[dict] = []
        per_doc: dict[str, int] = {}
        overflow: list[dict] = []

        for score, chunk in ranked:
            document_id = chunk.get("document_id", "")
            scored_chunk = dict(chunk)
            scored_chunk["_rerank_score"] = float(score)
            if per_doc.get(document_id, 0) < max_per_doc:
                result.append(scored_chunk)
                per_doc[document_id] = per_doc.get(document_id, 0) + 1
                if len(result) >= top_k:
                    break
            else:
                overflow.append(scored_chunk)

        if len(result) < top_k and overflow:
            for chunk in overflow:
                result.append(chunk)
                if len(result) >= top_k:
                    break

        return result
