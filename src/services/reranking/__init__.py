from .client import MODEL_NAME
from .factory import get_reranking_client


def rerank(
    query: str,
    chunks: list[dict],
    *,
    top_k: int = 5,
    max_per_doc: int = 2,
) -> list[dict]:
    return get_reranking_client().rerank(
        query,
        chunks,
        top_k=top_k,
        max_per_doc=max_per_doc,
    )
