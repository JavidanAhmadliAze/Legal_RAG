from .client import EMBEDDING_DIM, MODEL_NAME
from .factory import get_embedding_client


def embed_passages(
    texts: list[str],
    batch_size: int | None = None,
) -> list[list[float]]:
    return get_embedding_client().embed_passages(texts, batch_size=batch_size)


def embed_query(text: str) -> list[float]:
    return get_embedding_client().embed_query(text)
