from .client import ChunkDoc, INDEX_NAME
from .factory import get_opensearch_client


def get_client():
    return get_opensearch_client().get_client()


def get_async_client():
    return get_opensearch_client().get_async_client()


def ensure_index() -> None:
    get_opensearch_client().ensure_index()


def bulk_index(docs: list[ChunkDoc]) -> tuple[int, int]:
    return get_opensearch_client().bulk_index(docs)


def delete_document_chunks(document_id: str) -> None:
    get_opensearch_client().delete_document_chunks(document_id)
