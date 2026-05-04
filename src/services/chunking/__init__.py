from .structure import Chunk, chunk_by_structure


def chunk_documents(document_ids: list[str]) -> dict:
    from .factory import get_chunking_client

    return get_chunking_client().chunk_documents(document_ids)
