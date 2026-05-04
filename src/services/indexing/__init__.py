from .client import BLOCKED_STATUSES, DEFAULT_EMBED_BATCH, DEFAULT_RATE_LIMIT_S


def index_documents(
    document_ids: list[str],
    *,
    embed_batch: int | None = None,
) -> dict:
    from .factory import get_indexing_client

    return get_indexing_client().index_documents(
        document_ids,
        embed_batch=embed_batch,
    )


def get_indexed_document_ids() -> list[str]:
    from .factory import get_indexing_client

    return get_indexing_client().get_indexed_document_ids()


def find_expired_documents(
    document_ids: list[str],
    *,
    rate_limit_s: float | None = None,
) -> list[str]:
    from .factory import get_indexing_client

    return get_indexing_client().find_expired_documents(
        document_ids,
        rate_limit_s=rate_limit_s,
    )


def purge_expired_documents(expired_ids: list[str]) -> dict:
    from .factory import get_indexing_client

    return get_indexing_client().purge_expired_documents(expired_ids)
