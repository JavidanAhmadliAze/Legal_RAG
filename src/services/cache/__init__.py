from .factory import get_cache_client


def get(
    search_query: str,
    rerank_query: str,
    top_k: int,
    fetch_k: int,
    filters: dict,
) -> list[dict] | None:
    return get_cache_client().get(
        search_query,
        rerank_query,
        top_k,
        fetch_k,
        filters,
    )


def set(
    search_query: str,
    rerank_query: str,
    top_k: int,
    fetch_k: int,
    filters: dict,
    chunks: list[dict],
) -> None:
    get_cache_client().set(
        search_query,
        rerank_query,
        top_k,
        fetch_k,
        filters,
        chunks,
    )
