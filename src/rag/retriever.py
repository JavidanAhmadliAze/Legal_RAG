"""
Retrieval pipeline:
  1. BM25 search       → top fetch_k candidates
  2. k-NN search       → top fetch_k candidates
  3. Metadata filter   → applied to each set independently
  4. RRF combine       → merge filtered sets
  5. Cross-encoder     → rerank and return top_k
"""

from __future__ import annotations

import json

from opentelemetry import trace

from src.cache import redis_cache
from src.index.embeddings import embed_query
from src.index.opensearch_client import INDEX_NAME, get_client
from src.rag.reranker import rerank

_tracer = trace.get_tracer("legal-rag.retriever")


# ---------------------------------------------------------------------------
# OpenSearch helpers
# ---------------------------------------------------------------------------

def _os_filter_clauses(filters: dict) -> list[dict]:
    """Convert {field: value} to a list of OpenSearch term/range clauses."""
    clauses = []
    for field, value in filters.items():
        if isinstance(value, dict):
            clauses.append({"range": {field: value}})
        else:
            clauses.append({"term": {field: value}})
    return clauses


def _bm25_search(client, question: str, fetch_k: int, filters: dict) -> list[dict]:
    must: dict = {"match": {"text": {"query": question}}}
    body: dict = {
        "size": fetch_k,
        "_source": {"excludes": ["embedding"]},
    }
    if filters:
        body["query"] = {
            "bool": {"must": must, "filter": _os_filter_clauses(filters)}
        }
    else:
        body["query"] = must
    return client.search(index=INDEX_NAME, body=body)["hits"]["hits"]


def _knn_search(client, vec: list[float], fetch_k: int, filters: dict) -> list[dict]:
    knn_clause: dict = {"vector": vec, "k": fetch_k}
    if filters:
        filter_list = _os_filter_clauses(filters)
        knn_clause["filter"] = (
            {"bool": {"filter": filter_list}} if len(filter_list) > 1 else filter_list[0]
        )
    body = {
        "size": fetch_k,
        "query": {"knn": {"embedding": knn_clause}},
        "_source": {"excludes": ["embedding"]},
    }
    return client.search(index=INDEX_NAME, body=body)["hits"]["hits"]


# ---------------------------------------------------------------------------
# Post-retrieval metadata filter
# ---------------------------------------------------------------------------

def _metadata_filter(hits: list[dict], filters: dict) -> list[dict]:
    """Drop hits whose _source fields don't match every filter entry."""
    if not filters:
        return hits
    out = []
    for hit in hits:
        src = hit["_source"]
        match = True
        for field, value in filters.items():
            if isinstance(value, dict):
                v = src.get(field)
                if "gte" in value and v < value["gte"]:
                    match = False
                if "lte" in value and v > value["lte"]:
                    match = False
            elif src.get(field) != value:
                match = False
        if match:
            out.append(hit)
    return out


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------

def _rrf(results_list: list[list[dict]], k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}
    for results in results_list:
        for rank, hit in enumerate(results):
            _id = hit["_id"]
            scores[_id] = scores.get(_id, 0.0) + 1.0 / (k + rank + 1)
            docs[_id] = hit["_source"]
    return [docs[_id] for _id in sorted(scores, key=scores.__getitem__, reverse=True)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(
    question: str,
    *,
    top_k: int = 5,
    fetch_k: int = 20,
    filters: dict | None = None,
    rerank_query: str | None = None,
) -> list[dict]:
    """
    Full retrieval pipeline for *question*.

    Args:
        top_k:   Final number of chunks returned after reranking.
        fetch_k: Candidates fetched per retrieval method before filtering.
        filters: Optional metadata filters, e.g.
                 {"year": 2025} or {"year": {"gte": 2024}, "act_type": "Ustawa"}.
    """
    filters = filters or {}
    rerank_query = rerank_query or question

    cached = redis_cache.get(question, rerank_query, top_k, fetch_k, filters)
    if cached is not None:
        return cached

    client = get_client()

    with _tracer.start_as_current_span("retrieve") as span:
        span.set_attribute("retrieval.query", question)
        span.set_attribute("retrieval.rerank_query", rerank_query)
        span.set_attribute("retrieval.fetch_k", fetch_k)
        span.set_attribute("retrieval.top_k", top_k)
        if filters:
            span.set_attribute("retrieval.filters", json.dumps(filters))

        vec = embed_query(question)

        # 1 — BM25
        with _tracer.start_as_current_span("bm25_search") as bm25_span:
            bm25_hits = _bm25_search(client, question, fetch_k, filters)
            bm25_span.set_attribute("retrieval.bm25_hits", len(bm25_hits))

        # 2 — k-NN
        with _tracer.start_as_current_span("knn_search") as knn_span:
            knn_hits = _knn_search(client, vec, fetch_k, filters)
            knn_span.set_attribute("retrieval.knn_hits", len(knn_hits))

        # 3 — metadata filter each set independently
        with _tracer.start_as_current_span("metadata_filter") as mf_span:
            bm25_hits = _metadata_filter(bm25_hits, filters)
            knn_hits = _metadata_filter(knn_hits, filters)
            mf_span.set_attribute("retrieval.bm25_after_filter", len(bm25_hits))
            mf_span.set_attribute("retrieval.knn_after_filter", len(knn_hits))

        # 4 — RRF combine
        combined = _rrf([bm25_hits, knn_hits])
        span.set_attribute("retrieval.combined_count", len(combined))

        # 5 — cross-encoder rerank
        with _tracer.start_as_current_span("cross_encoder_rerank") as ce_span:
            reranked = rerank(rerank_query, combined, top_k=top_k)
            ce_span.set_attribute("retrieval.reranked_count", len(reranked))
            # Log per-chunk scores for precision monitoring
            for i, chunk in enumerate(reranked):
                ce_span.set_attribute(
                    f"retrieval.chunk_{i}.doc_id", chunk.get("document_id", "")
                )
                ce_span.set_attribute(
                    f"retrieval.chunk_{i}.page", chunk.get("page_num", -1)
                )
                ce_span.set_attribute(
                    f"retrieval.chunk_{i}.rerank_score",
                    chunk.get("_rerank_score", 0.0),
                )

        redis_cache.set(question, rerank_query, top_k, fetch_k, filters, reranked)
        return reranked
