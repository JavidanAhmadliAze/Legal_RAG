"""
Retrieval pipeline (async-only — OpenSearch BM25 + k-NN are external I/O):
  1. BM25 search       → top fetch_k candidates (text + title)
  2. k-NN search       → top fetch_k candidates
  3. Metadata filter   → year/act_type/law_domain applied Python-side
  4. RRF combine       → merge filtered sets
  5. Cross-encoder     → rerank and return top_k
"""

from __future__ import annotations

import asyncio
import json

from opentelemetry import trace

from src.services import cache
from src.services.embedding import embed_query
from src.services.opensearch import INDEX_NAME, get_async_client
from src.services.reranking import rerank

_tracer = trace.get_tracer("legal-rag.retriever")


# ---------------------------------------------------------------------------
# OpenSearch helpers
# ---------------------------------------------------------------------------

def _os_filter_clauses(filters: dict) -> list[dict]:
    """No metadata filters are pushed into the OpenSearch query.

    Pipeline order is strict:
      1. BM25 + k-NN run against the FULL corpus (no filters in the OS query).
      2. Metadata filters apply Python-side in _metadata_filter() — separately
         on the BM25 set and the k-NN set.
      3. RRF combines the two filtered sets.
      4. Cross-encoder reranks and selects top-k.

    Pushing any filter into OpenSearch would kill recall for queries whose
    target domain/year/act_type has zero (or nearly zero) docs in the index.
    """
    return []


# ---------------------------------------------------------------------------
# Post-retrieval metadata filter
# ---------------------------------------------------------------------------

def _metadata_filter(hits: list[dict], filters: dict) -> list[dict]:
    """Drop hits whose _source fields don't match every filter entry.

    Hard filters (year, act_type) drop non-matching hits unconditionally.

    Soft filter ``law_domain``: only applied when at least one hit actually
    matches it.  If zero hits in the BM25/k-NN result match the detected
    domain, the corpus has no docs of that domain near this query — we keep
    all hits and let the cross-encoder decide relevance, rather than zeroing
    out the result.

    Special filter ``title_any``: title must contain at least one of the
    given substrings (case-insensitive).
    """
    if not filters:
        return hits
    title_any = filters.get("title_any")
    law_domain = filters.get("law_domain")

    # Decide whether the soft law_domain filter has any matches in this batch.
    apply_law_domain = bool(law_domain) and any(
        (h.get("_source", {}).get("law_domain") == law_domain) for h in hits
    )

    out = []
    for hit in hits:
        src = hit["_source"]
        match = True
        for field, value in filters.items():
            if field in ("title_any", "law_domain"):
                continue
            if isinstance(value, dict):
                v = src.get(field)
                if "gte" in value and v < value["gte"]:
                    match = False
                if "lte" in value and v > value["lte"]:
                    match = False
            elif isinstance(value, (list, tuple)):
                if src.get(field) not in value:
                    match = False
            elif src.get(field) != value:
                match = False
        if not match:
            continue
        if apply_law_domain and src.get("law_domain") != law_domain:
            continue
        if title_any:
            title_l = (src.get("title") or "").lower()
            if not any(stem.lower() in title_l for stem in title_any):
                continue
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
# Public API — async only.  OpenSearch BM25 + k-NN run as native async I/O.
# Embedding + cross-encoder stay in to_thread because they're CPU-bound
# (PyTorch holds the GIL during inference).  Sync callers (eval scripts,
# CLI tools) wrap with asyncio.run(aretrieve(...)).
# ---------------------------------------------------------------------------

async def _abm25_search(client, question: str, fetch_k: int, filters: dict) -> list[dict]:
    with _tracer.start_as_current_span("bm25_search") as span:
        must: dict = {"match": {"text": {"query": question}}}
        body: dict = {"size": fetch_k, "_source": {"excludes": ["embedding"]}}
        os_clauses = _os_filter_clauses(filters) if filters else []
        body["query"] = (
            {"bool": {"must": must, "filter": os_clauses}}
            if os_clauses else must
        )
        res = await client.search(index=INDEX_NAME, body=body)
        hits = res["hits"]["hits"]
        span.set_attribute("retrieval.bm25_hits", len(hits))
        return hits


async def _aknn_search(client, vec: list[float], fetch_k: int, filters: dict) -> list[dict]:
    with _tracer.start_as_current_span("knn_search") as span:
        knn_clause: dict = {"vector": vec, "k": fetch_k}
        if filters:
            fl = _os_filter_clauses(filters)
            if len(fl) == 1:
                knn_clause["filter"] = fl[0]
            elif len(fl) > 1:
                knn_clause["filter"] = {"bool": {"filter": fl}}
        body = {
            "size": fetch_k,
            "query": {"knn": {"embedding": knn_clause}},
            "_source": {"excludes": ["embedding"]},
        }
        res = await client.search(index=INDEX_NAME, body=body)
        hits = res["hits"]["hits"]
        span.set_attribute("retrieval.knn_hits", len(hits))
        return hits


async def aretrieve(
    question: str,
    *,
    top_k: int = 5,
    fetch_k: int = 20,
    filters: dict | None = None,
    rerank_query: str | None = None,
    hyde_passage: str | None = None,
) -> list[dict]:
    """Async version of retrieve(). Used by the supervisor for parallel fan-out."""
    filters = filters or {}
    rerank_query = rerank_query or question

    cached = cache.get(question, rerank_query, top_k, fetch_k, filters)
    if cached is not None:
        return cached

    client = get_async_client()

    with _tracer.start_as_current_span("aretrieve") as span:
        span.set_attribute("retrieval.query", question)
        span.set_attribute("retrieval.rerank_query", rerank_query)
        span.set_attribute("retrieval.fetch_k", fetch_k)
        span.set_attribute("retrieval.top_k", top_k)
        span.set_attribute("retrieval.hyde", hyde_passage is not None)
        if filters:
            span.set_attribute("retrieval.filters", json.dumps(filters))

        # Embedding is CPU-bound (PyTorch) — to_thread.
        embed_text = hyde_passage if hyde_passage else question
        with _tracer.start_as_current_span("embed_query"):
            vec = await asyncio.to_thread(embed_query, embed_text)

        # BM25 and k-NN run TRULY in parallel as async I/O.
        bm25_hits, knn_hits = await asyncio.gather(
            _abm25_search(client, question, fetch_k, filters),
            _aknn_search(client, vec, fetch_k, filters),
        )
        span.set_attribute("retrieval.bm25_hits", len(bm25_hits))
        span.set_attribute("retrieval.knn_hits", len(knn_hits))

        with _tracer.start_as_current_span("metadata_filter") as mf_span:
            bm25_hits = _metadata_filter(bm25_hits, filters)
            knn_hits = _metadata_filter(knn_hits, filters)
            mf_span.set_attribute("retrieval.bm25_after_filter", len(bm25_hits))
            mf_span.set_attribute("retrieval.knn_after_filter", len(knn_hits))
        span.set_attribute("retrieval.bm25_after_filter", len(bm25_hits))
        span.set_attribute("retrieval.knn_after_filter", len(knn_hits))

        combined = _rrf([bm25_hits, knn_hits])
        span.set_attribute("retrieval.combined_count", len(combined))

        # Cross-encoder is CPU-bound — to_thread.
        with _tracer.start_as_current_span("cross_encoder_rerank") as ce_span:
            reranked = await asyncio.to_thread(rerank, rerank_query, combined, top_k=top_k, max_per_doc=5)
            ce_span.set_attribute("retrieval.reranked_count", len(reranked))
            for i, chunk in enumerate(reranked):
                ce_span.set_attribute(f"retrieval.chunk_{i}.doc_id",
                                      chunk.get("document_id", ""))
                ce_span.set_attribute(f"retrieval.chunk_{i}.page",
                                      chunk.get("page_num", -1))
                ce_span.set_attribute(f"retrieval.chunk_{i}.rerank_score",
                                      chunk.get("_rerank_score", 0.0))

        cache.set(question, rerank_query, top_k, fetch_k, filters, reranked)
        return reranked
