"""
OpenSearch client — index setup and chunk indexing.

Single index `legal_chunks` with:
  - text field  → BM25 full-text search (default OpenSearch behaviour)
  - embedding   → k-NN vector search (HNSW / cosine, via Lucene engine)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from opensearchpy import OpenSearch, helpers

from src.index.embeddings import EMBEDDING_DIM

INDEX_NAME = "legal_chunks"

_OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
_OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))

_INDEX_SETTINGS = {
    "settings": {
        "index": {
            "knn": True,
            "knn.algo_param.ef_search": 100,
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "analysis": {
            "analyzer": {
                "polish_standard": {
                    "type": "standard",
                    "stopwords": "_none_",
                }
            }
        },
    },
    "mappings": {
        "properties": {
            "document_id":       {"type": "keyword"},
            "source_url":        {"type": "keyword"},
            "chunk_index":       {"type": "integer"},
            "page_num":          {"type": "integer"},
            "anchor":            {"type": "keyword"},
            "title":             {"type": "text", "analyzer": "polish_standard"},
            "act_type":          {"type": "keyword"},
            "year":              {"type": "integer"},
            "pos":               {"type": "integer"},
            "status":            {"type": "keyword"},
            "announcement_date": {"type": "keyword"},
            "text": {
                "type": "text",
                "analyzer": "polish_standard",
            },
            "embedding": {
                "type": "knn_vector",
                "dimension": EMBEDDING_DIM,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "lucene",
                    "parameters": {"ef_construction": 128, "m": 16},
                },
            },
        }
    },
}


@dataclass
class ChunkDoc:
    document_id: str
    source_url: str
    chunk_index: int
    page_num: int
    anchor: str
    text: str
    embedding: list[float]
    # document-level enrichment
    title: str
    act_type: str
    year: int
    pos: int
    status: str
    announcement_date: str


def get_client() -> OpenSearch:
    return OpenSearch(
        hosts=[{"host": _OPENSEARCH_HOST, "port": _OPENSEARCH_PORT}],
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
    )


def ensure_index(client: OpenSearch) -> None:
    """Create the index if it doesn't exist."""
    if not client.indices.exists(index=INDEX_NAME):
        client.indices.create(index=INDEX_NAME, body=_INDEX_SETTINGS)
        print(f"Created index: {INDEX_NAME}")
    else:
        print(f"Index already exists: {INDEX_NAME}")


def bulk_index(client: OpenSearch, docs: list[ChunkDoc]) -> tuple[int, int]:
    """Bulk-index a list of ChunkDoc. Returns (success_count, error_count)."""
    actions = [
        {
            "_index": INDEX_NAME,
            "_id": f"{d.document_id}_{d.chunk_index}",
            "_source": {
                "document_id":       d.document_id,
                "source_url":        d.source_url,
                "chunk_index":       d.chunk_index,
                "page_num":          d.page_num,
                "anchor":            d.anchor,
                "text":              d.text,
                "embedding":         d.embedding,
                "title":             d.title,
                "act_type":          d.act_type,
                "year":              d.year,
                "pos":               d.pos,
                "status":            d.status,
                "announcement_date": d.announcement_date,
            },
        }
        for d in docs
    ]
    success, errors = helpers.bulk(client, actions, raise_on_error=False)
    return success, len(errors) if isinstance(errors, list) else 0
