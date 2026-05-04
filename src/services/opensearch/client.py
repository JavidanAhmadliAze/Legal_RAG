from __future__ import annotations

from dataclasses import dataclass

from opensearchpy import AsyncOpenSearch, OpenSearch, helpers

from src.services.config import (
    OpenSearchSettings,
    get_embedding_settings,
    get_opensearch_settings,
)

INDEX_NAME = get_opensearch_settings().index_name


@dataclass
class ChunkDoc:
    document_id: str
    source_url: str
    domain: str
    law_domain: str
    chunk_index: int
    page_num: int
    anchor: str
    subheading: str       # Rozdział / Dział heading for this chunk
    text: str
    embedding: list[float]
    title: str
    act_type: str
    year: int
    pos: int
    status: str
    announcement_date: str


def _build_index_settings() -> dict:
    opensearch_settings = get_opensearch_settings()
    embedding_settings = get_embedding_settings()
    return {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": opensearch_settings.ef_search,
                "number_of_shards": opensearch_settings.number_of_shards,
                "number_of_replicas": opensearch_settings.number_of_replicas,
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
                "document_id": {"type": "keyword"},
                "source_url": {"type": "keyword"},
                "domain": {"type": "keyword"},
                "law_domain": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "page_num": {"type": "integer"},
                "anchor": {"type": "keyword"},
                "subheading": {"type": "text", "analyzer": "polish_standard"},
                "title": {"type": "text", "analyzer": "polish_standard"},
                "act_type": {"type": "keyword"},
                "year": {"type": "integer"},
                "pos": {"type": "integer"},
                "status": {"type": "keyword"},
                "announcement_date": {"type": "keyword"},
                "text": {"type": "text", "analyzer": "polish_standard"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": embedding_settings.embedding_dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                        "parameters": {
                            "ef_construction": opensearch_settings.ef_construction,
                            "m": opensearch_settings.m,
                        },
                    },
                },
            }
        },
    }


class OpenSearchClient:
    def __init__(self, *, settings: OpenSearchSettings | None = None) -> None:
        self._settings = settings or get_opensearch_settings()
        self._client: OpenSearch | None = None
        self._async_client: AsyncOpenSearch | None = None

    def get_client(self) -> OpenSearch:
        if self._client is None:
            self._client = OpenSearch(
                hosts=[{"host": self._settings.host, "port": self._settings.port}],
                http_compress=self._settings.http_compress,
                use_ssl=self._settings.use_ssl,
                verify_certs=self._settings.verify_certs,
            )
        return self._client

    def get_async_client(self) -> AsyncOpenSearch:
        if self._async_client is None:
            self._async_client = AsyncOpenSearch(
                hosts=[{"host": self._settings.host, "port": self._settings.port}],
                http_compress=self._settings.http_compress,
                use_ssl=self._settings.use_ssl,
                verify_certs=self._settings.verify_certs,
            )
        return self._async_client

    def ensure_index(self) -> None:
        client = self.get_client()
        if not client.indices.exists(index=self._settings.index_name):
            client.indices.create(
                index=self._settings.index_name,
                body=_build_index_settings(),
            )
            print(f"Created index: {self._settings.index_name}")
        else:
            print(f"Index already exists: {self._settings.index_name}")

    def bulk_index(self, docs: list[ChunkDoc]) -> tuple[int, int]:
        client = self.get_client()
        actions = [
            {
                "_index": self._settings.index_name,
                "_id": f"{doc.document_id}_{doc.chunk_index}",
                "_source": {
                    "document_id": doc.document_id,
                    "source_url": doc.source_url,
                    "domain": doc.domain,
                    "law_domain": doc.law_domain,
                    "chunk_index": doc.chunk_index,
                    "page_num": doc.page_num,
                    "anchor": doc.anchor,
                    "subheading": doc.subheading,
                    "text": doc.text,
                    "embedding": doc.embedding,
                    "title": doc.title,
                    "act_type": doc.act_type,
                    "year": doc.year,
                    "pos": doc.pos,
                    "status": doc.status,
                    "announcement_date": doc.announcement_date,
                },
            }
            for doc in docs
        ]
        success, errors = helpers.bulk(
            client,
            actions,
            chunk_size=self._settings.bulk_chunk_size,
            max_chunk_bytes=self._settings.max_chunk_bytes,
            request_timeout=self._settings.request_timeout_s,
            raise_on_error=False,
        )
        return success, len(errors) if isinstance(errors, list) else 0

    def delete_document_chunks(self, document_id: str) -> None:
        self.get_client().delete_by_query(
            index=self._settings.index_name,
            body={"query": {"term": {"document_id": document_id}}},
            conflicts="proceed",
            refresh=True,
        )
