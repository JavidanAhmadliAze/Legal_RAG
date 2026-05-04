from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterable

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or not value.strip() else value


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or not value.strip() else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or not value.strip() else float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default

    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {value!r}")


def _env_csv(name: str, default: Iterable[str]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return tuple(default)
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _env_int_csv(name: str, default: Iterable[int]) -> tuple[int, ...]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return tuple(default)
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _resolve_path(value: str | None, default: Path, *, base_dir: Path) -> Path:
    if value is None or not value.strip():
        return default
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


@dataclass(frozen=True)
class StorageSettings:
    project_root: Path
    data_dir: Path
    raw_dir: Path
    parsed_dir: Path
    chunks_dir: Path


@dataclass(frozen=True)
class CacheSettings:
    redis_url: str
    ttl_s: int
    key_prefix: str


@dataclass(frozen=True)
class ChunkingSettings:
    max_chars: int
    overlap_chars: int


@dataclass(frozen=True)
class EmbeddingSettings:
    model_name: str
    embedding_dim: int
    batch_size: int
    normalize_embeddings: bool
    show_progress_bar: bool


@dataclass(frozen=True)
class FetcherSettings:
    default_years: tuple[int, ...]
    default_journals: tuple[str, ...]
    rate_limit_s: float
    retries: int
    backoff: float
    timeout_s: float
    follow_redirects: bool
    user_agent: str


@dataclass(frozen=True)
class PostgresSettings:
    dsn: str


@dataclass(frozen=True)
class IndexingSettings:
    embed_batch: int
    rate_limit_s: float
    blocked_statuses: frozenset[str]


@dataclass(frozen=True)
class LLMSettings:
    model: str
    base_url: str
    api_key_env_var: str
    temperature: float
    request_timeout_s: float
    streaming: bool
    translator_streaming: bool


@dataclass(frozen=True)
class MonitoringSettings:
    project_name: str
    port: int
    default_tracer_name: str
    instrument_langchain: bool
    instrument_openai: bool


@dataclass(frozen=True)
class OpenSearchSettings:
    host: str
    port: int
    index_name: str
    ef_search: int
    number_of_shards: int
    number_of_replicas: int
    ef_construction: int
    m: int
    bulk_chunk_size: int
    max_chunk_bytes: int
    request_timeout_s: int
    http_compress: bool
    use_ssl: bool
    verify_certs: bool


@dataclass(frozen=True)
class PdfParserSettings:
    x_tolerance: float
    y_tolerance: float


@dataclass(frozen=True)
class RerankingSettings:
    model_name: str
    max_length: int
    batch_size: int
    show_progress_bar: bool


@lru_cache(maxsize=1)
def get_storage_settings() -> StorageSettings:
    default_project_root = Path(__file__).resolve().parent.parent.parent
    project_root = _resolve_path(
        os.getenv("LEGAL_RAG_PROJECT_ROOT"),
        default_project_root,
        base_dir=default_project_root,
    )
    data_dir = _resolve_path(
        os.getenv("LEGAL_RAG_DATA_DIR"),
        project_root / "data",
        base_dir=project_root,
    )
    raw_dir = _resolve_path(
        os.getenv("LEGAL_RAG_RAW_DIR"),
        data_dir / "raw",
        base_dir=project_root,
    )
    parsed_dir = _resolve_path(
        os.getenv("LEGAL_RAG_PARSED_DIR"),
        data_dir / "parsed",
        base_dir=project_root,
    )
    chunks_dir = _resolve_path(
        os.getenv("LEGAL_RAG_CHUNKS_DIR"),
        data_dir / "chunks",
        base_dir=project_root,
    )
    return StorageSettings(
        project_root=project_root,
        data_dir=data_dir,
        raw_dir=raw_dir,
        parsed_dir=parsed_dir,
        chunks_dir=chunks_dir,
    )


@lru_cache(maxsize=1)
def get_cache_settings() -> CacheSettings:
    return CacheSettings(
        redis_url=_env_str(
            "CACHE_REDIS_URL",
            os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        ),
        ttl_s=_env_int("CACHE_TTL_S", 86_400),
        key_prefix=_env_str("CACHE_KEY_PREFIX", "rag:retrieve:"),
    )


@lru_cache(maxsize=1)
def get_chunking_settings() -> ChunkingSettings:
    return ChunkingSettings(
        max_chars=_env_int("CHUNKING_MAX_CHARS", 1_800),
        overlap_chars=_env_int("CHUNKING_OVERLAP_CHARS", 200),
    )


@lru_cache(maxsize=1)
def get_embedding_settings() -> EmbeddingSettings:
    return EmbeddingSettings(
        model_name=_env_str(
            "EMBEDDING_MODEL_NAME",
            "intfloat/multilingual-e5-large",
        ),
        embedding_dim=_env_int("EMBEDDING_DIM", 1024),
        batch_size=_env_int("EMBEDDING_BATCH_SIZE", 32),
        normalize_embeddings=_env_bool("EMBEDDING_NORMALIZE", True),
        show_progress_bar=_env_bool("EMBEDDING_SHOW_PROGRESS", True),
    )


@lru_cache(maxsize=1)
def get_fetcher_settings() -> FetcherSettings:
    return FetcherSettings(
        default_years=_env_int_csv(
            "FETCHER_DEFAULT_YEARS",
            [datetime.now().year],
        ),
        default_journals=tuple(
            journal.upper()
            for journal in _env_csv("FETCHER_DEFAULT_JOURNALS", ["DU", "MP"])
        ),
        rate_limit_s=_env_float("FETCHER_RATE_LIMIT_S", 0.5),
        retries=_env_int("FETCHER_RETRIES", 3),
        backoff=_env_float("FETCHER_BACKOFF", 2.0),
        timeout_s=_env_float("FETCHER_TIMEOUT_S", 30.0),
        follow_redirects=_env_bool("FETCHER_FOLLOW_REDIRECTS", True),
        user_agent=_env_str(
            "FETCHER_USER_AGENT",
            "LegalRAG/0.1 (research ingestion pipeline; non-commercial)",
        ),
    )


@lru_cache(maxsize=1)
def get_postgres_settings() -> PostgresSettings:
    return PostgresSettings(
        dsn=_env_str(
            "POSTGRES_DSN",
            "postgresql://legal:legal@localhost:5433/legal_rag",
        ),
    )


@lru_cache(maxsize=1)
def get_indexing_settings() -> IndexingSettings:
    return IndexingSettings(
        embed_batch=_env_int("INDEXING_EMBED_BATCH", 16),
        rate_limit_s=_env_float("INDEXING_RATE_LIMIT_S", 0.3),
        blocked_statuses=frozenset(
            status.casefold()
            for status in _env_csv(
                "INDEXING_BLOCKED_STATUSES",
                ["uchylony", "uznany za uchylony", "wygaśnięcie aktu"],
            )
        ),
    )


@lru_cache(maxsize=1)
def get_llm_settings() -> LLMSettings:
    return LLMSettings(
        model=_env_str("LLM_MODEL", "deepseek-chat"),
        base_url=_env_str("LLM_BASE_URL", "https://api.deepseek.com"),
        api_key_env_var=_env_str("LLM_API_KEY_ENV_VAR", "DEEPSEEK_API_KEY"),
        temperature=_env_float("LLM_TEMPERATURE", 0.0),
        request_timeout_s=_env_float("LLM_REQUEST_TIMEOUT_S", 60.0),
        streaming=_env_bool("LLM_STREAMING", True),
        translator_streaming=_env_bool("LLM_TRANSLATOR_STREAMING", False),
    )


@lru_cache(maxsize=1)
def get_monitoring_settings() -> MonitoringSettings:
    return MonitoringSettings(
        project_name=_env_str("MONITORING_PROJECT_NAME", "legal-rag"),
        port=_env_int("MONITORING_PORT", _env_int("PHOENIX_PORT", 6006)),
        default_tracer_name=_env_str(
            "MONITORING_TRACER_NAME",
            "legal-rag.retriever",
        ),
        instrument_langchain=_env_bool("MONITORING_INSTRUMENT_LANGCHAIN", True),
        instrument_openai=_env_bool("MONITORING_INSTRUMENT_OPENAI", True),
    )


@lru_cache(maxsize=1)
def get_opensearch_settings() -> OpenSearchSettings:
    return OpenSearchSettings(
        host=_env_str("OPENSEARCH_HOST", "localhost"),
        port=_env_int("OPENSEARCH_PORT", 9200),
        index_name=_env_str("OPENSEARCH_INDEX_NAME", "legal_chunks"),
        ef_search=_env_int("OPENSEARCH_EF_SEARCH", 100),
        number_of_shards=_env_int("OPENSEARCH_NUMBER_OF_SHARDS", 1),
        number_of_replicas=_env_int("OPENSEARCH_NUMBER_OF_REPLICAS", 0),
        ef_construction=_env_int("OPENSEARCH_EF_CONSTRUCTION", 128),
        m=_env_int("OPENSEARCH_M", 16),
        bulk_chunk_size=_env_int("OPENSEARCH_BULK_CHUNK_SIZE", 50),
        max_chunk_bytes=_env_int(
            "OPENSEARCH_MAX_CHUNK_BYTES",
            20 * 1024 * 1024,
        ),
        request_timeout_s=_env_int("OPENSEARCH_REQUEST_TIMEOUT_S", 120),
        http_compress=_env_bool("OPENSEARCH_HTTP_COMPRESS", True),
        use_ssl=_env_bool("OPENSEARCH_USE_SSL", False),
        verify_certs=_env_bool("OPENSEARCH_VERIFY_CERTS", False),
    )


@lru_cache(maxsize=1)
def get_pdf_parser_settings() -> PdfParserSettings:
    return PdfParserSettings(
        x_tolerance=_env_float("PDF_PARSER_X_TOLERANCE", 2.0),
        y_tolerance=_env_float("PDF_PARSER_Y_TOLERANCE", 3.0),
    )


@lru_cache(maxsize=1)
def get_reranking_settings() -> RerankingSettings:
    return RerankingSettings(
        model_name=_env_str(
            "RERANKING_MODEL_NAME",
            "sdadas/polish-reranker-bge-v2",
        ),
        max_length=_env_int("RERANKING_MAX_LENGTH", 1024),
        batch_size=_env_int("RERANKING_BATCH_SIZE", 8),
        show_progress_bar=_env_bool("RERANKING_SHOW_PROGRESS", False),
    )
