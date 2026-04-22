#!/usr/bin/env python3
"""
Embed and index all chunks into OpenSearch (BM25 + k-NN) and PostgreSQL.

Usage:
    python scripts/index_chunks.py                    # all documents
    python scripts/index_chunks.py WDU20220001105     # single document
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.index.embeddings import embed_passages
from src.index.opensearch_client import ChunkDoc, bulk_index, ensure_index, get_client
from src.index.postgres_client import ensure_schema, get_conn, upsert_chunks, upsert_document

_PROJECT_ROOT = Path(__file__).parent.parent
_CHUNKS_DIR = _PROJECT_ROOT / "data" / "chunks"
_RAW_DIR = _PROJECT_ROOT / "data" / "raw"

_EMBED_BATCH = 64  # chunks per embedding batch


def load_chunks(document_id: str) -> list[dict]:
    path = _CHUNKS_DIR / f"{document_id}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_meta(document_id: str) -> dict:
    path = _RAW_DIR / f"{document_id}.meta.json"
    return json.loads(path.read_text(encoding="utf-8"))


def index_document(
    document_id: str,
    os_client,
    pg_conn,
) -> None:
    chunks = load_chunks(document_id)
    if not chunks:
        print(f"  SKIP  {document_id} — no chunks found")
        return

    meta = load_meta(document_id)
    print(f"  {document_id}  {len(chunks)} chunks — embedding...", flush=True)

    # Embed all chunks in batches
    texts = [c["text"] for c in chunks]
    embeddings = embed_passages(texts, batch_size=_EMBED_BATCH)

    # Build OpenSearch docs
    os_docs = [
        ChunkDoc(
            document_id=document_id,
            source_url=c["source_url"],
            chunk_index=c["index"],
            page_num=c["page_num"],
            anchor=c["anchor"],
            text=c["text"],
            embedding=embeddings[i],
            title=meta["title"],
            act_type=meta["act_type"],
            year=meta["year"],
            pos=meta["pos"],
            status=meta["status"],
            announcement_date=meta["announcement_date"],
        )
        for i, c in enumerate(chunks)
    ]

    success, errors = bulk_index(os_client, os_docs)
    print(f"    OpenSearch: {success} indexed, {errors} errors")

    # PostgreSQL
    upsert_document(pg_conn, meta)
    upsert_chunks(pg_conn, chunks)
    print(f"    PostgreSQL: document + {len(chunks)} chunks saved")


def main() -> None:
    if len(sys.argv) > 1:
        doc_ids = sys.argv[1:]
    else:
        doc_ids = [p.stem for p in sorted(_CHUNKS_DIR.glob("*.jsonl"))]

    if not doc_ids:
        print("No chunk files found in data/chunks/")
        sys.exit(1)

    print("Connecting to OpenSearch and PostgreSQL...")
    os_client = get_client()
    ensure_index(os_client)

    pg_conn = get_conn()
    ensure_schema(pg_conn)

    for doc_id in doc_ids:
        index_document(doc_id, os_client, pg_conn)

    pg_conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
