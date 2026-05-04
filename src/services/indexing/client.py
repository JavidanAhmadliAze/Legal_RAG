from __future__ import annotations

import time
from urllib.parse import urlparse

from src.services.config import get_indexing_settings
from src.services.law_domain import classify_law_domain
from src.services.embedding import embed_passages
from src.services.indexing.postgres import (
    ensure_schema,
    get_conn,
    upsert_chunks,
    upsert_document,
)
from src.services.opensearch import ChunkDoc
from src.services.opensearch.factory import get_opensearch_client
from src.services.storage import load_chunk_records, load_document_meta
from src.sources.isap import VALID_STATUS, list_acts

_SETTINGS = get_indexing_settings()
DEFAULT_EMBED_BATCH = _SETTINGS.embed_batch
DEFAULT_RATE_LIMIT_S = _SETTINGS.rate_limit_s
BLOCKED_STATUSES = _SETTINGS.blocked_statuses


class IndexingClient:
    def __init__(self) -> None:
        self._opensearch = get_opensearch_client()

    def index_documents(
        self,
        document_ids: list[str],
        *,
        embed_batch: int | None = None,
    ) -> dict:
        selected_embed_batch = (
            get_indexing_settings().embed_batch
            if embed_batch is None
            else embed_batch
        )
        if not document_ids:
            summary = {
                "document_ids": [],
                "total_chunks": 0,
                "total_indexed": 0,
                "errors": 0,
            }
            print(f"Nothing to index. Summary: {summary}")
            return summary

        self._opensearch.ensure_index()

        pg_conn = get_conn()
        ensure_schema(pg_conn)

        processed: list[str] = []
        total_chunks = 0
        total_indexed = 0
        errors = 0

        try:
            for document_id in document_ids:
                chunks = load_chunk_records(document_id)
                if not chunks:
                    print(f"  [SKIP] No chunks found for {document_id}")
                    continue

                try:
                    meta = load_document_meta(document_id)
                except FileNotFoundError:
                    print(f"  [SKIP] Missing metadata for {document_id}")
                    errors += 1
                    continue

                law_domain = classify_law_domain(
                    meta.get("title", ""),
                    meta.get("act_type", ""),
                )
                print(
                    f"  {document_id}  {len(chunks)} chunks  [{law_domain}] - embedding...",
                    flush=True,
                )
                embeddings = embed_passages(
                    [chunk["text"] for chunk in chunks],
                    batch_size=selected_embed_batch,
                )

                docs = [
                    ChunkDoc(
                        document_id=document_id,
                        source_url=chunk["source_url"],
                        domain=(
                            urlparse(chunk["source_url"]).netloc
                            or chunk.get("domain", "")
                            or chunk["source_url"]
                        ),
                        law_domain=chunk.get("law_domain") or law_domain,
                        chunk_index=chunk["index"],
                        page_num=chunk["page_num"],
                        anchor=chunk.get("anchor", ""),
                        subheading=chunk.get("subheading", ""),
                        text=chunk["text"],
                        embedding=embeddings[index],
                        title=meta["title"],
                        act_type=meta["act_type"],
                        year=meta["year"],
                        pos=meta["pos"],
                        status=meta["status"],
                        announcement_date=meta["announcement_date"],
                    )
                    for index, chunk in enumerate(chunks)
                ]

                success, index_errors = self._opensearch.bulk_index(docs)
                total_chunks += len(chunks)
                total_indexed += success
                errors += index_errors

                upsert_document(pg_conn, meta)
                upsert_chunks(pg_conn, chunks)

                processed.append(document_id)
                print(f"  [OK] {document_id}: {len(chunks)} chunks, {success} indexed.")
        finally:
            pg_conn.close()

        summary = {
            "document_ids": processed,
            "total_chunks": total_chunks,
            "total_indexed": total_indexed,
            "errors": errors,
        }
        print(f"\nIndexing summary: {summary}")
        return summary

    def get_indexed_document_ids(self) -> list[str]:
        pg_conn = get_conn()
        try:
            with pg_conn.cursor() as cur:
                cur.execute("SELECT document_id FROM documents ORDER BY document_id")
                return [row["document_id"] for row in cur.fetchall()]
        finally:
            pg_conn.close()

    def find_expired_documents(
        self,
        document_ids: list[str],
        *,
        rate_limit_s: float | None = None,
    ) -> list[str]:
        selected_rate_limit = (
            get_indexing_settings().rate_limit_s
            if rate_limit_s is None
            else rate_limit_s
        )
        groups: dict[tuple[str, int], list[str]] = {}
        for document_id in document_ids:
            try:
                journal, year, _ = self._parse_document_id(document_id)
            except (ValueError, IndexError):
                print(f"  [WARN] Cannot parse doc_id: {document_id!r}, skipping.")
                continue
            groups.setdefault((journal, year), []).append(document_id)

        expired: list[str] = []

        for (journal, year), group_ids in sorted(groups.items()):
            print(f"[{journal} {year}] Checking {len(group_ids)} documents ...")
            try:
                acts = list_acts(year, journal=journal)
                status_map = {act.address: act.status for act in acts}
            except Exception as exc:
                print(f"  [ERROR] fetching ISAP {journal} {year}: {exc}")
                time.sleep(selected_rate_limit)
                continue

            time.sleep(selected_rate_limit)

            for document_id in group_ids:
                if document_id not in status_map:
                    print(f"  {document_id}: not found in ISAP -> expired")
                    expired.append(document_id)
                    continue

                status = status_map[document_id]
                if journal == "MP":
                    if status.lower() in BLOCKED_STATUSES:
                        print(f"  {document_id}: status={status!r} -> expired")
                        expired.append(document_id)
                elif status != VALID_STATUS:
                    print(f"  {document_id}: status={status!r} -> expired")
                    expired.append(document_id)

        print(f"\nResult: {len(expired)} expired out of {len(document_ids)} indexed.")
        return expired

    def purge_expired_documents(self, expired_ids: list[str]) -> dict:
        if not expired_ids:
            print("Nothing to purge - all indexed documents are still in force.")
            return {"checked": 0, "purged": 0, "errors": 0}

        pg_conn = get_conn()
        purged = 0
        errors = 0

        try:
            for document_id in expired_ids:
                try:
                    self._opensearch.delete_document_chunks(document_id)
                    print(f"  [OS]  Deleted chunks for {document_id}")
                except Exception as exc:
                    print(f"  [OS]  ERROR deleting {document_id}: {exc}")
                    errors += 1
                    continue

                try:
                    with pg_conn.cursor() as cur:
                        cur.execute(
                            "DELETE FROM documents WHERE document_id = %s",
                            (document_id,),
                        )
                    pg_conn.commit()
                    print(f"  [PG]  Deleted {document_id}")
                    purged += 1
                except Exception as exc:
                    print(f"  [PG]  ERROR deleting {document_id}: {exc}")
                    pg_conn.rollback()
                    errors += 1
        finally:
            pg_conn.close()

        summary = {
            "checked": len(expired_ids),
            "purged": purged,
            "errors": errors,
        }
        print(f"\nPurge complete: {summary}")
        return summary

    def _parse_document_id(self, document_id: str) -> tuple[str, int, int]:
        journal = document_id[1:3]
        year = int(document_id[3:7])
        pos = int(document_id[7:])
        return journal, year, pos
