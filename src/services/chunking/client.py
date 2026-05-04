from __future__ import annotations

from .structure import chunk_by_structure
from src.services.law_domain import classify_law_domain
from src.services.storage import load_document_meta, load_parsed_text, persist_chunks


class ChunkingClient:
    def chunk_documents(self, document_ids: list[str]) -> dict:
        processed: list[str] = []
        total_chunks = 0
        errors = 0

        for document_id in document_ids:
            try:
                meta = load_document_meta(document_id)
                text = load_parsed_text(document_id)
            except FileNotFoundError:
                print(f"  [SKIP] Missing files for {document_id}")
                errors += 1
                continue

            law_domain = classify_law_domain(
                meta.get("title", ""),
                meta.get("act_type", ""),
            )
            chunks = chunk_by_structure(
                text,
                document_id=document_id,
                source_url=meta["source_url"],
                law_domain=law_domain,
                title=meta.get("title", ""),
            )
            if not chunks:
                print(f"  [SKIP] No chunks produced for {document_id}")
                continue

            persist_chunks(document_id, chunks)
            processed.append(document_id)
            total_chunks += len(chunks)
            print(f"  [OK] {document_id}: {len(chunks)} chunks saved.")

        summary = {
            "document_ids": processed,
            "total_chunks": total_chunks,
            "errors": errors,
        }
        print(f"\nChunking summary: {summary}")
        return summary
