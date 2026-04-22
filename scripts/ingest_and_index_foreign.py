#!/usr/bin/env python3
"""
Full pipeline for foreign/immigration law acts from isap.sejm.gov.pl:
  1. Search DU journal for each year by foreign-law keywords
  2. Download PDFs (skip already saved)
  3. Parse PDF → clean text
  4. Chunk by page
  5. Index into OpenSearch (BM25 + k-NN) and PostgreSQL

Usage:
    python scripts/ingest_and_index_foreign.py                  # DU + MP, 2022-2026
    python scripts/ingest_and_index_foreign.py --years 2024 2025
    python scripts/ingest_and_index_foreign.py --journals DU MP
    python scripts/ingest_and_index_foreign.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunk.by_page import chunk_by_page
from src.fetch.http import fetch
from src.fetch.robots import can_fetch
from src.index.embeddings import embed_passages
from src.index.opensearch_client import ChunkDoc, bulk_index, ensure_index, get_client
from src.index.postgres_client import ensure_schema, get_conn, upsert_chunks, upsert_document
from src.io.storage import already_saved, save_document
from src.parse.pdf import parse_pdf
from src.sources.isap import VALID_STATUS, filter_foreign, filter_valid, list_acts

_PROJECT_ROOT = Path(__file__).parent.parent
_CHUNKS_DIR = _PROJECT_ROOT / "data" / "chunks"
_RAW_DIR = _PROJECT_ROOT / "data" / "raw"
_RATE_LIMIT = 0.5
_EMBED_BATCH = 64
_DEFAULT_YEARS = [2022, 2023, 2024, 2025, 2026]


# ---------------------------------------------------------------------------
# Stage 1 – ingest
# ---------------------------------------------------------------------------

def ingest_act(act) -> bool:
    """Download, parse and save one act. Returns True if newly saved."""
    if act.status != VALID_STATUS:
        raise ValueError(
            f"Refusing to ingest expired/invalid act {act.address} "
            f"(status={act.status!r}). Only '{VALID_STATUS}' is allowed."
        )
    if already_saved(act.address):
        return False
    if not can_fetch(act.pdf_url):
        print(f"    robots.txt blocks {act.address}")
        return False
    try:
        resp = fetch(act.pdf_url)
    except Exception as exc:
        print(f"    ERROR fetching {act.address}: {exc}")
        return False

    content_type = resp.headers.get("content-type", "application/pdf")
    text = parse_pdf(resp.content)
    save_document(
        document_id=act.address,
        source_url=act.pdf_url,
        content=resp.content,
        content_type=content_type,
        text=text,
        title=act.title,
        act_type=act.act_type,
        year=act.year,
        pos=act.pos,
        status=act.status,
        announcement_date=act.announcement_date,
    )
    time.sleep(_RATE_LIMIT)
    return True


# ---------------------------------------------------------------------------
# Stage 2 – chunk
# ---------------------------------------------------------------------------

def chunk_act(act) -> list[dict]:
    """Chunk a saved document. Returns chunk dicts (empty if already chunked)."""
    out_path = _CHUNKS_DIR / f"{act.address}.jsonl"
    if out_path.exists():
        return []   # already chunked

    text_path = _PROJECT_ROOT / "data" / "parsed" / f"{act.address}.txt"
    if not text_path.exists():
        return []

    meta_path = _RAW_DIR / f"{act.address}.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    chunks = chunk_by_page(
        text_path.read_text(encoding="utf-8"),
        document_id=act.address,
        source_url=meta["source_url"],
    )
    _CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    return chunks


# ---------------------------------------------------------------------------
# Stage 3 – index
# ---------------------------------------------------------------------------

def index_act(act, os_client, pg_conn) -> int:
    """Embed and index one act. Returns number of chunks indexed."""
    chunk_path = _CHUNKS_DIR / f"{act.address}.jsonl"
    if not chunk_path.exists():
        return 0

    chunks = [json.loads(l) for l in chunk_path.read_text(encoding="utf-8").splitlines()]
    if not chunks:
        return 0

    meta_path = _RAW_DIR / f"{act.address}.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    embeddings = embed_passages([c["text"] for c in chunks], batch_size=_EMBED_BATCH)

    os_docs = [
        ChunkDoc(
            document_id=act.address,
            source_url=chunks[i]["source_url"],
            chunk_index=chunks[i]["index"],
            page_num=chunks[i]["page_num"],
            anchor=chunks[i]["anchor"],
            text=chunks[i]["text"],
            embedding=embeddings[i],
            title=meta["title"],
            act_type=meta["act_type"],
            year=meta["year"],
            pos=meta["pos"],
            status=meta["status"],
            announcement_date=meta["announcement_date"],
        )
        for i in range(len(chunks))
    ]

    success, errors = bulk_index(os_client, os_docs)
    upsert_document(pg_conn, meta)
    upsert_chunks(pg_conn, chunks)
    return success


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", default=_DEFAULT_YEARS)
    parser.add_argument(
        "--journals",
        nargs="+",
        default=["DU", "MP"],
        help="ELI journal codes to scan (default: DU MP)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Collect matching acts across all years
    all_acts: list = []
    for journal in [j.upper() for j in args.journals]:
        for year in sorted(args.years):
            print(f"[{journal} {year}] Fetching act list...", end=" ", flush=True)
            acts = list_acts(year, journal=journal)
            valid = filter_valid(acts)
            foreign = [a for a in filter_foreign(valid) if a.has_pdf]
            skipped_expired = len(acts) - len(valid)
            print(
                f"{len(acts)} total → {skipped_expired} expired/invalid skipped "
                f"→ {len(foreign)} valid foreign-law acts with PDF"
            )
            all_acts.extend(foreign)

    # Deduplicate by address
    seen: set[str] = set()
    unique_acts = []
    for a in all_acts:
        if a.address not in seen:
            seen.add(a.address)
            unique_acts.append(a)

    print(f"\nTotal unique acts to process: {len(unique_acts)}\n")

    if args.dry_run:
        for a in unique_acts:
            status = "saved" if already_saved(a.address) else "new"
            print(f"  [{status}]  {a.address}  {a.act_type:20s}  {a.title[:70]}")
        return

    # Connect to stores
    os_client = get_client()
    ensure_index(os_client)
    pg_conn = get_conn()
    ensure_schema(pg_conn)

    ingested = chunked = indexed = skipped = 0

    for act in unique_acts:
        was_new = ingest_act(act)
        if was_new:
            ingested += 1
            print(f"  SAVED  {act.address}  {act.title[:65]}...")
        else:
            skipped += 1

        new_chunks = chunk_act(act)
        if new_chunks:
            chunked += len(new_chunks)

        n = index_act(act, os_client, pg_conn)
        if n:
            indexed += n
            print(f"    → {n} chunks indexed")

    pg_conn.close()
    print(f"\nDone.  Ingested: {ingested}  Skipped: {skipped}  "
          f"Chunks created: {chunked}  Chunks indexed: {indexed}")


if __name__ == "__main__":
    main()
