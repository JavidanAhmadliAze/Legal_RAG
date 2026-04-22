#!/usr/bin/env python3
"""
Chunk all parsed documents in data/parsed/ using the page-based chunker.
Output is written to data/chunks/<document_id>.jsonl — one JSON line per chunk.

Usage:
    python scripts/chunk_docs.py
    python scripts/chunk_docs.py WDU20220001105   # single document
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunk.by_page import chunk_by_page
from src.io.storage import _PROJECT_ROOT

_PARSED_DIR = _PROJECT_ROOT / "data" / "parsed"
_CHUNKS_DIR = _PROJECT_ROOT / "data" / "chunks"
_RAW_DIR = _PROJECT_ROOT / "data" / "raw"


def load_meta(document_id: str) -> dict:
    meta_path = _RAW_DIR / f"{document_id}.meta.json"
    return json.loads(meta_path.read_text(encoding="utf-8"))


def chunk_document(document_id: str) -> int:
    text_path = _PARSED_DIR / f"{document_id}.txt"
    if not text_path.exists():
        print(f"  SKIP  {document_id} — no parsed text found")
        return 0

    meta = load_meta(document_id)
    text = text_path.read_text(encoding="utf-8")

    chunks = chunk_by_page(
        text,
        document_id=document_id,
        source_url=meta["source_url"],
    )

    _CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _CHUNKS_DIR / f"{document_id}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    print(f"  OK    {document_id}  →  {len(chunks)} chunks  ({out_path.name})")
    return len(chunks)


def main() -> None:
    if len(sys.argv) > 1:
        doc_ids = sys.argv[1:]
    else:
        doc_ids = [p.stem for p in sorted(_PARSED_DIR.glob("*.txt"))]

    if not doc_ids:
        print("No documents found in data/parsed/")
        sys.exit(1)

    total = 0
    for doc_id in doc_ids:
        total += chunk_document(doc_id)
    print(f"\nDone. Total chunks: {total}")


if __name__ == "__main__":
    main()
