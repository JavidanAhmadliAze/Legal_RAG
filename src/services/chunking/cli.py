#!/usr/bin/env python3
"""
Chunk all parsed documents in data/parsed/ using the structure-aware chunker.
Output is written to data/chunks/<document_id>.jsonl — one JSON line per chunk.

Usage:
    python -m src.services.chunking.cli
    python -m src.services.chunking.cli WDU20220001105   # single document
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.chunking import chunk_documents
from src.services.storage import list_parsed_document_ids


def main() -> None:
    if len(sys.argv) > 1:
        doc_ids = sys.argv[1:]
    else:
        doc_ids = list_parsed_document_ids()

    if not doc_ids:
        print("No documents found in data/parsed/")
        sys.exit(1)

    summary = chunk_documents(doc_ids)
    print(f"\nDone. Total chunks: {summary['total_chunks']}")


if __name__ == "__main__":
    main()
