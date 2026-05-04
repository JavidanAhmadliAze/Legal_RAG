#!/usr/bin/env python3
"""
Embed and index all chunks into OpenSearch (BM25 + k-NN) and PostgreSQL.

Usage:
    python -m src.services.indexing.cli                    # all documents
    python -m src.services.indexing.cli WDU20220001105     # single document
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.indexing import index_documents
from src.services.storage import list_chunked_document_ids


def main() -> None:
    if len(sys.argv) > 1:
        doc_ids = sys.argv[1:]
    else:
        doc_ids = list_chunked_document_ids()

    if not doc_ids:
        print("No chunk files found in data/chunks/")
        sys.exit(1)

    summary = index_documents(doc_ids)
    print(
        "\nDone. "
        f"Indexed {summary['total_indexed']} chunks across "
        f"{len(summary['document_ids'])} documents."
    )


if __name__ == "__main__":
    main()
