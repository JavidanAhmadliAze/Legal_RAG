#!/usr/bin/env python3
"""
Download specific acts by journal reference (e.g. DU/2025/1079, MP/2026/370).

Usage:
    python scripts/ingest_specific.py DU/2025/1079 MP/2026/370 ...
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fetch.http import fetch
from src.fetch.robots import can_fetch
from src.io.storage import already_saved, save_document
from src.parse.pdf import parse_pdf

_API_BASE = "https://api.sejm.gov.pl/eli/acts"
_RATE_LIMIT = 0.5


def ingest_ref(ref: str) -> None:
    """Fetch and save a single act given a reference like 'DU/2025/1079'."""
    parts = ref.strip().split("/")
    if len(parts) != 3:
        print(f"  SKIP  {ref!r} — expected format JOURNAL/YEAR/POS")
        return
    journal, year, pos = parts[0].upper(), parts[1], parts[2]

    address = f"W{journal}{year}{int(pos):07d}"

    if already_saved(address):
        print(f"  SKIP  {ref} ({address}) — already saved")
        return

    # Fetch metadata
    meta_url = f"{_API_BASE}/{journal}/{year}/{pos}"
    try:
        meta_resp = fetch(meta_url)
        meta = meta_resp.json()
    except Exception as exc:
        print(f"  ERROR {ref} — metadata fetch failed: {exc}")
        return

    title = meta.get("title", "")
    act_type = meta.get("type", "")
    status = meta.get("status", "")
    announcement_date = meta.get("announcementDate", "")
    has_pdf = meta.get("textPDF", False)

    if not has_pdf:
        print(f"  SKIP  {ref} — no PDF available")
        return

    pdf_url = f"{_API_BASE}/{journal}/{year}/{pos}/text.pdf"

    if not can_fetch(pdf_url):
        print(f"  SKIP  {ref} — robots.txt disallows")
        return

    print(f"  GET   {ref}  {title[:70]}...")
    try:
        resp = fetch(pdf_url)
    except Exception as exc:
        print(f"  ERROR {ref}: {exc}")
        return

    content_type = resp.headers.get("content-type", "application/pdf")
    text = parse_pdf(resp.content)

    save_document(
        document_id=address,
        source_url=pdf_url,
        content=resp.content,
        content_type=content_type,
        text=text,
        title=title,
        act_type=act_type,
        year=int(year),
        pos=int(pos),
        status=status,
        announcement_date=announcement_date,
    )
    time.sleep(_RATE_LIMIT)


def main() -> None:
    refs = sys.argv[1:]
    if not refs:
        print("Usage: python scripts/ingest_specific.py JOURNAL/YEAR/POS ...")
        sys.exit(1)
    for ref in refs:
        ingest_ref(ref)
    print("\nDone.")


if __name__ == "__main__":
    main()
