#!/usr/bin/env python3
"""
Ingest recent Prawo pracy (labor law) acts from isap.sejm.gov.pl.

Usage:
    python scripts/ingest_labor.py                   # current year
    python scripts/ingest_labor.py --year 2024 2025  # explicit years
    python scripts/ingest_labor.py --dry-run         # list matches without downloading
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow running as `python scripts/ingest_labor.py` or `python ingest_labor.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fetch.robots import can_fetch
from src.fetch.http import fetch
from src.io.storage import already_saved, save_document
from src.parse.pdf import parse_pdf
from src.sources.isap import filter_labor, list_acts

_RATE_LIMIT_SECONDS = 0.5


def ingest_year(year: int, *, dry_run: bool) -> tuple[int, int]:
    """Ingest labor acts for *year*. Returns (downloaded, skipped) counts."""
    print(f"[{year}] Fetching act list from ISAP...")
    acts = list_acts(year)
    labor_acts = filter_labor(acts)
    pdf_acts = [a for a in labor_acts if a.has_pdf]

    print(f"[{year}] Found {len(acts)} total acts → {len(labor_acts)} labor-related → {len(pdf_acts)} with PDF")

    if dry_run:
        for act in pdf_acts:
            print(f"  {act.address}  {act.act_type:20s}  {act.title[:80]}")
        return 0, 0

    downloaded = skipped = 0
    for act in pdf_acts:
        if already_saved(act.address):
            print(f"  SKIP  {act.address} (already saved)")
            skipped += 1
            continue

        if not can_fetch(act.pdf_url):
            print(f"  SKIP  {act.address} (robots.txt disallows)")
            skipped += 1
            continue

        print(f"  GET   {act.address}  {act.title[:60]}...")
        try:
            resp = fetch(act.pdf_url)
        except Exception as exc:
            print(f"  ERROR {act.address}: {exc}")
            skipped += 1
            continue

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
        downloaded += 1
        time.sleep(_RATE_LIMIT_SECONDS)

    return downloaded, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Polish labor law acts from ISAP")
    parser.add_argument(
        "--year",
        type=int,
        nargs="+",
        default=[datetime.now().year],
        metavar="YEAR",
        help="Year(s) to ingest (default: current year)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching acts without downloading",
    )
    args = parser.parse_args()

    total_downloaded = total_skipped = 0
    for year in sorted(args.year):
        dl, sk = ingest_year(year, dry_run=args.dry_run)
        total_downloaded += dl
        total_skipped += sk

    if not args.dry_run:
        print(f"\nDone. Downloaded: {total_downloaded}  Skipped: {total_skipped}")


if __name__ == "__main__":
    main()
