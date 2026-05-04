from __future__ import annotations

import time

from src.services.config import get_fetcher_settings
from .http import fetch as http_fetch
from .robots import can_fetch as robots_can_fetch
from src.services.indexing.postgres import ensure_schema, get_conn
from src.services.pdf_parser.factory import get_pdf_parser_client
from src.services.storage import document_already_saved, persist_document
from src.sources.isap import filter_valid, list_acts


class FetcherClient:
    def __init__(self) -> None:
        self._pdf_parser = get_pdf_parser_client()

    def fetch(
        self,
        url: str,
        *,
        retries: int | None = None,
        backoff: float | None = None,
    ):
        return http_fetch(url, retries=retries, backoff=backoff)

    def can_fetch(self, url: str) -> bool:
        return robots_can_fetch(url)

    def list_new_acts(
        self,
        *,
        years: list[int] | None = None,
        journals: list[str] | None = None,
    ) -> list[dict]:
        settings = get_fetcher_settings()
        selected_years = list(years or settings.default_years)
        selected_journals = [
            journal.upper()
            for journal in (journals or settings.default_journals)
        ]

        pg_conn = get_conn()
        ensure_schema(pg_conn)
        try:
            with pg_conn.cursor() as cur:
                cur.execute("SELECT document_id FROM documents")
                existing: set[str] = {row["document_id"] for row in cur.fetchall()}
        finally:
            pg_conn.close()

        new_acts: list[dict] = []
        for journal in selected_journals:
            for year in sorted(selected_years):
                print(f"[{journal} {year}] Fetching act list from ISAP ...")
                try:
                    acts = list_acts(year, journal=journal)
                except Exception as exc:
                    print(f"  ERROR listing {journal} {year}: {exc}")
                    continue

                for act in filter_valid(acts):
                    if act.has_pdf and act.address not in existing:
                        new_acts.append(
                            {
                                "address": act.address,
                                "journal": act.journal,
                                "title": act.title,
                                "act_type": act.act_type,
                                "year": act.year,
                                "pos": act.pos,
                                "status": act.status,
                                "announcement_date": act.announcement_date,
                                "pdf_url": act.pdf_url,
                            }
                        )

        print(f"\nTotal new acts to ingest: {len(new_acts)}")
        return new_acts

    def download_and_parse_acts(
        self,
        acts: list[dict],
        *,
        rate_limit_s: float | None = None,
    ) -> list[str]:
        selected_rate_limit = (
            get_fetcher_settings().rate_limit_s
            if rate_limit_s is None
            else rate_limit_s
        )
        saved: list[str] = []

        for act in acts:
            address = act["address"]

            if document_already_saved(address):
                saved.append(address)
                continue

            if not self.can_fetch(act["pdf_url"]):
                print(f"  [SKIP] robots.txt blocks {address}")
                continue

            try:
                response = self.fetch(act["pdf_url"])
            except Exception as exc:
                print(f"  [ERROR] fetching {address}: {exc}")
                continue

            content_type = response.headers.get("content-type", "application/pdf")
            try:
                text = self._pdf_parser.parse_pdf(response.content)
            except Exception as exc:
                print(f"  [ERROR] parsing {address}: {exc}")
                continue

            persist_document(
                document_id=address,
                source_url=act["pdf_url"],
                content=response.content,
                content_type=content_type,
                text=text,
                title=act["title"],
                act_type=act["act_type"],
                year=act["year"],
                pos=act["pos"],
                status=act["status"],
                announcement_date=act["announcement_date"],
            )
            saved.append(address)
            print(f"  [SAVED] {address}: {act['title'][:65]}")
            time.sleep(selected_rate_limit)

        print(f"\nDownloaded+parsed {len(saved)}/{len(acts)} acts.")
        return saved
