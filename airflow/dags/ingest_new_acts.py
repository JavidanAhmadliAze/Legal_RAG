"""
Airflow DAG: ingest_new_acts
Schedule: on-demand (trigger manually or via REST API)

Orchestrates the ingestion pipeline by delegating each step to the service
layer under src/services/.

Trigger with optional conf overrides:
  {"years": [2025, 2026], "journals": ["DU", "MP"]}
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

# Make src/ importable from the mounted project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.services.chunking import chunk_documents
from src.services.fetcher import (
    DEFAULT_JOURNALS,
    DEFAULT_YEARS,
    download_and_parse_acts,
    list_new_acts,
)
from src.services.indexing import index_documents

default_args = {
    "owner": "legal-rag",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


@dag(
    dag_id="ingest_new_acts",
    description="Ingest new Polish legal acts from ISAP into OpenSearch + PostgreSQL",
    schedule=None,   # manual trigger only
    start_date=days_ago(1),
    default_args=default_args,
    catchup=False,
    tags=["ingestion", "isap"],
    params={
        "years": DEFAULT_YEARS,
        "journals": DEFAULT_JOURNALS,
    },
)
def ingest_new_acts():

    @task
    def list_new_acts_task(**context) -> list[dict]:
        params = context["params"]
        return list_new_acts(
            years=params.get("years", DEFAULT_YEARS),
            journals=params.get("journals", DEFAULT_JOURNALS),
        )

    @task
    def download_and_parse_task(new_acts: list[dict]) -> list[str]:
        return download_and_parse_acts(new_acts)

    @task
    def chunk_documents_task(saved_addresses: list[str]) -> dict:
        return chunk_documents(saved_addresses)

    @task
    def index_documents_task(chunk_summary: dict) -> dict:
        return index_documents(chunk_summary["document_ids"])

    new_acts = list_new_acts_task()
    saved_addresses = download_and_parse_task(new_acts)
    chunk_summary = chunk_documents_task(saved_addresses)
    index_documents_task(chunk_summary)


ingest_new_acts()
