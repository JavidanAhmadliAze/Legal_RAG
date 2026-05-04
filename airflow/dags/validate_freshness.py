"""
Airflow DAG: validate_freshness
Schedule: weekly (Sunday midnight)

Checks every indexed document against ISAP by delegating the freshness logic
to the service layer under src/services/.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.services.indexing import (
    find_expired_documents,
    get_indexed_document_ids,
    purge_expired_documents,
)

default_args = {
    "owner": "legal-rag",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}


@dag(
    dag_id="validate_freshness",
    description="Remove expired/repealed Polish legal acts from the index weekly",
    schedule="0 0 * * 0",   # every Sunday at midnight
    start_date=days_ago(1),
    default_args=default_args,
    catchup=False,
    tags=["validation", "freshness"],
)
def validate_freshness():

    @task
    def get_indexed_doc_ids() -> list[str]:
        return get_indexed_document_ids()

    @task
    def check_isap_status(doc_ids: list[str]) -> list[str]:
        return find_expired_documents(doc_ids)

    @task
    def purge_expired(expired_ids: list[str]) -> dict:
        return purge_expired_documents(expired_ids)

    doc_ids = get_indexed_doc_ids()
    expired = check_isap_status(doc_ids)
    purge_expired(expired)


validate_freshness()
