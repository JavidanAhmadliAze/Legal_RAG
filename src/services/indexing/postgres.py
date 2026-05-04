"""
PostgreSQL client — schema setup and storage for documents and chunks.
"""

from __future__ import annotations

import psycopg
from psycopg.rows import dict_row

from src.services.config import get_postgres_settings

_DDL = """
CREATE TABLE IF NOT EXISTS documents (
    document_id      TEXT PRIMARY KEY,
    source_url       TEXT NOT NULL,
    fetched_at       TIMESTAMPTZ,
    content_type     TEXT,
    title            TEXT,
    act_type         TEXT,
    year             INTEGER,
    pos              INTEGER,
    status           TEXT,
    announcement_date TEXT,
    raw_path         TEXT,
    parsed_path      TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    id               SERIAL PRIMARY KEY,
    document_id      TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    chunk_index      INTEGER NOT NULL,
    page_num         INTEGER,
    anchor           TEXT,
    text             TEXT NOT NULL,
    source_url       TEXT,
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_documents_year     ON documents(year);
CREATE INDEX IF NOT EXISTS idx_documents_act_type ON documents(act_type);
"""


def get_conn() -> psycopg.Connection:
    return psycopg.connect(get_postgres_settings().dsn, row_factory=dict_row)


def ensure_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(_DDL)
    conn.commit()


def upsert_document(conn: psycopg.Connection, meta: dict) -> None:
    sql = """
        INSERT INTO documents (
            document_id, source_url, fetched_at, content_type,
            title, act_type, year, pos, status, announcement_date,
            raw_path, parsed_path
        ) VALUES (
            %(document_id)s, %(source_url)s, %(fetched_at)s, %(content_type)s,
            %(title)s, %(act_type)s, %(year)s, %(pos)s, %(status)s,
            %(announcement_date)s, %(raw_path)s, %(parsed_path)s
        )
        ON CONFLICT (document_id) DO UPDATE SET
            title             = EXCLUDED.title,
            status            = EXCLUDED.status,
            announcement_date = EXCLUDED.announcement_date
    """
    with conn.cursor() as cur:
        cur.execute(sql, meta)
    conn.commit()


def upsert_chunks(
    conn: psycopg.Connection,
    chunks: list[dict],
) -> None:
    sql = """
        INSERT INTO chunks (document_id, chunk_index, page_num, anchor, text, source_url)
        VALUES (%(document_id)s, %(index)s, %(page_num)s, %(anchor)s, %(text)s, %(source_url)s)
        ON CONFLICT (document_id, chunk_index) DO UPDATE SET
            text   = EXCLUDED.text,
            anchor = EXCLUDED.anchor
    """
    with conn.cursor() as cur:
        cur.executemany(sql, chunks)
    conn.commit()
