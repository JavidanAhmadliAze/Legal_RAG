import src.services.chunking.client as chunking_client
import src.services.fetcher.client as fetcher_client
import src.services.indexing.client as indexing_client
from src.services.chunking.structure import Chunk
from src.sources.isap import ActMeta, VALID_STATUS


def _make_act(
    address: str,
    *,
    journal: str = "DU",
    year: int = 2025,
    pos: int = 1,
    status: str = VALID_STATUS,
    has_pdf: bool = True,
) -> ActMeta:
    return ActMeta(
        address=address,
        journal=journal,
        title=f"Act {address}",
        act_type="Ustawa",
        year=year,
        pos=pos,
        status=status,
        announcement_date="2025-01-01",
        has_pdf=has_pdf,
    )


def test_list_new_acts_filters_existing_and_requires_pdf(monkeypatch):
    class FakeCursor:
        def execute(self, _sql: str) -> None:
            return None

        def fetchall(self) -> list[dict]:
            return [{"document_id": "WDU20250000001"}]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    class FakeConn:
        def cursor(self) -> FakeCursor:
            return FakeCursor()

        def close(self) -> None:
            return None

    client = fetcher_client.FetcherClient()

    monkeypatch.setattr(fetcher_client, "get_conn", lambda: FakeConn())
    monkeypatch.setattr(fetcher_client, "ensure_schema", lambda _conn: None)
    monkeypatch.setattr(
        fetcher_client,
        "list_acts",
        lambda year, journal="DU": [
            _make_act("WDU20250000001", year=year, journal=journal, pos=1),
            _make_act(
                "WDU20250000002",
                year=year,
                journal=journal,
                pos=2,
                has_pdf=False,
            ),
            _make_act("WDU20250000003", year=year, journal=journal, pos=3),
        ],
    )

    result = client.list_new_acts(years=[2025], journals=["DU"])

    assert result == [
        {
            "address": "WDU20250000003",
            "journal": "DU",
            "title": "Act WDU20250000003",
            "act_type": "Ustawa",
            "year": 2025,
            "pos": 3,
            "status": VALID_STATUS,
            "announcement_date": "2025-01-01",
            "pdf_url": "https://api.sejm.gov.pl/eli/acts/DU/2025/3/text.pdf",
        }
    ]


def test_chunk_documents_returns_processed_ids(monkeypatch):
    saved: dict[str, list[Chunk]] = {}
    client = chunking_client.ChunkingClient()

    monkeypatch.setattr(
        chunking_client,
        "load_document_meta",
        lambda document_id: {"source_url": f"https://example.com/{document_id}.pdf"},
    )
    monkeypatch.setattr(
        chunking_client,
        "load_parsed_text",
        lambda _document_id: "Art. 1.\nTest.",
    )
    monkeypatch.setattr(
        chunking_client,
        "chunk_by_structure",
        lambda text, document_id, source_url: [
            Chunk(
                document_id=document_id,
                source_url=source_url,
                index=0,
                page_num=1,
                anchor="Art. 1",
                text=text,
            )
        ],
    )
    monkeypatch.setattr(
        chunking_client,
        "persist_chunks",
        lambda document_id, chunks: saved.setdefault(document_id, list(chunks)),
    )

    summary = client.chunk_documents(["WDU20250000003"])

    assert summary == {
        "document_ids": ["WDU20250000003"],
        "total_chunks": 1,
        "errors": 0,
    }
    assert saved["WDU20250000003"][0].anchor == "Art. 1"


def test_index_documents_embeds_and_persists(monkeypatch):
    class FakeConn:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    fake_conn = FakeConn()
    client = indexing_client.IndexingClient()
    captured: dict[str, object] = {}

    monkeypatch.setattr(client._opensearch, "ensure_index", lambda: None)
    monkeypatch.setattr(indexing_client, "get_conn", lambda: fake_conn)
    monkeypatch.setattr(indexing_client, "ensure_schema", lambda _conn: None)
    monkeypatch.setattr(
        indexing_client,
        "load_chunk_records",
        lambda _document_id: [
            {
                "document_id": "WDU20250000003",
                "source_url": "https://example.com/doc.pdf",
                "index": 0,
                "page_num": 1,
                "anchor": "Art. 1",
                "text": "Test chunk",
            }
        ],
    )
    monkeypatch.setattr(
        indexing_client,
        "load_document_meta",
        lambda _document_id: {
            "document_id": "WDU20250000003",
            "source_url": "https://example.com/doc.pdf",
            "fetched_at": "2025-01-01T00:00:00+00:00",
            "content_type": "application/pdf",
            "title": "Act title",
            "act_type": "Ustawa",
            "year": 2025,
            "pos": 3,
            "status": VALID_STATUS,
            "announcement_date": "2025-01-01",
            "raw_path": "/tmp/raw.pdf",
            "parsed_path": "/tmp/parsed.txt",
        },
    )
    monkeypatch.setattr(
        indexing_client,
        "embed_passages",
        lambda texts, batch_size: [[0.1, 0.2]],
    )

    def fake_bulk_index(docs):
        captured["docs"] = docs
        return len(docs), 0

    monkeypatch.setattr(client._opensearch, "bulk_index", fake_bulk_index)
    monkeypatch.setattr(
        indexing_client,
        "upsert_document",
        lambda _conn, meta: captured.setdefault("meta", meta),
    )
    monkeypatch.setattr(
        indexing_client,
        "upsert_chunks",
        lambda _conn, chunks: captured.setdefault("chunks", chunks),
    )

    summary = client.index_documents(["WDU20250000003"])

    assert summary == {
        "document_ids": ["WDU20250000003"],
        "total_chunks": 1,
        "total_indexed": 1,
        "errors": 0,
    }
    assert fake_conn.closed is True
    assert captured["docs"][0].chunk_index == 0
    assert captured["meta"]["title"] == "Act title"
    assert captured["chunks"][0]["anchor"] == "Art. 1"


def test_find_expired_documents_handles_du_and_mp_statuses(monkeypatch):
    client = indexing_client.IndexingClient()
    monkeypatch.setattr(indexing_client.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        indexing_client,
        "list_acts",
        lambda year, journal="DU": (
            [
                _make_act("WDU20250000001", year=year, journal=journal, pos=1),
                _make_act(
                    "WDU20250000002",
                    year=year,
                    journal=journal,
                    pos=2,
                    status="uchylony",
                ),
            ]
            if journal == "DU"
            else [
                _make_act(
                    "WMP20250000003",
                    year=year,
                    journal=journal,
                    pos=3,
                    status="wygaśnięcie aktu",
                )
            ]
        ),
    )

    expired = client.find_expired_documents(
        ["WDU20250000001", "WDU20250000002", "WMP20250000003"]
    )

    assert expired == ["WDU20250000002", "WMP20250000003"]
