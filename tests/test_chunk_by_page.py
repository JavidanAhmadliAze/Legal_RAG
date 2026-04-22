from src.chunk.by_page import chunk_by_page


def test_chunk_by_page_supports_monitor_polski_headers():
    text = """MONITOR POLSKI
RZECZYPOSPOLITEJ POLSKIEJ
Warszawa, dnia 10 kwietnia 2026 r.
Poz. 370
KOMUNIKAT
Treść strony pierwszej

Monitor Polski – 2 – Poz. 370
Art. 1. Treść strony drugiej
"""
    chunks = chunk_by_page(
        text,
        document_id="WMP20260000370",
        source_url="https://api.sejm.gov.pl/eli/acts/MP/2026/370/text.pdf",
    )
    assert len(chunks) == 2
    assert chunks[0].page_num == 1
    assert chunks[1].page_num == 2
    assert "Monitor Polski" in chunks[1].text
    assert chunks[1].anchor.lower().startswith("art.")
