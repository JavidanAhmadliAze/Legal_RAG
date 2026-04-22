import json
from pathlib import Path

import pytest

from src.io.storage import already_saved, save_document


@pytest.fixture(autouse=True)
def tmp_data_dirs(tmp_path, monkeypatch):
    """Redirect data dirs to a temp directory so tests don't pollute data/."""
    import src.io.storage as storage_mod

    monkeypatch.setattr(storage_mod, "_RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(storage_mod, "_PARSED_DIR", tmp_path / "parsed")


def test_save_document_creates_files():
    rec = save_document(
        document_id="WDU20250000001",
        source_url="https://example.com/test.pdf",
        content=b"%PDF-1.4 fake",
        content_type="application/pdf",
        text="Artykuł 1. Pracownik ma prawo do wynagrodzenia.",
        title="Ustawa testowa",
        act_type="Ustawa",
        year=2025,
        pos=1,
        status="obowiązujący",
        announcement_date="2025-01-01",
    )
    assert Path(rec.raw_path).exists()
    assert Path(rec.parsed_path).exists()
    meta = json.loads(Path(rec.raw_path).with_suffix(".meta.json").read_text())
    assert meta["document_id"] == "WDU20250000001"
    assert meta["title"] == "Ustawa testowa"


def test_already_saved_false_before_save():
    assert not already_saved("WDU20250099999")


def test_already_saved_true_after_save():
    save_document(
        document_id="WDU20250000002",
        source_url="https://example.com/test2.pdf",
        content=b"%PDF fake",
        content_type="application/pdf",
        text="Treść.",
        title="Test",
        act_type="Ustawa",
        year=2025,
        pos=2,
        status="obowiązujący",
        announcement_date="2025-01-02",
    )
    assert already_saved("WDU20250000002")


def test_parsed_text_encoding():
    polish_text = "Pracownik ma prawo do wynagrodzenia: ąćęłńóśźż"
    rec = save_document(
        document_id="WDU20250000003",
        source_url="https://example.com/test3.pdf",
        content=b"%PDF fake",
        content_type="application/pdf",
        text=polish_text,
        title="Test diacritics",
        act_type="Ustawa",
        year=2025,
        pos=3,
        status="obowiązujący",
        announcement_date="2025-01-03",
    )
    saved = Path(rec.parsed_path).read_text(encoding="utf-8")
    assert saved == polish_text
