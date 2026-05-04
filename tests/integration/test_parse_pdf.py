import pytest

from src.services.pdf_parser import parse_pdf


def test_parse_pdf_returns_string(sample_pdf_bytes):
    result = parse_pdf(sample_pdf_bytes)
    assert isinstance(result, str)


def test_parse_pdf_preserves_polish_diacritics(sample_pdf_bytes):
    result = parse_pdf(sample_pdf_bytes)
    # Polish diacritics must survive round-trip through pdfplumber
    polish_chars = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
    assert any(ch in result for ch in polish_chars), (
        "No Polish diacritics found — possible encoding problem"
    )


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Download a small real ISAP PDF once for the test session."""
    import httpx

    url = "https://api.sejm.gov.pl/eli/acts/DU/2025/1746/text.pdf"
    resp = httpx.get(url, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    return resp.content
