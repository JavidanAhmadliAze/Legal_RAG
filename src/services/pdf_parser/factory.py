from __future__ import annotations

from .client import PdfParserClient

_client: PdfParserClient | None = None


def get_pdf_parser_client() -> PdfParserClient:
    global _client
    if _client is None:
        _client = PdfParserClient()
    return _client
