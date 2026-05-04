from __future__ import annotations

import io

import pdfplumber

from src.services.config import PdfParserSettings, get_pdf_parser_settings


class PdfParserClient:
    def __init__(self, *, settings: PdfParserSettings | None = None) -> None:
        self._settings = settings or get_pdf_parser_settings()

    def parse_pdf(self, content: bytes) -> str:
        pages: list[str] = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                text = page.extract_text(
                    x_tolerance=self._settings.x_tolerance,
                    y_tolerance=self._settings.y_tolerance,
                )
                if text:
                    pages.append(text.strip())
        return "\n\n".join(pages)
