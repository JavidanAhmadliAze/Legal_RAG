import io

import pdfplumber


def parse_pdf(content: bytes) -> str:
    """Extract plain text from PDF bytes, preserving Polish diacritics (UTF-8)."""
    pages: list[str] = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=3)
            if text:
                pages.append(text.strip())
    return "\n\n".join(pages)
