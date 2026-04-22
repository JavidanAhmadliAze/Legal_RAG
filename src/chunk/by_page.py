"""
Page-based chunker for Polish legal acts from isap.sejm.gov.pl.

Each page in the PDF is separated by a header of the form:
    "Dziennik Ustaw – N – Poz. XXXX"
or  "Monitor Polski – N – Poz. XXXX"

One chunk = one page.  The first block (cover page) is included as page 1.
"""

import re
from dataclasses import dataclass, field

_JOURNAL_NAMES = r"(?:Dziennik Ustaw|Monitor Polski)"

_PAGE_HEADER_RE = re.compile(
    rf"{_JOURNAL_NAMES}\s*[–\-]\s*(\d+)\s*[–\-]\s*Poz\.\s*(\d+)",
    re.IGNORECASE,
)

# Structural anchors: Rozdział, Art., §
_ANCHOR_RE = re.compile(r"(Rozdział\s+\w+|Art\.\s*\d+[\w]*|§\s*\d+)", re.IGNORECASE)


@dataclass
class Chunk:
    document_id: str
    source_url: str
    index: int        # 0-based position in document
    page_num: int     # page number as printed in the PDF
    anchor: str       # first structural marker found in this chunk (or "")
    text: str = field(repr=False)


def chunk_by_page(
    text: str,
    *,
    document_id: str,
    source_url: str,
) -> list[Chunk]:
    """
    Split *text* into one chunk per printed page, using the
    printed journal header as the page boundary.
    """
    # Split on the page-break header; keep the delimiter via a capture group
    parts = re.split(
        rf"((?:^|\n){_JOURNAL_NAMES}\s*[–\-]\s*\d+\s*[–\-]\s*Poz\.\s*\d+\n?)",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    chunks: list[Chunk] = []
    page_num = 1
    index = 0
    pending_header = ""

    for part in parts:
        header_match = _PAGE_HEADER_RE.search(part)
        if header_match:
            # This part is a page-break header — record the page number
            pending_header = part.strip()
            page_num = int(header_match.group(1))
        else:
            body = part.strip()
            if not body:
                continue
            # Combine header + body so each chunk is self-contained
            full_text = f"{pending_header}\n{body}".strip() if pending_header else body
            anchor_match = _ANCHOR_RE.search(full_text)
            anchor = anchor_match.group(0).strip() if anchor_match else ""
            chunks.append(
                Chunk(
                    document_id=document_id,
                    source_url=source_url,
                    index=index,
                    page_num=page_num,
                    anchor=anchor,
                    text=full_text,
                )
            )
            index += 1
            pending_header = ""
            page_num += 1  # fallback if next part has no header

    return chunks
