"""
Page-based chunker for Polish legal acts from isap.sejm.gov.pl.

Each page in the PDF is separated by a header of the form:
    "Dziennik Ustaw – N – Poz. XXXX"
or  "Monitor Polski – N – Poz. XXXX"

One chunk = one page, unless the page exceeds MAX_CHARS, in which case it is
split further at paragraph boundaries.

Size constraints:
  - Embedding model (multilingual-e5-base): 512 tokens ≈ 1,500 chars
  - Cross-encoder (polish-reranker-roberta-v3): 256 tokens total (query + passage)
    → effective passage budget ≈ 700 chars
  - DeepSeek LLM: 64K tokens — not a constraint at chunk level

MAX_CHARS is set to the embedding limit (1,500).  The cross-encoder will still
truncate passages longer than ~700 chars, but keeping chunks at 1,500 prevents
total loss of long-tail content at the embedding/retrieval stage.
"""

import re
from dataclasses import dataclass, field

# Hard ceiling: must fit the embedding model's 512-token window.
# ~3 chars/token for Polish legal text → 512 * 3 = 1,536 → round down.
MAX_CHARS = 1_500

_JOURNAL_NAMES = r"(?:Dziennik Ustaw|Monitor Polski)"

_PAGE_HEADER_RE = re.compile(
    rf"{_JOURNAL_NAMES}\s*[–\-]\s*(\d+)\s*[–\-]\s*Poz\.\s*(\d+)",
    re.IGNORECASE,
)

# Structural anchors: Rozdział, Art., §
_ANCHOR_RE = re.compile(r"(Rozdział\s+\w+|Art\.\s*\d+[\w]*|§\s*\d+)", re.IGNORECASE)

# Paragraph split points, ordered from strongest to weakest boundary
_PARA_SPLIT_RE = re.compile(r"\n{2,}|\n(?=\d+[.)]\s)|(?<=\.)\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ])")


def _split_long_text(text: str) -> list[str]:
    """Split *text* into segments no longer than MAX_CHARS at paragraph boundaries."""
    if len(text) <= MAX_CHARS:
        return [text]

    segments: list[str] = []
    current_start = 0

    while current_start < len(text):
        end = current_start + MAX_CHARS
        if end >= len(text):
            segments.append(text[current_start:].strip())
            break

        # Find the last paragraph boundary before the limit
        window = text[current_start:end]
        best_split = -1
        for m in _PARA_SPLIT_RE.finditer(window):
            best_split = m.start()

        if best_split > MAX_CHARS // 4:
            cut = current_start + best_split
        else:
            # No good boundary found — hard cut at MAX_CHARS
            cut = end

        segment = text[current_start:cut].strip()
        if segment:
            segments.append(segment)
        current_start = cut

    return [s for s in segments if s]


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
    Split *text* into chunks: one per printed page, sub-split if the page
    exceeds MAX_CHARS so every chunk fits the embedding model's token window.
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
            pending_header = part.strip()
            page_num = int(header_match.group(1))
        else:
            body = part.strip()
            if not body:
                continue
            full_text = f"{pending_header}\n{body}".strip() if pending_header else body
            for segment in _split_long_text(full_text):
                anchor_match = _ANCHOR_RE.search(segment)
                anchor = anchor_match.group(0).strip() if anchor_match else ""
                chunks.append(
                    Chunk(
                        document_id=document_id,
                        source_url=source_url,
                        index=index,
                        page_num=page_num,
                        anchor=anchor,
                        text=segment,
                    )
                )
                index += 1
            pending_header = ""
            page_num += 1

    return chunks
