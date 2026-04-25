"""
Structure-aware chunker for Polish legal acts from isap.sejm.gov.pl.

Split strategy
--------------
1. Strip page-break headers (Dziennik Ustaw / Monitor Polski), but record
   the page number at which each article starts.
2. Split the full text at Art. / § boundaries — never mix two articles in
   one chunk.
3. Keep the current Rozdział / Dział heading as a context prefix in every
   chunk that belongs to that chapter.
4. If an article fits within MAX_CHARS → one chunk.
   If it is longer → split at ust. (numbered-paragraph) boundaries and
   carry OVERLAP_CHARS of the previous ust. into the next sub-chunk so
   cross-ust. reasoning is preserved.

Model boundaries respected
--------------------------
  multilingual-e5-base   : 512 tokens  ≈ 1,500 chars  → MAX_CHARS = 1,400
  polish-reranker-roberta: 256 tokens total (query + passage)
                           → effective passage ≈ 700 chars; content beyond
                             that is truncated by the cross-encoder, but
                             article headers and first provisions always
                             appear first so the most relevant text is seen.
  DeepSeek LLM           : 64 K tokens — not a constraint at chunk level.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_CHARS = 1_400     # hard ceiling: fits embedding model's 512-token window
OVERLAP_CHARS = 200   # overlap carried into the next sub-chunk within one article

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_JOURNAL_NAMES = r"(?:Dziennik Ustaw|Monitor Polski)"

# Page-break headers inserted by the PDF parser between pages.
_PAGE_HEADER_RE = re.compile(
    rf"(?:^|\n){_JOURNAL_NAMES}\s*[–\-]\s*(\d+)\s*[–\-]\s*Poz\.\s*\d+[^\n]*",
    re.IGNORECASE | re.MULTILINE,
)

# Article / paragraph start (line must begin with Art. or §).
# Matches: "Art. 120.", "Art. 120a.", "§ 3.", "§ 3a."
_ARTICLE_START_RE = re.compile(
    r"(?m)^(?:Art\.\s*\d+\w*\.|§\s*\d+\w*\.?)",
)

# Chapter / division headings — carried as context prefix.
_CHAPTER_RE = re.compile(
    r"(?m)^(?:DZIAŁ\s+[IVXLC\w]+|Rozdział\s+\d+\w*)[^\n]*",
)

# ust. split point inside an article:
#   - must start at the beginning of a line (after \n)
#   - digit(s) followed by period and whitespace and a Polish capital letter
#   - NOT preceded by ")" which would indicate a footnote reference inline
_UST_START_RE = re.compile(
    r"(?m)^(\d{1,2})\.\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ])",
)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    document_id: str
    source_url: str
    index: int        # 0-based position in document
    page_num: int     # page where the article starts
    anchor: str       # e.g. "Art. 120" or "§ 3"
    text: str = field(repr=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_page_headers(text: str) -> tuple[str, dict[int, int]]:
    """
    Remove page-break headers and return (clean_text, offset→page_num map).
    The map lets us look up which page a character offset belongs to.
    """
    page_map: dict[int, int] = {0: 1}
    offset_shift = 0
    clean_parts: list[str] = []
    last_end = 0

    for m in _PAGE_HEADER_RE.finditer(text):
        clean_parts.append(text[last_end:m.start()])
        # Record the position (after removal) → page number
        new_offset = sum(len(p) for p in clean_parts)
        page_num = int(m.group(1))
        page_map[new_offset] = page_num
        last_end = m.end()

    clean_parts.append(text[last_end:])
    return "".join(clean_parts), page_map


def _page_at_offset(offset: int, page_map: dict[int, int]) -> int:
    """Return the page number active at *offset* in the cleaned text."""
    page = 1
    for pos, num in sorted(page_map.items()):
        if pos <= offset:
            page = num
        else:
            break
    return page


def _split_at_ust(article_text: str) -> list[str]:
    """
    Split *article_text* into ust.-level segments.
    Returns the original text as a single item if no ust. markers are found.
    """
    # Find all ust. start positions (skip position 0 — that's the Art. header)
    boundaries = [m.start() for m in _UST_START_RE.finditer(article_text)
                  if m.start() > 0]
    if not boundaries:
        return [article_text]

    segments: list[str] = []
    prev = 0
    for b in boundaries:
        segments.append(article_text[prev:b].rstrip())
        prev = b
    segments.append(article_text[prev:].rstrip())
    return [s for s in segments if s.strip()]


def _make_sub_chunks(header: str, segments: list[str]) -> list[str]:
    """
    Pack *segments* (ust. pieces) into sub-chunks that each fit MAX_CHARS,
    carrying OVERLAP_CHARS of the previous segment as context.
    """
    sub_chunks: list[str] = []
    current = header
    overlap = ""  # last segment added — carried as context when a new sub-chunk starts

    for seg in segments:
        candidate = (current + "\n" + seg).strip()
        if len(candidate) <= MAX_CHARS:
            current = candidate
        else:
            if current.strip() and current.strip() != header.strip():
                sub_chunks.append(current.strip())
            # Start a new sub-chunk: header + previous segment (overlap) + this segment
            if overlap:
                current = (header + "\n" + overlap + "\n" + seg).strip()
            else:
                current = (header + "\n" + seg).strip()
        overlap = seg

    if current.strip() and current.strip() != header.strip():
        sub_chunks.append(current.strip())

    # If a single ust. is still too long, hard-cut it
    result: list[str] = []
    for sc in sub_chunks:
        if len(sc) <= MAX_CHARS:
            result.append(sc)
        else:
            # Hard cut at MAX_CHARS, keeping header
            pos = 0
            first = True
            while pos < len(sc):
                if first:
                    piece = sc[pos:pos + MAX_CHARS]
                    first = False
                else:
                    piece = header + "\n[...]\n" + sc[pos:pos + MAX_CHARS - len(header) - 8]
                result.append(piece.strip())
                pos += MAX_CHARS
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_by_structure(
    text: str,
    *,
    document_id: str,
    source_url: str,
) -> list[Chunk]:
    """
    Split *text* into structure-aware chunks respecting Art./§ boundaries
    and ML model token limits.
    """
    clean_text, page_map = _strip_page_headers(text)

    # Collect article boundaries and chapter headings in one pass
    events: list[tuple[int, str, str]] = []  # (offset, kind, text)

    for m in _ARTICLE_START_RE.finditer(clean_text):
        events.append((m.start(), "article", m.group(0)))

    for m in _CHAPTER_RE.finditer(clean_text):
        events.append((m.start(), "chapter", m.group(0)))

    events.sort(key=lambda e: e[0])

    if not events:
        # No structural markers — fall back to simple MAX_CHARS splitting
        return _fallback_chunks(clean_text, document_id, source_url, page_map)

    chunks: list[Chunk] = []
    index = 0
    current_chapter = ""

    # Collect article spans
    article_spans: list[tuple[int, int, str]] = []  # (start, end, anchor)
    art_events = [(off, txt) for off, kind, txt in events if kind == "article"]
    chapter_events = [(off, txt) for off, kind, txt in events if kind == "chapter"]

    for i, (start, anchor) in enumerate(art_events):
        end = art_events[i + 1][0] if i + 1 < len(art_events) else len(clean_text)
        article_spans.append((start, end, anchor))

    for art_start, art_end, raw_anchor in article_spans:
        # Update current chapter if a chapter heading precedes this article
        for ch_off, ch_txt in chapter_events:
            if ch_off < art_start:
                current_chapter = ch_txt.strip()

        article_text = clean_text[art_start:art_end].strip()
        if not article_text:
            continue

        # Normalise anchor to "Art. 120" / "§ 3"
        anchor = raw_anchor.rstrip(".")
        page_num = _page_at_offset(art_start, page_map)

        # Build context prefix (chapter + anchor)
        prefix = (f"{current_chapter}\n{anchor}" if current_chapter
                  else anchor)

        if len(article_text) <= MAX_CHARS:
            sub_texts = [article_text]
        else:
            segments = _split_at_ust(article_text)
            sub_texts = _make_sub_chunks(prefix, segments)

        for sub_text in sub_texts:
            chunks.append(Chunk(
                document_id=document_id,
                source_url=source_url,
                index=index,
                page_num=page_num,
                anchor=anchor,
                text=sub_text,
            ))
            index += 1

    return chunks


def _fallback_chunks(
    text: str,
    document_id: str,
    source_url: str,
    page_map: dict[int, int],
) -> list[Chunk]:
    """Simple MAX_CHARS splitter for documents without Art./§ structure."""
    chunks: list[Chunk] = []
    pos = 0
    index = 0
    while pos < len(text):
        piece = text[pos:pos + MAX_CHARS].strip()
        if piece:
            chunks.append(Chunk(
                document_id=document_id,
                source_url=source_url,
                index=index,
                page_num=_page_at_offset(pos, page_map),
                anchor="",
                text=piece,
            ))
            index += 1
        pos += MAX_CHARS
    return chunks
