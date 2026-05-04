"""
Structure-aware chunker for Polish legal acts from isap.sejm.gov.pl.

Split strategy
--------------
1. Strip page-break headers (Dziennik Ustaw / Monitor Polski), recording page numbers.
2. Split the full text at Art. / § boundaries — never mix two articles in one chunk.
3. Keep the current Rozdział / Dział heading as a context prefix in every chunk
   that belongs to that chapter.
4. If an article fits within MAX_CHARS → one chunk (variable size is intentional).
5. If longer → split progressively:
     a. ust. (numbered-paragraph) boundaries  — preferred
     b. double-newline (paragraph) boundaries — fallback
     c. sentence boundaries                   — last resort
     d. hard character cut                    — emergency only
   Carry OVERLAP_CHARS of the previous segment into the next chunk to preserve
   cross-boundary reasoning.
6. Fallback for documents with no Art./§ markers: paragraph-aware splitting
   (double-newline → sentence → hard cut), no raw character slicing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from src.services.config import get_chunking_settings

_SETTINGS = get_chunking_settings()
MAX_CHARS = _SETTINGS.max_chars
OVERLAP_CHARS = _SETTINGS.overlap_chars

_JOURNAL_NAMES = r"(?:Dziennik Ustaw|Monitor Polski)"
_PAGE_HEADER_RE = re.compile(
    rf"(?:^|\n){_JOURNAL_NAMES}\s*[–\-]\s*(\d+)\s*[–\-]\s*Poz\.\s*\d+[^\n]*",
    re.IGNORECASE | re.MULTILINE,
)
_ARTICLE_START_RE = re.compile(r"(?m)^(?:Art\.\s*\d+\w*\.|§\s*\d+\w*\.?)")
_CHAPTER_RE = re.compile(r"(?m)^(?:DZIAŁ\s+[IVXLC\w]+|Rozdział\s+\d+\w*)[^\n]*")
_UST_START_RE = re.compile(r"(?m)^(\d{1,2})\.\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ])")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻA-Z])")


@dataclass
class Chunk:
    document_id: str
    source_url: str
    domain: str
    law_domain: str
    title: str
    subheading: str       # current Rozdział / Dział heading, empty if none
    index: int
    page_num: int
    anchor: str           # Art. X / § X that starts this chunk
    text: str = field(repr=False)


def _strip_page_headers(text: str) -> tuple[str, dict[int, int]]:
    page_map: dict[int, int] = {0: 1}
    clean_parts: list[str] = []
    last_end = 0

    for match in _PAGE_HEADER_RE.finditer(text):
        clean_parts.append(text[last_end:match.start()])
        new_offset = sum(len(part) for part in clean_parts)
        page_map[new_offset] = int(match.group(1))
        last_end = match.end()

    clean_parts.append(text[last_end:])
    return "".join(clean_parts), page_map


def _page_at_offset(offset: int, page_map: dict[int, int]) -> int:
    page = 1
    for pos, num in sorted(page_map.items()):
        if pos <= offset:
            page = num
        else:
            break
    return page


def _split_at_ust(text: str) -> list[str]:
    boundaries = [
        m.start() for m in _UST_START_RE.finditer(text) if m.start() > 0
    ]
    if not boundaries:
        return [text]
    segments: list[str] = []
    prev = 0
    for b in boundaries:
        segments.append(text[prev:b].rstrip())
        prev = b
    segments.append(text[prev:].rstrip())
    return [s for s in segments if s.strip()]


def _split_at_paragraphs(text: str) -> list[str]:
    """Split on blank lines (double newline), return non-empty paragraphs."""
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]


def _split_at_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _pack_segments(segments: list[str], header: str, *, depth: int = 0) -> list[str]:
    """
    Pack a list of text segments into chunks ≤ MAX_CHARS.

    Each chunk starts with the header for context. Carries OVERLAP_CHARS of the
    previous segment into the next chunk. Segments that are individually too large
    are recursively split at the next granularity level (depth 0→paragraphs,
    1→sentences, 2→hard cut).
    """
    result: list[str] = []
    current = header
    overlap = ""

    for seg in segments:
        full_seg = (f"{header}\n{overlap}\n{seg}" if overlap else f"{header}\n{seg}").strip()

        if len(full_seg) <= MAX_CHARS:
            current = full_seg
        else:
            if current.strip() and current.strip() != header.strip():
                result.append(current.strip())

            candidate = (
                f"{header}\n{overlap}\n{seg}" if overlap else f"{header}\n{seg}"
            ).strip()

            if len(candidate) <= MAX_CHARS:
                current = candidate
            else:
                # Segment itself is too large — recurse at finer granularity.
                sub = _force_split(seg, header, depth=depth)
                result.extend(sub[:-1])
                current = sub[-1] if sub else header

        overlap = seg[-OVERLAP_CHARS:].strip() if OVERLAP_CHARS > 0 else ""

    if current.strip() and current.strip() != header.strip():
        result.append(current.strip())

    return result or [header]


def _force_split(text: str, header: str, *, depth: int) -> list[str]:
    """Split *text* at progressively finer granularity until it fits."""
    if depth == 0:
        segs = _split_at_paragraphs(text)
        if len(segs) > 1:
            return _pack_segments(segs, header, depth=1)
    if depth <= 1:
        segs = _split_at_sentences(text)
        if len(segs) > 1:
            return _pack_segments(segs, header, depth=2)
    # Hard cut — absolute last resort.
    pieces: list[str] = []
    budget = MAX_CHARS - len(header) - 8  # room for "\n[...]\n"
    if budget < 1:
        budget = MAX_CHARS
    pos = 0
    first = True
    while pos < len(text):
        if first:
            piece = text[pos : pos + MAX_CHARS]
            first = False
        else:
            piece = f"{header}\n[...]\n{text[pos : pos + budget]}"
        pieces.append(piece.strip())
        pos += MAX_CHARS if first else budget
    return pieces


def _chunk_article(
    article_text: str,
    *,
    header: str,
    document_id: str,
    source_url: str,
    domain: str,
    law_domain: str,
    title: str,
    subheading: str,
    start_index: int,
    page_num: int,
    anchor: str,
) -> list[Chunk]:
    if len(article_text) <= MAX_CHARS:
        sub_texts = [article_text]
    else:
        ust_segs = _split_at_ust(article_text)
        if len(ust_segs) > 1:
            sub_texts = _pack_segments(ust_segs, header, depth=0)
        else:
            sub_texts = _force_split(article_text, header, depth=0)

    chunks: list[Chunk] = []
    for i, text in enumerate(sub_texts):
        chunks.append(
            Chunk(
                document_id=document_id,
                source_url=source_url,
                domain=domain,
                law_domain=law_domain,
                title=title,
                subheading=subheading,
                index=start_index + i,
                page_num=page_num,
                anchor=anchor,
                text=text,
            )
        )
    return chunks


def chunk_by_structure(
    text: str,
    *,
    document_id: str,
    source_url: str,
    law_domain: str = "other",
    title: str = "",
) -> list[Chunk]:
    domain = urlparse(source_url).netloc or source_url
    clean_text, page_map = _strip_page_headers(text)

    art_events = [
        (m.start(), m.group(0)) for m in _ARTICLE_START_RE.finditer(clean_text)
    ]
    chapter_events = [
        (m.start(), m.group(0).strip()) for m in _CHAPTER_RE.finditer(clean_text)
    ]

    if not art_events:
        return _fallback_chunks(
            clean_text, document_id, source_url, domain, law_domain, title, page_map
        )

    chunks: list[Chunk] = []
    index = 0
    current_chapter = ""

    for i, (art_start, raw_anchor) in enumerate(art_events):
        art_end = art_events[i + 1][0] if i + 1 < len(art_events) else len(clean_text)

        for ch_off, ch_txt in chapter_events:
            if ch_off < art_start:
                current_chapter = ch_txt

        article_text = clean_text[art_start:art_end].strip()
        if not article_text:
            continue

        anchor = raw_anchor.rstrip(".")
        page_num = _page_at_offset(art_start, page_map)
        header = f"{current_chapter}\n{anchor}" if current_chapter else anchor

        new_chunks = _chunk_article(
            article_text,
            header=header,
            document_id=document_id,
            source_url=source_url,
            domain=domain,
            law_domain=law_domain,
            title=title,
            subheading=current_chapter,
            start_index=index,
            page_num=page_num,
            anchor=anchor,
        )
        chunks.extend(new_chunks)
        index += len(new_chunks)

    return chunks


def _fallback_chunks(
    text: str,
    document_id: str,
    source_url: str,
    domain: str,
    law_domain: str,
    title: str,
    page_map: dict[int, int],
) -> list[Chunk]:
    """Paragraph-aware fallback for documents with no Art./§ markers."""
    paragraphs = _split_at_paragraphs(text)
    if not paragraphs:
        return []

    def _make(t: str, pos: int, idx: int) -> Chunk:
        return Chunk(
            document_id=document_id,
            source_url=source_url,
            domain=domain,
            law_domain=law_domain,
            title=title,
            subheading="",
            index=idx,
            page_num=_page_at_offset(pos, page_map),
            anchor="",
            text=t,
        )

    chunks: list[Chunk] = []
    index = 0
    current = ""
    current_pos = 0

    for para in paragraphs:
        candidate = (current + "\n\n" + para).strip() if current else para
        if len(candidate) <= MAX_CHARS:
            current = candidate
        else:
            if current:
                chunks.append(_make(current, current_pos, index))
                index += 1
                current_pos += len(current)

            if len(para) <= MAX_CHARS:
                current = para
            else:
                for piece in _force_split(para, "", depth=0):
                    if not piece:
                        continue
                    chunks.append(_make(piece, current_pos, index))
                    index += 1
                    current_pos += len(piece)
                current = ""

    if current:
        chunks.append(_make(current, current_pos, index))

    return chunks
