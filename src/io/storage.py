import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_RAW_DIR = _PROJECT_ROOT / "data" / "raw"
_PARSED_DIR = _PROJECT_ROOT / "data" / "parsed"


@dataclass
class DocumentRecord:
    document_id: str
    source_url: str
    fetched_at: str        # ISO-8601
    content_type: str
    title: str
    act_type: str
    year: int
    pos: int
    status: str
    announcement_date: str
    raw_path: str
    parsed_path: str


def save_document(
    *,
    document_id: str,
    source_url: str,
    content: bytes,
    content_type: str,
    text: str,
    title: str,
    act_type: str,
    year: int,
    pos: int,
    status: str,
    announcement_date: str,
) -> DocumentRecord:
    """Persist raw bytes, parsed text, and metadata; return the record."""
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    _PARSED_DIR.mkdir(parents=True, exist_ok=True)

    ext = "pdf" if "pdf" in content_type else "html"
    raw_path = _RAW_DIR / f"{document_id}.{ext}"
    parsed_path = _PARSED_DIR / f"{document_id}.txt"

    raw_path.write_bytes(content)
    parsed_path.write_text(text, encoding="utf-8")

    record = DocumentRecord(
        document_id=document_id,
        source_url=source_url,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        content_type=content_type,
        title=title,
        act_type=act_type,
        year=year,
        pos=pos,
        status=status,
        announcement_date=announcement_date,
        raw_path=str(raw_path),
        parsed_path=str(parsed_path),
    )
    meta_path = _RAW_DIR / f"{document_id}.meta.json"
    meta_path.write_text(
        json.dumps(asdict(record), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return record


def already_saved(document_id: str) -> bool:
    """True if this document was already ingested (skip duplicate downloads)."""
    return (_RAW_DIR / f"{document_id}.meta.json").exists()
