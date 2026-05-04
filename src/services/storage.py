from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from src.services.config import get_storage_settings
from src.services.chunking.structure import Chunk
from src.io import storage as storage_io


def raw_dir() -> Path:
    return get_storage_settings().raw_dir


def parsed_dir() -> Path:
    return get_storage_settings().parsed_dir


def chunks_dir() -> Path:
    return get_storage_settings().chunks_dir


def document_already_saved(document_id: str) -> bool:
    return storage_io.already_saved(document_id)


def persist_document(**kwargs):
    return storage_io.save_document(**kwargs)


def load_document_meta(document_id: str) -> dict:
    meta_path = raw_dir() / f"{document_id}.meta.json"
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_parsed_text(document_id: str) -> str:
    text_path = parsed_dir() / f"{document_id}.txt"
    return text_path.read_text(encoding="utf-8")


def persist_chunks(document_id: str, chunks: Iterable[Chunk]) -> Path:
    chunk_records = list(chunks)
    out_dir = chunks_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{document_id}.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for chunk in chunk_records:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    return out_path


def load_chunk_records(document_id: str) -> list[dict]:
    path = chunks_dir() / f"{document_id}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def list_parsed_document_ids() -> list[str]:
    return [path.stem for path in sorted(parsed_dir().glob("*.txt"))]


def list_chunked_document_ids() -> list[str]:
    return [path.stem for path in sorted(chunks_dir().glob("*.jsonl"))]
