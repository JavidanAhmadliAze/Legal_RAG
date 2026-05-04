#!/usr/bin/env python3
"""
Generate a retrieval ground-truth dataset from currently-indexed chunks.

For each indexed document, sample N "answerable" chunks (>=MIN_CHARS, not §1
preamble), ask the LLM to write an English question that THIS chunk uniquely
answers plus a Polish keyword query for BM25, then label the source chunk
relevance=3 in the ground truth.

Output is schema-compatible with src/eval/cli/eval_retrieval.py.

Usage
-----
    python -m src.eval.cli.generate_synthetic_eval
    python -m src.eval.cli.generate_synthetic_eval --per-doc 2 --limit-docs 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

for line in (_PROJECT_ROOT / ".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

from langchain_core.messages import HumanMessage, SystemMessage

from src.services.llm import get_llm_client
from src.services.opensearch import INDEX_NAME, get_client


_MIN_CHARS = 300
_DEFAULT_PER_DOC = 1
_DEFAULT_OUT = _PROJECT_ROOT / "data" / "eval" / "ground_truth.json"

_SYSTEM = (
    "You generate evaluation questions for a Polish legal-document retrieval "
    "system. Given a single excerpt from a Polish statute, write ONE concrete "
    "user question (in English) whose factual answer is contained in this "
    "excerpt and not trivially answered by other passages of similar laws. "
    "Then write a short Polish keyword query (2–6 tokens, no punctuation) "
    "suitable for BM25 against Polish statute text.\n\n"
    "Return ONLY valid JSON with this exact shape:\n"
    "{\"query\": \"...\", \"polish_query\": \"...\"}"
)

_USER_TMPL = (
    "Title: {title}\n"
    "Anchor: {anchor}\n"
    "Law domain: {law_domain}\n\n"
    "Excerpt:\n{text}\n"
)


def _scan_index(client) -> list[dict]:
    body = {"size": 1000, "_source": {"excludes": ["embedding"]}, "query": {"match_all": {}}}
    res = client.search(index=INDEX_NAME, body=body, scroll="2m")
    hits = list(res["hits"]["hits"])
    sid = res.get("_scroll_id")
    while sid:
        res = client.scroll(scroll_id=sid, scroll="2m")
        new = res["hits"]["hits"]
        if not new:
            break
        hits.extend(new)
        sid = res.get("_scroll_id")
    return [h["_source"] | {"_id": h["_id"]} for h in hits]


def _pick_chunks_per_doc(chunks: list[dict], per_doc: int) -> list[dict]:
    """For each doc, pick up to per_doc chunks: longest, not §1, not first chunk."""
    by_doc: dict[str, list[dict]] = {}
    for c in chunks:
        by_doc.setdefault(c["document_id"], []).append(c)

    picked: list[dict] = []
    for doc_id, group in by_doc.items():
        cand = [
            c for c in group
            if len(c.get("text", "")) >= _MIN_CHARS
            and c.get("anchor", "").strip() not in ("§ 1", "§1", "Art. 1")
            and c.get("chunk_index", 0) > 0
        ]
        if not cand:
            cand = [c for c in group if len(c.get("text", "")) >= _MIN_CHARS] or group
        cand.sort(key=lambda c: len(c.get("text", "")), reverse=True)
        picked.extend(cand[:per_doc])
    return picked


def _generate_query(llm, chunk: dict, retries: int = 2) -> dict[str, str] | None:
    user = _USER_TMPL.format(
        title=chunk.get("title", "")[:300],
        anchor=chunk.get("anchor", ""),
        law_domain=chunk.get("law_domain", ""),
        text=chunk.get("text", "")[:1500],
    )
    for attempt in range(retries + 1):
        try:
            resp = llm.invoke([SystemMessage(content=_SYSTEM), HumanMessage(content=user)])
            txt = resp.content.strip()
            m = re.search(r"\{.*\}", txt, re.DOTALL)
            if not m:
                continue
            obj = json.loads(m.group(0))
            q = obj.get("query", "").strip()
            pq = obj.get("polish_query", "").strip()
            if q and pq:
                return {"query": q, "polish_query": pq}
        except Exception as e:
            if attempt == retries:
                print(f"    LLM error after {retries+1} attempts: {e}", file=sys.stderr)
        time.sleep(1.0)
    return None


def _entry_id(chunk: dict, idx: int) -> str:
    doc = chunk["document_id"]
    page = chunk.get("page_num", 0)
    ci = chunk.get("chunk_index", idx)
    return f"{doc}_p{page}_c{ci}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-doc", type=int, default=_DEFAULT_PER_DOC,
                        help="Number of queries to generate per indexed document")
    parser.add_argument("--limit-docs", type=int, default=None,
                        help="If set, only process the first N distinct documents")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args()

    client = get_client()
    print(f"Scanning {INDEX_NAME}...", flush=True)
    chunks = _scan_index(client)
    print(f"  loaded {len(chunks)} chunks across "
          f"{len({c['document_id'] for c in chunks})} docs", flush=True)

    picks = _pick_chunks_per_doc(chunks, args.per_doc)
    if args.limit_docs:
        seen: set[str] = set()
        limited = []
        for c in picks:
            if c["document_id"] in seen and len(seen) >= args.limit_docs:
                continue
            seen.add(c["document_id"])
            if len(seen) > args.limit_docs:
                break
            limited.append(c)
        picks = limited

    print(f"Generating queries for {len(picks)} chunks...", flush=True)
    llm = get_llm_client().get_chat_model()

    dataset: list[dict[str, Any]] = []
    for i, ch in enumerate(picks, 1):
        gen = _generate_query(llm, ch)
        if not gen:
            print(f"  [{i}/{len(picks)}] {ch['document_id']} :: {ch.get('anchor','?')}  SKIP", flush=True)
            continue
        chunk_key = f"{ch['document_id']}:{ch.get('page_num', 0)}"
        entry = {
            "id": _entry_id(ch, i),
            "query": gen["query"],
            "polish_query": gen["polish_query"],
            "topic": f"{ch.get('law_domain','')} :: {ch.get('anchor','')}",
            "law_domain": ch.get("law_domain", ""),
            "expected_acts": [ch["document_id"]],
            "is_out_of_scope": False,
            "ground_truth": {chunk_key: 3},
            "source_chunk": {
                "document_id": ch["document_id"],
                "page_num": ch.get("page_num"),
                "anchor": ch.get("anchor", ""),
                "chunk_index": ch.get("chunk_index"),
            },
        }
        dataset.append(entry)
        print(f"  [{i}/{len(picks)}] {ch['document_id']} :: {ch.get('anchor','?')}  →  {gen['query'][:80]}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dataset, ensure_ascii=False, indent=2))
    print(f"\nWrote {len(dataset)} queries → {args.out}")


if __name__ == "__main__":
    main()
