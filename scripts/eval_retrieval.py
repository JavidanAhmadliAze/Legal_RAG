#!/usr/bin/env python3
"""
Batch retrieval evaluation — NDCG and MRR.

Runs the full retrieval pipeline for every query in a ground-truth dataset,
reports per-query and aggregate NDCG@k and MRR, and emits all metrics as
OpenTelemetry spans visible in Phoenix at http://localhost:6006.

Usage
-----
    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --dataset data/eval/ground_truth.json --k 10
    python scripts/eval_retrieval.py --query-id temporary_protection_absence
"""

from __future__ import annotations

import argparse 
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

for line in Path(".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

from src.eval.document_retrieval import evaluate_document_retrieval
from src.monitoring.tracing import init as init_tracing
from src.rag.chain import _split_into_subquestions, _TRANSLATE_PROMPT, _dedupe_chunks
from src.rag.query_parser import parse_query
from src.rag.retriever import retrieve

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from opentelemetry import trace

init_tracing(project_name="legal-rag-eval", port=6006)

_tracer = trace.get_tracer("legal-rag.eval_retrieval")

_DEFAULT_DATASET = Path(__file__).parent.parent / "data" / "eval" / "ground_truth.json"
_DEFAULT_K = 10


def _chunk_id(chunk: dict) -> str:
    return f"{chunk.get('document_id', '')}:{chunk.get('page_num', '')}"


def _retrieve_for_query(question: str, top_k: int) -> list[dict]:
    """Run the same multi-unit retrieval logic as the RAG chain."""
    translator = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        streaming=False,
        temperature=0,
    )
    translate_chain = _TRANSLATE_PROMPT | translator | StrOutputParser()

    units = _split_into_subquestions(question)
    search_units = list(units) + ([question] if len(units) > 1 else [])

    collected: list[dict] = []
    seen_queries: set[str] = set()
    per_unit_k = 6 if len(search_units) > 1 else top_k

    for unit in search_units:
        parsed = parse_query(unit)
        polish = translate_chain.invoke({
            "question": unit,
            "preserve_hint": parsed.preserved_hint() or "none",
        })
        final_q = parsed.augment(polish)
        if final_q in seen_queries:
            continue
        seen_queries.add(final_q)
        collected.extend(retrieve(final_q, top_k=per_unit_k, rerank_query=final_q))

    chunks = _dedupe_chunks(collected)
    chunks.sort(key=lambda c: c.get("_rerank_score", -99), reverse=True)
    return chunks[:top_k]


def run_one(entry: dict, k: int) -> dict:
    query = entry["query"]
    ground_truth: dict[str, int] = entry["ground_truth"]

    with _tracer.start_as_current_span("eval.query") as span:
        span.set_attribute("eval.query_id", entry["id"])
        span.set_attribute("eval.query", query)
        span.set_attribute("eval.k", k)

        with _tracer.start_as_current_span("eval.retrieve"):
            chunks = _retrieve_for_query(query, top_k=k)

        retrieved_ids = [_chunk_id(c) for c in chunks]
        result = evaluate_document_retrieval(retrieved_ids, ground_truth)

        # Emit per-query metrics as span attributes — visible in Phoenix
        span.set_attribute("eval.ndcg", result.ndcg)
        span.set_attribute("eval.xdcg", result.xdcg)
        span.set_attribute("eval.mrr", result.mrr)
        span.set_attribute("eval.fidelity", result.fidelity)
        span.set_attribute("eval.holes", result.holes)
        span.set_attribute("eval.passed", result.passed)
        span.set_attribute("eval.max_relevance", result.max_relevance)
        span.set_attribute("eval.retrieved_ids", json.dumps(retrieved_ids))

        # Emit per-chunk rerank scores so Phoenix shows ranking quality
        for i, chunk in enumerate(chunks):
            span.set_attribute(f"eval.chunk_{i}.id", _chunk_id(chunk))
            span.set_attribute(f"eval.chunk_{i}.rerank_score",
                               chunk.get("_rerank_score", 0.0))
            span.set_attribute(f"eval.chunk_{i}.ground_truth_rel",
                               ground_truth.get(_chunk_id(chunk), 0))

        relevant_hits = [
            (rank + 1, cid, ground_truth.get(cid, 0))
            for rank, cid in enumerate(retrieved_ids)
            if ground_truth.get(cid, 0) > 0
        ]

    return {
        "id": entry["id"],
        "ndcg": result.ndcg,
        "xdcg": result.xdcg,
        "mrr": result.mrr,
        "fidelity": result.fidelity,
        "holes": result.holes,
        "passed": result.passed,
        "hits": relevant_hits,
        "retrieved_ids": retrieved_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--k", type=int, default=_DEFAULT_K,
                        help="Cutoff rank for NDCG@k and MRR@k")
    parser.add_argument("--query-id", default=None,
                        help="Run only the entry with this id")
    args = parser.parse_args()

    dataset: list[dict] = json.loads(args.dataset.read_text())
    if args.query_id:
        dataset = [e for e in dataset if e["id"] == args.query_id]
        if not dataset:
            sys.exit(f"No entry with id={args.query_id!r}")

    print(f"Evaluating {len(dataset)} queries  |  k={args.k}\n{'=' * 64}")

    results = []

    with _tracer.start_as_current_span("eval.batch") as batch_span:
        batch_span.set_attribute("eval.dataset", str(args.dataset))
        batch_span.set_attribute("eval.k", args.k)
        batch_span.set_attribute("eval.query_count", len(dataset))

        for entry in dataset:
            print(f"\n[{entry['id']}]")
            print(f"  Query: {entry['query'][:80]}...")
            r = run_one(entry, args.k)
            results.append(r)
            print(f"  NDCG@{args.k}: {r['ndcg']:.4f}  |  MRR@{args.k}: {r['mrr']:.4f}"
                  f"  |  Fidelity: {r['fidelity']:.4f}  |  Holes: {r['holes']}")
            if r["hits"]:
                for rank, cid, rel in r["hits"]:
                    print(f"    rank {rank:2d}  rel={rel}  {cid}")
            else:
                print("    *** no relevant chunk in top-k ***")

        # Aggregate metrics on the parent span
        mean_ndcg = sum(r["ndcg"] for r in results) / len(results)
        mean_mrr = sum(r["mrr"] for r in results) / len(results)
        mean_fid = sum(r["fidelity"] for r in results) / len(results)
        passed = sum(1 for r in results if r["passed"])

        batch_span.set_attribute("eval.mean_ndcg", round(mean_ndcg, 4))
        batch_span.set_attribute("eval.mean_mrr", round(mean_mrr, 4))
        batch_span.set_attribute("eval.mean_fidelity", round(mean_fid, 4))
        batch_span.set_attribute("eval.passed", passed)
        batch_span.set_attribute("eval.total", len(results))

    if len(results) > 1:
        print(f"\n{'=' * 64}")
        print(f"Aggregate over {len(results)} queries  (k={args.k})")
        print(f"  Mean NDCG@{args.k} : {mean_ndcg:.4f}")
        print(f"  Mean MRR@{args.k}  : {mean_mrr:.4f}")
        print(f"  Mean Fidelity : {mean_fid:.4f}")
        print(f"  Passed        : {passed}/{len(results)}")
        print(f"\nAll traces sent to Phoenix → http://localhost:6006")


if __name__ == "__main__":
    main()
