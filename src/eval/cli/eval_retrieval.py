#!/usr/bin/env python3
"""
Batch retrieval evaluation against data/eval/ground_truth.json.

Runs the production retrieval pipeline (BM25 + k-NN + RRF + cross-encoder
rerank, the same pipeline the supervisor sub-agents call) for every query in
the dataset and computes the FULL set of retrieval metrics:

    Precision@k, Recall@k, Hit@k, MRR@k, MAP@k, NDCG@k, XDCG@k, Fidelity, Holes

Per-query and aggregate metrics are emitted as OpenTelemetry spans visible in
Phoenix at http://localhost:6006.

Usage
-----
    python -m src.eval.cli.eval_retrieval
    python -m src.eval.cli.eval_retrieval --k 10
    python -m src.eval.cli.eval_retrieval --query-id eu_long_term_5year_clock
    python -m src.eval.cli.eval_retrieval --dataset data/eval/ground_truth.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env so DEEPSEEK_API_KEY etc. are available for the translate step.
for line in (_PROJECT_ROOT / ".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

from langchain_core.output_parsers import StrOutputParser
from opentelemetry import trace

import asyncio

from src.eval.document_retrieval import evaluate_document_retrieval
from src.services.agents.prompts import TRANSLATE_PROMPT
from src.services.agents.nodes.query_parser import _extract_filters
from src.services.agents.nodes.retriever import aretrieve
from src.services.llm import get_llm_client
from src.services.monitoring import init as init_tracing

# Use the externally-running Phoenix (don't relaunch a local one).
os.environ.setdefault("PHOENIX_HOST", "localhost")
# Skip LangChain/OpenAI auto-instrumentation in the eval — only the manual
# eval.* spans matter, and the heavy chat-model wrapping adds significant
# per-call overhead and span volume.
init_tracing(
    project_name="legal-rag-eval",
    port=6006,
    instrument_langchain=False,
    instrument_openai=False,
)
_tracer = trace.get_tracer("legal-rag.eval_retrieval")

_DEFAULT_DATASET = Path(__file__).parent.parent / "data" / "eval" / "ground_truth.json"
_DEFAULT_K = 15            # match _TOTAL_TOP_K used by the supervisor
_FETCH_K = 50              # match production; larger pool feeds reranker more candidates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_id(chunk: dict) -> str:
    return f"{chunk.get('document_id', '')}:{chunk.get('page_num', '')}"


_translate_chain = None


def _translate(question: str) -> str:
    """English/PL question → focused Polish keyword query (same as supervisor)."""
    global _translate_chain
    if _translate_chain is None:
        llm = get_llm_client().get_translator_model()
        _translate_chain = TRANSLATE_PROMPT | llm | StrOutputParser()
    return _translate_chain.invoke({"question": question, "preserve_hint": "none"})


async def _aretrieve_one(query: str, top_k: int) -> list[dict]:
    """Run the production retrieval pipeline for a single query (async)."""
    polish = _translate(query)
    filters = _extract_filters(polish) or {}
    return await aretrieve(
        polish,
        top_k=top_k,
        fetch_k=_FETCH_K,
        rerank_query=polish,
        filters=filters or None,
    )


def _aggregate(results: list[dict], k: int) -> dict[str, float]:
    """Mean every metric across queries."""
    if not results:
        return {}
    keys = [
        "precision_at_k", "recall_at_k", "hit_at_k", "map_at_k",
        "mrr", "ndcg", "xdcg", "fidelity",
    ]
    agg = {f"mean_{k}": round(sum(r[k] for r in results) / len(results), 4) for k in keys}
    agg["passed"] = sum(1 for r in results if r["passed"])
    agg["total"] = len(results)
    agg["holes_count"] = sum(1 for r in results if r["holes"])
    return agg


# ---------------------------------------------------------------------------
# Per-query runner
# ---------------------------------------------------------------------------

async def run_one(entry: dict, k: int) -> dict:
    qid = entry["id"]
    query = entry["query"]
    ground_truth: dict[str, int] = entry.get("ground_truth", {})
    is_oos = entry.get("is_out_of_scope", False)

    with _tracer.start_as_current_span("eval.query") as span:
        span.set_attribute("eval.query_id", qid)
        span.set_attribute("eval.query", query)
        span.set_attribute("eval.k", k)
        span.set_attribute("eval.is_out_of_scope", is_oos)

        with _tracer.start_as_current_span("eval.retrieve"):
            chunks = await asyncio.wait_for(_aretrieve_one(query, top_k=k), timeout=300)

        # Ground-truth labels key chunks as "{doc_id}:{page_num}". The retriever
        # may return several chunks from the same page (different chunk_index
        # within one page) — dedup by chunk-id, keeping rank order, so a single
        # labelled page is counted at most once.
        seen_ids: set[str] = set()
        retrieved_ids: list[str] = []
        for c in chunks:
            cid = _chunk_id(c)
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            retrieved_ids.append(cid)
        result = evaluate_document_retrieval(retrieved_ids, ground_truth, k=k)

        # All metrics → span attributes (visible in Phoenix)
        span.set_attribute("eval.precision_at_k", result.precision_at_k)
        span.set_attribute("eval.recall_at_k", result.recall_at_k)
        span.set_attribute("eval.hit_at_k", result.hit_at_k)
        span.set_attribute("eval.map_at_k", result.map_at_k)
        span.set_attribute("eval.mrr", result.mrr)
        span.set_attribute("eval.ndcg", result.ndcg)
        span.set_attribute("eval.xdcg", result.xdcg)
        span.set_attribute("eval.fidelity", result.fidelity)
        span.set_attribute("eval.max_relevance", result.max_relevance)
        span.set_attribute("eval.holes", result.holes)
        span.set_attribute("eval.passed", result.passed)
        span.set_attribute("eval.retrieved_ids", json.dumps(retrieved_ids))

        # Per-chunk diagnostics
        for i, c in enumerate(chunks[:k]):
            cid = _chunk_id(c)
            span.set_attribute(f"eval.chunk_{i}.id", cid)
            span.set_attribute(f"eval.chunk_{i}.rerank_score",
                               c.get("_rerank_score", 0.0))
            span.set_attribute(f"eval.chunk_{i}.ground_truth_rel",
                               ground_truth.get(cid, 0))

        relevant_hits = [
            (rank + 1, cid, ground_truth.get(cid, 0))
            for rank, cid in enumerate(retrieved_ids)
            if ground_truth.get(cid, 0) > 0
        ]

    return {
        "id": qid,
        "is_out_of_scope": is_oos,
        "precision_at_k": result.precision_at_k,
        "recall_at_k": result.recall_at_k,
        "hit_at_k": result.hit_at_k,
        "map_at_k": result.map_at_k,
        "mrr": result.mrr,
        "ndcg": result.ndcg,
        "xdcg": result.xdcg,
        "fidelity": result.fidelity,
        "max_relevance": result.max_relevance,
        "holes": result.holes,
        "passed": result.passed,
        "hits": relevant_hits,
        "retrieved_ids": retrieved_ids,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def amain() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--k", type=int, default=_DEFAULT_K,
                        help="Cutoff rank for all @k metrics")
    parser.add_argument("--query-id", default=None,
                        help="Run only the entry with this id")
    parser.add_argument("--save", type=Path, default=None,
                        help="Write per-query results to this JSON file")
    args = parser.parse_args()

    dataset: list[dict] = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.query_id:
        dataset = [e for e in dataset if e["id"] == args.query_id]
        if not dataset:
            sys.exit(f"No entry with id={args.query_id!r}")

    print(f"Evaluating {len(dataset)} queries  |  k={args.k}\n{'=' * 72}")

    results: list[dict] = []

    with _tracer.start_as_current_span("eval.batch") as batch_span:
        batch_span.set_attribute("eval.dataset", str(args.dataset))
        batch_span.set_attribute("eval.k", args.k)
        batch_span.set_attribute("eval.query_count", len(dataset))

        for entry in dataset:
            print(f"\n[{entry['id']}]  {'(OOS)' if entry.get('is_out_of_scope') else ''}", flush=True)
            print(f"  Query: {entry['query'][:90]}...", flush=True)
            try:
                r = await run_one(entry, args.k)
            except asyncio.TimeoutError:
                print(f"  *** TIMEOUT after 120s — skipping ***", flush=True)
                r = {
                    "id": entry["id"], "is_out_of_scope": entry.get("is_out_of_scope", False),
                    "precision_at_k": 0.0, "recall_at_k": 0.0, "hit_at_k": 0.0,
                    "map_at_k": 0.0, "mrr": 0.0, "ndcg": 0.0, "xdcg": 0.0,
                    "fidelity": 0.0, "max_relevance": 0, "holes": True, "passed": False,
                    "hits": [], "retrieved_ids": [], "timed_out": True,
                }
            results.append(r)
            if args.save:
                args.save.write_text(json.dumps({
                    "k": args.k, "aggregate": _aggregate(results, args.k),
                    "per_query": results,
                }, ensure_ascii=False, indent=2))
            print(
                f"  P@{args.k}={r['precision_at_k']:.3f}  "
                f"R@{args.k}={r['recall_at_k']:.3f}  "
                f"Hit={r['hit_at_k']:.0f}  "
                f"MAP@{args.k}={r['map_at_k']:.3f}  "
                f"MRR={r['mrr']:.3f}  "
                f"NDCG@{args.k}={r['ndcg']:.3f}  "
                f"XDCG@{args.k}={r['xdcg']:.3f}"
            )
            if r["hits"]:
                for rank, cid, rel in r["hits"][:5]:
                    print(f"    rank {rank:2d}  rel={rel}  {cid}")
            elif not r["is_out_of_scope"]:
                print("    *** no relevant chunk in top-k ***")

        agg = _aggregate(results, args.k)
        for key, val in agg.items():
            batch_span.set_attribute(f"eval.{key}", val)

    if len(results) > 1:
        print(f"\n{'=' * 72}")
        print(f"Aggregate over {len(results)} queries  (k={args.k})")
        print(f"  Mean Precision@{args.k}  : {agg['mean_precision_at_k']:.4f}")
        print(f"  Mean Recall@{args.k}     : {agg['mean_recall_at_k']:.4f}")
        print(f"  Mean Hit@{args.k}        : {agg['mean_hit_at_k']:.4f}")
        print(f"  Mean MAP@{args.k}        : {agg['mean_map_at_k']:.4f}")
        print(f"  Mean MRR@{args.k}        : {agg['mean_mrr']:.4f}")
        print(f"  Mean NDCG@{args.k}       : {agg['mean_ndcg']:.4f}")
        print(f"  Mean XDCG@{args.k}       : {agg['mean_xdcg']:.4f}")
        print(f"  Passed                   : {agg['passed']}/{agg['total']}")
        print(f"  Holes (no relevant hit)  : {agg['holes_count']}/{agg['total']}")
        print("\nAll traces sent to Phoenix → http://localhost:6006")

    if args.save:
        args.save.write_text(json.dumps({
            "k": args.k,
            "aggregate": agg,
            "per_query": results,
        }, ensure_ascii=False, indent=2))
        print(f"Saved per-query results → {args.save}")


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
