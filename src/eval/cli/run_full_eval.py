#!/usr/bin/env python3
"""
Run the FULL evaluation suite over the entire ground-truth dataset and emit a
single metrics file.

Per query, this runner exercises every evaluator wired into ``src/eval``:

    Retrieval (process)
      - Document Retrieval  : Precision@k, Recall@k, Hit@k, MRR@k, MAP@k,
                              NDCG@k, XDCG@k, Fidelity, Holes
      - Retrieval (LLM)     : 1-5 score from an LLM judge

    Generation (process)
      - Groundedness        : every claim grounded in retrieved chunks
      - Relevance           : answer addresses the question
      - Response Completeness : alignment with the ground-truth reference
      - Hard rules          : must_mention / must_not_mention / expected_acts
                              applied as boolean checks against the answer

Output: ``data/eval/results/run_<UTC-timestamp>.json`` containing
  - per-query metrics
  - per-cohort aggregates (by difficulty, query_type, law_domain)
  - global aggregates

Usage:
    python -m src.eval.cli.run_full_eval
    python -m src.eval.cli.run_full_eval --only eu_long_term_5year_clock
    python -m src.eval.cli.run_full_eval --skip-generation   # retrieval only
    python -m src.eval.cli.run_full_eval --k 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env so DEEPSEEK_API_KEY etc. reach the LLM-judge calls and translator.
_ENV_PATH = _PROJECT_ROOT / ".env"
if _ENV_PATH.exists():
    for line in _ENV_PATH.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

from langchain_core.output_parsers import StrOutputParser
from opentelemetry import trace

from src.eval import (
    evaluate_document_retrieval,
    evaluate_groundedness,
    evaluate_relevance,
    evaluate_response_completeness,
    evaluate_retrieval,
)
from src.services.agents.chain import build_chain
from src.services.agents.prompts import TRANSLATE_PROMPT
from src.services.agents.nodes.query_parser import _extract_filters
from src.services.agents.nodes.retriever import aretrieve
from src.services.llm import get_llm_client
from src.services.monitoring import init as init_tracing

os.environ.setdefault("PHOENIX_HOST", "localhost")
init_tracing(
    project_name="legal-rag-eval",
    port=6006,
    instrument_langchain=False,
    instrument_openai=False,
)
_tracer = trace.get_tracer("legal-rag.run_full_eval")

_DEFAULT_DATASET = _PROJECT_ROOT / "data" / "eval" / "ground_truth.json"
_RESULTS_DIR = _PROJECT_ROOT / "data" / "eval" / "results"
_DEFAULT_K = 15           # matches the supervisor's _TOTAL_TOP_K
_FETCH_K = 50             # matches production retriever's candidate pool size


# ---------------------------------------------------------------------------
# Retrieval pipeline (matches what the supervisor does for one sub-query)
# ---------------------------------------------------------------------------

def _chunk_id(chunk: dict) -> str:
    return f"{chunk.get('document_id', '')}:{chunk.get('page_num', '')}"


_translate_chain = None


def _translate(question: str) -> str:
    global _translate_chain
    if _translate_chain is None:
        llm = get_llm_client().get_translator_model()
        _translate_chain = TRANSLATE_PROMPT | llm | StrOutputParser()
    return _translate_chain.invoke({"question": question, "preserve_hint": "none"})


async def _retrieve(query: str, top_k: int) -> list[dict]:
    polish = _translate(query)
    filters = _extract_filters(polish) or {}
    return await aretrieve(
        polish,
        top_k=top_k,
        fetch_k=_FETCH_K,
        rerank_query=polish,
        filters=filters or None,
    )


# ---------------------------------------------------------------------------
# Hard programmatic checks against the generated answer
# ---------------------------------------------------------------------------

def _hard_rules(entry: dict, response: str, retrieved: list[dict]) -> dict:
    """must_mention / must_not_mention / expected_acts presence checks."""
    response_l = response.lower()
    out = {
        "must_mention_passed": True,
        "must_mention_missing": [],
        "must_not_mention_passed": True,
        "must_not_mention_hit": [],
        "expected_acts_cited": [],
        "expected_acts_passed": True,
    }

    for needle in entry.get("must_mention") or []:
        if needle.lower() not in response_l:
            out["must_mention_missing"].append(needle)
    out["must_mention_passed"] = not out["must_mention_missing"]

    for needle in entry.get("must_not_mention") or []:
        if needle.lower() in response_l:
            out["must_not_mention_hit"].append(needle)
    out["must_not_mention_passed"] = not out["must_not_mention_hit"]

    expected_acts = entry.get("expected_acts") or []
    if expected_acts:
        retrieved_acts = {c.get("document_id") for c in retrieved}
        out["expected_acts_cited"] = sorted(set(expected_acts) & retrieved_acts)
        out["expected_acts_passed"] = bool(out["expected_acts_cited"])

    return out


# ---------------------------------------------------------------------------
# Per-query runner
# ---------------------------------------------------------------------------

def _print_summary(r: dict) -> None:
    """One-line per-query summary printed during the run."""
    if "error" in r:
        print(f"      ERROR: {r['error']}")
        return
    bits: list[str] = []
    if "document_retrieval" in r:
        dr = r["document_retrieval"]
        bits.append(f"NDCG={dr['ndcg']:.2f} Hit={dr['hit_at_k']:.0f}")
    if "groundedness" in r:
        bits.append(f"Grnd={r['groundedness']['score']}")
    if "relevance" in r:
        bits.append(f"Rel={r['relevance']['score']}")
    if "response_completeness" in r:
        bits.append(f"Comp={r['response_completeness']['score']}")
    if "hard_rules" in r:
        hr = r["hard_rules"]
        flags = (
            ("M" if hr["must_mention_passed"] else "m")
            + ("N" if hr["must_not_mention_passed"] else "n")
            + ("A" if hr["expected_acts_passed"] else "a")
        )
        bits.append(f"HR={flags}")
    if bits:
        print("      " + "  ".join(bits))


async def _run_one(entry: dict, k: int, skip_generation: bool, chain) -> dict:
    qid = entry["id"]
    query = entry["query"]
    is_oos = entry.get("is_out_of_scope", False)
    labels: dict[str, int] = entry.get("ground_truth", {}) or {}
    ref_answer: str | None = entry.get("ground_truth_answer")

    result: dict = {
        "id": qid,
        "query": query,
        "topic": entry.get("topic"),
        "query_type": entry.get("query_type"),
        "difficulty": entry.get("difficulty"),
        "law_domain": entry.get("law_domain"),
        "is_out_of_scope": is_oos,
    }

    with _tracer.start_as_current_span("eval.full.query") as span:
        span.set_attribute("eval.query_id", qid)
        span.set_attribute("eval.k", k)

        # ---- Retrieve ---------------------------------------------------
        t0 = time.perf_counter()
        try:
            chunks = await _retrieve(query, top_k=k)
        except Exception as exc:
            result["error"] = f"retrieval failed: {exc}"
            return result
        result["retrieval_latency_s"] = round(time.perf_counter() - t0, 3)
        retrieved_ids = [_chunk_id(c) for c in chunks]
        result["retrieved_chunk_ids"] = retrieved_ids
        result["retrieved_count"] = len(chunks)

        # ---- Document Retrieval (process eval) --------------------------
        if labels and not is_oos:
            dr = evaluate_document_retrieval(retrieved_ids, labels)
            result["document_retrieval"] = {
                "precision_at_k": dr.precision_at_k,
                "recall_at_k": dr.recall_at_k,
                "hit_at_k": dr.hit_at_k,
                "mrr": dr.mrr,
                "map_at_k": dr.map_at_k,
                "ndcg": dr.ndcg,
                "xdcg": dr.xdcg,
                "fidelity": dr.fidelity,
                "max_relevance": dr.max_relevance,
                "holes": dr.holes,
                "passed": dr.passed,
            }

        # ---- Retrieval LLM judge ---------------------------------------
        try:
            ret = evaluate_retrieval(query, chunks)
            result["retrieval_judge"] = {
                "score": ret.score, "passed": ret.passed, "reason": ret.reason,
            }
        except Exception as exc:
            result["retrieval_judge_error"] = str(exc)

        if skip_generation:
            return result

        # ---- Generation ------------------------------------------------
        t0 = time.perf_counter()
        try:
            response = "".join(chain.stream(query))
        except Exception as exc:
            result["error"] = f"generation failed: {exc}"
            return result
        result["generation_latency_s"] = round(time.perf_counter() - t0, 3)
        result["response"] = response

        # ---- Hard programmatic rules -----------------------------------
        result["hard_rules"] = _hard_rules(entry, response, chunks)

        # ---- Groundedness LLM judge ------------------------------------
        try:
            grd = evaluate_groundedness(response, chunks)
            result["groundedness"] = {
                "score": grd.score, "passed": grd.passed, "reason": grd.reason,
            }
        except Exception as exc:
            result["groundedness_error"] = str(exc)

        # ---- Relevance LLM judge ---------------------------------------
        try:
            rel = evaluate_relevance(query, response)
            result["relevance"] = {
                "score": rel.score, "passed": rel.passed, "reason": rel.reason,
            }
        except Exception as exc:
            result["relevance_error"] = str(exc)

        # ---- Response Completeness (process eval) ----------------------
        if ref_answer:
            try:
                rc = evaluate_response_completeness(response, ref_answer)
                result["response_completeness"] = {
                    "score": rc.score, "passed": rc.passed, "reason": rc.reason,
                }
            except Exception as exc:
                result["response_completeness_error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_NUMERIC_PATHS: list[tuple[str, ...]] = [
    ("document_retrieval", "precision_at_k"),
    ("document_retrieval", "recall_at_k"),
    ("document_retrieval", "hit_at_k"),
    ("document_retrieval", "mrr"),
    ("document_retrieval", "map_at_k"),
    ("document_retrieval", "ndcg"),
    ("document_retrieval", "xdcg"),
    ("document_retrieval", "fidelity"),
    ("retrieval_judge", "score"),
    ("groundedness", "score"),
    ("relevance", "score"),
    ("response_completeness", "score"),
    ("retrieval_latency_s",),
    ("generation_latency_s",),
]

_BOOLEAN_PATHS: list[tuple[str, ...]] = [
    ("document_retrieval", "passed"),
    ("document_retrieval", "holes"),
    ("retrieval_judge", "passed"),
    ("groundedness", "passed"),
    ("relevance", "passed"),
    ("response_completeness", "passed"),
    ("hard_rules", "must_mention_passed"),
    ("hard_rules", "must_not_mention_passed"),
    ("hard_rules", "expected_acts_passed"),
]


def _dig(d: dict, path: tuple[str, ...]):
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _aggregate(results: list[dict]) -> dict:
    agg: dict = {"count": len(results)}
    for path in _NUMERIC_PATHS:
        vals = [_dig(r, path) for r in results]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if vals:
            agg["_".join(path) + "_mean"] = round(statistics.mean(vals), 4)
            if len(vals) > 1:
                agg["_".join(path) + "_stdev"] = round(statistics.pstdev(vals), 4)
    for path in _BOOLEAN_PATHS:
        vals = [_dig(r, path) for r in results]
        vals = [v for v in vals if isinstance(v, bool)]
        if vals:
            agg["_".join(path) + "_rate"] = round(sum(vals) / len(vals), 4)
    return agg


def _aggregate_by(results: list[dict], field: str) -> dict[str, dict]:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        key = r.get(field) or "(unset)"
        by[key].append(r)
    return {k: _aggregate(v) for k, v in by.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full RAG evaluation suite over the dataset.",
    )
    parser.add_argument(
        "--dataset", type=Path, default=_DEFAULT_DATASET,
        help="Path to ground_truth.json (default: data/eval/ground_truth.json).",
    )
    parser.add_argument(
        "--only", action="append", default=None,
        help="Limit run to these query ids (repeatable).",
    )
    parser.add_argument(
        "--k", type=int, default=_DEFAULT_K,
        help=f"top-k retrieval (default: {_DEFAULT_K}).",
    )
    parser.add_argument(
        "--skip-generation", action="store_true",
        help="Run retrieval evaluators only (skip generation + LLM judges on it).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=_RESULTS_DIR,
        help="Directory to write run_<timestamp>.json into.",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}")
        print("Run `python -m src.eval.cli.build_eval_dataset` first.")
        sys.exit(1)

    entries = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.only:
        wanted = set(args.only)
        entries = [e for e in entries if e["id"] in wanted]
        if not entries:
            print(f"No entries match --only {args.only!r}")
            sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chain = None if args.skip_generation else build_chain()

    async def _run_all() -> list[dict]:
        out: list[dict] = []
        for i, entry in enumerate(entries, 1):
            print(f"  [{i:2d}/{len(entries)}] {entry['id']}: {entry['query'][:70]}…")
            r = await _run_one(entry, args.k, args.skip_generation, chain)
            out.append(r)
            _print_summary(r)
        return out

    results = asyncio.run(_run_all())

    # ---- Aggregate ------------------------------------------------------
    summary = {
        "global": _aggregate(results),
        "by_difficulty": _aggregate_by(results, "difficulty"),
        "by_query_type": _aggregate_by(results, "query_type"),
        "by_law_domain": _aggregate_by(results, "law_domain"),
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = {
        "run_timestamp_utc": timestamp,
        "dataset": str(args.dataset),
        "k": args.k,
        "skip_generation": args.skip_generation,
        "summary": summary,
        "results": results,
    }
    out_path = args.output_dir / f"run_{timestamp}.json"
    out_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    # Stable "latest" pointer for dashboards / CI.
    latest = args.output_dir / "latest.json"
    latest.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")

    print()
    print(f"Wrote per-run results : {out_path}")
    print(f"Wrote latest pointer  : {latest}")
    print()
    _print_global_summary_pct(summary)


# Metrics that are fractions in [0,1] — display as percent in the printout.
_PCT_METRIC_SUFFIXES = (
    "precision_at_k_mean", "recall_at_k_mean", "hit_at_k_mean",
    "mrr_mean", "map_at_k_mean", "ndcg_mean", "xdcg_mean", "fidelity_mean",
    "_passed_rate", "_holes_rate",
)


def _is_pct_metric(name: str) -> bool:
    return any(name.endswith(s) for s in _PCT_METRIC_SUFFIXES)


def _print_global_summary_pct(summary: dict) -> None:
    """Print the global summary; ratio metrics shown as %."""
    print("Global summary (ratio metrics shown as %):")
    for k, v in summary["global"].items():
        if isinstance(v, (int, float)) and _is_pct_metric(k):
            print(f"  {k:50s} {v * 100:6.2f}%")
        else:
            print(f"  {k:50s} {v}")

    for cohort_name in ("by_difficulty", "by_query_type", "by_law_domain"):
        print()
        print(f"{cohort_name}:")
        for label, agg in summary[cohort_name].items():
            n = agg.get("count", 0)
            ndcg = agg.get("document_retrieval_ndcg_mean")
            hit = agg.get("document_retrieval_hit_at_k_mean")
            prec = agg.get("document_retrieval_precision_at_k_mean")
            rec = agg.get("document_retrieval_recall_at_k_mean")
            judge = agg.get("retrieval_judge_score_mean")
            parts = [f"n={n:2d}"]
            if ndcg is not None:
                parts.append(f"NDCG={ndcg * 100:5.1f}%")
            if hit is not None:
                parts.append(f"Hit@k={hit * 100:5.1f}%")
            if prec is not None:
                parts.append(f"P@k={prec * 100:5.1f}%")
            if rec is not None:
                parts.append(f"R@k={rec * 100:5.1f}%")
            if judge is not None:
                parts.append(f"Judge={judge:.2f}/5")
            print(f"  {label:20s}  " + "  ".join(parts))


if __name__ == "__main__":
    main()
