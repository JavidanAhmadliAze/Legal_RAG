#!/usr/bin/env python3
"""
RAG evaluation runner.

Usage
-----
# Evaluate without ground truth (Retrieval, Groundedness, Relevance):
    python -m src.eval.cli.evaluate --query "Who can apply for Karta Polaka?"

# Full evaluation with ground truth labels (all five evaluators):
    python -m src.eval.cli.evaluate \
        --query "Who can apply for Karta Polaka?" \
        --ground-truth-answer "Polish-origin foreigners who meet..." \
        --ground-truth-labels '{"WDU20220001264:12": 3, "WDU20220001421:5": 2}'

Ground-truth labels format: JSON mapping "doc_id:page_num" (or any chunk id)
to relevance 0-3.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dotenv import load_dotenv  # type: ignore[import-untyped]

load_dotenv()

from src.services.monitoring import init as init_tracing

init_tracing(project_name="legal-rag-eval", port=6006)

from src.eval import (
    evaluate_document_retrieval,
    evaluate_groundedness,
    evaluate_relevance,
    evaluate_response_completeness,
    evaluate_retrieval,
)
from opentelemetry import trace

import asyncio

from src.services.agents.chain import build_chain
from src.services.agents.retriever import aretrieve

_tracer = trace.get_tracer("legal-rag.evaluator")


def _chunk_id(chunk: dict) -> str:
    return f"{chunk.get('document_id', '')}:{chunk.get('page_num', '')}"


def run(
    query: str,
    ground_truth_answer: str | None,
    ground_truth_labels: dict[str, int] | None,
) -> None:
    print(f"\nQuery: {query}\n{'=' * 60}")

    with _tracer.start_as_current_span("evaluation") as eval_span:
        eval_span.set_attribute("eval.query", query)

        # --- Retrieval -------------------------------------------------------
        chunks = asyncio.run(aretrieve(query, top_k=5))
        retrieved_ids = [_chunk_id(c) for c in chunks]

        # 1. Document Retrieval (only when ground-truth labels provided)
        if ground_truth_labels:
            with _tracer.start_as_current_span("eval.document_retrieval") as span:
                dr = evaluate_document_retrieval(retrieved_ids, ground_truth_labels)
                span.set_attribute("eval.precision_at_k", dr.precision_at_k)
                span.set_attribute("eval.recall_at_k", dr.recall_at_k)
                span.set_attribute("eval.hit_at_k", dr.hit_at_k)
                span.set_attribute("eval.map_at_k", dr.map_at_k)
                span.set_attribute("eval.mrr", dr.mrr)
                span.set_attribute("eval.ndcg", dr.ndcg)
                span.set_attribute("eval.xdcg", dr.xdcg)
                span.set_attribute("eval.fidelity", dr.fidelity)
                span.set_attribute("eval.max_relevance", dr.max_relevance)
                span.set_attribute("eval.holes", dr.holes)
                span.set_attribute("eval.passed", dr.passed)
            k = len(retrieved_ids)
            print(
                f"\n[Document Retrieval]  {'PASS' if dr.passed else 'FAIL'}  (k={k})\n"
                f"  Precision@k  : {dr.precision_at_k:.4f}\n"
                f"  Recall@k     : {dr.recall_at_k:.4f}\n"
                f"  Hit@k        : {dr.hit_at_k:.4f}\n"
                f"  MAP@k        : {dr.map_at_k:.4f}\n"
                f"  MRR@k        : {dr.mrr:.4f}\n"
                f"  NDCG@k       : {dr.ndcg:.4f}\n"
                f"  XDCG@k       : {dr.xdcg:.4f}\n"
                f"  Max Relevance: {dr.max_relevance}\n"
                f"  Holes        : {dr.holes}"
            )

        # 2. Retrieval quality (LLM judge)
        with _tracer.start_as_current_span("eval.retrieval") as span:
            ret = evaluate_retrieval(query, chunks)
            span.set_attribute("eval.score", ret.score)
            span.set_attribute("eval.passed", ret.passed)
            span.set_attribute("eval.reason", ret.reason)
        print(
            f"\n[Retrieval]  {'PASS' if ret.passed else 'FAIL'}  (score {ret.score}/5)\n"
            f"  {ret.reason.strip()}"
        )

        # --- Generation ------------------------------------------------------
        chain = build_chain()
        response = "".join(chain.stream(query))
        print(f"\nGenerated response:\n{response}\n{'-' * 60}")

        # 3. Groundedness (LLM judge)
        with _tracer.start_as_current_span("eval.groundedness") as span:
            grd = evaluate_groundedness(response, chunks)
            span.set_attribute("eval.score", grd.score)
            span.set_attribute("eval.passed", grd.passed)
            span.set_attribute("eval.reason", grd.reason)
        print(
            f"\n[Groundedness]  {'PASS' if grd.passed else 'FAIL'}  (score {grd.score}/5)\n"
            f"  {grd.reason.strip()}"
        )

        # 4. Relevance (LLM judge)
        with _tracer.start_as_current_span("eval.relevance") as span:
            rel = evaluate_relevance(query, response)
            span.set_attribute("eval.score", rel.score)
            span.set_attribute("eval.passed", rel.passed)
            span.set_attribute("eval.reason", rel.reason)
        print(
            f"\n[Relevance]  {'PASS' if rel.passed else 'FAIL'}  (score {rel.score}/5)\n"
            f"  {rel.reason.strip()}"
        )

        # 5. Response Completeness (only when ground-truth answer provided)
        if ground_truth_answer:
            with _tracer.start_as_current_span("eval.response_completeness") as span:
                rc = evaluate_response_completeness(response, ground_truth_answer)
                span.set_attribute("eval.score", rc.score)
                span.set_attribute("eval.passed", rc.passed)
                span.set_attribute("eval.reason", rc.reason)
            print(
                f"\n[Response Completeness]  {'PASS' if rc.passed else 'FAIL'}  (score {rc.score}/5)\n"
                f"  {rc.reason.strip()}"
            )

        eval_span.set_attribute("eval.retrieval_score", ret.score)
        eval_span.set_attribute("eval.groundedness_score", grd.score)
        eval_span.set_attribute("eval.relevance_score", rel.score)

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluators.")
    parser.add_argument("--query", required=True, help="Question to evaluate.")
    parser.add_argument(
        "--ground-truth-answer",
        default=None,
        help="Reference answer text (enables Response Completeness evaluator).",
    )
    parser.add_argument(
        "--ground-truth-labels",
        default=None,
        help=(
            'JSON mapping chunk-id to relevance 0-3, e.g. \'{"WDU2022:12": 3}\'. '
            "Enables Document Retrieval evaluator."
        ),
    )
    args = parser.parse_args()

    labels: dict[str, int] | None = None
    if args.ground_truth_labels:
        labels = json.loads(args.ground_truth_labels)

    run(args.query, args.ground_truth_answer, labels)


if __name__ == "__main__":
    main()
