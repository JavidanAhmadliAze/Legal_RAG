"""
Document Retrieval evaluator — process evaluation with ground-truth labels.

Modern RAG-retrieval metrics
----------------------------
Precision@k     : fraction of the top-k that are relevant.
Recall@k        : fraction of all relevant docs that appear in the top-k.
                  (a.k.a. "fidelity" — kept under both names for back-compat.)
Hit@k           : 1.0 if at least one relevant doc is in top-k, else 0.0.
                  (Also called Success@k.)
MRR@k           : 1 / rank of the first relevant doc in top-k (0 otherwise).
MAP@k           : Mean Average Precision — average of Precision@i over the
                  ranks i where a relevant doc appears, divided by min(R, k).
NDCG@k          : graded gain (linear), normalised by ideal ordering.
XDCG@k          : graded gain (exponential, 2^rel − 1), normalised.
Max Relevance   : highest graded label present in the retrieved set.
Holes           : True when no relevant doc was retrieved at all.

Why each metric matters for RAG
-------------------------------
Precision@k    — top-k goes to the LLM as context; noise hurts grounding.
Recall@k       — if the relevant chunk is missing, the LLM can't ground.
Hit@k          — minimum bar: was anything useful surfaced at all?
MRR@k          — penalises burying the right answer at rank N.
MAP@k          — single-number summary that respects rank for multi-relevant queries.
NDCG / XDCG    — the only metrics that respect graded relevance (1–3 scale).

Ground-truth format
-------------------
``ground_truth`` maps doc-id → integer relevance on a 0–3 scale
(0 = not relevant, 1 = marginally, 2 = relevant, 3 = highly).
Documents absent from the mapping are treated as relevance 0.

Pass/Fail
---------
Passes when ``NDCG@k ≥ ndcg_threshold`` AND ``Holes == False``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DocumentRetrievalResult:
    # Set-based metrics
    precision_at_k: float
    recall_at_k: float
    hit_at_k: float
    fidelity: float          # alias for recall over full retrieved set (back-compat)

    # Rank-aware metrics
    mrr: float
    map_at_k: float
    ndcg: float
    xdcg: float

    # Diagnostics
    max_relevance: float
    holes: bool
    passed: bool


def evaluate_document_retrieval(
    retrieved_ids: list[str],
    ground_truth: dict[str, int],
    *,
    k: int | None = None,
    ndcg_threshold: float = 0.5,
) -> DocumentRetrievalResult:
    """
    Args:
        retrieved_ids:  Ordered list of document/chunk IDs from retrieval.
        ground_truth:   Mapping {doc_id: relevance 0-3}.
        k:              Cutoff rank. If None, defaults to len(retrieved_ids).
        ndcg_threshold: Minimum NDCG@k for a Pass verdict.
    """
    # Dedupe retrieved IDs while preserving rank order — multiple chunks from
    # the same (document_id, page_num) collapse to one. Without this step,
    # duplicates inflate Recall, MAP, and NDCG above 1.0.
    seen: set[str] = set()
    retrieved_unique: list[str] = []
    for doc_id in retrieved_ids:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        retrieved_unique.append(doc_id)

    if k is None:
        k = len(retrieved_unique)
    top_k = retrieved_unique[:k]

    relevant_ids = {doc_id for doc_id, rel in ground_truth.items() if rel > 0}
    n_relevant = len(relevant_ids)

    # Per-rank graded relevances (in retrieved order)
    rels_at_k = [ground_truth.get(doc_id, 0) for doc_id in top_k]
    n_hits_at_k = sum(1 for r in rels_at_k if r > 0)

    # ----- Precision@k -------------------------------------------------------
    precision_at_k = n_hits_at_k / k if k > 0 else 0.0

    # ----- Recall@k / Fidelity ----------------------------------------------
    recall_at_k = n_hits_at_k / n_relevant if n_relevant else 1.0
    fidelity = recall_at_k  # alias kept for back-compat

    # ----- Hit@k (Success@k) -------------------------------------------------
    hit_at_k = 1.0 if n_hits_at_k > 0 else 0.0

    # ----- MRR@k -------------------------------------------------------------
    mrr = 0.0
    for rank, rel in enumerate(rels_at_k):
        if rel > 0:
            mrr = 1.0 / (rank + 1)
            break

    # ----- MAP@k -------------------------------------------------------------
    # AP@k = (1/min(R,k)) * Σ_{i where rel_i>0} Precision@i
    if n_relevant == 0:
        map_at_k = 1.0
    else:
        running_hits = 0
        precision_sum = 0.0
        for i, rel in enumerate(rels_at_k, start=1):
            if rel > 0:
                running_hits += 1
                precision_sum += running_hits / i
        denom = min(n_relevant, k)
        map_at_k = precision_sum / denom if denom else 0.0

    # ----- NDCG@k / XDCG@k ---------------------------------------------------
    def _dcg(rels: list[int], *, exponential: bool) -> float:
        total = 0.0
        for rank, rel in enumerate(rels):
            gain = (2**rel - 1) if exponential else rel
            total += gain / math.log2(rank + 2)
        return total

    ideal_rels = sorted(ground_truth.values(), reverse=True)[:k]

    dcg = _dcg(rels_at_k, exponential=False)
    idcg = _dcg(ideal_rels, exponential=False)
    ndcg = dcg / idcg if idcg > 0 else 0.0

    xdcg_raw = _dcg(rels_at_k, exponential=True)
    ixdcg = _dcg(ideal_rels, exponential=True)
    xdcg = xdcg_raw / ixdcg if ixdcg > 0 else 0.0

    # ----- Diagnostics -------------------------------------------------------
    max_relevance = float(max(rels_at_k, default=0))
    holes = max_relevance == 0
    passed = (ndcg >= ndcg_threshold) and (not holes)

    return DocumentRetrievalResult(
        precision_at_k=round(precision_at_k, 4),
        recall_at_k=round(recall_at_k, 4),
        hit_at_k=round(hit_at_k, 4),
        fidelity=round(fidelity, 4),
        mrr=round(mrr, 4),
        map_at_k=round(map_at_k, 4),
        ndcg=round(ndcg, 4),
        xdcg=round(xdcg, 4),
        max_relevance=max_relevance,
        holes=holes,
        passed=passed,
    )
