"""
Document Retrieval evaluator — process evaluation with ground-truth labels.

Metrics
-------
Fidelity     : fraction of relevant documents that were retrieved (recall).
NDCG         : normalised discounted cumulative gain (linear gain).
XDCG         : extended DCG using exponential gain  (2^rel − 1).
Max Relevance: highest relevance score present in the retrieved set.
Holes        : True when no relevant document was retrieved at all.

Ground-truth format
-------------------
``ground_truth`` is a mapping from document-id to integer relevance on a
0-3 scale (0 = not relevant, 1 = marginally, 2 = relevant, 3 = highly).
Documents absent from the mapping are treated as relevance 0.

Pass/Fail
---------
Passes when NDCG ≥ ``ndcg_threshold`` AND Holes is False.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DocumentRetrievalResult:
    fidelity: float
    ndcg: float
    xdcg: float
    mrr: float
    max_relevance: float
    holes: bool
    passed: bool


def evaluate_document_retrieval(
    retrieved_ids: list[str],
    ground_truth: dict[str, int],
    *,
    ndcg_threshold: float = 0.5,
) -> DocumentRetrievalResult:
    """
    Args:
        retrieved_ids: Ordered list of document/chunk IDs returned by retrieval.
        ground_truth:  Mapping {doc_id: relevance 0-3}.
        ndcg_threshold: Minimum NDCG for a Pass verdict.
    """
    k = len(retrieved_ids)
    relevant_ids = {doc_id for doc_id, rel in ground_truth.items() if rel > 0}

    # --- Fidelity (recall) ---------------------------------------------------
    retrieved_set = set(retrieved_ids)
    fidelity = (
        len(retrieved_set & relevant_ids) / len(relevant_ids)
        if relevant_ids
        else 1.0
    )

    # --- Max Relevance -------------------------------------------------------
    rels_in_retrieved = [ground_truth.get(doc_id, 0) for doc_id in retrieved_ids]
    max_relevance = float(max(rels_in_retrieved, default=0))

    # --- Holes ---------------------------------------------------------------
    holes = max_relevance == 0

    # --- DCG helpers ---------------------------------------------------------
    def _dcg(rels: list[int], *, exponential: bool) -> float:
        total = 0.0
        for rank, rel in enumerate(rels):
            gain = (2**rel - 1) if exponential else rel
            total += gain / math.log2(rank + 2)
        return total

    ideal_rels = sorted(ground_truth.values(), reverse=True)[:k]

    # --- NDCG ----------------------------------------------------------------
    dcg = _dcg(rels_in_retrieved, exponential=False)
    idcg = _dcg(ideal_rels, exponential=False)
    ndcg = dcg / idcg if idcg > 0 else 0.0

    # --- XDCG ----------------------------------------------------------------
    xdcg_val = _dcg(rels_in_retrieved, exponential=True)
    ixdcg = _dcg(ideal_rels, exponential=True)
    xdcg = xdcg_val / ixdcg if ixdcg > 0 else 0.0

    # --- MRR -----------------------------------------------------------------
    mrr = 0.0
    for rank, doc_id in enumerate(retrieved_ids):
        if ground_truth.get(doc_id, 0) > 0:
            mrr = 1.0 / (rank + 1)
            break

    passed = (ndcg >= ndcg_threshold) and (not holes)

    return DocumentRetrievalResult(
        fidelity=round(fidelity, 4),
        ndcg=round(ndcg, 4),
        xdcg=round(xdcg, 4),
        mrr=round(mrr, 4),
        max_relevance=max_relevance,
        holes=holes,
        passed=passed,
    )
