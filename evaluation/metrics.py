
"""Reusable binary-relevance retrieval evaluation metrics."""

import math
from statistics import mean


DEFAULT_K_VALUES = (1, 3, 5, 10)


def prepare_inputs(retrieved_ids, relevant_ids, k):
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer.")

    retrieved = list(retrieved_ids)
    relevant = set(relevant_ids)

    if not relevant:
        raise ValueError(
            "Retrieval metrics require relevant ground-truth chunks."
        )

    if len(retrieved) != len(set(retrieved)):
        raise ValueError("Duplicate retrieved chunk IDs.")

    return retrieved[:k], relevant


def hit_rate_at_k(retrieved_ids, relevant_ids, k):
    retrieved, relevant = prepare_inputs(
        retrieved_ids, relevant_ids, k
    )

    return float(any(cid in relevant for cid in retrieved))


def precision_at_k(retrieved_ids, relevant_ids, k):
    retrieved, relevant = prepare_inputs(
        retrieved_ids, relevant_ids, k
    )

    return sum(cid in relevant for cid in retrieved) / k


def recall_at_k(retrieved_ids, relevant_ids, k):
    retrieved, relevant = prepare_inputs(
        retrieved_ids, relevant_ids, k
    )

    return sum(cid in relevant for cid in retrieved) / len(relevant)


def reciprocal_rank_at_k(retrieved_ids, relevant_ids, k):
    retrieved, relevant = prepare_inputs(
        retrieved_ids, relevant_ids, k
    )

    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(retrieved_ids, relevant_ids, k):
    retrieved, relevant = prepare_inputs(
        retrieved_ids, relevant_ids, k
    )

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, cid in enumerate(retrieved, start=1)
        if cid in relevant
    )

    ideal_hits = min(len(relevant), k)

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg


def evaluate_retrieval(
    retrieved_ids,
    relevant_ids,
    k_values=DEFAULT_K_VALUES
):
    retrieved_ids = list(retrieved_ids)
    relevant_ids = set(relevant_ids)

    if not relevant_ids:
        raise ValueError(
            "Unanswerable questions must be evaluated separately."
        )

    results = {}

    for k in k_values:
        results[f"hit_at_{k}"] = hit_rate_at_k(
            retrieved_ids, relevant_ids, k
        )

        results[f"precision_at_{k}"] = precision_at_k(
            retrieved_ids, relevant_ids, k
        )

        results[f"recall_at_{k}"] = recall_at_k(
            retrieved_ids, relevant_ids, k
        )

        results[f"rr_at_{k}"] = reciprocal_rank_at_k(
            retrieved_ids, relevant_ids, k
        )

        results[f"ndcg_at_{k}"] = ndcg_at_k(
            retrieved_ids, relevant_ids, k
        )

    return results


def aggregate_metrics(question_results):
    if not question_results:
        raise ValueError("No question results to aggregate.")

    keys = set(question_results[0])

    if any(set(row) != keys for row in question_results):
        raise ValueError("Inconsistent metric keys.")

    summary = {}

    for key in sorted(keys):
        if key.startswith("rr_at_"):
            output_key = key.replace("rr_at_", "mrr_at_", 1)
        elif key.startswith("hit_at_"):
            output_key = key.replace("hit_at_", "hit_rate_at_", 1)
        else:
            output_key = key

        summary[output_key] = mean(
            row[key] for row in question_results
        )

    return summary
