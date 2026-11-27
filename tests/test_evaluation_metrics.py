
import math

import pytest

from evaluation.metrics import (
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
    ndcg_at_k,
    evaluate_retrieval,
    aggregate_metrics,
)


RETRIEVED = ["A", "B", "C", "D", "E"]
RELEVANT = ["B", "D", "F"]


def test_hit_rate():
    assert hit_rate_at_k(RETRIEVED, RELEVANT, 1) == 0
    assert hit_rate_at_k(RETRIEVED, RELEVANT, 5) == 1


def test_precision():
    assert precision_at_k(
        RETRIEVED, RELEVANT, 5
    ) == pytest.approx(2 / 5)


def test_recall():
    assert recall_at_k(
        RETRIEVED, RELEVANT, 5
    ) == pytest.approx(2 / 3)


def test_reciprocal_rank():
    assert reciprocal_rank_at_k(
        RETRIEVED, RELEVANT, 5
    ) == pytest.approx(0.5)


def test_ndcg():
    expected = (
        1 / math.log2(3) + 1 / math.log2(5)
    ) / (
        1 + 1 / math.log2(3) + 1 / math.log2(4)
    )

    assert ndcg_at_k(
        RETRIEVED, RELEVANT, 5
    ) == pytest.approx(expected)


def test_perfect_retrieval():
    results = evaluate_retrieval(
        ["A", "B"], ["A", "B"], (2,)
    )

    assert all(value == 1.0 for value in results.values())


def test_no_relevant_retrieval():
    results = evaluate_retrieval(
        ["A", "B"], ["C"], (2,)
    )

    assert all(value == 0.0 for value in results.values())


def test_empty_retrieval():
    results = evaluate_retrieval([], ["A"], (5,))

    assert all(value == 0.0 for value in results.values())


def test_unanswerable_question():
    with pytest.raises(ValueError):
        evaluate_retrieval(["A"], [], (1,))


def test_duplicate_retrieved_ids():
    with pytest.raises(ValueError):
        evaluate_retrieval(["A", "A"], ["A"], (2,))


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_invalid_k(k):
    with pytest.raises(ValueError):
        evaluate_retrieval(["A"], ["A"], (k,))


def test_short_retrieval_precision():
    assert precision_at_k(
        ["A"], ["A"], 5
    ) == pytest.approx(0.2)


def test_aggregate_metrics():
    first = evaluate_retrieval(
        ["A"], ["A"], (1,)
    )

    second = evaluate_retrieval(
        ["B"], ["A"], (1,)
    )

    summary = aggregate_metrics([first, second])

    assert summary["hit_rate_at_1"] == 0.5
    assert summary["precision_at_1"] == 0.5
    assert summary["recall_at_1"] == 0.5
    assert summary["mrr_at_1"] == 0.5
    assert summary["ndcg_at_1"] == 0.5


def test_empty_aggregation():
    with pytest.raises(ValueError):
        aggregate_metrics([])


def test_inconsistent_aggregation():
    with pytest.raises(ValueError):
        aggregate_metrics([
            {"hit_at_1": 1.0},
            {"hit_at_5": 1.0}
        ])
