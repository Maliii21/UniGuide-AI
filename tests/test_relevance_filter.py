
import math

import pytest

from src.relevance_filter import (
    check_relevance,
    DEFAULT_THRESHOLD,
)


# ==========================================
# Test Fixtures
# ==========================================

@pytest.fixture
def sample_results():

    return [
        {
            "chunk_id": "C001",
            "text": "Course registration policy.",
            "document_title": "Registration",
            "page": 1,
            "reranker_score": 4.2
        },
        {
            "chunk_id": "C002",
            "text": "Course withdrawal policy.",
            "document_title": "Withdrawal",
            "page": 2,
            "reranker_score": -6.5
        },
        {
            "chunk_id": "C003",
            "text": "Attendance requirements.",
            "document_title": "Attendance",
            "page": 3,
            "reranker_score": -8.0
        }
    ]


# ==========================================
# 1. Default Threshold Tests
# ==========================================

def test_default_threshold():

    assert DEFAULT_THRESHOLD == -5.0


def test_default_threshold_used(sample_results):

    decision = check_relevance(sample_results)

    assert decision["threshold"] == -5.0


# ==========================================
# 2. Acceptance and Rejection Tests
# ==========================================

def test_relevant_results_accepted(sample_results):

    decision = check_relevance(sample_results)

    assert decision["passed"] is True

    assert decision["reason"] == (
        "relevance_threshold_passed"
    )


def test_irrelevant_results_rejected():

    results = [
        {"chunk_id": "C001", "reranker_score": -8.0},
        {"chunk_id": "C002", "reranker_score": -10.0}
    ]

    decision = check_relevance(results)

    assert decision["passed"] is False
    assert decision["reason"] == "low_relevance"
    assert decision["results"] == []
    assert decision["max_score"] == -8.0


def test_maximum_score(sample_results):

    decision = check_relevance(sample_results)

    assert decision["max_score"] == pytest.approx(4.2)


def test_maximum_score_not_first():

    results = [
        {"chunk_id": "C001", "reranker_score": -8.0},
        {"chunk_id": "C002", "reranker_score": 3.0}
    ]

    decision = check_relevance(results)

    assert decision["passed"] is True
    assert decision["max_score"] == 3.0


# ==========================================
# 3. Threshold Boundary Tests
# ==========================================

@pytest.mark.parametrize(
    "score, expected",
    [
        (-4.999, True),
        (-5.0, True),
        (-5.001, False)
    ]
)
def test_threshold_boundary(score, expected):

    results = [
        {
            "chunk_id": "C001",
            "reranker_score": score
        }
    ]

    decision = check_relevance(results)

    assert decision["passed"] is expected


def test_custom_threshold_accepts(sample_results):

    decision = check_relevance(
        sample_results,
        threshold=4.0
    )

    assert decision["passed"] is True
    assert decision["threshold"] == 4.0


def test_custom_threshold_rejects(sample_results):

    decision = check_relevance(
        sample_results,
        threshold=5.0
    )

    assert decision["passed"] is False
    assert decision["results"] == []


# ==========================================
# 4. Query-Level Filtering Tests
# ==========================================

def test_all_results_retained_when_accepted(
    sample_results
):

    decision = check_relevance(sample_results)

    assert len(decision["results"]) == 3

    # Low-scoring chunks are also retained
    # when the query-level gate passes.
    assert decision["results"][1]["reranker_score"] == -6.5
    assert decision["results"][2]["reranker_score"] == -8.0


def test_result_order_preserved(sample_results):

    decision = check_relevance(sample_results)

    assert [
        item["chunk_id"]
        for item in decision["results"]
    ] == ["C001", "C002", "C003"]


def test_empty_results():

    decision = check_relevance([])

    assert decision["passed"] is False
    assert decision["max_score"] is None
    assert decision["reason"] == "no_results"
    assert decision["results"] == []


# ==========================================
# 5. Score Validation Tests
# ==========================================

def test_missing_reranker_score():

    results = [
        {"chunk_id": "C001"}
    ]

    with pytest.raises(ValueError):

        check_relevance(results)


@pytest.mark.parametrize(
    "invalid_score",
    [
        math.nan,
        math.inf,
        -math.inf,
        None,
        "4.2",
        True,
        False
    ]
)
def test_invalid_reranker_score(invalid_score):

    results = [
        {
            "chunk_id": "C001",
            "reranker_score": invalid_score
        }
    ]

    with pytest.raises(ValueError):

        check_relevance(results)


def test_invalid_score_among_valid_results():

    results = [
        {"chunk_id": "C001", "reranker_score": 5.0},
        {"chunk_id": "C002", "reranker_score": math.nan}
    ]

    with pytest.raises(ValueError):

        check_relevance(results)


# ==========================================
# 6. Threshold Validation Tests
# ==========================================

@pytest.mark.parametrize(
    "invalid_threshold",
    [
        math.nan,
        math.inf,
        -math.inf,
        None,
        "-5",
        True,
        False
    ]
)
def test_invalid_threshold(
    sample_results,
    invalid_threshold
):

    with pytest.raises(ValueError):

        check_relevance(
            sample_results,
            threshold=invalid_threshold
        )


def test_integer_threshold(sample_results):

    decision = check_relevance(
        sample_results,
        threshold=-5
    )

    assert decision["passed"] is True
    assert isinstance(decision["threshold"], float)


# ==========================================
# 7. Metadata Preservation Tests
# ==========================================

def test_metadata_preservation(sample_results):

    decision = check_relevance(sample_results)

    assert decision["results"] == sample_results


def test_original_results_not_modified(sample_results):

    original = [
        item.copy()
        for item in sample_results
    ]

    check_relevance(sample_results)

    assert sample_results == original


def test_returned_results_are_copies(sample_results):

    decision = check_relevance(sample_results)

    decision["results"][0]["text"] = "Modified text"

    assert sample_results[0]["text"] == (
        "Course registration policy."
    )


# ==========================================
# 8. Output Contract Test
# ==========================================

def test_output_structure(sample_results):

    decision = check_relevance(sample_results)

    assert set(decision.keys()) == {
        "passed",
        "max_score",
        "threshold",
        "reason",
        "results"
    }
