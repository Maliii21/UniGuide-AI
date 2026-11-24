
import math


DEFAULT_THRESHOLD = -5.0


def check_relevance(
    results,
    threshold=DEFAULT_THRESHOLD
):
    """
    Check query-level retrieval relevance.

    This is a heuristic relevance gate,
    not a guarantee of answerability.
    """

    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
    ):
        raise ValueError(
            "Threshold must be a finite number."
        )

    if not results:

        return {
            "passed": False,
            "max_score": None,
            "threshold": float(threshold),
            "reason": "no_results",
            "results": []
        }

    scores = []

    for result in results:

        if "reranker_score" not in result:
            raise ValueError(
                "Reranker score is required."
            )

        score = result["reranker_score"]

        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise ValueError(
                "Invalid reranker score."
            )

        scores.append(float(score))

    max_score = max(scores)

    passed = max_score >= threshold

    return {
        "passed": passed,
        "max_score": max_score,
        "threshold": float(threshold),
        "reason": (
            "relevance_threshold_passed"
            if passed
            else "low_relevance"
        ),
        "results": (
            [result.copy() for result in results]
            if passed
            else []
        )
    }


def main():

    # Test 1: Preserve low-scoring supporting chunks
    results = [
        {
            "chunk_id": "A",
            "reranker_score": 2.392507
        },
        {
            "chunk_id": "B",
            "reranker_score": -2.896008
        },
        {
            "chunk_id": "C",
            "reranker_score": -6.0
        }
    ]

    decision = check_relevance(results)

    assert decision["passed"] is True
    assert len(decision["results"]) == 3

    print("Test 1: Relevant context retained - PASSED")

    # Test 2: Reject low-relevance context
    results = [
        {
            "chunk_id": "D",
            "reranker_score": -11.118126
        },
        {
            "chunk_id": "E",
            "reranker_score": -11.126480
        }
    ]

    decision = check_relevance(results)

    assert decision["passed"] is False
    assert decision["results"] == []

    print("Test 2: Low-relevance context rejected - PASSED")

    # Test 3: Negative but acceptable score
    results = [
        {
            "chunk_id": "F",
            "reranker_score": -2.922903
        }
    ]

    decision = check_relevance(results)

    assert decision["passed"] is True

    print("Test 3: Negative relevant score accepted - PASSED")

    # Test 4: Empty retrieval results
    decision = check_relevance([])

    assert decision["passed"] is False
    assert decision["max_score"] is None
    assert decision["reason"] == "no_results"

    print("Test 4: Empty results handled - PASSED")

    # Test 5: Configurable threshold
    decision = check_relevance(
        results,
        threshold=0.0
    )

    assert decision["passed"] is False

    print("Test 5: Configurable threshold - PASSED")

    # Test 6: Threshold boundary
    decision = check_relevance([
        {
            "chunk_id": "G",
            "reranker_score": -5.0
        }
    ])

    assert decision["passed"] is True

    print("Test 6: Threshold boundary - PASSED")

    # Test 7: Missing score
    try:

        check_relevance([
            {"chunk_id": "H"}
        ])

    except ValueError:

        print("Test 7: Missing score rejected - PASSED")

    else:

        raise AssertionError(
            "Missing score should raise ValueError."
        )

    # Test 8: Invalid score
    try:

        check_relevance([
            {
                "chunk_id": "I",
                "reranker_score": float("nan")
            }
        ])

    except ValueError:

        print("Test 8: Invalid score rejected - PASSED")

    else:

        raise AssertionError(
            "Invalid score should raise ValueError."
        )

    print("\nAll 8 relevance filter tests passed.")


if __name__ == "__main__":
    main()
