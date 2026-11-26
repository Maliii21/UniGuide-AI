
"""Tests for thread-safe shared retrieval resources."""

import time

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import Mock, patch

import pytest

from src.shared_retrieval import SharedRetrievalEngine
from src.relevance_filter import DEFAULT_THRESHOLD


@pytest.fixture
def mock_engine():

    engine = Mock()

    engine.retrieve_with_relevance.return_value = {
        "relevance_passed": True,
        "results": [
            {
                "chunk_id": "AIU-001-P001-C001",
                "text": "University policy."
            }
        ],
        "context": "University policy."
    }

    return engine


def test_engine_initialization():

    with patch(
        "src.shared_retrieval.RetrievalEngine"
    ) as factory:

        SharedRetrievalEngine(device="cpu")

        factory.assert_called_once_with(device="cpu")


def test_dependency_injection(mock_engine):

    with patch(
        "src.shared_retrieval.RetrievalEngine"
    ) as factory:

        shared = SharedRetrievalEngine(
            retrieval_engine=mock_engine
        )

        shared.retrieve_with_relevance("Test question")

        factory.assert_not_called()


def test_default_parameters(mock_engine):

    shared = SharedRetrievalEngine(
        retrieval_engine=mock_engine
    )

    shared.retrieve_with_relevance("Test question")

    mock_engine.retrieve_with_relevance.assert_called_once_with(
        query="Test question",
        top_k=5,
        candidate_k=20,
        rrf_k=60,
        threshold=DEFAULT_THRESHOLD
    )


def test_custom_parameters(mock_engine):

    shared = SharedRetrievalEngine(
        retrieval_engine=mock_engine
    )

    shared.retrieve_with_relevance(
        query="Test question",
        top_k=3,
        candidate_k=10,
        rrf_k=30,
        threshold=-4.0
    )

    mock_engine.retrieve_with_relevance.assert_called_once_with(
        query="Test question",
        top_k=3,
        candidate_k=10,
        rrf_k=30,
        threshold=-4.0
    )


def test_result_preserved(mock_engine):

    shared = SharedRetrievalEngine(
        retrieval_engine=mock_engine
    )

    result = shared.retrieve_with_relevance(
        "Test question"
    )

    assert result == (
        mock_engine.retrieve_with_relevance.return_value
    )


def test_independent_results(mock_engine):

    shared = SharedRetrievalEngine(
        retrieval_engine=mock_engine
    )

    first = shared.retrieve_with_relevance("Question A")
    second = shared.retrieve_with_relevance("Question B")

    assert first is not second
    assert first["results"] is not second["results"]

    first["results"][0]["text"] = "Modified text"

    assert second["results"][0]["text"] == (
        "University policy."
    )

    assert mock_engine.retrieve_with_relevance.return_value[
        "results"
    ][0]["text"] == "University policy."


def test_engine_reused(mock_engine):

    shared = SharedRetrievalEngine(
        retrieval_engine=mock_engine
    )

    for i in range(5):

        shared.retrieve_with_relevance(
            f"Question {i}"
        )

    assert (
        mock_engine.retrieve_with_relevance.call_count == 5
    )


def test_concurrent_requests_are_serialized():

    engine = Mock()

    active_requests = 0
    max_active_requests = 0

    barrier = Barrier(5)

    def retrieve(query, **kwargs):

        nonlocal active_requests
        nonlocal max_active_requests

        active_requests += 1

        max_active_requests = max(
            max_active_requests,
            active_requests
        )

        try:

            time.sleep(0.03)

            return {"query": query}

        finally:

            active_requests -= 1

    engine.retrieve_with_relevance.side_effect = retrieve

    shared = SharedRetrievalEngine(
        retrieval_engine=engine
    )

    questions = [
        f"Question {i}" for i in range(5)
    ]

    def worker(question):

        barrier.wait(timeout=5)

        return shared.retrieve_with_relevance(question)

    with ThreadPoolExecutor(max_workers=5) as executor:

        results = list(
            executor.map(worker, questions)
        )

    assert max_active_requests == 1
    assert active_requests == 0

    assert [r["query"] for r in results] == questions

    assert engine.retrieve_with_relevance.call_count == 5


def test_lock_released_after_exception(mock_engine):

    mock_engine.retrieve_with_relevance.side_effect = [
        RuntimeError("Retrieval failed"),
        {"context": "Recovered successfully"}
    ]

    shared = SharedRetrievalEngine(
        retrieval_engine=mock_engine
    )

    with pytest.raises(
        RuntimeError,
        match="Retrieval failed"
    ):

        shared.retrieve_with_relevance("Question A")

    result = shared.retrieve_with_relevance("Question B")

    assert result["context"] == "Recovered successfully"
