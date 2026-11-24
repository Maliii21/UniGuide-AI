
from unittest.mock import Mock

import numpy as np
import pytest

import src.retrieval as retrieval_module

from src.retrieval import RetrievalEngine


# ==========================================
# Test Fixtures
# ==========================================

@pytest.fixture
def sample_metadata():

    return [
        {
            "chunk_id": "C001",
            "text": "Students must register for courses.",
            "document_title": "Registration Policy",
            "page": 1,
            "source_url": "https://example.com/registration",
            "document_date": "2026",
            "version_notes": "Current version",
            "review_notes": "Verified"
        },
        {
            "chunk_id": "C002",
            "text": "Students may withdraw from courses.",
            "document_title": "Withdrawal Policy",
            "page": 2,
            "source_url": "https://example.com/withdrawal",
            "document_date": "2026",
            "version_notes": "",
            "review_notes": ""
        }
    ]


@pytest.fixture
def dependencies(monkeypatch, sample_metadata):

    embeddings = np.array(
        [[1, 0, 0], [0, 1, 0]],
        dtype=np.float32
    )

    config = {
        "model_name": "test-embedding-model",
        "embedding_dimension": 3
    }

    index = Mock()
    index.d = 3
    index.ntotal = 2

    model = Mock()
    model.get_embedding_dimension.return_value = 3

    load_data = Mock(
        return_value=(
            embeddings,
            sample_metadata,
            config
        )
    )

    load_index = Mock(return_value=index)
    load_model = Mock(return_value=model)

    monkeypatch.setattr(
        retrieval_module,
        "load_embedding_data",
        load_data
    )

    monkeypatch.setattr(
        retrieval_module,
        "load_index",
        load_index
    )

    monkeypatch.setattr(
        retrieval_module,
        "load_model",
        load_model
    )

    return {
        "index": index,
        "model": model,
        "config": config,
        "load_data": load_data,
        "load_index": load_index,
        "load_model": load_model
    }


@pytest.fixture
def engine(dependencies):

    return RetrievalEngine(device="cpu")


@pytest.fixture
def ranked_results(sample_metadata):

    return [
        {
            **sample_metadata[0],
            "rank": 1,
            "reranker_score": 4.2
        },
        {
            **sample_metadata[1],
            "rank": 2,
            "reranker_score": -6.0
        }
    ]


# ==========================================
# 1. Initialization Tests
# ==========================================

def test_engine_initialization(
    engine,
    dependencies,
    sample_metadata
):

    assert engine.device == "cpu"
    assert engine.metadata == sample_metadata
    assert engine.index is dependencies["index"]
    assert engine.model is dependencies["model"]
    assert engine.config == dependencies["config"]

    assert engine.keyword_engine.num_documents == 2


def test_dependencies_loaded(
    engine,
    dependencies
):

    dependencies["load_data"].assert_called_once_with()

    dependencies["load_index"].assert_called_once_with()

    dependencies["load_model"].assert_called_once_with(
        model_name="test-embedding-model",
        device="cpu"
    )


def test_model_index_dimension_mismatch(dependencies):

    dependencies[
        "model"
    ].get_embedding_dimension.return_value = 768

    with pytest.raises(
        ValueError,
        match="dimensions do not match"
    ):

        RetrievalEngine()


def test_reranker_initially_not_loaded(engine):

    assert engine.reranker is None


# ==========================================
# 2. Semantic Search Tests
# ==========================================

def test_semantic_search(
    engine,
    monkeypatch,
    sample_metadata
):

    query_embedding = np.array(
        [1, 0, 0],
        dtype=np.float32
    )

    embed_query = Mock(
        return_value=query_embedding
    )

    expected = [
        {
            **sample_metadata[0],
            "similarity_score": 1.0
        }
    ]

    vector_search = Mock(return_value=expected)

    monkeypatch.setattr(
        retrieval_module,
        "embed_query",
        embed_query
    )

    monkeypatch.setattr(
        retrieval_module,
        "search",
        vector_search
    )

    results = engine.semantic_search(
        "Course registration",
        top_k=1
    )

    embed_query.assert_called_once_with(
        "Course registration",
        engine.model
    )

    kwargs = vector_search.call_args.kwargs

    assert kwargs["index"] is engine.index

    np.testing.assert_array_equal(
        kwargs["query_embedding"],
        query_embedding
    )

    assert kwargs["metadata"] == sample_metadata
    assert kwargs["top_k"] == 1

    assert vector_search.call_count == 1
    assert results == expected


# ==========================================
# 3. Keyword Search Tests
# ==========================================

def test_keyword_search(engine):

    results = engine.keyword_search(
        "registration",
        top_k=2
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "C001"
    assert results[0]["bm25_score"] > 0


def test_keyword_search_no_matches(engine):

    results = engine.keyword_search(
        "zzzxxyy",
        top_k=5
    )

    assert results == []


# ==========================================
# 4. Hybrid Search Tests
# ==========================================

def test_hybrid_search(
    engine,
    sample_metadata
):

    semantic_results = [
        {
            **sample_metadata[0],
            "similarity_score": 0.9
        },
        {
            **sample_metadata[1],
            "similarity_score": 0.8
        }
    ]

    keyword_results = [
        {
            **sample_metadata[1],
            "bm25_score": 5.0
        }
    ]

    engine.semantic_search = Mock(
        return_value=semantic_results
    )

    engine.keyword_search = Mock(
        return_value=keyword_results
    )

    results = engine.hybrid_search(
        "Course withdrawal",
        top_k=2,
        candidate_k=10,
        rrf_k=60
    )

    engine.semantic_search.assert_called_once_with(
        "Course withdrawal",
        top_k=10
    )

    engine.keyword_search.assert_called_once_with(
        "Course withdrawal",
        top_k=10
    )

    assert [
        item["chunk_id"] for item in results
    ] == ["C002", "C001"]

    assert results[0]["rrf_score"] == pytest.approx(
        1 / 62 + 1 / 61
    )


def test_hybrid_empty_results(engine):

    engine.semantic_search = Mock(return_value=[])
    engine.keyword_search = Mock(return_value=[])

    assert engine.hybrid_search("Unknown topic") == []


@pytest.mark.parametrize(
    "method",
    ["hybrid_search", "reranked_search"]
)
@pytest.mark.parametrize(
    "candidate_k",
    [0, -1, 1.5, True]
)
def test_invalid_candidate_k(
    engine,
    method,
    candidate_k
):

    with pytest.raises(ValueError):

        getattr(engine, method)(
            "Course registration",
            top_k=2,
            candidate_k=candidate_k
        )


@pytest.mark.parametrize(
    "method",
    ["hybrid_search", "reranked_search"]
)
def test_candidate_k_smaller_than_top_k(
    engine,
    method
):

    with pytest.raises(ValueError):

        getattr(engine, method)(
            "Course registration",
            top_k=5,
            candidate_k=2
        )


# ==========================================
# 5. Lazy Loading Tests
# ==========================================

def test_lazy_reranker_loading(
    engine,
    monkeypatch
):

    fake_reranker = Mock()

    constructor = Mock(
        return_value=fake_reranker
    )

    monkeypatch.setattr(
        retrieval_module,
        "Reranker",
        constructor
    )

    assert engine.reranker is None

    first = engine.load_reranker()
    second = engine.load_reranker()

    assert first is fake_reranker
    assert second is first

    constructor.assert_called_once_with(
        device="cpu",
        use_title=False
    )


def test_keyword_mode_does_not_load_reranker(engine):

    engine.retrieve(
        "registration",
        mode="keyword"
    )

    assert engine.reranker is None


# ==========================================
# 6. Reranked Search Tests
# ==========================================

def test_reranked_search(
    engine,
    sample_metadata,
    ranked_results
):

    candidates = [
        item.copy()
        for item in sample_metadata
    ]

    engine.hybrid_search = Mock(
        return_value=candidates
    )

    fake_reranker = Mock()

    fake_reranker.rerank.return_value = (
        ranked_results[:1]
    )

    engine.load_reranker = Mock(
        return_value=fake_reranker
    )

    results = engine.reranked_search(
        "Course registration",
        top_k=1,
        candidate_k=10,
        rrf_k=30
    )

    engine.hybrid_search.assert_called_once_with(
        query="Course registration",
        top_k=10,
        candidate_k=10,
        rrf_k=30
    )

    engine.load_reranker.assert_called_once_with()

    fake_reranker.rerank.assert_called_once_with(
        query="Course registration",
        results=candidates,
        top_k=1
    )

    assert results == ranked_results[:1]


def test_reranked_search_empty_candidates(engine):

    engine.hybrid_search = Mock(return_value=[])

    engine.load_reranker = Mock()

    results = engine.reranked_search(
        "Unknown topic"
    )

    assert results == []

    engine.load_reranker.assert_not_called()


# ==========================================
# 7. Retrieval Mode Tests
# ==========================================

@pytest.mark.parametrize(
    "mode, method",
    [
        ("semantic", "semantic_search"),
        ("keyword", "keyword_search"),
        ("hybrid", "hybrid_search"),
        ("reranked", "reranked_search")
    ]
)
def test_retrieval_mode_dispatch(
    engine,
    sample_metadata,
    mode,
    method
):

    mocked_search = Mock(
        return_value=[
            item.copy()
            for item in sample_metadata
        ]
    )

    setattr(engine, method, mocked_search)

    results = engine.retrieve(
        "  Course registration  ",
        top_k=2,
        mode=mode,
        candidate_k=10,
        rrf_k=30
    )

    if mode in ("semantic", "keyword"):

        mocked_search.assert_called_once_with(
            "Course registration",
            top_k=2
        )

    else:

        mocked_search.assert_called_once_with(
            "Course registration",
            top_k=2,
            candidate_k=10,
            rrf_k=30
        )

    assert [
        item["rank"] for item in results
    ] == [1, 2]

    assert all(
        item["retrieval_mode"] == mode
        for item in results
    )


def test_default_retrieval_mode(engine):

    engine.hybrid_search = Mock(return_value=[])

    engine.retrieve("Course registration")

    engine.hybrid_search.assert_called_once_with(
        "Course registration",
        top_k=5,
        candidate_k=20,
        rrf_k=60
    )


@pytest.mark.parametrize(
    "mode",
    ["unknown", "", None, "hybrid_rerank"]
)
def test_invalid_retrieval_mode(engine, mode):

    with pytest.raises(ValueError):

        engine.retrieve(
            "Course registration",
            mode=mode
        )


@pytest.mark.parametrize(
    "query",
    ["", "   ", None, 123]
)
def test_invalid_query(engine, query):

    with pytest.raises(ValueError):

        engine.retrieve(query)


@pytest.mark.parametrize(
    "top_k",
    [0, -1, 1.5, True]
)
def test_invalid_top_k(engine, top_k):

    with pytest.raises(ValueError):

        engine.retrieve(
            "Course registration",
            top_k=top_k
        )


def test_retrieve_empty_results(engine):

    engine.hybrid_search = Mock(return_value=[])

    assert engine.retrieve("Unknown topic") == []


# ==========================================
# 8. Context Building Tests
# ==========================================

def test_context_contains_source_metadata(
    engine,
    ranked_results
):

    context = engine.build_context(
        ranked_results
    )

    assert "[Source 1]" in context
    assert "[Source 2]" in context

    for result in ranked_results:

        assert result["chunk_id"] in context
        assert result["document_title"] in context
        assert result["source_url"] in context
        assert result["text"] in context

        assert f"Page: {result['page']}" in context

    assert "Document date: 2026" in context
    assert "Version notes: Current version" in context
    assert "Review notes: Verified" in context


def test_context_source_order(
    engine,
    ranked_results
):

    context = engine.build_context(
        ranked_results
    )

    assert context.index("[Source 1]") < (
        context.index("[Source 2]")
    )


def test_empty_context(engine):

    assert engine.build_context([]) == ""


def test_optional_context_metadata(
    engine,
    ranked_results
):

    result = ranked_results[0].copy()

    result.pop("document_date")
    result.pop("version_notes")
    result.pop("review_notes")

    context = engine.build_context([result])

    assert result["text"] in context
    assert "Document date: " in context
    assert "Version notes: " in context
    assert "Review notes: " in context


def test_context_does_not_modify_results(
    engine,
    ranked_results
):

    original = [
        item.copy()
        for item in ranked_results
    ]

    engine.build_context(ranked_results)

    assert ranked_results == original


# ==========================================
# 9. Relevance Integration Tests
# ==========================================

def test_relevance_accepted(
    engine,
    ranked_results
):

    engine.retrieve = Mock(
        return_value=ranked_results
    )

    output = engine.retrieve_with_relevance(
        "  Course registration  ",
        top_k=2,
        candidate_k=10,
        rrf_k=30
    )

    engine.retrieve.assert_called_once_with(
        query="  Course registration  ",
        top_k=2,
        mode="reranked",
        candidate_k=10,
        rrf_k=30
    )

    assert output["query"] == "Course registration"
    assert output["retrieval_mode"] == "reranked"
    assert output["relevance_passed"] is True
    assert output["max_score"] == 4.2
    assert output["threshold"] == -5.0

    assert output["reason"] == (
        "relevance_threshold_passed"
    )

    assert output["retrieved_count"] == 2
    assert output["retained_count"] == 2
    assert output["results"] == ranked_results

    assert "C001" in output["context"]
    assert "C002" in output["context"]


def test_relevance_rejected(
    engine,
    ranked_results
):

    for result in ranked_results:

        result["reranker_score"] = -10.0

    engine.retrieve = Mock(
        return_value=ranked_results
    )

    output = engine.retrieve_with_relevance(
        "Unrelated question"
    )

    assert output["relevance_passed"] is False
    assert output["reason"] == "low_relevance"
    assert output["max_score"] == -10.0

    assert output["retrieved_count"] == 2
    assert output["retained_count"] == 0

    assert output["results"] == []
    assert output["context"] == ""


def test_relevance_empty_results(engine):

    engine.retrieve = Mock(return_value=[])

    output = engine.retrieve_with_relevance(
        "Unknown topic"
    )

    assert output["relevance_passed"] is False
    assert output["reason"] == "no_results"
    assert output["max_score"] is None
    assert output["retrieved_count"] == 0
    assert output["retained_count"] == 0
    assert output["results"] == []
    assert output["context"] == ""


def test_custom_relevance_threshold(
    engine,
    ranked_results
):

    engine.retrieve = Mock(
        return_value=ranked_results
    )

    output = engine.retrieve_with_relevance(
        "Course registration",
        threshold=5.0
    )

    assert output["threshold"] == 5.0
    assert output["relevance_passed"] is False
    assert output["context"] == ""


def test_relevance_output_contract(
    engine,
    ranked_results
):

    engine.retrieve = Mock(
        return_value=ranked_results
    )

    output = engine.retrieve_with_relevance(
        "Course registration"
    )

    assert set(output.keys()) == {
        "query",
        "retrieval_mode",
        "relevance_passed",
        "max_score",
        "threshold",
        "reason",
        "retrieved_count",
        "retained_count",
        "results",
        "context"
    }


# ==========================================
# 10. Pipeline Integration Test
# ==========================================

def test_complete_retrieval_pipeline(
    engine,
    monkeypatch,
    sample_metadata
):

    query_embedding = np.array(
        [1, 0, 0],
        dtype=np.float32
    )

    monkeypatch.setattr(
        retrieval_module,
        "embed_query",
        Mock(return_value=query_embedding)
    )

    semantic_results = [
        {
            **sample_metadata[0],
            "similarity_score": 0.9
        },
        {
            **sample_metadata[1],
            "similarity_score": 0.8
        }
    ]

    monkeypatch.setattr(
        retrieval_module,
        "search",
        Mock(return_value=semantic_results)
    )

    def fake_rerank(query, results, top_k):

        output = []

        for result in results[:top_k]:

            item = result.copy()
            item["reranker_score"] = 4.0

            output.append(item)

        return output

    fake_reranker = Mock()

    fake_reranker.rerank.side_effect = fake_rerank

    monkeypatch.setattr(
        retrieval_module,
        "Reranker",
        Mock(return_value=fake_reranker)
    )

    output = engine.retrieve_with_relevance(
        "Course registration",
        top_k=2,
        candidate_k=2
    )

    assert output["relevance_passed"] is True
    assert output["retrieved_count"] == 2
    assert output["retained_count"] == 2

    assert output["results"][0]["chunk_id"] == "C001"

    assert [
        item["rank"]
        for item in output["results"]
    ] == [1, 2]

    assert all(
        item["retrieval_mode"] == "reranked"
        for item in output["results"]
    )

    assert all(
        "rrf_score" in item
        for item in output["results"]
    )

    assert all(
        "reranker_score" in item
        for item in output["results"]
    )

    assert "C001" in output["context"]
    assert "C002" in output["context"]

    assert engine.reranker is fake_reranker
