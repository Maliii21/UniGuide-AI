
import pytest

from src.rank_fusion import reciprocal_rank_fusion


# ==========================================
# Test Fixtures
# ==========================================

@pytest.fixture
def semantic_results():

    return [
        {
            "chunk_id": "C001",
            "text": "Course registration policy.",
            "similarity_score": 0.90
        },
        {
            "chunk_id": "C002",
            "text": "Course withdrawal policy.",
            "similarity_score": 0.85
        },
        {
            "chunk_id": "C003",
            "text": "Attendance requirements.",
            "similarity_score": 0.80
        }
    ]


@pytest.fixture
def keyword_results():

    return [
        {
            "chunk_id": "C002",
            "text": "Course withdrawal policy.",
            "bm25_score": 8.0
        },
        {
            "chunk_id": "C004",
            "text": "University dress code.",
            "bm25_score": 6.0
        },
        {
            "chunk_id": "C001",
            "text": "Course registration policy.",
            "bm25_score": 4.0
        }
    ]


# ==========================================
# 1. Ranking Tests
# ==========================================

def test_fusion_ranking(
    semantic_results,
    keyword_results
):

    results = reciprocal_rank_fusion(
        semantic_results,
        keyword_results,
        top_k=4,
        k=60
    )

    assert [
        item["chunk_id"] for item in results
    ] == [
        "C002",
        "C001",
        "C004",
        "C003"
    ]


def test_rrf_formula(
    semantic_results,
    keyword_results
):

    results = reciprocal_rank_fusion(
        semantic_results,
        keyword_results,
        top_k=4,
        k=60
    )

    expected = {
        "C001": 1 / 61 + 1 / 63,
        "C002": 1 / 62 + 1 / 61,
        "C003": 1 / 63,
        "C004": 1 / 62
    }

    for result in results:

        assert result["rrf_score"] == pytest.approx(
            expected[result["chunk_id"]]
        )


def test_scores_descending(
    semantic_results,
    keyword_results
):

    results = reciprocal_rank_fusion(
        semantic_results,
        keyword_results,
        top_k=4
    )

    scores = [
        result["rrf_score"]
        for result in results
    ]

    assert scores == sorted(
        scores,
        reverse=True
    )


def test_final_rank(
    semantic_results,
    keyword_results
):

    results = reciprocal_rank_fusion(
        semantic_results,
        keyword_results,
        top_k=4
    )

    assert [
        result["rank"] for result in results
    ] == [1, 2, 3, 4]


# ==========================================
# 2. Duplicate Handling Tests
# ==========================================

def test_unique_chunk_ids(
    semantic_results,
    keyword_results
):

    results = reciprocal_rank_fusion(
        semantic_results,
        keyword_results,
        top_k=10
    )

    ids = [
        result["chunk_id"]
        for result in results
    ]

    assert len(ids) == len(set(ids))
    assert len(ids) == 4


def test_duplicate_within_same_list(
    semantic_results
):

    duplicated = [
        semantic_results[0],
        semantic_results[0].copy()
    ]

    results = reciprocal_rank_fusion(
        duplicated,
        [],
        top_k=5,
        k=60
    )

    assert len(results) == 1

    assert results[0]["rrf_score"] == (
        pytest.approx(1 / 61)
    )


# ==========================================
# 3. Top K Tests
# ==========================================

def test_top_k_limit(
    semantic_results,
    keyword_results
):

    results = reciprocal_rank_fusion(
        semantic_results,
        keyword_results,
        top_k=2
    )

    assert len(results) == 2


def test_top_k_larger_than_results(
    semantic_results,
    keyword_results
):

    results = reciprocal_rank_fusion(
        semantic_results,
        keyword_results,
        top_k=100
    )

    assert len(results) == 4


# ==========================================
# 4. Empty Results Tests
# ==========================================

def test_empty_semantic_results(
    keyword_results
):

    results = reciprocal_rank_fusion(
        [],
        keyword_results,
        top_k=5
    )

    assert [
        item["chunk_id"] for item in results
    ] == ["C002", "C004", "C001"]


def test_empty_keyword_results(
    semantic_results
):

    results = reciprocal_rank_fusion(
        semantic_results,
        [],
        top_k=5
    )

    assert [
        item["chunk_id"] for item in results
    ] == ["C001", "C002", "C003"]


def test_both_results_empty():

    assert reciprocal_rank_fusion(
        [],
        [],
        top_k=5
    ) == []


# ==========================================
# 5. Score and Metadata Tests
# ==========================================

def test_original_scores_preserved(
    semantic_results,
    keyword_results
):

    results = reciprocal_rank_fusion(
        semantic_results,
        keyword_results,
        top_k=4
    )

    result = next(
        item for item in results
        if item["chunk_id"] == "C002"
    )

    assert result["similarity_score"] == (
        pytest.approx(0.85)
    )

    assert result["bm25_score"] == (
        pytest.approx(8.0)
    )


def test_original_ranks_preserved(
    semantic_results,
    keyword_results
):

    results = reciprocal_rank_fusion(
        semantic_results,
        keyword_results,
        top_k=4
    )

    result = next(
        item for item in results
        if item["chunk_id"] == "C002"
    )

    assert result["semantic_rank"] == 2
    assert result["keyword_rank"] == 1


def test_missing_source_rank(
    semantic_results,
    keyword_results
):

    results = reciprocal_rank_fusion(
        semantic_results,
        keyword_results,
        top_k=4
    )

    by_id = {
        item["chunk_id"]: item
        for item in results
    }

    assert by_id["C003"]["keyword_rank"] is None
    assert by_id["C004"]["semantic_rank"] is None

    assert by_id["C003"]["bm25_score"] is None
    assert by_id["C004"]["similarity_score"] is None


def test_metadata_preservation(
    semantic_results,
    keyword_results
):

    results = reciprocal_rank_fusion(
        semantic_results,
        keyword_results,
        top_k=4
    )

    assert results[0]["text"] == (
        "Course withdrawal policy."
    )


def test_original_results_not_modified(
    semantic_results,
    keyword_results
):

    original_semantic = [
        item.copy()
        for item in semantic_results
    ]

    original_keyword = [
        item.copy()
        for item in keyword_results
    ]

    reciprocal_rank_fusion(
        semantic_results,
        keyword_results,
        top_k=4
    )

    assert semantic_results == original_semantic
    assert keyword_results == original_keyword


# ==========================================
# 6. Parameter Validation Tests
# ==========================================

@pytest.mark.parametrize(
    "top_k",
    [0, -1, 1.5]
)
def test_invalid_top_k(
    semantic_results,
    keyword_results,
    top_k
):

    with pytest.raises(ValueError):

        reciprocal_rank_fusion(
            semantic_results,
            keyword_results,
            top_k=top_k
        )


@pytest.mark.parametrize(
    "k",
    [0, -1, 1.5]
)
def test_invalid_k(
    semantic_results,
    keyword_results,
    k
):

    with pytest.raises(ValueError):

        reciprocal_rank_fusion(
            semantic_results,
            keyword_results,
            top_k=4,
            k=k
        )


def test_custom_k(
    semantic_results,
    keyword_results
):

    results = reciprocal_rank_fusion(
        semantic_results,
        keyword_results,
        top_k=4,
        k=10
    )

    assert results[0]["chunk_id"] == "C002"

    assert results[0]["rrf_score"] == (
        pytest.approx(1 / 12 + 1 / 11)
    )


# ==========================================
# 7. Deterministic Ranking Tests
# ==========================================

def test_tie_breaking():

    semantic = [
        {
            "chunk_id": "C002",
            "text": "First document.",
            "similarity_score": 0.9
        }
    ]

    keyword = [
        {
            "chunk_id": "C001",
            "text": "Second document.",
            "bm25_score": 5.0
        }
    ]

    results = reciprocal_rank_fusion(
        semantic,
        keyword,
        top_k=2,
        k=60
    )

    assert results[0]["rrf_score"] == (
        results[1]["rrf_score"]
    )

    assert [
        item["chunk_id"] for item in results
    ] == ["C001", "C002"]
