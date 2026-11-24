
import numpy as np
import pytest
import faiss

from src.vector_store import (
    build_index,
    search,
    file_hash,
)


# ==========================================
# Test Fixtures
# ==========================================

@pytest.fixture
def sample_embeddings():

    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0]
        ],
        dtype=np.float32
    )


@pytest.fixture
def sample_metadata():

    return [
        {
            "chunk_id": "C001",
            "text": "Course registration policy.",
            "document_title": "Registration",
            "page": 1
        },
        {
            "chunk_id": "C002",
            "text": "Course withdrawal policy.",
            "document_title": "Withdrawal",
            "page": 2
        },
        {
            "chunk_id": "C003",
            "text": "Attendance requirements.",
            "document_title": "Attendance",
            "page": 3
        },
        {
            "chunk_id": "C004",
            "text": "University dress code.",
            "document_title": "Dress Code",
            "page": 4
        }
    ]


@pytest.fixture
def sample_index(sample_embeddings):

    return build_index(sample_embeddings)


@pytest.fixture
def query_embedding():

    return np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32
    )


# ==========================================
# 1. Index Creation Tests
# ==========================================

def test_build_index(sample_index):

    assert sample_index.ntotal == 4


def test_index_dimension(sample_index):

    assert sample_index.d == 3


def test_index_type(sample_index):

    assert isinstance(
        sample_index,
        faiss.IndexFlatIP
    )


def test_index_metric(sample_index):

    assert (
        sample_index.metric_type
        == faiss.METRIC_INNER_PRODUCT
    )


# ==========================================
# 2. Search Tests
# ==========================================

def test_self_retrieval(
    sample_index,
    sample_metadata,
    query_embedding
):

    results = search(
        sample_index,
        query_embedding,
        sample_metadata,
        top_k=1
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "C001"

    assert results[0]["similarity_score"] == (
        pytest.approx(1.0)
    )


def test_similarity_ranking(
    sample_index,
    sample_metadata,
    query_embedding
):

    results = search(
        sample_index,
        query_embedding,
        sample_metadata,
        top_k=4
    )

    ids = [
        result["chunk_id"]
        for result in results
    ]

    scores = [
        result["similarity_score"]
        for result in results
    ]

    assert ids == [
        "C001",
        "C002",
        "C003",
        "C004"
    ]

    assert scores == pytest.approx(
        [1.0, 0.8, 0.0, -1.0]
    )


def test_top_k(
    sample_index,
    sample_metadata,
    query_embedding
):

    results = search(
        sample_index,
        query_embedding,
        sample_metadata,
        top_k=2
    )

    assert len(results) == 2


def test_top_k_larger_than_index(
    sample_index,
    sample_metadata,
    query_embedding
):

    results = search(
        sample_index,
        query_embedding,
        sample_metadata,
        top_k=100
    )

    assert len(results) == 4


@pytest.mark.parametrize("top_k", [0, -1, 1.5])
def test_invalid_top_k(
    top_k,
    sample_index,
    sample_metadata,
    query_embedding
):

    with pytest.raises(ValueError):

        search(
            sample_index,
            query_embedding,
            sample_metadata,
            top_k=top_k
        )


# ==========================================
# 3. Query Validation Tests
# ==========================================

def test_invalid_query_dimension(
    sample_index,
    sample_metadata
):

    query = np.array(
        [1.0, 0.0],
        dtype=np.float32
    )

    with pytest.raises(ValueError):

        search(
            sample_index,
            query,
            sample_metadata
        )


@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf, -np.inf]
)
def test_invalid_query_values(
    invalid_value,
    sample_index,
    sample_metadata
):

    query = np.array(
        [invalid_value, 0.0, 1.0],
        dtype=np.float32
    )

    with pytest.raises(ValueError):

        search(
            sample_index,
            query,
            sample_metadata
        )


def test_non_normalized_query(
    sample_index,
    sample_metadata
):

    query = np.array(
        [2.0, 0.0, 0.0],
        dtype=np.float32
    )

    with pytest.raises(ValueError):

        search(
            sample_index,
            query,
            sample_metadata
        )


def test_zero_query_vector(
    sample_index,
    sample_metadata
):

    query = np.zeros(
        3,
        dtype=np.float32
    )

    with pytest.raises(ValueError):

        search(
            sample_index,
            query,
            sample_metadata
        )


# ==========================================
# 4. Metadata Tests
# ==========================================

def test_metadata_count_mismatch(
    sample_index,
    sample_metadata,
    query_embedding
):

    with pytest.raises(ValueError):

        search(
            sample_index,
            query_embedding,
            sample_metadata[:2]
        )


def test_metadata_preservation(
    sample_index,
    sample_metadata,
    query_embedding
):

    results = search(
        sample_index,
        query_embedding,
        sample_metadata,
        top_k=1
    )

    result = results[0]

    for key, value in sample_metadata[0].items():

        assert result[key] == value


def test_original_metadata_not_modified(
    sample_index,
    sample_metadata,
    query_embedding
):

    original = [
        item.copy()
        for item in sample_metadata
    ]

    search(
        sample_index,
        query_embedding,
        sample_metadata,
        top_k=4
    )

    assert sample_metadata == original


# ==========================================
# 5. Index Persistence Tests
# ==========================================

def test_index_save_and_reload(
    tmp_path,
    sample_index,
    sample_metadata,
    query_embedding
):

    path = tmp_path / "test_index.faiss"

    faiss.write_index(
        sample_index,
        str(path)
    )

    assert path.exists()

    loaded_index = faiss.read_index(
        str(path)
    )

    assert loaded_index.ntotal == 4
    assert loaded_index.d == 3

    original_results = search(
        sample_index,
        query_embedding,
        sample_metadata,
        top_k=4
    )

    loaded_results = search(
        loaded_index,
        query_embedding,
        sample_metadata,
        top_k=4
    )

    assert original_results == loaded_results


# ==========================================
# 6. File Integrity Tests
# ==========================================

def test_file_hash_consistency(tmp_path):

    path = tmp_path / "sample.txt"

    path.write_text(
        "Original content",
        encoding="utf-8"
    )

    first_hash = file_hash(path)
    second_hash = file_hash(path)

    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_file_hash_detects_changes(tmp_path):

    path = tmp_path / "sample.txt"

    path.write_text(
        "Original content",
        encoding="utf-8"
    )

    original_hash = file_hash(path)

    path.write_text(
        "Modified content",
        encoding="utf-8"
    )

    modified_hash = file_hash(path)

    assert original_hash != modified_hash
