# python -m pytest tests/test_retrieval.py -v -rs
#
# Most tests use a fake query model and hand-built vectors, so they run
# offline. Tests that need the real multilingual-e5-base model are skipped
# automatically when it cannot be loaded (-rs shows the reason).
from unittest.mock import Mock

import numpy as np
import pytest

from src.embeddings import (
    EMBEDDING_DIMENSION,
    QUERY_PREFIX,
    embed_passages,
    load_model,
    save_index,
)
from src.retrieval import (
    Retriever,
    cosine_similarity,
    format_context,
    retrieve,
    top_k_indices,
)


def basis_vector(*values):
    vector = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    vector[:len(values)] = values
    return vector


def make_chunks(texts):
    return [
        {
            "document_id": f"AIU-{number:03d}",
            "page_id": f"AIU-{number:03d}-P001",
            "chunk_id": f"AIU-{number:03d}-P001-C001",
            "document_title": f"Policy {number}",
            "source": f"policy-{number}.pdf",
            "source_url": f"https://aiu.edu.eg/policy-{number}.pdf",
            "language": "English",
            "page": number,
            "token_count": 10,
            "text": text,
        }
        for number, text in enumerate(texts, start=1)
    ]


def make_index(scores):
    """Unit vectors whose cosine similarity with basis_vector(1) is `scores`."""
    vectors = np.zeros((len(scores), EMBEDDING_DIMENSION), dtype=np.float32)

    for row, score in enumerate(scores):
        vectors[row, 0] = score
        vectors[row, row + 1] = np.sqrt(1 - score ** 2)

    return vectors


@pytest.fixture
def query_model():
    model = Mock()
    model.encode.return_value = basis_vector(1)
    return model


@pytest.fixture(scope="module")
def real_model():
    try:
        return load_model()
    except Exception as error:
        pytest.skip(f"multilingual-e5-base could not be loaded ({type(error).__name__}): {error}")


# Test 1: Cosine similarity formula
def test_cosine_similarity_matches_manual_formula():
    query = np.array([3.0, 4.0])
    matrix = np.array([[3.0, 4.0], [4.0, -3.0], [-6.0, -8.0], [0.0, 0.0]])

    scores = cosine_similarity(query, matrix)

    assert np.allclose(scores, [1.0, 0.0, -1.0, 0.0], atol=1e-12)


def test_cosine_similarity_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="Shape mismatch"):
        cosine_similarity(np.ones(3), np.ones((2, 4)))


# Test 2: Top-K selection
def test_top_k_is_ordered_stable_and_clamped():
    scores = np.array([0.2, 0.9, 0.9, 0.1])

    assert top_k_indices(scores, 3).tolist() == [1, 2, 0]
    assert top_k_indices(scores, 10).tolist() == [1, 2, 0, 3]
    assert top_k_indices(scores, 0).tolist() == []


# Test 3: Basic semantic search (controlled similarities)
def test_similarity_search_ranks_by_cosine(query_model):
    chunks = make_chunks([
        "University email account policy.",
        "Course registration requirements.",
        "Students must attend at least 75% of lectures.",
        "Attendance warnings are issued to students.",
    ])
    vectors = make_index([0.15, 0.42, 0.92, 0.81])

    results = retrieve("minimum attendance?", query_model, chunks, vectors, top_k=4)

    assert [result["text"] for result in results] == [
        "Students must attend at least 75% of lectures.",
        "Attendance warnings are issued to students.",
        "Course registration requirements.",
        "University email account policy.",
    ]
    assert np.allclose([result["score"] for result in results], [0.92, 0.81, 0.42, 0.15], atol=1e-6)
    assert [result["rank"] for result in results] == [1, 2, 3, 4]


# Test 4: Metadata is returned and inputs are not changed
def test_results_keep_metadata_without_vectors(query_model):
    chunks = make_chunks(["First.", "Second."])
    vectors = make_index([0.5, 0.9])

    result = retrieve("question", query_model, chunks, vectors, top_k=1)[0]

    assert result["chunk_id"] == "AIU-002-P001-C001"
    assert result["source_url"] == "https://aiu.edu.eg/policy-2.pdf"
    assert result["page"] == 2
    assert "vector" not in result
    assert "score" not in chunks[1] and "rank" not in chunks[1]


# Test 5: The query uses the E5 "query: " prefix
def test_query_prefix_is_applied(query_model):
    chunks = make_chunks(["First."])

    retrieve("  Where is the library?  ", query_model, chunks, make_index([0.5]))

    assert query_model.encode.call_args.args[0] == QUERY_PREFIX + "Where is the library?"


# Test 6: Minimum score
def test_min_score_filters_weak_matches_and_renumbers(query_model):
    chunks = make_chunks(["A", "B", "C", "D"])
    vectors = make_index([0.5, 0.9, 0.7, 0.1])

    results = retrieve("q", query_model, chunks, vectors, top_k=4, min_score=0.6)

    assert [result["text"] for result in results] == ["B", "C"]
    assert [result["rank"] for result in results] == [1, 2]
    assert retrieve("q", query_model, chunks, vectors, top_k=4, min_score=0.99) == []


# Test 7: Empty question, bad top_k, mismatched index
def test_empty_question_and_bad_top_k_skip_encoding():
    chunks = make_chunks(["A"])
    vectors = make_index([0.5])

    assert retrieve("   ", None, chunks, vectors) == []
    assert retrieve("question", None, chunks, vectors, top_k=0) == []


def test_rejects_embeddings_that_do_not_match_chunks(query_model):
    with pytest.raises(ValueError, match="do not match"):
        retrieve("q", query_model, make_chunks(["A", "B"]), make_index([0.5]))


# Test 8: Prompt context
def test_format_context_includes_title_page_and_text():
    results = [
        {**make_chunks(["Students must attend 75%."])[0], "rank": 1, "score": 0.9},
        {**make_chunks(["x", "Warnings are issued."])[1], "rank": 2, "score": 0.8},
    ]

    context = format_context(results)

    assert "[Source 1: Policy 1, page 1]\nStudents must attend 75%." in context
    assert "[Source 2: Policy 2, page 2]\nWarnings are issued." in context
    assert format_context([]) == ""


# Test 9: Retriever loads a saved index
def test_retriever_searches_a_saved_index(tmp_path, query_model):
    chunks = make_chunks(["A", "B", "C"])
    vectors = make_index([0.3, 0.95, 0.6])
    save_index(chunks, vectors, tmp_path)

    retriever = Retriever(tmp_path, model=query_model)
    results = retriever.search("question", top_k=2)

    assert [result["text"] for result in results] == ["B", "C"]
    assert retriever.config["chunk_count"] == 3

    with pytest.raises(FileNotFoundError):
        Retriever(tmp_path / "missing", model=query_model)


# Test 10: English semantic search with the real model
def test_real_model_english_query_finds_attendance_rule(real_model):
    passages = [
        "Students must attend at least 75% of lectures.",
        "Attendance warnings are issued to students.",
        "Course registration requirements.",
        "University email account policy.",
    ]
    chunks = make_chunks(passages)
    vectors = embed_passages(passages, real_model)

    results = retrieve(
        "What is the minimum attendance percentage?",
        real_model, chunks, vectors, top_k=4,
    )

    assert results[0]["text"] == passages[0]
    assert results[-1]["text"] in (passages[2], passages[3])


# Test 11: Arabic question retrieves English text (cross-lingual)
def test_real_model_arabic_query_finds_english_text(real_model):
    passages = [
        "Students must attend at least 75% of lectures.",
        "Course registration requirements.",
        "University email account policy.",
    ]
    chunks = make_chunks(passages)
    vectors = embed_passages(passages, real_model)

    results = retrieve(
        "ما هو الحد الأدنى لنسبة الحضور؟",
        real_model, chunks, vectors, top_k=1,
    )

    assert results[0]["text"] == passages[0]
