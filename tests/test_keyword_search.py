
import math

import pytest

from src.keyword_search import BM25Search, tokenize


# ==========================================
# Test Fixtures
# ==========================================

@pytest.fixture
def sample_metadata():

    return [
        {
            "chunk_id": "C001",
            "text": "Students register courses during registration.",
            "document_title": "Course Registration",
            "page": 1
        },
        {
            "chunk_id": "C002",
            "text": "Students may withdraw from courses.",
            "document_title": "Course Withdrawal",
            "page": 2
        },
        {
            "chunk_id": "C003",
            "text": "Students must attend classes regularly.",
            "document_title": "Attendance Policy",
            "page": 3
        },
        {
            "chunk_id": "C004",
            "text": "Transfer students need a minimum CGPA.",
            "document_title": "Transfer Admission",
            "page": 4
        },
        {
            "chunk_id": "C005",
            "text": "GPA determines the credit hours limit.",
            "document_title": "Credit Hours",
            "page": 5
        }
    ]


@pytest.fixture
def engine(sample_metadata):

    return BM25Search(sample_metadata)


# ==========================================
# 1. Tokenization Tests
# ==========================================

def test_case_normalization(engine):

    assert tokenize("COURSE") == (
        tokenize("course")
    )


def test_punctuation_removal(engine):

    tokens = tokenize(
        "Course, registration!"
    )

    assert tokens == ["course", "register"]


def test_stopword_removal(engine):

    tokens = tokenize(
        "the course and the student"
    )

    assert "the" not in tokens
    assert "and" not in tokens
    assert "course" in tokens
    assert "student" in tokens


@pytest.mark.parametrize(
    "word, expected",
    [
        ("students", "student"),
        ("courses", "course"),
        ("withdrawal", "withdraw"),
        ("withdrawn", "withdraw"),
        ("registration", "register"),
        ("registered", "register"),
        ("transferred", "transfer"),
        ("requirements", "requirement"),
        ("hours", "hour"),
        ("examinations", "exam")
    ]
)
def test_term_normalization(
    engine,
    word,
    expected
):

    assert tokenize(word) == [expected]


def test_gpa_cgpa_distinction(engine):

    assert tokenize("GPA") == ["gpa"]
    assert tokenize("CGPA") == ["cgpa"]

    assert tokenize("GPA") != (
        tokenize("CGPA")
    )


def test_numeric_tokens(engine):

    tokens = tokenize("21 credit hours")

    assert "21" in tokens
    assert "credit" in tokens
    assert "hour" in tokens


def test_empty_tokenization(engine):

    assert tokenize("") == []


# ==========================================
# 2. BM25 Ranking Tests
# ==========================================

def test_registration_ranking(engine):

    results = engine.search(
        "course registration",
        top_k=3
    )

    assert results[0]["chunk_id"] == "C001"


def test_withdrawal_ranking(engine):

    results = engine.search(
        "course withdrawal",
        top_k=3
    )

    assert results[0]["chunk_id"] == "C002"


def test_attendance_ranking(engine):

    results = engine.search(
        "attend classes",
        top_k=3
    )

    assert results[0]["chunk_id"] == "C003"


def test_gpa_retrieval(engine):

    results = engine.search(
        "GPA",
        top_k=5
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "C005"


def test_cgpa_retrieval(engine):

    results = engine.search(
        "CGPA",
        top_k=5
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "C004"


def test_scores_descending(engine):

    results = engine.search(
        "student course registration",
        top_k=5
    )

    scores = [
        result["bm25_score"]
        for result in results
    ]

    assert len(scores) > 1

    assert scores == sorted(
        scores,
        reverse=True
    )


def test_positive_scores(engine):

    results = engine.search(
        "course",
        top_k=5
    )

    assert len(results) > 0

    assert all(
        result["bm25_score"] > 0
        for result in results
    )


def test_no_matching_terms(engine):

    results = engine.search(
        "zzzxxyy",
        top_k=5
    )

    assert results == []


def test_stopwords_only_query(engine):

    results = engine.search(
        "the and is",
        top_k=5
    )

    assert results == []


def test_repeated_query_terms(engine):

    first = engine.search(
        "course",
        top_k=5
    )

    second = engine.search(
        "course course course",
        top_k=5
    )

    assert first == second


# ==========================================
# 3. BM25 Mathematical Validation
# ==========================================

def test_bm25_formula():

    metadata = [
        {
            "chunk_id": "C001",
            "text": "fiber fiber cable"
        },
        {
            "chunk_id": "C002",
            "text": "water pipe"
        }
    ]

    engine = BM25Search(metadata)

    results = engine.search(
        "fiber",
        top_k=2
    )

    # Known corpus statistics
    N = 2
    df = 1
    tf = 2
    doc_length = 3
    avg_length = 2.5

    k1 = 1.5
    b = 0.75

    idf = math.log(
        1 + (N - df + 0.5) / (df + 0.5)
    )

    expected_score = idf * (
        tf * (k1 + 1)
    ) / (
        tf + k1 * (
            1 - b + b * doc_length / avg_length
        )
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "C001"

    assert results[0]["bm25_score"] == (
        pytest.approx(expected_score)
    )


# ==========================================
# 4. Input Validation Tests
# ==========================================

def test_empty_corpus():

    with pytest.raises(ValueError):
        BM25Search([])


def test_corpus_without_tokens():

    metadata = [
        {"chunk_id": "C001", "text": ""}
    ]

    with pytest.raises(ValueError):
        BM25Search(metadata)


@pytest.mark.parametrize(
    "query",
    ["", "   "]
)
def test_empty_query(engine, query):

    with pytest.raises(ValueError):
        engine.search(query)


@pytest.mark.parametrize(
    "top_k",
    [0, -1, 1.5]
)
def test_invalid_top_k(engine, top_k):

    with pytest.raises(ValueError):

        engine.search(
            "course",
            top_k=top_k
        )


def test_top_k_limit(engine):

    results = engine.search(
        "student",
        top_k=2
    )

    assert len(results) == 2


def test_top_k_larger_than_corpus(engine):

    results = engine.search(
        "student",
        top_k=100
    )

    assert len(results) == 4


# ==========================================
# 5. Metadata Preservation Tests
# ==========================================

def test_metadata_preservation(
    engine,
    sample_metadata
):

    results = engine.search(
        "registration",
        top_k=1
    )

    for key, value in sample_metadata[0].items():

        assert results[0][key] == value


def test_original_metadata_not_modified(
    engine,
    sample_metadata
):

    original = [
        item.copy()
        for item in sample_metadata
    ]

    engine.search(
        "course registration",
        top_k=5
    )

    assert sample_metadata == original
