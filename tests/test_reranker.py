
import numpy as np
import pytest

from unittest.mock import Mock

from src.reranker import Reranker


# ==========================================
# Fake Tokenizer
# ==========================================

class FakeTokenizer:

    def __init__(self):

        self.word_to_id = {}
        self.id_to_word = {}

    def encode(
        self,
        text,
        text_pair=None,
        add_special_tokens=True,
        **kwargs
    ):

        words = text.split()

        if text_pair is not None:
            words += text_pair.split()

        ids = []

        for word in words:

            if word not in self.word_to_id:

                token_id = len(self.word_to_id) + 100

                self.word_to_id[word] = token_id
                self.id_to_word[token_id] = word

            ids.append(self.word_to_id[word])

        if add_special_tokens:

            count = self.num_special_tokens_to_add(
                pair=text_pair is not None
            )

            ids += [0] * count

        return ids

    def decode(self, token_ids, **kwargs):

        return " ".join(
            self.id_to_word[token_id]
            for token_id in token_ids
        )

    def num_special_tokens_to_add(self, pair=False):

        return 3 if pair else 2


# ==========================================
# Test Fixtures
# ==========================================

@pytest.fixture
def reranker():

    # Avoid loading the real Cross-Encoder
    instance = Reranker.__new__(Reranker)

    instance.max_length = 512
    instance.batch_size = 8
    instance.use_title = False

    instance.model = Mock()
    instance.model.tokenizer = FakeTokenizer()

    instance.model.predict.return_value = np.array(
        [0.2, 0.9, -0.5],
        dtype=np.float32
    )

    return instance


@pytest.fixture
def sample_results():

    return [
        {
            "chunk_id": "C001",
            "text": "Course registration policy.",
            "document_title": "Registration",
            "page": 1,
            "rank": 1
        },
        {
            "chunk_id": "C002",
            "text": "Course withdrawal policy.",
            "document_title": "Withdrawal",
            "page": 2,
            "rank": 2
        },
        {
            "chunk_id": "C003",
            "text": "Attendance requirements.",
            "document_title": "Attendance",
            "page": 3,
            "rank": 3
        }
    ]


# ==========================================
# 1. Ranking Tests
# ==========================================

def test_reranking_order(reranker, sample_results):

    results = reranker.rerank(
        "How can I withdraw?",
        sample_results,
        top_k=3
    )

    assert [
        item["chunk_id"] for item in results
    ] == ["C002", "C001", "C003"]


def test_reranker_scores(reranker, sample_results):

    results = reranker.rerank(
        "Course withdrawal",
        sample_results,
        top_k=3
    )

    assert [
        item["reranker_score"] for item in results
    ] == pytest.approx([0.9, 0.2, -0.5])


def test_final_ranks(reranker, sample_results):

    results = reranker.rerank(
        "Course withdrawal",
        sample_results,
        top_k=3
    )

    assert [
        item["rank"] for item in results
    ] == [1, 2, 3]


def test_original_ranks(reranker, sample_results):

    results = reranker.rerank(
        "Course withdrawal",
        sample_results,
        top_k=3
    )

    assert [
        item["original_rank"] for item in results
    ] == [2, 1, 3]


def test_top_k_limit(reranker, sample_results):

    results = reranker.rerank(
        "Course withdrawal",
        sample_results,
        top_k=2
    )

    assert len(results) == 2


def test_top_k_larger_than_results(
    reranker,
    sample_results
):

    results = reranker.rerank(
        "Course withdrawal",
        sample_results,
        top_k=100
    )

    assert len(results) == 3


def test_stable_tie_breaking(
    reranker,
    sample_results
):

    reranker.model.predict.return_value = np.array(
        [1.0, 1.0, 1.0]
    )

    results = reranker.rerank(
        "University policy",
        sample_results,
        top_k=3
    )

    assert [
        item["chunk_id"] for item in results
    ] == ["C001", "C002", "C003"]


# ==========================================
# 2. Metadata Preservation Tests
# ==========================================

def test_metadata_preservation(
    reranker,
    sample_results
):

    results = reranker.rerank(
        "Course withdrawal",
        sample_results
    )

    assert results[0]["text"] == (
        sample_results[1]["text"]
    )

    assert results[0]["document_title"] == (
        "Withdrawal"
    )

    assert results[0]["page"] == 2


def test_original_results_not_modified(
    reranker,
    sample_results
):

    original = [
        item.copy()
        for item in sample_results
    ]

    reranker.rerank(
        "Course withdrawal",
        sample_results
    )

    assert sample_results == original


# ==========================================
# 3. Token Safety Tests
# ==========================================

def test_short_passage_single_window(reranker):

    passage = "Students must attend classes."

    windows, token_count = reranker.build_windows(
        "Attendance rules",
        passage
    )

    assert windows == [passage]
    assert token_count == 9


def test_long_passage_multiple_windows(reranker):

    passage = " ".join(
        f"word{i}" for i in range(600)
    )

    windows, token_count = reranker.build_windows(
        "Course withdrawal",
        passage
    )

    assert token_count > 512
    assert len(windows) == 2


def test_windows_respect_token_limit(reranker):

    query = "Course withdrawal"

    passage = " ".join(
        f"word{i}" for i in range(1200)
    )

    windows, _ = reranker.build_windows(
        query,
        passage
    )

    for window in windows:

        tokens = reranker.model.tokenizer.encode(
            query,
            text_pair=window,
            add_special_tokens=True
        )

        assert len(tokens) <= 512


def test_full_passage_coverage(reranker):

    words = [
        f"word{i}" for i in range(1200)
    ]

    passage = " ".join(words)

    windows, _ = reranker.build_windows(
        "Course withdrawal",
        passage
    )

    covered_words = set()

    for window in windows:
        covered_words.update(window.split())

    assert covered_words == set(words)


def test_window_overlap(reranker):

    passage = " ".join(
        f"word{i}" for i in range(600)
    )

    windows, _ = reranker.build_windows(
        "Course withdrawal",
        passage
    )

    first = windows[0].split()
    second = windows[1].split()

    assert first[-64:] == second[:64]


def test_exact_token_limit(reranker):

    query = "Course withdrawal"

    passage = " ".join(
        f"word{i}" for i in range(507)
    )

    windows, token_count = reranker.build_windows(
        query,
        passage
    )

    assert token_count == 512
    assert windows == [passage]


def test_query_exceeds_token_budget(reranker):

    query = "word " * 510

    with pytest.raises(ValueError):

        reranker.build_windows(
            query,
            "University policy."
        )


# ==========================================
# 4. Sliding Window Score Aggregation
# ==========================================

def test_maximum_window_score(reranker):

    passage = " ".join(
        f"word{i}" for i in range(600)
    )

    records = [
        {
            "chunk_id": "C001",
            "text": passage,
            "document_title": "Long Document",
            "rank": 1
        }
    ]

    reranker.model.predict.return_value = np.array(
        [-4.0, 6.0]
    )

    results = reranker.rerank(
        "Course withdrawal",
        records,
        top_k=1
    )

    result = results[0]

    assert result["reranker_score"] == 6.0
    assert result["reranker_best_window"] == 2
    assert result["reranker_window_count"] == 2
    assert result["reranker_windowed"] is True
    assert result["reranker_truncated"] is False

    assert result["text"] == passage


def test_multiple_chunk_window_mapping(reranker):

    records = [
        {
            "chunk_id": "C001",
            "text": "Short passage.",
            "document_title": "Short Document",
            "rank": 1
        },
        {
            "chunk_id": "C002",
            "text": " ".join(
                f"word{i}" for i in range(600)
            ),
            "document_title": "Long Document",
            "rank": 2
        }
    ]

    # One score for the short chunk,
    # two scores for the long chunk.
    reranker.model.predict.return_value = np.array(
        [2.0, -4.0, 6.0]
    )

    results = reranker.rerank(
        "Course withdrawal",
        records,
        top_k=2
    )

    assert results[0]["chunk_id"] == "C002"
    assert results[0]["reranker_score"] == 6.0

    assert results[1]["chunk_id"] == "C001"
    assert results[1]["reranker_score"] == 2.0


def test_normal_passage_window_metadata(
    reranker,
    sample_results
):

    results = reranker.rerank(
        "Course withdrawal",
        sample_results
    )

    for result in results:

        assert result["reranker_window_count"] == 1
        assert result["reranker_best_window"] == 1
        assert result["reranker_windowed"] is False
        assert result["reranker_truncated"] is False

        assert result["reranker_input_tokens"] <= 512


# ==========================================
# 5. Model Input Tests
# ==========================================

def test_text_only_input(reranker, sample_results):

    reranker.rerank(
        "Course withdrawal",
        sample_results
    )

    pairs = reranker.model.predict.call_args.args[0]

    assert pairs[0] == (
        "Course withdrawal",
        sample_results[0]["text"]
    )


def test_title_inclusion(reranker, sample_results):

    reranker.use_title = True

    reranker.rerank(
        "Course withdrawal",
        sample_results
    )

    pairs = reranker.model.predict.call_args.args[0]

    assert pairs[0][1] == (
        "Registration\nCourse registration policy."
    )


# ==========================================
# 6. Input Validation Tests
# ==========================================

@pytest.mark.parametrize(
    "query",
    ["", "   ", None]
)
def test_invalid_query(
    reranker,
    sample_results,
    query
):

    with pytest.raises(ValueError):

        reranker.rerank(
            query,
            sample_results
        )


@pytest.mark.parametrize(
    "top_k",
    [0, -1, 1.5]
)
def test_invalid_top_k(
    reranker,
    sample_results,
    top_k
):

    with pytest.raises(ValueError):

        reranker.rerank(
            "Course withdrawal",
            sample_results,
            top_k=top_k
        )


def test_empty_results(reranker):

    assert reranker.rerank(
        "Course withdrawal",
        []
    ) == []

    reranker.model.predict.assert_not_called()


@pytest.mark.parametrize(
    "text",
    ["", "   ", None]
)
def test_invalid_passage(
    reranker,
    sample_results,
    text
):

    sample_results[0]["text"] = text

    with pytest.raises(ValueError):

        reranker.rerank(
            "Course withdrawal",
            sample_results
        )


# ==========================================
# 7. Score Validation Tests
# ==========================================

@pytest.mark.parametrize(
    "invalid_score",
    [np.nan, np.inf, -np.inf]
)
def test_invalid_model_scores(
    reranker,
    sample_results,
    invalid_score
):

    reranker.model.predict.return_value = np.array(
        [0.2, invalid_score, -0.5]
    )

    with pytest.raises(ValueError):

        reranker.rerank(
            "Course withdrawal",
            sample_results
        )


def test_score_count_mismatch(
    reranker,
    sample_results
):

    reranker.model.predict.return_value = np.array(
        [0.2, 0.9]
    )

    with pytest.raises(ValueError):

        reranker.rerank(
            "Course withdrawal",
            sample_results
        )
