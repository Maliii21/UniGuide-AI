
"""
Unit tests for the UniGuide-AI citation validator.

All tests run locally without external API calls.
"""

import pytest

from src.citation_validator import (
    extract_citations,
    validate_citations,
    CitationValidationResult
)


VALID_ID_1 = "AIU-003-P002-C002"
VALID_ID_2 = "AIU-009-P001-C002"
INVALID_ID = "AIU-001-P001-C002"


@pytest.fixture
def retrieved_chunks():

    return [
        {
            "chunk_id": VALID_ID_1,
            "text": "Summer registration is optional."
        },
        {
            "chunk_id": VALID_ID_2,
            "text": "Summer tuition information."
        }
    ]


class TestCitationExtraction:

    def test_extract_single_citation(self):

        answer = f"Answer [{VALID_ID_1}]"

        assert extract_citations(answer) == [
            VALID_ID_1
        ]

    def test_extract_multiple_citations(self):

        answer = (
            f"Answer [{VALID_ID_1}] "
            f"[{VALID_ID_2}]"
        )

        assert extract_citations(answer) == [
            VALID_ID_1,
            VALID_ID_2
        ]

    def test_duplicate_citations_removed(self):

        answer = (
            f"Answer [{VALID_ID_1}] "
            f"[{VALID_ID_1}]"
        )

        assert extract_citations(answer) == [
            VALID_ID_1
        ]

    def test_no_citations(self):

        assert extract_citations(
            "Answer without citations."
        ) == []

    def test_unbracketed_id_not_extracted(self):

        assert extract_citations(
            f"Answer {VALID_ID_1}"
        ) == []

    def test_invalid_input_type(self):

        with pytest.raises(TypeError):
            extract_citations(None)


class TestCitationValidation:

    def test_valid_single_citation(
        self, retrieved_chunks
    ):

        result = validate_citations(
            f"Answer [{VALID_ID_1}]",
            retrieved_chunks
        )

        assert isinstance(
            result, CitationValidationResult
        )
        assert result.status == "valid"
        assert result.passed
        assert result.valid_ids == [VALID_ID_1]
        assert result.invalid_ids == []
        assert result.citation_validity == 1.0

    def test_multiple_valid_citations(
        self, retrieved_chunks
    ):

        result = validate_citations(
            f"Answer [{VALID_ID_1}] [{VALID_ID_2}]",
            retrieved_chunks
        )

        assert result.passed
        assert result.citation_count == 2
        assert result.valid_citation_count == 2
        assert result.citation_validity == 1.0

    def test_invalid_citation(
        self, retrieved_chunks
    ):

        result = validate_citations(
            f"Answer [{INVALID_ID}]",
            retrieved_chunks
        )

        assert result.status == "invalid_citations"
        assert not result.passed
        assert result.invalid_ids == [INVALID_ID]
        assert result.citation_validity == 0.0

    def test_mixed_valid_and_invalid_citations(
        self, retrieved_chunks
    ):

        result = validate_citations(
            f"Answer [{VALID_ID_1}] [{INVALID_ID}]",
            retrieved_chunks
        )

        assert result.status == "invalid_citations"
        assert not result.passed
        assert result.valid_ids == [VALID_ID_1]
        assert result.invalid_ids == [INVALID_ID]
        assert result.citation_count == 2
        assert result.valid_citation_count == 1
        assert result.citation_validity == 0.5

    def test_missing_citations(
        self, retrieved_chunks
    ):

        result = validate_citations(
            "Summer registration is optional.",
            retrieved_chunks
        )

        assert result.status == "missing_citations"
        assert not result.passed
        assert result.citation_count == 0

    @pytest.mark.parametrize(
        "answer",
        ["", " ", "\n\t"]
    )
    def test_empty_answer(
        self, answer, retrieved_chunks
    ):

        result = validate_citations(
            answer,
            retrieved_chunks
        )

        assert result.status == "empty_answer"
        assert not result.passed

    @pytest.mark.parametrize(
        "answer",
        [
            f"Answer {VALID_ID_1}",
            f"Answer ({VALID_ID_1})",
            f"Answer [{VALID_ID_1}",
            f"Answer {VALID_ID_1}]",
            f"Answer [{VALID_ID_1}, {VALID_ID_2}]",
            "Answer [AIU-003-P002-C02]"
        ]
    )
    def test_malformed_citations(
        self, answer, retrieved_chunks
    ):

        result = validate_citations(
            answer,
            retrieved_chunks
        )

        assert result.status == "malformed_citations"
        assert result.malformed_citations
        assert not result.passed

    def test_duplicate_citation_count(
        self, retrieved_chunks
    ):

        result = validate_citations(
            f"Answer [{VALID_ID_1}] [{VALID_ID_1}]",
            retrieved_chunks
        )

        assert result.passed
        assert result.citation_count == 1
        assert result.valid_citation_count == 1

    def test_duplicate_retrieved_ids(
        self, retrieved_chunks
    ):

        chunks = retrieved_chunks + [
            retrieved_chunks[0]
        ]

        result = validate_citations(
            f"Answer [{VALID_ID_1}]",
            chunks
        )

        assert result.passed
        assert len(result.retrieved_ids) == 2

    def test_empty_retrieved_chunks(self):

        result = validate_citations(
            f"Answer [{VALID_ID_1}]",
            []
        )

        assert result.status == "invalid_citations"
        assert not result.passed

    def test_invalid_answer_type(
        self, retrieved_chunks
    ):

        with pytest.raises(TypeError):

            validate_citations(
                None,
                retrieved_chunks
            )

    def test_validation_does_not_modify_chunks(
        self, retrieved_chunks
    ):

        from copy import deepcopy

        original = deepcopy(retrieved_chunks)

        validate_citations(
            f"Answer [{VALID_ID_1}]",
            retrieved_chunks
        )

        assert retrieved_chunks == original


class TestCitationRegression:

    def test_real_rag_hallucination_case(self):
        """
        Reproduces the citation failure observed
        during the real RAG integration experiment.
        """

        answer = (
            "No, registration in the Summer "
            "semester is optional "
            "[AIU-001-P001-C002] "
            "[AIU-003-P002-C002]."
        )

        retrieved_chunks = [
            {"chunk_id": "AIU-009-P001-C002"},
            {"chunk_id": "AIU-011-P002-C001"},
            {"chunk_id": "AIU-003-P002-C002"},
            {"chunk_id": "AIU-020-P002-C001"},
            {"chunk_id": "AIU-020-P002-C002"}
        ]

        result = validate_citations(
            answer,
            retrieved_chunks
        )

        assert result.status == "invalid_citations"
        assert not result.passed
        assert result.citation_count == 2
        assert result.valid_citation_count == 1
        assert result.citation_validity == 0.5

        assert result.invalid_ids == [
            "AIU-001-P001-C002"
        ]

        assert result.valid_ids == [
            "AIU-003-P002-C002"
        ]
