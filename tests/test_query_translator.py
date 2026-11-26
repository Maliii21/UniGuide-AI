
"""
Unit tests for the UniGuide-AI query translation component.

All LLM requests are mocked.
No real API requests are made.
"""

from unittest.mock import Mock

import pytest

from src.llm import LLMResponse
from src.query_translator import (
    QueryTranslator,
    contains_arabic
)


ARABIC_QUESTION = "هل التسجيل الصيفي إجباري؟"

ENGLISH_QUESTION = "Is summer registration mandatory?"

MIXED_QUESTION = "ما هو الحد الأدنى للـ CGPA للتخرج؟"

MIXED_TRANSLATION = (
    "What is the minimum CGPA required for graduation?"
)

MODEL = "gemini-3.1-flash-lite"


# --------------------------------------------------
# Fixtures
# --------------------------------------------------

@pytest.fixture
def llm():

    return Mock()


@pytest.fixture
def translator(llm):

    return QueryTranslator(llm)


# --------------------------------------------------
# Language detection
# --------------------------------------------------

class TestLanguageDetection:

    @pytest.mark.parametrize(
        "text",
        [
            ARABIC_QUESTION,
            MIXED_QUESTION,
            "التسجيل",
            "AIU الجامعة"
        ]
    )
    def test_detects_arabic(self, text):

        assert contains_arabic(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            ENGLISH_QUESTION,
            "CGPA 2.0",
            "12345",
            "",
            "   ",
            "?!"
        ]
    )
    def test_non_arabic_text(self, text):

        assert contains_arabic(text) is False

    @pytest.mark.parametrize(
        "text",
        [None, 123, [], {}]
    )
    def test_invalid_input(self, text):

        with pytest.raises(TypeError):

            contains_arabic(text)


# --------------------------------------------------
# Question validation
# --------------------------------------------------

class TestQuestionValidation:

    @pytest.mark.parametrize(
        "question",
        [None, "", "   ", 123, [], {}]
    )
    def test_invalid_question(
        self, translator, llm, question
    ):

        with pytest.raises(ValueError):

            translator.translate(question)

        llm.translate_query.assert_not_called()


# --------------------------------------------------
# English bypass
# --------------------------------------------------

class TestEnglishBypass:

    def test_english_question(self, translator, llm):

        result = translator.translate(
            ENGLISH_QUESTION
        )

        assert result.success
        assert result.status == "not_required"
        assert result.original_question == ENGLISH_QUESTION
        assert result.retrieval_query == ENGLISH_QUESTION
        assert result.translation_required is False
        assert result.latency_seconds == 0.0
        assert result.llm_response is None
        assert result.error_message is None

        llm.translate_query.assert_not_called()

    def test_whitespace_removed(self, translator, llm):

        result = translator.translate(
            f"  {ENGLISH_QUESTION}  "
        )

        assert result.original_question == ENGLISH_QUESTION
        assert result.retrieval_query == ENGLISH_QUESTION

        llm.translate_query.assert_not_called()


# --------------------------------------------------
# Successful translation
# --------------------------------------------------

class TestSuccessfulTranslation:

    def test_arabic_translation(self, translator, llm):

        response = LLMResponse(
            status="success",
            model=MODEL,
            answer=ENGLISH_QUESTION,
            total_elapsed_seconds=0.5,
            attempt_count=1
        )

        llm.translate_query.return_value = response

        result = translator.translate(ARABIC_QUESTION)

        assert result.success
        assert result.status == "success"
        assert result.original_question == ARABIC_QUESTION
        assert result.retrieval_query == ENGLISH_QUESTION
        assert result.translation_required is True
        assert result.latency_seconds == 0.5
        assert result.llm_response is response
        assert result.error_message is None

        llm.translate_query.assert_called_once_with(
            ARABIC_QUESTION
        )

    def test_mixed_language(self, translator, llm):

        llm.translate_query.return_value = LLMResponse(
            status="success",
            model=MODEL,
            answer=MIXED_TRANSLATION
        )

        result = translator.translate(MIXED_QUESTION)

        assert result.success
        assert result.original_question == MIXED_QUESTION
        assert result.retrieval_query == MIXED_TRANSLATION
        assert "CGPA" in result.retrieval_query

        llm.translate_query.assert_called_once_with(
            MIXED_QUESTION
        )

    def test_translation_whitespace(
        self, translator, llm
    ):

        llm.translate_query.return_value = LLMResponse(
            status="success",
            model=MODEL,
            answer=f"  {ENGLISH_QUESTION}  "
        )

        result = translator.translate(ARABIC_QUESTION)

        assert result.success
        assert result.retrieval_query == ENGLISH_QUESTION


# --------------------------------------------------
# Translation API errors
# --------------------------------------------------

class TestTranslationErrors:

    @pytest.mark.parametrize(
        "status",
        [
            "rate_limited",
            "authentication_error",
            "model_unavailable",
            "timeout",
            "service_unavailable",
            "empty_response",
            "error"
        ]
    )
    def test_api_failure(
        self, translator, llm, status
    ):

        response = LLMResponse(
            status=status,
            model=MODEL,
            total_elapsed_seconds=1.5,
            error_message="Translation request failed."
        )

        llm.translate_query.return_value = response

        result = translator.translate(ARABIC_QUESTION)

        assert not result.success
        assert result.status == "translation_error"
        assert result.retrieval_query is None
        assert result.original_question == ARABIC_QUESTION
        assert result.translation_required is True
        assert result.latency_seconds == 1.5
        assert result.llm_response is response
        assert result.error_message == (
            "Translation request failed."
        )

    def test_default_error_message(
        self, translator, llm
    ):

        llm.translate_query.return_value = LLMResponse(
            status="error",
            model=MODEL
        )

        result = translator.translate(ARABIC_QUESTION)

        assert not result.success
        assert result.error_message == (
            "Unable to translate the question."
        )


# --------------------------------------------------
# Invalid translation output
# --------------------------------------------------

class TestInvalidTranslation:

    @pytest.mark.parametrize(
        "answer",
        [
            None,
            "",
            "   ",
            ARABIC_QUESTION,
            MIXED_QUESTION,
            "12345",
            "?!",
            "INSUFFICIENT_CONTEXT"
        ]
    )
    def test_invalid_translation(
        self, translator, llm, answer
    ):

        response = LLMResponse(
            status="success",
            model=MODEL,
            answer=answer,
            total_elapsed_seconds=0.8
        )

        llm.translate_query.return_value = response

        result = translator.translate(ARABIC_QUESTION)

        assert not result.success
        assert result.status == "invalid_translation"
        assert result.retrieval_query is None
        assert result.translation_required is True
        assert result.latency_seconds == 0.8
        assert result.llm_response is response
        assert result.error_message is not None
