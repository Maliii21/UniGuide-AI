
"""
Unit tests for the UniGuide-AI RAG pipeline.

Retrieval and LLM components are mocked.
No model loading or external API calls are required.
"""

from unittest.mock import Mock, patch

import pytest

from src.rag_pipeline import (
    RAGPipeline,
    RAGResponse,
    insufficient_context_message
)

from src.llm import LLMResponse


VALID_ID = "AIU-003-P002-C002"
INVALID_ID = "AIU-001-P001-C002"

QUESTION = "Is summer registration mandatory?"

VALID_ANSWER = (
    "Summer registration is optional. "
    "[AIU-003-P002-C002]"
)


@pytest.fixture
def retrieval_engine():

    engine = Mock()

    engine.retrieve_with_relevance.return_value = {
        "relevance_passed": True,
        "max_score": 4.5,
        "threshold": -5.0,
        "reason": "passed",
        "retrieved_count": 1,
        "retained_count": 1,
        "results": [
            {
                "chunk_id": VALID_ID,
                "text": "Summer registration is optional."
            }
        ],
        "context": (
            "Chunk ID: AIU-003-P002-C002\n"
            "Summer registration is optional."
        )
    }

    return engine


@pytest.fixture
def llm():

    model = Mock()

    model.generate_answer.return_value = LLMResponse(
        status="success",
        model="gemini-3.1-flash-lite",
        answer=VALID_ANSWER,
        attempt_count=1
    )

    return model


@pytest.fixture
def pipeline(retrieval_engine, llm):

    with RAGPipeline(
        retrieval_engine=retrieval_engine,
        llm=llm
    ) as instance:

        yield instance


# ==========================================
# Input validation
# ==========================================

class TestInputValidation:

    @pytest.mark.parametrize(
        "question",
        ["", " ", "\n\t"]
    )
    def test_empty_question(
        self, pipeline, question,
        retrieval_engine, llm
    ):

        with pytest.raises(ValueError):

            pipeline.ask(question)

        retrieval_engine.retrieve_with_relevance.assert_not_called()
        llm.generate_answer.assert_not_called()

    def test_question_whitespace_removed(
        self, pipeline, retrieval_engine
    ):

        result = pipeline.ask(
            f"  {QUESTION}  "
        )

        assert result.question == QUESTION

        kwargs = (
            retrieval_engine.retrieve_with_relevance
            .call_args.kwargs
        )

        assert kwargs["query"] == QUESTION


# ==========================================
# Successful RAG execution
# ==========================================

class TestSuccessfulPipeline:

    def test_successful_answer(self, pipeline):

        result = pipeline.ask(QUESTION)

        assert isinstance(result, RAGResponse)
        assert result.success
        assert result.status == "success"
        assert result.answer == VALID_ANSWER
        assert result.error_message is None

    def test_retrieval_metadata(self, pipeline):

        result = pipeline.ask(QUESTION)

        assert result.relevance_passed
        assert result.max_relevance_score == 4.5
        assert result.relevance_threshold == -5.0
        assert result.relevance_reason == "passed"
        assert result.retrieved_count == 1
        assert result.retained_count == 1

        assert (
            result.retrieved_chunks[0]["chunk_id"]
            == VALID_ID
        )

    def test_context_forwarded_to_llm(
        self, pipeline, retrieval_engine, llm
    ):

        pipeline.ask(QUESTION)

        expected_context = (
            retrieval_engine.retrieve_with_relevance
            .return_value["context"]
        )

        llm.generate_answer.assert_called_once_with(
            question=QUESTION,
            context=expected_context
        )

    def test_retrieval_parameters(
        self, pipeline, retrieval_engine
    ):

        pipeline.ask(
            QUESTION,
            top_k=3,
            candidate_k=10,
            rrf_k=40,
            threshold=-4.0
        )

        retrieval_engine.retrieve_with_relevance.assert_called_once_with(
            query=QUESTION,
            top_k=3,
            candidate_k=10,
            rrf_k=40,
            threshold=-4.0
        )

    def test_latency_recorded(self, pipeline):

        result = pipeline.ask(QUESTION)

        assert result.retrieval_latency_seconds >= 0
        assert result.total_latency_seconds >= 0

        assert (
            result.total_latency_seconds
            >= result.retrieval_latency_seconds
        )

    def test_llm_response_preserved(
        self, pipeline, llm
    ):

        result = pipeline.ask(QUESTION)

        assert (
            result.llm_response
            is llm.generate_answer.return_value
        )

    def test_valid_citation_metadata(self, pipeline):

        result = pipeline.ask(QUESTION)

        assert result.citation_validation is not None
        assert result.citation_validation.passed

        assert result.citation_validation.valid_ids == [
            VALID_ID
        ]

        assert (
            result.citation_validation.citation_validity
            == 1.0
        )


# ==========================================
# Relevance filtering
# ==========================================

class TestRelevanceFiltering:

    def test_irrelevant_question_skips_llm(
        self, pipeline, retrieval_engine, llm
    ):

        retrieval_engine.retrieve_with_relevance.return_value.update({
            "relevance_passed": False,
            "max_score": -11.1,
            "reason": "below_threshold",
            "retrieved_count": 5,
            "retained_count": 0,
            "results": [],
            "context": ""
        })

        result = pipeline.ask(
            "What is the weather in Alexandria?"
        )

        assert result.status == "insufficient_context"
        assert not result.success
        assert not result.relevance_passed
        assert result.retained_count == 0
        assert result.llm_response is None
        assert result.citation_validation is None
        assert result.answer

        llm.generate_answer.assert_not_called()

    def test_empty_chunks_skip_llm(
        self, pipeline, retrieval_engine, llm
    ):

        retrieval_engine.retrieve_with_relevance.return_value.update({
            "results": [],
            "retained_count": 0
        })

        result = pipeline.ask(QUESTION)

        assert result.status == "insufficient_context"

        llm.generate_answer.assert_not_called()

    @pytest.mark.parametrize(
        "context",
        ["", " ", "\n\t"]
    )
    def test_empty_context_skips_llm(
        self, pipeline, retrieval_engine, llm,
        context
    ):

        retrieval_engine.retrieve_with_relevance.return_value[
            "context"
        ] = context

        result = pipeline.ask(QUESTION)

        assert result.status == "insufficient_context"

        llm.generate_answer.assert_not_called()


# ==========================================
# Citation validation
# ==========================================

class TestCitationIntegration:

    @pytest.mark.parametrize(
        "answer, expected_status",
        [
            (
                f"Answer [{INVALID_ID}]",
                "invalid_citations"
            ),
            (
                f"Answer [{VALID_ID}] [{INVALID_ID}]",
                "invalid_citations"
            ),
            (
                "Answer without citations.",
                "missing_citations"
            ),
            (
                f"Answer ({VALID_ID})",
                "malformed_citations"
            ),
            (
                f"Answer [{VALID_ID}, {INVALID_ID}]",
                "malformed_citations"
            )
        ]
    )
    def test_invalid_answers_blocked(
        self, pipeline, llm,
        answer, expected_status
    ):

        llm.generate_answer.return_value.answer = answer

        result = pipeline.ask(QUESTION)

        assert result.status == "citation_error"
        assert not result.success
        assert result.answer is None
        assert result.error_message

        assert result.citation_validation is not None
        assert not result.citation_validation.passed

        assert (
            result.citation_validation.status
            == expected_status
        )

        llm.generate_answer.assert_called_once()

    def test_mixed_citation_validity(
        self, pipeline, llm
    ):

        llm.generate_answer.return_value.answer = (
            f"Answer [{VALID_ID}] [{INVALID_ID}]"
        )

        result = pipeline.ask(QUESTION)

        validation = result.citation_validation

        assert validation.citation_validity == 0.5
        assert validation.valid_ids == [VALID_ID]
        assert validation.invalid_ids == [INVALID_ID]

        assert result.answer is None

    def test_duplicate_valid_citations(
        self, pipeline, llm
    ):

        llm.generate_answer.return_value.answer = (
            f"Answer [{VALID_ID}] [{VALID_ID}]"
        )

        result = pipeline.ask(QUESTION)

        assert result.success
        assert result.citation_validation.passed
        assert result.citation_validation.citation_count == 1


# ==========================================
# Abstention handling
# ==========================================

class TestAbstention:

    @pytest.mark.parametrize(
        "question, expected_text",
        [
            (
                "What is the exact transcript fee?",
                "sufficient"
            ),
            (
                "\u0645\u0627 \u0647\u064a "
                "\u0627\u0644\u0631\u0633\u0648\u0645\u061f",
                "\u0644\u0627 \u062a\u062a\u0648\u0641\u0631"
            )
        ]
    )
    
    def test_bilingual_abstention(
        self, pipeline, llm,
        question, expected_text
    ):

        # Configure translation for the Arabic test case.
        llm.translate_query.return_value = LLMResponse(
            status="success",
            model="gemini-3.1-flash-lite",
            answer="What are the fees?",
            total_elapsed_seconds=0.5,
            attempt_count=1
        )

        # Simulate insufficient evidence during generation.
        llm.generate_answer.return_value.answer = (
            "INSUFFICIENT_CONTEXT"
        )

        result = pipeline.ask(question)


        assert result.status == "insufficient_context"
        assert result.relevance_passed
        assert result.llm_response.success
        assert result.citation_validation is None
        assert result.error_message is None

        assert expected_text in result.answer
        assert result.answer != "INSUFFICIENT_CONTEXT"

        llm.generate_answer.assert_called_once()

    def test_abstention_with_whitespace(
        self, pipeline, llm
    ):

        llm.generate_answer.return_value.answer = (
            "  INSUFFICIENT_CONTEXT\n"
        )

        result = pipeline.ask(QUESTION)

        assert result.status == "insufficient_context"
        assert result.citation_validation is None

    def test_abstention_marker_must_be_exact(
        self, pipeline, llm
    ):

        llm.generate_answer.return_value.answer = (
            "INSUFFICIENT_CONTEXT. Additional text."
        )

        result = pipeline.ask(QUESTION)

        assert result.status == "citation_error"
        assert result.answer is None

    def test_english_fallback_message(self):

        answer = insufficient_context_message(
            "What is the transcript fee?"
        )

        assert "sufficient" in answer

    def test_arabic_fallback_message(self):

        answer = insufficient_context_message(
            "\u0645\u0627 \u0647\u064a "
            "\u0627\u0644\u0631\u0633\u0648\u0645\u061f"
        )

        assert (
            "\u0644\u0627 \u062a\u062a\u0648\u0641\u0631"
            in answer
        )


# ==========================================
# LLM error handling
# ==========================================

class TestLLMErrors:

    @pytest.mark.parametrize(
        "code",
        [401, 429, 503, 504]
    )
    def test_generation_failure(
        self, pipeline, llm, code
    ):

        llm.generate_answer.return_value = LLMResponse(
            status="error",
            model="gemini-3.1-flash-lite",
            error_code=code,
            error_message="Simulated API failure.",
            attempt_count=1
        )

        result = pipeline.ask(QUESTION)

        assert result.status == "llm_error"
        assert not result.success
        assert result.answer is None
        assert result.error_message == "Simulated API failure."
        assert result.llm_response.error_code == code
        assert result.citation_validation is None

        llm.generate_answer.assert_called_once()

    def test_empty_llm_response(
        self, pipeline, llm
    ):

        llm.generate_answer.return_value = LLMResponse(
            status="empty_response",
            model="gemini-3.1-flash-lite",
            answer=None,
            error_message="Empty response.",
            attempt_count=1
        )

        result = pipeline.ask(QUESTION)

        assert result.status == "llm_error"
        assert result.answer is None
        assert result.citation_validation is None


# ==========================================
# Pipeline lifecycle
# ==========================================

class TestPipelineLifecycle:

    def test_injected_llm_not_closed(
        self, pipeline, llm
    ):

        pipeline.close()

        llm.close.assert_not_called()

    def test_owned_llm_closed_once(
        self, retrieval_engine
    ):

        with patch(
            "src.rag_pipeline.GeminiLLM"
        ) as factory:

            instance = RAGPipeline(
                retrieval_engine=retrieval_engine
            )

            instance.close()
            instance.close()

            factory.return_value.close.assert_called_once()

    def test_ask_after_close(
        self, pipeline, retrieval_engine, llm
    ):

        pipeline.close()

        with pytest.raises(RuntimeError):

            pipeline.ask(QUESTION)

        retrieval_engine.retrieve_with_relevance.assert_not_called()
        llm.generate_answer.assert_not_called()

    def test_context_manager_closes_owned_llm(
        self, retrieval_engine
    ):

        with patch(
            "src.rag_pipeline.GeminiLLM"
        ) as factory:

            with RAGPipeline(
                retrieval_engine=retrieval_engine
            ) as instance:

                assert isinstance(
                    instance, RAGPipeline
                )

            factory.return_value.close.assert_called_once()

# ==========================================
# Response time regression tests
# ==========================================

class TestResponseTimeRegression:

    @pytest.mark.parametrize(
        "scenario, expected_status",
        [
            ("success", "success"),
            ("citation_error", "citation_error"),
            ("llm_error", "llm_error"),
            ("abstention", "insufficient_context"),
            ("relevance_rejection", "insufficient_context")
        ]
    )
    def test_total_latency_recorded_for_all_paths(
        self,
        pipeline,
        retrieval_engine,
        llm,
        scenario,
        expected_status
    ):

        if scenario == "citation_error":

            llm.generate_answer.return_value.answer = (
                "Answer [AIU-999-P001-C001]"
            )

        elif scenario == "llm_error":

            llm.generate_answer.return_value = LLMResponse(
                status="error",
                model="gemini-3.1-flash-lite",
                error_code=429,
                error_message="Simulated API failure."
            )

        elif scenario == "abstention":

            llm.generate_answer.return_value.answer = (
                "INSUFFICIENT_CONTEXT"
            )

        elif scenario == "relevance_rejection":

            retrieval_engine.retrieve_with_relevance.return_value.update({
                "relevance_passed": False,
                "max_score": -11.1,
                "reason": "below_threshold",
                "retained_count": 0,
                "results": [],
                "context": ""
            })

        with patch(
            "src.rag_pipeline.time.perf_counter",
            side_effect=[100.0, 100.0, 101.5, 104.0]
        ):

            result = pipeline.ask(QUESTION)

        assert result.status == expected_status

        assert result.retrieval_latency_seconds == (
            pytest.approx(1.5)
        )

        assert result.total_latency_seconds == (
            pytest.approx(4.0)
        )


# ==========================================
# Automatic query translation integration
# ==========================================

class TestQueryTranslationIntegration:

    ARABIC_QUESTION = "هل التسجيل الصيفي إجباري؟"

    ARABIC_ANSWER = (
        "لا، التسجيل الصيفي اختياري. "
        f"[{VALID_ID}]"
    )

    MODEL = "gemini-3.1-flash-lite"

    def configure_translation(self, llm):

        response = LLMResponse(
            status="success",
            model=self.MODEL,
            answer=QUESTION,
            total_elapsed_seconds=0.5,
            attempt_count=1
        )

        llm.translate_query.return_value = response

        return response

    # ------------------------------------------
    # Successful Arabic integration
    # ------------------------------------------

    def test_arabic_question_integration(
        self, pipeline, retrieval_engine, llm
    ):

        translation_response = (
            self.configure_translation(llm)
        )

        llm.generate_answer.return_value.answer = (
            self.ARABIC_ANSWER
        )

        result = pipeline.ask(self.ARABIC_QUESTION)

        assert result.success
        assert result.status == "success"

        assert result.question == self.ARABIC_QUESTION
        assert result.retrieval_query == QUESTION
        assert result.translation_required is True

        assert result.translation_latency_seconds == 0.5
        assert result.translation_result.success

        assert (
            result.translation_result.llm_response
            is translation_response
        )

        assert result.answer == self.ARABIC_ANSWER
        assert result.citation_validation.passed

        assert result.citation_validation.valid_ids == [
            VALID_ID
        ]

        llm.translate_query.assert_called_once_with(
            self.ARABIC_QUESTION
        )

        retrieval_engine.retrieve_with_relevance.assert_called_once_with(
            query=QUESTION,
            top_k=5,
            candidate_k=20,
            rrf_k=60,
            threshold=-5.0
        )

        llm.generate_answer.assert_called_once_with(
            question=self.ARABIC_QUESTION,
            context=result.context
        )

    # ------------------------------------------
    # English bypass
    # ------------------------------------------

    def test_english_translation_bypass(
        self, pipeline, retrieval_engine, llm
    ):

        result = pipeline.ask(QUESTION)

        assert result.success
        assert result.question == QUESTION
        assert result.retrieval_query == QUESTION

        assert result.translation_required is False
        assert result.translation_latency_seconds == 0.0

        assert (
            result.translation_result.status
            == "not_required"
        )

        assert result.translation_result.llm_response is None

        llm.translate_query.assert_not_called()

        retrieval_engine.retrieve_with_relevance.assert_called_once_with(
            query=QUESTION,
            top_k=5,
            candidate_k=20,
            rrf_k=60,
            threshold=-5.0
        )

        llm.generate_answer.assert_called_once_with(
            question=QUESTION,
            context=result.context
        )

    # ------------------------------------------
    # Mixed-language questions
    # ------------------------------------------

    def test_mixed_language_question(
        self, pipeline, retrieval_engine, llm
    ):

        question = "هل Summer registration إجباري؟"

        self.configure_translation(llm)

        llm.generate_answer.return_value.answer = (
            self.ARABIC_ANSWER
        )

        result = pipeline.ask(question)

        assert result.success
        assert result.question == question
        assert result.retrieval_query == QUESTION
        assert result.translation_required is True

        llm.translate_query.assert_called_once_with(
            question
        )

        llm.generate_answer.assert_called_once_with(
            question=question,
            context=result.context
        )

    # ------------------------------------------
    # Translation API failures
    # ------------------------------------------

    @pytest.mark.parametrize(
        "status",
        [
            "rate_limited",
            "authentication_error",
            "timeout",
            "service_unavailable",
            "empty_response"
        ]
    )
    def test_translation_failure_stops_pipeline(
        self, pipeline, retrieval_engine, llm, status
    ):

        llm.translate_query.return_value = LLMResponse(
            status=status,
            model=self.MODEL,
            total_elapsed_seconds=0.8,
            error_message="Simulated translation failure."
        )

        result = pipeline.ask(self.ARABIC_QUESTION)

        assert not result.success
        assert result.status == "llm_error"

        assert result.question == self.ARABIC_QUESTION
        assert result.retrieval_query is None
        assert result.translation_required is True

        assert result.translation_latency_seconds == 0.8

        assert (
            result.translation_result.status
            == "translation_error"
        )

        assert result.answer is None
        assert result.error_message
        assert result.llm_response is None
        assert result.citation_validation is None

        assert result.retrieved_count == 0
        assert result.retained_count == 0
        assert result.retrieval_latency_seconds == 0.0

        llm.translate_query.assert_called_once_with(
            self.ARABIC_QUESTION
        )

        retrieval_engine.retrieve_with_relevance.assert_not_called()

        llm.generate_answer.assert_not_called()

    # ------------------------------------------
    # Invalid translation outputs
    # ------------------------------------------

    @pytest.mark.parametrize(
        "translation",
        [
            None,
            "",
            "12345",
            ARABIC_QUESTION,
            "INSUFFICIENT_CONTEXT"
        ]
    )
    def test_invalid_translation_stops_pipeline(
        self, pipeline, retrieval_engine, llm,
        translation
    ):

        llm.translate_query.return_value = LLMResponse(
            status="success",
            model=self.MODEL,
            answer=translation
        )

        result = pipeline.ask(self.ARABIC_QUESTION)

        assert not result.success
        assert result.status == "llm_error"

        assert result.retrieval_query is None
        assert result.translation_required is True

        assert (
            result.translation_result.status
            == "invalid_translation"
        )

        assert result.answer is None
        assert result.error_message

        retrieval_engine.retrieve_with_relevance.assert_not_called()

        llm.generate_answer.assert_not_called()

    # ------------------------------------------
    # Relevance rejection after translation
    # ------------------------------------------

    def test_relevance_rejection_after_translation(
        self, pipeline, retrieval_engine, llm
    ):

        question = "ما هو الطقس في الإسكندرية؟"

        translated_query = (
            "What is the weather in Alexandria?"
        )

        llm.translate_query.return_value = LLMResponse(
            status="success",
            model=self.MODEL,
            answer=translated_query,
            total_elapsed_seconds=0.5
        )

        retrieval_engine.retrieve_with_relevance.return_value.update({
            "relevance_passed": False,
            "max_score": -11.1,
            "reason": "below_threshold",
            "retrieved_count": 5,
            "retained_count": 0,
            "results": [],
            "context": ""
        })

        result = pipeline.ask(question)

        assert result.status == "insufficient_context"
        assert not result.success

        assert result.translation_required is True
        assert result.translation_result.success
        assert result.retrieval_query == translated_query

        assert not result.relevance_passed
        assert result.retained_count == 0

        assert "لا تتوفر" in result.answer

        assert result.llm_response is None
        assert result.citation_validation is None

        llm.translate_query.assert_called_once_with(
            question
        )

        retrieval_engine.retrieve_with_relevance.assert_called_once_with(
            query=translated_query,
            top_k=5,
            candidate_k=20,
            rrf_k=60,
            threshold=-5.0
        )

        llm.generate_answer.assert_not_called()

    # ------------------------------------------
    # Citation validation after translation
    # ------------------------------------------

    def test_invalid_citation_after_translation(
        self, pipeline, llm
    ):

        self.configure_translation(llm)

        llm.generate_answer.return_value.answer = (
            f"التسجيل الصيفي اختياري. [{INVALID_ID}]"
        )

        result = pipeline.ask(self.ARABIC_QUESTION)

        assert result.translation_result.success
        assert result.status == "citation_error"
        assert not result.success
        assert result.answer is None

        assert not result.citation_validation.passed

        assert result.citation_validation.invalid_ids == [
            INVALID_ID
        ]

    # ------------------------------------------
    # Generation failure after translation
    # ------------------------------------------

    def test_generation_failure_after_translation(
        self, pipeline, retrieval_engine, llm
    ):

        self.configure_translation(llm)

        llm.generate_answer.return_value = LLMResponse(
            status="rate_limited",
            model=self.MODEL,
            error_code=429,
            error_message="API usage limit reached."
        )

        result = pipeline.ask(self.ARABIC_QUESTION)

        assert result.translation_result.success
        assert result.relevance_passed

        assert result.status == "llm_error"
        assert not result.success
        assert result.answer is None

        assert result.llm_response.error_code == 429
        assert result.citation_validation is None

        llm.translate_query.assert_called_once()

        retrieval_engine.retrieve_with_relevance.assert_called_once()

        llm.generate_answer.assert_called_once()

    # ------------------------------------------
    # Translation and total response time
    # ------------------------------------------

    def test_total_latency_includes_translation(
        self, pipeline, llm
    ):

        self.configure_translation(llm)

        llm.generate_answer.return_value.answer = (
            self.ARABIC_ANSWER
        )

        with patch(
            "src.rag_pipeline.time.perf_counter",
            side_effect=[
                100.0,
                100.5,
                102.0,
                104.0
            ]
        ):

            result = pipeline.ask(self.ARABIC_QUESTION)

        assert result.success

        assert result.translation_latency_seconds == (
            pytest.approx(0.5)
        )

        assert result.retrieval_latency_seconds == (
            pytest.approx(1.5)
        )

        assert result.total_latency_seconds == (
            pytest.approx(4.0)
        )

    def test_translation_failure_latency(
        self, pipeline, retrieval_engine, llm
    ):

        llm.translate_query.return_value = LLMResponse(
            status="timeout",
            model=self.MODEL,
            total_elapsed_seconds=2.0,
            error_message="Translation timed out."
        )

        with patch(
            "src.rag_pipeline.time.perf_counter",
            side_effect=[100.0, 102.0]
        ):

            result = pipeline.ask(self.ARABIC_QUESTION)

        assert result.status == "llm_error"

        assert result.translation_latency_seconds == (
            pytest.approx(2.0)
        )

        assert result.total_latency_seconds == (
            pytest.approx(2.0)
        )

        assert result.retrieval_latency_seconds == 0.0

        retrieval_engine.retrieve_with_relevance.assert_not_called()

        llm.generate_answer.assert_not_called()
