
"""
RAG pipeline for UniGuide-AI.

Connects document retrieval, relevance filtering
and grounded LLM answer generation.
"""

import re
import time

from dataclasses import dataclass, field
from typing import Optional

from .retrieval import RetrievalEngine
from .relevance_filter import DEFAULT_THRESHOLD
from .llm import GeminiLLM, LLMResponse
from .query_translator import (
    QueryTranslator,
    TranslationResult
)

from .citation_validator import (
    CitationValidationResult,
    validate_citations
)

# --------------------------------------------------
# Pipeline Configuration
# --------------------------------------------------

DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_K = 20
DEFAULT_RRF_K = 60

# --------------------------------------------------
# Response Structure
# --------------------------------------------------

@dataclass
class RAGResponse:
    """Structured response returned by the RAG pipeline."""

    question: str
    status: str

    answer: Optional[str] = None

    relevance_passed: bool = False
    max_relevance_score: Optional[float] = None
    relevance_threshold: Optional[float] = None
    relevance_reason: Optional[str] = None

    retrieved_count: int = 0
    retained_count: int = 0

    retrieved_chunks: list = field(default_factory=list)

    context: str = ""

    retrieval_latency_seconds: float = 0.0
    total_latency_seconds: float = 0.0

    llm_response: Optional[LLMResponse] = None

    citation_validation: Optional[CitationValidationResult] = None

    error_message: Optional[str] = None
    retrieval_query: Optional[str] = None

    translation_required: bool = False

    translation_latency_seconds: float = 0.0

    translation_result: Optional[TranslationResult] = None

    @property
    def success(self):
        return self.status == "success"

# --------------------------------------------------
# Language-aware fallback
# --------------------------------------------------

def insufficient_context_message(question):
    """Return a fallback message in English or Arabic."""

    if re.search(r"[\u0600-\u06FF]", question):
        return (
            "لا تتوفر معلومات كافية في وثائق الجامعة "
            "المسترجعة للإجابة عن هذا السؤال."
        )

    return (
        "The retrieved university documents do not "
        "contain sufficient relevant information "
        "to answer this question."
    )

# --------------------------------------------------
# RAG Pipeline
# --------------------------------------------------

class RAGPipeline:
    """Coordinate retrieval and LLM answer generation."""

    def __init__(
        self,
        retrieval_engine=None,
        llm=None,
        device="cpu"
    ):

        self._owns_llm = llm is None
        self._closed = False

        self.retrieval_engine = (
            retrieval_engine
            if retrieval_engine is not None
            else RetrievalEngine(device=device)
        )

        self.llm = (
            llm if llm is not None else GeminiLLM()
        )
        self.query_translator = QueryTranslator(self.llm)

    def ask(
        self,
        question,
        top_k=DEFAULT_TOP_K,
        candidate_k=DEFAULT_CANDIDATE_K,
        rrf_k=DEFAULT_RRF_K,
        threshold=DEFAULT_THRESHOLD
    ):
        """Retrieve relevant evidence and generate an answer."""

        if self._closed:
            raise RuntimeError("The RAG pipeline is closed.")

        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                "Question must be a non-empty string."
            )

        
        question = question.strip()

        total_start = time.perf_counter()

        # Step 0: Prepare the retrieval query.
        translation = self.query_translator.translate(question)

        # Stop if translation failed.
        if not translation.success:

            return RAGResponse(
                question=question,
                status="llm_error",
                retrieval_query=None,
                translation_required=(
                    translation.translation_required
                ),
                translation_latency_seconds=(
                    translation.latency_seconds
                ),
                translation_result=translation,
                error_message=(
                    "Unable to translate the question. "
                    "Please try again or ask in English."
                ),
                total_latency_seconds=round(
                    time.perf_counter() - total_start, 4
                )
            )

        retrieval_query = translation.retrieval_query

        retrieval_start = time.perf_counter()

        # Step 1: Retrieve and filter relevant chunks.
        retrieval = self.retrieval_engine.retrieve_with_relevance(
            query=retrieval_query,
            top_k=top_k,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            threshold=threshold
        )


        retrieval_latency = (
            time.perf_counter() - retrieval_start
        )

        # Step 2: Prepare response metadata.
        
        result = RAGResponse(
            question=question,
            status="pending",
            retrieval_query=retrieval_query,
            translation_required=(
                translation.translation_required
            ),
            translation_latency_seconds=(
                translation.latency_seconds
            ),
            translation_result=translation,
            relevance_passed=retrieval["relevance_passed"],
            max_relevance_score=retrieval["max_score"],
            relevance_threshold=retrieval["threshold"],
            relevance_reason=retrieval["reason"],
            retrieved_count=retrieval["retrieved_count"],
            retained_count=retrieval["retained_count"],
            retrieved_chunks=retrieval["results"],
            context=retrieval["context"],
            retrieval_latency_seconds=round(
                retrieval_latency, 4
            )
        )

        # Step 3: Reject irrelevant or empty context.
        if (
            not result.relevance_passed
            or not result.retrieved_chunks
            or not result.context.strip()
        ):

            result.status = "insufficient_context"

            result.answer = insufficient_context_message(
                question
            )

            result.total_latency_seconds = round(
                time.perf_counter() - total_start, 4
            )

            # No generation API request is made.
            return result

        # Step 4: Generate an answer using retrieved context.
        llm_result = self.llm.generate_answer(
            question=question,
            context=result.context
        )

        result.llm_response = llm_result

        # Step 5: Handle generation results.
        if llm_result.success:

            generated_answer = llm_result.answer or ""

            # Handle explicit LLM abstention.
            if generated_answer.strip() == "INSUFFICIENT_CONTEXT":

                result.status = "insufficient_context"

                result.answer = insufficient_context_message(
                    question
                )

                result.total_latency_seconds = round(
                    time.perf_counter() - total_start, 4
                )

                return result

            # Step 6: Validate generated citations.
            validation = validate_citations(
                answer=generated_answer,
                retrieved_chunks=result.retrieved_chunks
            )

            result.citation_validation = validation

            if validation.passed:

                result.status = "success"
                result.answer = generated_answer

            else:

                result.status = "citation_error"
                result.answer = None

                result.error_message = (
                    "The generated answer could not "
                    "be verified against the retrieved "
                    "university sources."
                )

        else:

            result.status = "llm_error"
            result.error_message = llm_result.error_message

        # Record total pipeline execution time.
        result.total_latency_seconds = round(
            time.perf_counter() - total_start, 4
        )

        return result

    def close(self):
        """Release owned LLM resources."""

        if self._closed:
            return

        if self._owns_llm:
            self.llm.close()

        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()