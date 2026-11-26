
"""
Unit tests for UniGuide-AI conversation management.

No external API calls or model loading are required.
"""

import pytest

from src.conversation import (
    ChatMessage,
    ConversationManager
)

from src.rag_pipeline import RAGResponse
from src.citation_validator import validate_citations


QUESTION = "Is summer registration mandatory?"

VALID_ID = "AIU-003-P002-C002"
OTHER_ID = "AIU-009-P001-C002"

ANSWER = (
    "Summer registration is optional. "
    "[AIU-003-P002-C002]"
)


@pytest.fixture
def conversation():

    return ConversationManager()


@pytest.fixture
def chunks():

    return [
        {
            "chunk_id": VALID_ID,
            "document_title": "Academic Regulations",
            "page": 2,
            "source_url": "https://example.com/academic.pdf",
            "text": "Summer registration is optional."
        },
        {
            "chunk_id": OTHER_ID,
            "document_title": "Tuition Policy",
            "page": 1,
            "source_url": "https://example.com/tuition.pdf",
            "text": "Summer tuition information."
        }
    ]


@pytest.fixture
def successful_response(chunks):

    return RAGResponse(
        question=QUESTION,
        status="success",
        answer=ANSWER,
        retrieved_chunks=chunks,
        citation_validation=validate_citations(
            ANSWER,
            chunks
        ),
        total_latency_seconds=2.5
    )


# ==========================================
# Initialization and user messages
# ==========================================

class TestUserMessages:

    def test_empty_conversation(self, conversation):

        assert conversation.is_empty
        assert conversation.message_count == 0
        assert conversation.get_history() == []

    def test_add_user_message(self, conversation):

        message = conversation.add_user_message(
            QUESTION
        )

        assert isinstance(message, ChatMessage)
        assert message.role == "user"
        assert message.content == QUESTION
        assert message.timestamp

        assert conversation.message_count == 1
        assert not conversation.is_empty

    def test_whitespace_removed(self, conversation):

        message = conversation.add_user_message(
            f"  {QUESTION}  "
        )

        assert message.content == QUESTION

    @pytest.mark.parametrize(
        "content",
        ["", " ", "\n\t"]
    )
    def test_empty_message_rejected(
        self, conversation, content
    ):

        with pytest.raises(ValueError):

            conversation.add_user_message(content)

        assert conversation.is_empty

    @pytest.mark.parametrize(
        "content",
        [None, 123, []]
    )
    def test_invalid_message_type(
        self, conversation, content
    ):

        with pytest.raises(TypeError):

            conversation.add_user_message(content)

        assert conversation.is_empty

    def test_arabic_message(self, conversation):

        question = (
            "\u0645\u0627 \u0647\u064a "
            "\u0627\u0644\u0631\u0633\u0648\u0645\u061f"
        )

        message = conversation.add_user_message(
            question
        )

        assert message.content == question


# ==========================================
# Assistant responses
# ==========================================

class TestAssistantResponses:

    def test_successful_response(
        self, conversation, successful_response
    ):

        message = conversation.add_assistant_response(
            successful_response
        )

        assert message.role == "assistant"
        assert message.content == ANSWER
        assert message.status == "success"
        assert message.response_time == 2.5

        assert conversation.message_count == 1

    def test_message_order(
        self, conversation, successful_response
    ):

        conversation.add_user_message(QUESTION)

        conversation.add_assistant_response(
            successful_response
        )

        history = conversation.get_history()

        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].role == "assistant"

    def test_insufficient_context(self, conversation):

        response = RAGResponse(
            question=QUESTION,
            status="insufficient_context",
            answer="Insufficient information."
        )

        message = conversation.add_assistant_response(
            response
        )

        assert message.status == "insufficient_context"
        assert message.content == response.answer
        assert message.sources == []

    def test_citation_error_blocks_answer(
        self, conversation
    ):

        response = RAGResponse(
            question=QUESTION,
            status="citation_error",
            answer="Unverified generated answer."
        )

        message = conversation.add_assistant_response(
            response
        )

        assert message.status == "citation_error"

        assert (
            "Unverified generated answer."
            not in message.content
        )

        assert message.sources == []

    def test_llm_error_message(self, conversation):

        response = RAGResponse(
            question=QUESTION,
            status="llm_error",
            error_message="API usage limit reached."
        )

        message = conversation.add_assistant_response(
            response
        )

        assert message.status == "llm_error"
        assert message.content == response.error_message
        assert message.sources == []

    def test_llm_error_without_message(
        self, conversation
    ):

        response = RAGResponse(
            question=QUESTION,
            status="llm_error"
        )

        message = conversation.add_assistant_response(
            response
        )

        assert message.content == (
            "Unable to generate an answer."
        )

    def test_unknown_status(self, conversation):

        response = RAGResponse(
            question=QUESTION,
            status="unknown",
            answer="Unexpected answer."
        )

        message = conversation.add_assistant_response(
            response
        )

        assert message.content == (
            "Unable to process the response."
        )

        assert message.sources == []


# ==========================================
# Source management
# ==========================================

class TestSourceManagement:

    def test_only_cited_sources_stored(
        self, conversation, successful_response
    ):

        message = conversation.add_assistant_response(
            successful_response
        )

        assert len(message.sources) == 1

        assert message.sources[0]["chunk_id"] == (
            VALID_ID
        )

    def test_source_metadata(
        self, conversation, successful_response
    ):

        message = conversation.add_assistant_response(
            successful_response
        )

        source = message.sources[0]

        assert source["document_title"] == (
            "Academic Regulations"
        )

        assert source["page"] == 2

        assert source["source_url"] == (
            "https://example.com/academic.pdf"
        )

    def test_multiple_cited_sources(
        self, conversation, chunks
    ):

        answer = (
            f"Answer [{VALID_ID}] [{OTHER_ID}]"
        )

        response = RAGResponse(
            question=QUESTION,
            status="success",
            answer=answer,
            retrieved_chunks=chunks,
            citation_validation=validate_citations(
                answer,
                chunks
            )
        )

        message = conversation.add_assistant_response(
            response
        )

        assert len(message.sources) == 2

        assert {
            source["chunk_id"]
            for source in message.sources
        } == {VALID_ID, OTHER_ID}

    def test_duplicate_sources_removed(
        self, conversation, successful_response
    ):

        successful_response.retrieved_chunks.append(
            successful_response.retrieved_chunks[0]
        )

        message = conversation.add_assistant_response(
            successful_response
        )

        assert len(message.sources) == 1

    def test_no_sources_without_validation(
        self, conversation, successful_response
    ):

        successful_response.citation_validation = None

        message = conversation.add_assistant_response(
            successful_response
        )

        assert message.sources == []

    def test_no_sources_with_failed_validation(
        self, conversation, successful_response
    ):

        successful_response.citation_validation = (
            validate_citations(
                "Answer [AIU-999-P001-C001]",
                successful_response.retrieved_chunks
            )
        )

        message = conversation.add_assistant_response(
            successful_response
        )

        assert message.sources == []


# ==========================================
# History protection
# ==========================================

class TestHistoryProtection:

    def test_history_returns_independent_copy(
        self, conversation
    ):

        conversation.add_user_message(QUESTION)

        history = conversation.get_history()

        history[0].content = "Modified externally."

        assert conversation.get_history()[0].content == (
            QUESTION
        )

    def test_returned_message_is_independent(
        self, conversation
    ):

        message = conversation.add_user_message(
            QUESTION
        )

        message.content = "Modified externally."

        assert conversation.get_history()[0].content == (
            QUESTION
        )

    def test_history_list_cannot_be_modified(
        self, conversation
    ):

        conversation.add_user_message(QUESTION)

        history = conversation.get_history()

        history.clear()

        assert conversation.message_count == 1

    def test_nested_sources_are_protected(
        self, conversation, successful_response
    ):

        conversation.add_assistant_response(
            successful_response
        )

        history = conversation.get_history()

        history[0].sources[0]["page"] = 999

        assert (
            conversation.get_history()[0]
            .sources[0]["page"]
            == 2
        )


# ==========================================
# Session management
# ==========================================

class TestSessionManagement:

    def test_conversation_isolation(
        self, conversation
    ):

        another_conversation = ConversationManager()

        conversation.add_user_message(QUESTION)

        assert conversation.message_count == 1
        assert another_conversation.is_empty

    def test_clear_conversation(
        self, conversation, successful_response
    ):

        conversation.add_user_message(QUESTION)

        conversation.add_assistant_response(
            successful_response
        )

        conversation.clear()

        assert conversation.is_empty
        assert conversation.message_count == 0
        assert conversation.get_history() == []

    def test_clear_empty_conversation(
        self, conversation
    ):

        conversation.clear()

        assert conversation.is_empty

    def test_new_message_after_clear(
        self, conversation
    ):

        conversation.add_user_message(QUESTION)

        conversation.clear()

        conversation.add_user_message(
            "What is the attendance policy?"
        )

        assert conversation.message_count == 1

        assert conversation.get_history()[0].content == (
            "What is the attendance policy?"
        )

    def test_clear_does_not_affect_other_session(
        self, conversation
    ):

        another_conversation = ConversationManager()

        conversation.add_user_message(QUESTION)

        another_conversation.add_user_message(
            "Another question."
        )

        conversation.clear()

        assert conversation.is_empty
        assert another_conversation.message_count == 1
