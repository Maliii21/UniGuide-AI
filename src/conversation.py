
"""
Session-based conversation management for UniGuide-AI.

Stores user questions, assistant responses,
validated sources, and response metadata.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from copy import deepcopy
from typing import Optional

from .rag_pipeline import RAGResponse


@dataclass
class ChatMessage:
    """Represents one message in a conversation."""

    role: str
    content: str

    timestamp: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    status: Optional[str] = None

    sources: list = field(default_factory=list)

    response_time: Optional[float] = None


class ConversationManager:
    """Manages the conversation within a user session."""

    def __init__(self):

        self._messages = []

    def add_user_message(self, content):

        if not isinstance(content, str):
            raise TypeError("Message must be a string.")

        content = content.strip()

        if not content:
            raise ValueError("Message cannot be empty.")

        message = ChatMessage(
            role="user",
            content=content
        )

        self._messages.append(message)

        return deepcopy(message)

    def add_assistant_response(self, response: RAGResponse):

        sources = []

        # Store only validated, cited sources.
        if (
            response.status == "success"
            and response.citation_validation is not None
            and response.citation_validation.passed
        ):

            valid_ids = set(
                response.citation_validation.valid_ids
            )

            seen_ids = set()

            for chunk in response.retrieved_chunks:

                chunk_id = chunk["chunk_id"]

                if (
                    chunk_id in valid_ids
                    and chunk_id not in seen_ids
                ):

                    sources.append({
                        "chunk_id": chunk_id,
                        "document_title": chunk.get(
                            "document_title", ""
                        ),
                        "page": chunk.get("page"),
                        "source_url": chunk.get(
                            "source_url", ""
                        )
                    })

                    seen_ids.add(chunk_id)

        # Never display an unverified generated answer.
        if response.status == "citation_error":

            content = (
                "The generated answer could not be "
                "verified against the university sources. "
                "Please try rephrasing your question."
            )

        elif response.status == "llm_error":

            content = (
                response.error_message
                or "Unable to generate an answer."
            )

        elif response.status in (
            "success", "insufficient_context"
        ):

            content = (
                response.answer
                or "No answer is available."
            )

        else:

            content = "Unable to process the response."

        message = ChatMessage(
            role="assistant",
            content=content,
            status=response.status,
            sources=sources,
            response_time=response.total_latency_seconds
        )

        self._messages.append(message)

        return deepcopy(message)

    def get_history(self):
        """Return an independent copy of chat history."""

        return deepcopy(self._messages)

    def clear(self):
        """Clear the current conversation."""

        self._messages.clear()

    @property
    def message_count(self):

        return len(self._messages)

    @property
    def is_empty(self):

        return not self._messages
