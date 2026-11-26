
"""
Citation validation for UniGuide-AI.

Validates generated citation IDs against
the retrieved university document chunks.
"""

import re

from dataclasses import dataclass, field


CHUNK_ID_PATTERN = r"AIU-\d{3}-P\d{3}-C\d{3}"

CITATION_PATTERN = re.compile(
    rf"\[({CHUNK_ID_PATTERN})\]"
)

CHUNK_ID_REGEX = re.compile(CHUNK_ID_PATTERN)


@dataclass
class CitationValidationResult:
    """Structured citation validation result."""

    status: str

    cited_ids: list = field(default_factory=list)
    valid_ids: list = field(default_factory=list)
    invalid_ids: list = field(default_factory=list)
    retrieved_ids: list = field(default_factory=list)

    malformed_citations: bool = False
    citation_count: int = 0
    valid_citation_count: int = 0
    citation_validity: float = 0.0

    @property
    def passed(self):
        return self.status == "valid"


def extract_citations(answer):
    """Extract unique, correctly formatted citation IDs."""

    if not isinstance(answer, str):
        raise TypeError("Answer must be a string.")

    citations = CITATION_PATTERN.findall(answer)

    return list(dict.fromkeys(citations))


def validate_citations(answer, retrieved_chunks):
    """Validate citations against retrieved chunk IDs."""

    if not isinstance(answer, str):
        raise TypeError("Answer must be a string.")

    retrieved_ids = list(dict.fromkeys(
        chunk["chunk_id"]
        for chunk in retrieved_chunks
    ))

    retrieved_set = set(retrieved_ids)

    cited_ids = extract_citations(answer)

    valid_ids = [
        citation
        for citation in cited_ids
        if citation in retrieved_set
    ]

    invalid_ids = [
        citation
        for citation in cited_ids
        if citation not in retrieved_set
    ]

    # Detect IDs outside the required bracket format.
    remaining_text = CITATION_PATTERN.sub("", answer)

    malformed_citations = bool(
        CHUNK_ID_REGEX.search(remaining_text)
        or re.search(r"AIU-", remaining_text)
    )

    citation_count = len(cited_ids)
    valid_citation_count = len(valid_ids)

    citation_validity = (
        valid_citation_count / citation_count
        if citation_count
        else 0.0
    )

    if not answer.strip():
        status = "empty_answer"

    elif malformed_citations:
        status = "malformed_citations"

    elif not cited_ids:
        status = "missing_citations"

    elif invalid_ids:
        status = "invalid_citations"

    else:
        status = "valid"

    return CitationValidationResult(
        status=status,
        cited_ids=cited_ids,
        valid_ids=valid_ids,
        invalid_ids=invalid_ids,
        retrieved_ids=retrieved_ids,
        malformed_citations=malformed_citations,
        citation_count=citation_count,
        valid_citation_count=valid_citation_count,
        citation_validity=citation_validity
    )
