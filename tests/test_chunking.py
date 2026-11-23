import pytest
import tiktoken

from src.chunking import ENCODING_NAME, chunk_document, chunk_documents


@pytest.fixture(scope="module")
def encoding():
    return tiktoken.get_encoding(ENCODING_NAME)


def test_token_limit_and_exact_overlap_preserve_unicode_text(encoding):
    text = " ".join(["Regulations 家庭 🙂 العربية apply."] * 150)
    chunks = chunk_document("policy.pdf", [{"page": 1, "text": text}], encoding)
    token_lists = [encoding.encode_ordinary(chunk["text"]) for chunk in chunks]

    assert len(chunks) > 1
    assert max(map(len, token_lists)) == 500
    assert len(chunks[0]["text"].split()) < 500
    restored = token_lists[0].copy()
    for left, right in zip(token_lists, token_lists[1:]):
        assert left[-75:] == right[:75]
        restored.extend(right[75:])
    assert encoding.decode(restored, errors="strict") == text


def test_paragraph_boundaries_and_starting_page_metadata(encoding):
    first = "Rules" + " apply" * 380
    second = "Further" + " conditions" * 800
    pages = [{"page": 2, "text": first}, {"page": 3, "text": second}]

    chunks = chunk_document("policy.pdf", pages, encoding)

    assert chunks[0]["text"] == first + "\n\n"
    assert [chunk["page"] for chunk in chunks[:3]] == [2, 2, 3]
    assert chunk_document("policy.pdf", pages, encoding) == chunks
    for number, chunk in enumerate(chunks, 1):
        assert set(chunk) == {"chunk_id", "source", "page", "text"}
        assert chunk["source"] == "policy.pdf"
        assert chunk["chunk_id"] == f"policy.pdf#p{chunk['page']}_c{number}"


def test_separate_documents_do_not_share_text_or_ids():
    chunks = chunk_documents([
        {"source": "a.pdf", "page": 1, "text": "First policy."},
        {"source": "b.pdf", "page": 1, "text": "Second policy."},
    ])

    assert [chunk["text"] for chunk in chunks] == ["First policy.", "Second policy."]
    assert len({chunk["chunk_id"] for chunk in chunks}) == 2


def test_empty_document_produces_no_chunks(encoding):
    assert chunk_document("empty.pdf", [], encoding) == []
    assert chunk_document("empty.pdf", [{"page": 1, "text": ""}], encoding) == []
