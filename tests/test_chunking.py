# python -m pytest tests/test_chunking.py -v
import json
import pytest

from src.chunking import (
    load_tokenizer,
    load_documents,
    count_tokens,
    split_text,
    create_chunks,
    save_chunks,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


@pytest.fixture(scope="session")
def tokenizer():
    return load_tokenizer()


@pytest.fixture
def sample_documents():
    return [
        {
            "document_id": "AIU-TEST",
            "page_id": "AIU-TEST-P001",
            "document_title": "Attendance Policy",
            "source": "attendance.pdf",
            "source_url": "https://aiu.edu.eg",
            "page": 1,
            "language": "English",
            "extraction_method": "native_text",
            "image_review_status": "not_required",
            "text": (
                "Students must attend their lectures. " * 150
            )
        },
        {
            "document_id": "AIU-TEST",
            "page_id": "AIU-TEST-P002",
            "document_title": "Attendance Policy",
            "source": "attendance.pdf",
            "source_url": "https://aiu.edu.eg",
            "page": 2,
            "language": "Arabic",
            "extraction_method": "native_text",
            "image_review_status": "not_required",
            "text": "يجب على الطلاب حضور المحاضرات."
        }
    ]


# Test 1: Token counting
def test_count_tokens(tokenizer):
    text = "AIU Attendance Policy"

    assert count_tokens(text, tokenizer) > 0


# Test 2: Empty text
def test_empty_text(tokenizer):
    assert split_text("", tokenizer) == []
    assert split_text("   ", tokenizer) == []


# Test 3: Short text
def test_short_text(tokenizer):
    text = "Students must attend lectures."

    chunks = split_text(text, tokenizer)

    assert len(chunks) == 1
    assert chunks[0] == text


# Test 4: Long text splitting
def test_long_text(tokenizer):
    text = "Students must attend lectures. " * 200

    chunks = split_text(text, tokenizer)

    assert len(chunks) > 1


# Test 5: Maximum chunk size
def test_maximum_chunk_size(tokenizer):
    text = "University academic policies. " * 300

    chunks = split_text(text, tokenizer)

    assert all(
        count_tokens(chunk, tokenizer) <= CHUNK_SIZE
        for chunk in chunks
    )


# Test 6: Overlapping content
# Test 6: Overlapping content
def test_overlap(tokenizer):
    text = " ".join(
        f"Student number {i} attends lectures."
        for i in range(200)
    )

    chunks = split_text(
        text,
        tokenizer,
        chunk_size=100,
        overlap=20
    )

    assert len(chunks) > 1

    for first, second in zip(chunks, chunks[1:]):
        max_length = min(len(first), len(second))

        shared_text = ""

        for n in range(max_length, 0, -1):
            if first[-n:] == second[:n]:
                shared_text = first[-n:]
                break

        assert shared_text.strip(), (
            "No overlapping text found between chunks"
        )

        overlap_tokens = count_tokens(
            shared_text,
            tokenizer
        )

        assert overlap_tokens >= 15, (
            f"Insufficient overlap: {overlap_tokens} tokens"
        )

        assert overlap_tokens <= 25, (
            f"Excessive overlap: {overlap_tokens} tokens"
        )

# Test 7: Arabic text preservation
def test_arabic_text(tokenizer):
    text = "يجب على الطلاب حضور المحاضرات."

    chunks = split_text(text, tokenizer)

    assert len(chunks) == 1
    assert chunks[0] == text


# Test 8: Metadata preservation
def test_metadata_preservation(
    sample_documents,
    tokenizer
):
    chunks = create_chunks(
        sample_documents,
        tokenizer
    )

    originals = {
        doc["page_id"]: doc
        for doc in sample_documents
    }

    for chunk in chunks:
        original = originals[chunk["page_id"]]

        for key, value in original.items():
            if key != "text":
                assert chunk[key] == value


# Test 9: Unique chunk IDs
def test_unique_chunk_ids(
    sample_documents,
    tokenizer
):
    chunks = create_chunks(
        sample_documents,
        tokenizer
    )

    ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    assert len(ids) == len(set(ids))


# Test 10: Page boundaries
def test_page_boundaries(
    sample_documents,
    tokenizer
):
    chunks = create_chunks(
        sample_documents,
        tokenizer
    )

    originals = {
        doc["page_id"]: doc["text"]
        for doc in sample_documents
    }

    for chunk in chunks:
        assert chunk["text"] in originals[
            chunk["page_id"]
        ]


# Test 11: Invalid chunk size
def test_invalid_chunk_size(tokenizer):
    with pytest.raises(ValueError):
        split_text(
            "Test text",
            tokenizer,
            chunk_size=0
        )


# Test 12: Invalid overlap
def test_invalid_overlap(tokenizer):
    with pytest.raises(ValueError):
        split_text(
            "Test text",
            tokenizer,
            chunk_size=100,
            overlap=100
        )

    with pytest.raises(ValueError):
        split_text(
            "Test text",
            tokenizer,
            chunk_size=100,
            overlap=-1
        )


# Test 13: Duplicate page IDs
def test_duplicate_page_ids(
    sample_documents,
    tokenizer
):
    documents = [
        sample_documents[0],
        sample_documents[0]
    ]

    with pytest.raises(ValueError):
        create_chunks(documents, tokenizer)


# Test 14: JSON saving
def test_save_chunks(
    sample_documents,
    tokenizer,
    tmp_path
):
    chunks = create_chunks(
        sample_documents,
        tokenizer
    )

    output_file = tmp_path / "chunks.json"

    save_chunks(chunks, output_file)

    assert output_file.exists()

    with open(
        output_file,
        "r",
        encoding="utf-8"
    ) as file:
        saved_chunks = json.load(file)

    assert saved_chunks == chunks


# Test 15: Text coverage
def test_text_coverage(tokenizer):
    text = " ".join(
        f"Academic policy section {i}."
        for i in range(200)
    )

    chunks = split_text(text, tokenizer)

    covered = [False] * len(text)
    previous_start = -1

    for chunk in chunks:
        start = text.find(
            chunk,
            previous_start + 1
        )

        assert start >= 0

        end = start + len(chunk)

        for position in range(start, end):
            covered[position] = True

        previous_start = start

    assert all(
        covered[i] or char.isspace()
        for i, char in enumerate(text)
    )


# Test 16: Embedding input length
def test_embedding_input_length(
    sample_documents,
    tokenizer
):
    chunks = create_chunks(
        sample_documents,
        tokenizer
    )

    for chunk in chunks:
        embedding_text = "passage: " + chunk["text"]

        tokens = tokenizer.encode(
            embedding_text,
            add_special_tokens=True,
            truncation=False
        )

        assert len(tokens) <= tokenizer.model_max_length