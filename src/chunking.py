import json
import re
from pathlib import Path
from collections import Counter

from transformers import AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "documents.json"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "chunks.json"

MODEL_NAME = "intfloat/multilingual-e5-base"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 60


def load_tokenizer(model_name=MODEL_NAME):
    """Load the embedding model tokenizer."""

    return AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True
    )


def load_documents(input_path=INPUT_FILE):
    """Load processed documents from JSON."""

    with open(input_path, "r", encoding="utf-8") as file:
        documents = json.load(file)

    if not isinstance(documents, list):
        raise ValueError("Documents must be a JSON list.")

    return documents


def count_tokens(text, tokenizer):
    """Count tokens without adding special tokens."""

    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
            verbose=False
        )
    )


def split_text(
    text,
    tokenizer,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):
    """Split text into overlapping, boundary-aware chunks."""

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be between 0 and chunk_size - 1."
        )

    if not text or not text.strip():
        return []

    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required.")

    text = text.strip()

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        verbose=False
    )

    offsets = encoded["offset_mapping"]
    total_tokens = len(encoded["input_ids"])

    # Identify paragraph and sentence boundaries.
    boundaries = {
        match.end()
        for match in re.finditer(
            r"\n+|[.!?؟؛](?=\s|$)",
            text
        )
    }

    chunks = []
    start = 0

    while start < total_tokens:

        end = min(start + chunk_size, total_tokens)

        # Prefer a natural boundary near the chunk end.
        if end < total_tokens:

            minimum_end = start + max(
                overlap + 1,
                int(chunk_size * 0.8)
            )

            for candidate in range(end, minimum_end - 1, -1):

                char_end = offsets[candidate - 1][1]

                if char_end in boundaries:
                    end = candidate
                    break

        # Preserve the original text using character offsets.
        while end > start:

            char_start = offsets[start][0]
            char_end = offsets[end - 1][1]

            chunk_text = text[char_start:char_end].strip()

            if count_tokens(chunk_text, tokenizer) <= chunk_size:
                break

            end -= 1

        if end <= start:
            raise ValueError("Unable to create a valid chunk.")

        if chunk_text:
            chunks.append(chunk_text)

        if end >= total_tokens:
            break

        # Keep overlapping tokens and guarantee forward progress.
        start = max(start + 1, end - overlap)

    return chunks


def create_chunks(
    documents,
    tokenizer,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):
    """Create chunks while preserving document metadata."""

    chunks = []
    seen_page_ids = set()

    for document in documents:

        page_id = document["page_id"]

        if page_id in seen_page_ids:
            raise ValueError(f"Duplicate page ID: {page_id}")

        seen_page_ids.add(page_id)

        text_chunks = split_text(
            document["text"],
            tokenizer,
            chunk_size,
            overlap
        )

        for index, chunk_text in enumerate(
            text_chunks,
            start=1
        ):

            # Preserve all original document metadata.
            chunk = document.copy()

            chunk["chunk_id"] = (
                f"{page_id}-C{index:03d}"
            )

            chunk["chunk_index"] = index

            chunk["text"] = chunk_text

            chunk["token_count"] = count_tokens(
                chunk_text,
                tokenizer
            )

            chunks.append(chunk)

    return chunks


def save_chunks(chunks, output_path=OUTPUT_FILE):
    """Save chunks to a UTF-8 JSON file."""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w", encoding="utf-8") as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )


def print_statistics(documents, chunks):
    """Display chunking statistics."""

    token_counts = [
        chunk["token_count"]
        for chunk in chunks
    ]

    document_counts = Counter(
        chunk["document_id"]
        for chunk in chunks
    )

    print("\n" + "=" * 50)
    print("CHUNKING STATISTICS")
    print("=" * 50)

    print("Total documents:", len({
        doc["document_id"]
        for doc in documents
    }))

    print("Total pages:", len(documents))
    print("Total chunks:", len(chunks))

    if token_counts:

        print("Minimum tokens:", min(token_counts))
        print("Maximum tokens:", max(token_counts))

        print(
            "Average tokens:",
            round(
                sum(token_counts) / len(token_counts),
                2
            )
        )

    print("Chunk size:", CHUNK_SIZE)
    print("Chunk overlap:", CHUNK_OVERLAP)

    print("\nChunks per document:")

    for document_id, count in sorted(
        document_counts.items()
    ):
        print(f"{document_id}: {count}")

    print("\nOutput saved to:", OUTPUT_FILE)


if __name__ == "__main__":

    print("Loading tokenizer...")
    tokenizer = load_tokenizer()

    print("Loading documents...")
    documents = load_documents()

    print("Creating chunks...")
    chunks = create_chunks(documents, tokenizer)

    print("Saving chunks...")
    save_chunks(chunks)

    print_statistics(documents, chunks)