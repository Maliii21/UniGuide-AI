# python src/update_source_inventory.py
import csv
import json
from pathlib import Path
from collections import Counter


BASE_DIR = Path(__file__).resolve().parent.parent

SOURCES_FILE = BASE_DIR / "data" / "sources.csv"

CHUNKS_FILE = (
    BASE_DIR / "data" / "processed" / "chunks.json"
)


def update_chunk_counts(
    sources_file=SOURCES_FILE,
    chunks_file=CHUNKS_FILE
):
    """Update source inventory with generated chunk counts."""

    sources_file = Path(sources_file)

    # Load generated chunks.
    with open(chunks_file, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    # Count chunks for each document.
    chunk_counts = Counter(
        chunk["document_id"]
        for chunk in chunks
    )

    # Load the existing source inventory.
    with open(
        sources_file,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames or "document_id" not in fieldnames:
        raise ValueError("Invalid source inventory.")

    document_ids = [
        row["document_id"]
        for row in rows
    ]

    # Validate document IDs.
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("Duplicate document IDs found.")

    if set(document_ids) != set(chunk_counts):
        raise ValueError(
            "Document IDs do not match between sources and chunks."
        )

    # Add the column if it does not exist.
    if "chunk_count" not in fieldnames:
        fieldnames.append("chunk_count")

    # Update chunk counts.
    for row in rows:

        document_id = row["document_id"]

        row["chunk_count"] = str(
            chunk_counts[document_id]
        )

    # Validate the total before saving.
    total_chunks = sum(
        int(row["chunk_count"])
        for row in rows
    )

    if total_chunks != len(chunks):
        raise ValueError("Chunk count mismatch.")

    # Write to a temporary file first.
    temp_file = sources_file.with_suffix(".csv.tmp")

    with open(
        temp_file,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    # Replace the original after successful writing.
    temp_file.replace(sources_file)

    print("=" * 50)
    print("SOURCE INVENTORY UPDATED")
    print("=" * 50)

    print("Total documents:", len(rows))
    print("Total chunks:", total_chunks)

    print("\nChunks per document:")

    for document_id, count in sorted(
        chunk_counts.items()
    ):
        print(f"{document_id}: {count}")

    print("\nUpdated file:", sources_file)


if __name__ == "__main__":
    update_chunk_counts()