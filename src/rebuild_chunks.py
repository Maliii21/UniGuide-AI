"""Write the two required Student 1 preprocessing outputs."""

if __package__:
    from .chunking import chunk_documents
    from .document_loader import DOCUMENTS_PATH, PROCESSED_FOLDER, document_records, read_documents, write_json
else:
    from chunking import chunk_documents
    from document_loader import DOCUMENTS_PATH, PROCESSED_FOLDER, document_records, read_documents, write_json

CHUNKS_PATH = PROCESSED_FOLDER / "chunks.json"


def main():
    try:
        documents, skipped = read_documents()
        for path, reason in skipped:
            print(f"Skipped {path.name}: {reason}")
        records = document_records(documents)
        chunks = chunk_documents(records)
        if not chunks:
            raise ValueError("No usable document chunks in data/raw.")
        write_json(DOCUMENTS_PATH, records)
        write_json(CHUNKS_PATH, chunks)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Unable to rebuild Student 1 outputs: {error}")
        return 1
    print(f"Processed {len(documents)} document(s), skipped {len(skipped)}, "
          f"created {len(records)} page record(s) and {len(chunks)} chunk(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
