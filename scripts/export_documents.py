"""Export page text for local inspection before cleaning and chunking."""

from src.document_loader import PROCESSED_FOLDER, PROJECT_ROOT, read_documents
from src.rebuild_chunks import atomic_write, chunk_json

DOCUMENTS_PATH = PROCESSED_FOLDER / "documents.json"


def main():
    documents, skipped = read_documents()
    for path, reason in skipped:
        print(f"Skipped {path.name}: {reason}")

    records = [
        {
            "source": path.name,
            "source_path": path.relative_to(PROJECT_ROOT).as_posix(),
            "page": part.page,
            "text": part.text,
        }
        for path, parts in sorted(documents, key=lambda document: document[0].as_posix())
        for part in sorted(parts, key=lambda part: part.page or 0)
    ]
    atomic_write(DOCUMENTS_PATH, chunk_json(records))
    print(f"Exported {len(records)} record(s) from {len(documents)} document(s) to {DOCUMENTS_PATH}.")


if __name__ == "__main__":
    main()
