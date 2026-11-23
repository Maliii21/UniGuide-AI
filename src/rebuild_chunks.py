import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

if __package__:
    from .chunking import CHUNK_OVERLAP, CHUNK_SIZE, chunk_document
    from .document_loader import PROJECT_ROOT, read_documents
else:
    from chunking import CHUNK_OVERLAP, CHUNK_SIZE, chunk_document
    from document_loader import PROJECT_ROOT, read_documents

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "document_chunks.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME, device="cpu")


def model_input_limit(model):
    def fits(text):
        tokens = model.tokenizer(text, truncation=False, verbose=False)["input_ids"]
        return len(tokens) <= model.max_seq_length

    return fits


def atomic_write(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as staged:
        staged_path = Path(staged.name)
        try:
            staged.write(contents)
            staged.close()
            os.replace(staged_path, path)
        finally:
            staged.close()
            staged_path.unlink(missing_ok=True)


def chunk_json(chunks):
    return (json.dumps(chunks, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def build_chunks(model, *, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    documents, skipped = read_documents()
    for path, reason in skipped:
        print(f"Skipped {path.name}: {reason}")

    fits = model_input_limit(model)
    chunks = []
    processed = 0
    for path, parts in documents:
        try:
            passages = chunk_document(path, parts, fits=fits, chunk_size=chunk_size, overlap=overlap)
        except ValueError as error:
            skipped.append((path, str(error)))
            print(f"Skipped {path.name}: {error}")
            continue
        chunks.extend(passages)
        processed += 1
        print(f"{path.name}: {len(passages)} chunk(s)")
    if not chunks:
        raise ValueError("No usable document chunks in data/raw.")
    return chunks, processed, len(skipped)


def chunk_settings(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Maximum words per chunk.")
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP, help="Target overlap in words.")
    args = parser.parse_args()
    if not 0 <= args.overlap < args.chunk_size:
        parser.error("Chunk size must be positive and overlap must be smaller.")
    return args


def main():
    args = chunk_settings("Rebuild UniGuide AI's processed chunks.")
    try:
        chunks, processed, skipped = build_chunks(load_model(), chunk_size=args.chunk_size, overlap=args.overlap)
        atomic_write(CHUNKS_PATH, chunk_json(chunks))
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Unable to rebuild chunks: {error}")
        return 1
    print(f"Processed {processed} document(s), skipped {skipped}, created {len(chunks)} chunk(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
