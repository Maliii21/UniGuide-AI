# python src/embeddings.py
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_FILE = BASE_DIR / "data" / "processed" / "chunks.json"
OUTPUT_DIR = BASE_DIR / "data" / "embeddings"

MODEL_NAME = "intfloat/multilingual-e5-base"
DEVICE = "cpu"
BATCH_SIZE = 16


def load_model(model_name=MODEL_NAME, device=DEVICE):
    """Load the embedding model."""
    return SentenceTransformer(model_name, device=device)


def load_chunks(file_path=CHUNKS_FILE):
    """Load and validate processed text chunks."""
    with open(file_path, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Chunks must be a non-empty list.")

    seen_ids = set()

    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("Each chunk must be a dictionary.")

        chunk_id = chunk.get("chunk_id")
        text = chunk.get("text")

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("Missing or invalid chunk ID.")

        if chunk_id in seen_ids:
            raise ValueError(f"Duplicate chunk ID: {chunk_id}")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Empty or invalid text: {chunk_id}")

        seen_ids.add(chunk_id)

    return chunks


def embed_documents(chunks, model, batch_size=BATCH_SIZE):
    """Generate normalized embeddings for document chunks."""
    if not chunks:
        raise ValueError("No chunks provided.")

    texts = [
        f"passage: {chunk['text']}"
        for chunk in chunks
    ]

    # Prevent silent truncation
    for chunk, text in zip(chunks, texts):
        token_ids = model.tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False,
            verbose=False
        )

        if len(token_ids) > model.max_seq_length:
            raise ValueError(
                f"Chunk exceeds model token limit: "
                f"{chunk['chunk_id']}"
            )

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return np.asarray(embeddings, dtype=np.float32)


def embed_query(query, model):
    """Generate a normalized embedding for a user query."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    text = f"query: {query.strip()}"

    token_ids = model.tokenizer.encode(
        text,
        add_special_tokens=True,
        truncation=False,
        verbose=False
    )

    if len(token_ids) > model.max_seq_length:
        raise ValueError("Query exceeds model token limit.")

    embedding = model.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embedding = np.asarray(embedding, dtype=np.float32)

    validate_embeddings(
        embedding,
        expected_count=1,
        expected_dim=model.get_embedding_dimension()
    )

    return embedding[0]


def validate_embeddings(
    embeddings,
    expected_count,
    expected_dim
):
    """Validate embedding shape, values and normalization."""
    expected_shape = (expected_count, expected_dim)

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Invalid embedding shape: {embeddings.shape}. "
            f"Expected: {expected_shape}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinity.")

    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError("Embeddings are not normalized.")

    return True


def save_embeddings(
    embeddings,
    chunks,
    output_dir=OUTPUT_DIR,
    model_name=MODEL_NAME
):
    """Save embeddings, chunk records and model information."""
    output_dir = Path(output_dir)

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D matrix.")

    validate_embeddings(
        embeddings,
        expected_count=len(chunks),
        expected_dim=embeddings.shape[1]
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(
        output_dir / "embeddings.npy",
        embeddings,
        allow_pickle=False
    )

    with open(
        output_dir / "metadata.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    config = {
        "model_name": model_name,
        "embedding_dimension": int(embeddings.shape[1]),
        "number_of_embeddings": int(embeddings.shape[0]),
        "normalized": True,
        "dtype": str(embeddings.dtype),
        "document_prefix": "passage: ",
        "query_prefix": "query: "
    }

    with open(
        output_dir / "config.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(config, file, indent=2)

    print(f"Embeddings saved to: {output_dir}")


def main():
    print("Loading processed chunks...")
    chunks = load_chunks()
    print(f"Total chunks: {len(chunks)}")

    print("\nLoading embedding model...")
    model = load_model()

    dimension = model.get_embedding_dimension()

    print(f"Model: {MODEL_NAME}")
    print(f"Device: {model.device}")
    print(f"Embedding dimension: {dimension}")
    print(f"Maximum sequence length: {model.max_seq_length}")

    print("\nGenerating document embeddings...")
    embeddings = embed_documents(chunks, model)

    print("\nValidating embeddings...")
    validate_embeddings(
        embeddings,
        expected_count=len(chunks),
        expected_dim=dimension
    )

    print("Validation passed.")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Data type: {embeddings.dtype}")

    print("\nSaving embeddings and metadata...")
    save_embeddings(embeddings, chunks)

    print("\nEmbedding pipeline completed successfully.")


if __name__ == "__main__":
    main()
