import hashlib
import io
import json

import numpy as np

if __package__:
    from .rebuild_chunks import CHUNKS_PATH, MODEL_NAME, PROJECT_ROOT, atomic_write, chunk_json, load_model, model_input_limit
else:
    from rebuild_chunks import CHUNKS_PATH, MODEL_NAME, PROJECT_ROOT, atomic_write, chunk_json, load_model, model_input_limit

EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "document_embeddings.npy"
EMBEDDING_DIMENSION = 384


def validate_chunks(chunks):
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("No document chunks found. Rebuild the knowledge base.")
    required = ("chunk_id", "source", "text")
    if any(not isinstance(chunk, dict) or any(
        not isinstance(chunk.get(key), str) or not chunk[key].strip() for key in required
    ) for chunk in chunks):
        raise ValueError("Invalid chunk metadata. Rebuild the knowledge base.")
    if len({chunk["chunk_id"] for chunk in chunks}) != len(chunks):
        raise ValueError("Chunk IDs must be unique.")


def validate_vectors(vectors, count):
    if vectors.shape != (count, EMBEDDING_DIMENSION):
        raise ValueError(f"Expected {count} rows of {EMBEDDING_DIMENSION} values, got {vectors.shape}.")
    if vectors.dtype.kind != "f" or not np.isfinite(vectors).all():
        raise ValueError("Embeddings must contain finite floating-point values.")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError("Embeddings must be normalized to unit length.")


def generate_embeddings(chunks, model):
    validate_chunks(chunks)
    texts = [chunk["text"] for chunk in chunks]
    if not all(map(model_input_limit(model), texts)):
        raise ValueError("A chunk exceeds the model input limit. Rebuild the knowledge base.")
    vectors = np.asarray(model.encode(
        texts, batch_size=8, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    ), dtype=np.float32)
    validate_vectors(vectors, len(chunks))
    return vectors


def manifest_path(embeddings_path):
    return embeddings_path.with_suffix(".manifest.json")


def index_manifest(chunk_bytes, vector_bytes):
    return {
        "model": MODEL_NAME,
        "chunks_sha256": hashlib.sha256(chunk_bytes).hexdigest(),
        "embeddings_sha256": hashlib.sha256(vector_bytes).hexdigest(),
    }


def save_knowledge_base(chunks, vectors, chunks_path=CHUNKS_PATH, embeddings_path=EMBEDDINGS_PATH):
    validate_chunks(chunks)
    validate_vectors(vectors, len(chunks))
    chunk_bytes = chunk_json(chunks)
    buffer = io.BytesIO()
    np.save(buffer, vectors, allow_pickle=False)
    vector_bytes = buffer.getvalue()
    manifest = index_manifest(chunk_bytes, vector_bytes)
    # Publish the manifest last so readers can detect mixed files during a rebuild.
    atomic_write(chunks_path, chunk_bytes)
    atomic_write(embeddings_path, vector_bytes)
    atomic_write(manifest_path(embeddings_path), json.dumps(manifest, indent=2).encode("utf-8"))


def main():
    try:
        chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        validate_chunks(chunks)
        vectors = generate_embeddings(chunks, load_model())
        save_knowledge_base(chunks, vectors)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Unable to generate embeddings: {error}")
        return 1
    print(f"Saved {vectors.shape[0]} embeddings with {vectors.shape[1]} dimensions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
