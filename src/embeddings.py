"""Embedding generation and index storage for UniGuide AI.

Pipeline: data/processed/chunks.json -> multilingual-e5-base -> 768-dimension
unit vectors -> saved in data/embeddings/ so they can be reloaded without
re-embedding.

E5 models need a prefix on every input:
    "passage: " for document chunks and "query: " for student questions.
Each chunk is embedded together with its document title ("Attendance Policy",
then a new line, then the chunk text). Many chunks never mention the policy
name, and adding it made the right document rank higher in our tests.
Vectors are L2-normalized, so cosine similarity equals the dot product.

Index files (written by save_index, read by load_index):
    embeddings.npy         float32 matrix, shape (number of chunks, 768)
    chunks_metadata.json   the chunks, in the same row order as the matrix
    embedding_config.json  model name, dimension, prefixes, chunk count

Run from the project root:
    python src/embeddings.py
"""

import json
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_FILE = BASE_DIR / "data" / "processed" / "chunks.json"
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"

VECTORS_NAME = "embeddings.npy"
METADATA_NAME = "chunks_metadata.json"
CONFIG_NAME = "embedding_config.json"

# Must stay identical to MODEL_NAME in chunking.py (a test checks this).
MODEL_NAME = "intfloat/multilingual-e5-base"
EMBEDDING_DIMENSION = 768
MAX_INPUT_TOKENS = 512
# Room for the document title, the "passage: " prefix and the model's two
# special tokens, which are added in front of every chunk before embedding.
SPECIAL_TOKEN_RESERVE = 64
MAX_CHUNK_TOKENS = MAX_INPUT_TOKENS - SPECIAL_TOKEN_RESERVE

PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "
BATCH_SIZE = 16

REQUIRED_FIELDS = (
    "chunk_id",
    "document_id",
    "page",
    "source_url",
    "token_count",
    "text",
)


def load_model(model_name=MODEL_NAME):
    """Load the Sentence Transformers model (downloads it on first use)."""

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def load_chunks(chunks_path=CHUNKS_FILE):
    """Read the chunks produced by chunking.py."""

    chunks_path = Path(chunks_path)

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_path}. "
            "Run: python src/document_loader.py, then python src/chunking.py"
        )

    with open(chunks_path, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not isinstance(chunks, list):
        raise ValueError("Chunks must be a JSON list.")

    return chunks


def validate_chunks(chunks):
    """Check that every chunk can be embedded and traced back to its source."""

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("There are no chunks to embed.")

    seen_ids = set()

    for position, chunk in enumerate(chunks):

        if not isinstance(chunk, dict):
            raise ValueError(f"Chunk {position} must be a dictionary.")

        missing = [name for name in REQUIRED_FIELDS if name not in chunk]

        if missing:
            raise ValueError(
                f"Chunk {position} is missing fields: {', '.join(missing)}"
            )

        chunk_id = chunk["chunk_id"]

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(f"Chunk {position} has an invalid chunk_id.")

        if chunk_id in seen_ids:
            raise ValueError(f"Duplicate chunk ID: {chunk_id}")

        seen_ids.add(chunk_id)

        if not isinstance(chunk["text"], str) or not chunk["text"].strip():
            raise ValueError(f"Chunk {chunk_id} has empty text.")

        token_count = chunk["token_count"]

        if type(token_count) is not int or token_count <= 0:
            raise ValueError(f"Chunk {chunk_id} has an invalid token_count.")

        if token_count > MAX_CHUNK_TOKENS:
            raise ValueError(
                f"Chunk {chunk_id} has {token_count} tokens; "
                f"the limit is {MAX_CHUNK_TOKENS}."
            )

    return chunks


def validate_embeddings(vectors, expected_count=None):
    """Check shape, dimension, finite values and unit length."""

    vectors = np.asarray(vectors)

    if vectors.ndim != 2 or vectors.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected embeddings of shape (n, {EMBEDDING_DIMENSION}), "
            f"got {vectors.shape}."
        )

    if expected_count is not None and vectors.shape[0] != expected_count:
        raise ValueError(
            f"Expected {expected_count} embeddings, got {vectors.shape[0]}."
        )

    if not np.all(np.isfinite(vectors)):
        raise ValueError("Embeddings contain NaN or infinite values.")

    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3):
        raise ValueError("Embeddings are not normalized to unit length.")

    return vectors.astype(np.float32, copy=False)


def embed_passages(texts, model, batch_size=BATCH_SIZE):
    """Embed document passages. Returns a float32 matrix (len(texts), 768)."""

    if isinstance(texts, str) or texts is None or len(texts) == 0:
        raise ValueError("Provide a non-empty list of texts to embed.")

    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Every text to embed must be a non-empty string.")

    vectors = model.encode(
        [PASSAGE_PREFIX + text for text in texts],
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return validate_embeddings(vectors, expected_count=len(texts))


def embed_query(question, model):
    """Embed one student question. Returns a float32 vector of shape (768,)."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    vector = model.encode(
        QUERY_PREFIX + question.strip(),
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    vector = np.asarray(vector)

    if vector.ndim != 1:
        raise ValueError(f"Expected a single query vector, got {vector.shape}.")

    return validate_embeddings(vector[np.newaxis, :], expected_count=1)[0]


def passage_text(chunk):
    """The text that is embedded for a chunk: document title, then the chunk."""

    title = str(chunk.get("document_title") or "").strip()

    return f"{title}\n{chunk['text']}" if title else chunk["text"]


def embed_chunks(chunks, model, batch_size=BATCH_SIZE):
    """Embed every chunk, keeping the same order as the chunk list."""

    validate_chunks(chunks)

    return embed_passages(
        [passage_text(chunk) for chunk in chunks],
        model,
        batch_size,
    )


def save_index(chunks, vectors, directory=EMBEDDINGS_DIR, model_name=MODEL_NAME):
    """Save vectors, chunk metadata and configuration to disk."""

    validate_chunks(chunks)
    vectors = validate_embeddings(vectors, expected_count=len(chunks))

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    np.save(directory / VECTORS_NAME, vectors)

    with open(directory / METADATA_NAME, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    config = {
        "model_name": model_name,
        "dimension": EMBEDDING_DIMENSION,
        "chunk_count": len(chunks),
        "normalized": True,
        "includes_document_title": True,
        "passage_prefix": PASSAGE_PREFIX,
        "query_prefix": QUERY_PREFIX,
    }

    with open(directory / CONFIG_NAME, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    return directory


def load_index(directory=EMBEDDINGS_DIR):
    """Reload a saved index. Returns (chunks, vectors, config)."""

    directory = Path(directory)

    vectors_path = directory / VECTORS_NAME
    metadata_path = directory / METADATA_NAME
    config_path = directory / CONFIG_NAME

    if not all(path.exists() for path in (vectors_path, metadata_path, config_path)):
        raise FileNotFoundError(
            f"Embedding index not found in {directory}. "
            "Run: python src/embeddings.py"
        )

    vectors = np.load(vectors_path)

    with open(metadata_path, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    validate_chunks(chunks)
    vectors = validate_embeddings(vectors, expected_count=len(chunks))

    if config.get("chunk_count") != len(chunks):
        raise ValueError("Index configuration does not match the saved chunks.")

    return chunks, vectors, config


def build_index(
    chunks_path=CHUNKS_FILE,
    directory=EMBEDDINGS_DIR,
    model=None,
    batch_size=BATCH_SIZE,
):
    """Embed all chunks and save the index. Returns (chunks, vectors)."""

    chunks = validate_chunks(load_chunks(chunks_path))

    if model is None:
        print("Loading embedding model...")
        model = load_model()

    print(f"Embedding {len(chunks)} chunks...")
    vectors = embed_chunks(chunks, model, batch_size)

    save_index(chunks, vectors, directory)

    return chunks, vectors


def print_statistics(chunks, vectors, directory=EMBEDDINGS_DIR):
    """Display a short summary of the saved index."""

    print("\n" + "=" * 50)
    print("EMBEDDING STATISTICS")
    print("=" * 50)

    print("Model:", MODEL_NAME)
    print("Chunks:", len(chunks))
    print("Embedding matrix:", vectors.shape)
    print("Values stored:", vectors.size)
    print("Languages:", sorted({chunk.get("language", "unknown") for chunk in chunks}))
    print("Index saved to:", directory)


if __name__ == "__main__":

    chunks, vectors = build_index()
    print_statistics(chunks, vectors)
