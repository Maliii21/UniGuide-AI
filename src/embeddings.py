"""Embedding layer for UniGuide AI (Step 5 of the plan).

Turns text into fixed-length vectors with a Sentence Transformers model.
Document chunks and student questions MUST use the same model, so both paths
go through this module and share MODEL_NAME and the same normalization.

All vectors are float32, 384-dimensional and L2-normalized (unit length).
The index build/save/load logic stays in generate_embeddings.py and
semantic_search.py; this module re-exports it so the rest of the project
has one import point for everything embedding-related.
"""

import numpy as np

if __package__:
    from .generate_embeddings import (
        EMBEDDING_DIMENSION, EMBEDDINGS_PATH, generate_embeddings, save_knowledge_base, validate_vectors,
    )
    from .rebuild_chunks import MODEL_NAME, load_model
else:
    from generate_embeddings import (
        EMBEDDING_DIMENSION, EMBEDDINGS_PATH, generate_embeddings, save_knowledge_base, validate_vectors,
    )
    from rebuild_chunks import MODEL_NAME, load_model

__all__ = [
    "EMBEDDING_DIMENSION", "EMBEDDINGS_PATH", "MODEL_NAME", "embed_query", "embed_texts",
    "generate_embeddings", "load_model", "normalize_rows", "save_knowledge_base", "validate_vectors",
]


def normalize_rows(vectors):
    """Scale each row to unit length (L2 norm = 1) using NumPy.

    Accepts a 1-D vector or a 2-D matrix. Raises ValueError for a zero row,
    because a zero vector has no direction and cosine similarity is undefined.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if not np.all(norms > 0):
        raise ValueError("Cannot normalize a zero vector.")
    return vectors / norms


def embed_texts(texts, model, batch_size=8):
    """Embed a list of passages. Returns a (len(texts), 384) float32 array."""
    if isinstance(texts, str) or not len(texts):
        raise ValueError("Provide a non-empty list of texts to embed.")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Every text to embed must be a non-empty string.")
    vectors = np.asarray(model.encode(
        list(texts), batch_size=batch_size, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    ), dtype=np.float32)
    vectors = normalize_rows(vectors)  # guards against a model that skips normalization
    validate_vectors(vectors, len(texts))
    return vectors


def embed_query(question, model):
    """Embed one student question. Returns a (384,) float32 unit vector.

    Uses the same model and normalization as the document chunks, which is
    what makes question-to-chunk similarity scores meaningful.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")
    vector = np.asarray(model.encode(
        question.strip(), convert_to_numpy=True, normalize_embeddings=True,
    ), dtype=np.float32)
    if vector.shape != (EMBEDDING_DIMENSION,):
        raise ValueError(f"Expected a {EMBEDDING_DIMENSION}-value query vector, got {vector.shape}.")
    vector = normalize_rows(vector)
    validate_vectors(vector[np.newaxis, :], 1)
    return vector

