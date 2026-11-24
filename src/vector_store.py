
import json
import hashlib
from pathlib import Path

import faiss
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent

EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"

EMBEDDINGS_FILE = EMBEDDINGS_DIR / "embeddings.npy"
METADATA_FILE = EMBEDDINGS_DIR / "metadata.json"
CONFIG_FILE = EMBEDDINGS_DIR / "config.json"
INDEX_FILE = EMBEDDINGS_DIR / "index.faiss"
MANIFEST_FILE = EMBEDDINGS_DIR / "index_manifest.json"


def file_hash(path):
    """Calculate a file fingerprint."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_embedding_data():
    """Load and validate embeddings, metadata and configuration."""

    embeddings = np.load(EMBEDDINGS_FILE, allow_pickle=False)

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        config = json.load(file)

    if embeddings.ndim != 2 or embeddings.size == 0:
        raise ValueError("Invalid embedding matrix.")

    if not isinstance(metadata, list):
        raise ValueError("Invalid metadata format.")

    if len(embeddings) != len(metadata):
        raise ValueError("Embeddings and metadata count mismatch.")

    if embeddings.shape[0] != config["number_of_embeddings"]:
        raise ValueError("Embedding count does not match configuration.")

    if embeddings.shape[1] != config["embedding_dimension"]:
        raise ValueError("Embedding dimension mismatch.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid values.")

    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError("Embeddings must be normalized.")

    chunk_ids = []

    for record in metadata:
        if not isinstance(record, dict):
            raise ValueError("Invalid metadata record.")

        chunk_id = record.get("chunk_id")
        text = record.get("text")

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("Invalid chunk ID.")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Missing text for {chunk_id}.")

        chunk_ids.append(chunk_id)

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs detected.")

    embeddings = np.ascontiguousarray(
        embeddings, dtype=np.float32
    )

    return embeddings, metadata, config


def build_index(embeddings):
    """Create an exact inner-product FAISS index."""

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    if index.ntotal != len(embeddings):
        raise ValueError("Index vector count mismatch.")

    return index


def save_index(index):
    """Save the index and its source fingerprints."""

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_FILE))

    manifest = {
        "index_type": "IndexFlatIP",
        "similarity_metric": "cosine",
        "embedding_dimension": index.d,
        "number_of_vectors": index.ntotal,
        "embeddings_hash": file_hash(EMBEDDINGS_FILE),
        "metadata_hash": file_hash(METADATA_FILE),
        "config_hash": file_hash(CONFIG_FILE),
        "index_hash": file_hash(INDEX_FILE)
    }

    with open(MANIFEST_FILE, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print(f"Index saved to: {INDEX_FILE}")


def load_index():
    """Load a saved index and verify its consistency."""

    if not INDEX_FILE.exists() or not MANIFEST_FILE.exists():
        raise FileNotFoundError("Build the FAISS index first.")

    with open(MANIFEST_FILE, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    files_to_check = {
        "embeddings_hash": EMBEDDINGS_FILE,
        "metadata_hash": METADATA_FILE,
        "config_hash": CONFIG_FILE,
        "index_hash": INDEX_FILE
    }

    for key, path in files_to_check.items():
        if file_hash(path) != manifest[key]:
            raise ValueError(
                f"File changed: {path.name}. Rebuild the index."
            )

    index = faiss.read_index(str(INDEX_FILE))

    if index.ntotal != manifest["number_of_vectors"]:
        raise ValueError("Saved index count mismatch.")

    if index.d != manifest["embedding_dimension"]:
        raise ValueError("Saved index dimension mismatch.")

    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError("Incorrect similarity metric.")

    return index


def search(index, query_embedding, metadata, top_k=5):
    """Retrieve the most similar chunks."""

    if index.ntotal == 0:
        raise ValueError("The index is empty.")

    if index.ntotal != len(metadata):
        raise ValueError("Index and metadata count mismatch.")

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    query = np.asarray(query_embedding, dtype=np.float32)

    if query.ndim == 1:
        query = query.reshape(1, -1)

    if query.shape != (1, index.d):
        raise ValueError("Query embedding dimension mismatch.")

    if not np.isfinite(query).all():
        raise ValueError("Invalid query embedding.")

    if not np.allclose(
        np.linalg.norm(query, axis=1), 1.0, atol=1e-5
    ):
        raise ValueError("Query embedding must be normalized.")

    query = np.ascontiguousarray(query)

    top_k = min(top_k, index.ntotal)

    scores, indices = index.search(query, top_k)

    results = []

    for score, position in zip(scores[0], indices[0]):

        if position < 0:
            continue

        result = metadata[int(position)].copy()

        result["similarity_score"] = float(score)

        results.append(result)

    return results


def main():

    print("Loading embedding data...")

    embeddings, metadata, config = load_embedding_data()

    print("Model:", config["model_name"])
    print("Embedding shape:", embeddings.shape)
    print("Metadata records:", len(metadata))

    print("\nBuilding FAISS index...")

    index = build_index(embeddings)

    print("Index type: IndexFlatIP")
    print("Similarity metric: Cosine similarity")
    print("Index dimension:", index.d)
    print("Total indexed vectors:", index.ntotal)

    print("\nSaving FAISS index...")

    save_index(index)

    print("\nReloading saved index...")

    loaded_index = load_index()

    print("Reloaded vectors:", loaded_index.ntotal)

    print("\nRunning self-retrieval test...")

    results = search(
        loaded_index,
        embeddings[0],
        metadata,
        top_k=5
    )

    for rank, result in enumerate(results, start=1):
        print(
            f"Rank {rank}: "
            f"{result['chunk_id']} | "
            f"Score: {result['similarity_score']:.6f}"
        )

    if not np.isclose(
        results[0]["similarity_score"], 1.0, atol=1e-5
    ):
        raise ValueError("Self-retrieval test failed.")

    if not any(
        result["chunk_id"] == metadata[0]["chunk_id"]
        for result in results
    ):
        raise ValueError("Original chunk was not retrieved.")

    print("\nVector store built and verified successfully.")


if __name__ == "__main__":
    main()
