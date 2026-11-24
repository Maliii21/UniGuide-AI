"""Retrieval for UniGuide AI.

Pipeline: question -> "query: " embedding -> cosine similarity against every
chunk (NumPy) -> Top-K chunks with scores and source metadata.

The similarity is written out with NumPy dot products and vector norms:

    cos(q, d) = (q . d) / (||q|| * ||d||)

Usage from the LLM / UI code:

    retriever = Retriever()                 # loads the model and the index once
    results = retriever.search("What is the minimum attendance?", top_k=5)
    context = format_context(results)       # text for the LLM prompt

Command line (from the project root):
    python src/retrieval.py "What is the minimum attendance?" --top-k 5
"""

import argparse

import numpy as np

if __package__:
    from .embeddings import EMBEDDINGS_DIR, embed_query, load_index, load_model
else:
    from embeddings import EMBEDDINGS_DIR, embed_query, load_index, load_model


def cosine_similarity(query, matrix):
    """Cosine similarity between one query vector and every row of a matrix.

    query:  shape (dim,)
    matrix: shape (n, dim)
    Returns an array of shape (n,) with values in [-1, 1]. A zero-length
    vector has no direction, so its score is defined as 0.
    """

    query = np.asarray(query, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)

    if query.ndim != 1 or matrix.ndim != 2 or matrix.shape[1] != query.shape[0]:
        raise ValueError(
            f"Shape mismatch: query {query.shape} vs matrix {matrix.shape}."
        )

    dot_products = matrix @ query
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)

    return np.divide(
        dot_products,
        norms,
        out=np.zeros_like(dot_products),
        where=norms > 0,
    )


def top_k_indices(scores, k):
    """Indices of the k highest scores, best first.

    The sort is stable, so equal scores keep their original chunk order and
    results are reproducible.
    """

    if k <= 0:
        return np.array([], dtype=int)

    return np.argsort(-np.asarray(scores), kind="stable")[:k]


def retrieve(question, model, chunks, vectors, top_k=5, min_score=None):
    """Return the Top-K chunks for a question.

    Each result is a copy of the chunk metadata plus "score" (cosine
    similarity) and "rank" (1 = best). min_score drops weak matches so the
    app can say that no relevant document was found.
    """

    if not isinstance(question, str) or not question.strip() or top_k <= 0:
        return []

    vectors = np.asarray(vectors)

    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise ValueError("Embeddings do not match the chunk list.")

    query = embed_query(question, model)
    scores = cosine_similarity(query, vectors)

    results = []

    for index in top_k_indices(scores, top_k):

        score = float(scores[index])

        if min_score is not None and score < min_score:
            continue

        results.append({**chunks[index], "score": score})

    for rank, result in enumerate(results, start=1):
        result["rank"] = rank

    return results


class Retriever:
    """Loads the model and the saved index once, then answers many searches."""

    def __init__(self, directory=EMBEDDINGS_DIR, model=None):

        self.chunks, self.vectors, self.config = load_index(directory)
        self.model = model if model is not None else load_model(
            self.config["model_name"]
        )

    def search(self, question, top_k=5, min_score=None):

        return retrieve(
            question,
            self.model,
            self.chunks,
            self.vectors,
            top_k,
            min_score,
        )


def format_context(results):
    """Build the source block for the LLM prompt and the answer citations."""

    blocks = []

    for result in results:

        title = result.get("document_title") or result.get("source", "Unknown document")

        header = f"[Source {result['rank']}: {title}, page {result['page']}]"

        blocks.append(f"{header}\n{result['text']}")

    return "\n\n".join(blocks)


def print_results(question, results):
    """Display results in the terminal."""

    print(f"\nQuestion: {question}")

    if not results:
        print("No matching chunks found.")

    for result in results:

        print(f"\n--- Result {result['rank']} ---")
        print(f"Score:    {result['score']:.4f}")
        print(f"Document: {result.get('document_title', result.get('source'))}")
        print(f"Page:     {result['page']}")
        print(f"Chunk:    {result['chunk_id']}")
        print(f"URL:      {result['source_url']}")
        print(result["text"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Search the UniGuide AI index.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=None)
    arguments = parser.parse_args()

    retriever = Retriever()

    print_results(
        arguments.question,
        retriever.search(arguments.question, arguments.top_k, arguments.min_score),
    )
