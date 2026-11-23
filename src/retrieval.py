
"""Retrieval engine for UniGuide AI (Step 6 of the plan).

Pipeline: question -> embedding -> cosine similarity against every chunk
(NumPy) -> Top-K chunks with scores and source metadata.

The cosine similarity is written out with NumPy dot products and vector
norms, as the assignment requires:

    cos(q, d) = (q . d) / (||q|| * ||d||)

Student 3 (LLM + UI) can use it like this:

    retriever = Retriever()                       # load model and index once
    results = retriever.search("What is the attendance rule?", top_k=3)
    context = format_context(results)             # text to put in the LLM prompt
"""

import numpy as np

if __package__:
    from .embeddings import EMBEDDING_DIMENSION, EMBEDDINGS_PATH, embed_query, load_model
    from .semantic_search import CHUNKS_PATH, add_continuation, load_knowledge_base, select_passages
else:
    from embeddings import EMBEDDING_DIMENSION, EMBEDDINGS_PATH, embed_query, load_model
    from semantic_search import CHUNKS_PATH, add_continuation, load_knowledge_base, select_passages


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
        raise ValueError(f"Shape mismatch: query {query.shape} vs matrix {matrix.shape}.")
    dots = matrix @ query                                   # q . d for every chunk
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    return np.divide(dots, norms, out=np.zeros_like(dots), where=norms > 0)


def top_k_indices(scores, k):
    """Indices of the k highest scores, best first.

    The sort is stable, so equal scores keep their original chunk order and
    results are reproducible.
    """
    if k <= 0:
        return np.array([], dtype=int)
    return np.argsort(-np.asarray(scores), kind="stable")[:k]


def retrieve(question, model, chunks, embeddings, top_k=3, *, min_score=None, diversify=True):
    """Return the Top-K chunks for a question, each with its cosine score.

    Each result is the chunk's metadata plus "score" (raw cosine similarity)
    and "rank" (1 = best). With diversify=True the Top-K list is re-ranked to
    favor topical filenames and avoid near-duplicate passages, and one
    continuation passage may be attached under "context". Set
    diversify=False for plain cosine order. Scores are always the unadjusted
    cosine values. min_score drops weak matches so the app can say that no
    useful document was found.
    """
    if not isinstance(question, str) or not question.strip() or top_k <= 0:
        return []
    embeddings = np.asarray(embeddings)
    if embeddings.shape != (len(chunks), EMBEDDING_DIMENSION):
        raise ValueError("Embedding dimensions or row count do not match the knowledge base.")

    query = embed_query(question, model)
    scores = cosine_similarity(query, embeddings)
    ranked = top_k_indices(scores, len(scores))             # full ranking, best first
    chosen = select_passages(question, chunks, embeddings, scores, ranked, top_k) if diversify else ranked[:top_k]
    results = [{**chunks[index], "score": float(scores[index])} for index in chosen]
    if diversify and results:
        add_continuation(chunks, scores, ranked, chosen, results)
    if min_score is not None:
        results = [result for result in results if result["score"] >= min_score]
    for rank, result in enumerate(results, start=1):
        result["rank"] = rank
    return results


class Retriever:
    """Loads the model and the verified index once, then answers many searches."""

    def __init__(self, chunks_path=CHUNKS_PATH, embeddings_path=EMBEDDINGS_PATH, model=None):
        self.chunks, self.embeddings = load_knowledge_base(chunks_path, embeddings_path)
        self.model = model if model is not None else load_model()

    def search(self, question, top_k=3, *, min_score=None, diversify=True):
        return retrieve(question, self.model, self.chunks, self.embeddings, top_k,
                        min_score=min_score, diversify=diversify)


def _page_label(passage):
    page = passage.get("page_start")
    if page is None:
        return ""
    end = passage.get("page_end") or page
    return f", page {page}" if page == end else f", pages {page}-{end}"


def format_context(results):
    """Build the source block that goes into the LLM prompt (and the UI citations)."""
    blocks = []
    for result in results:
        for passage, note in [(result, ""), *((p, " (continuation)") for p in result.get("context", []))]:
            header = f"[Source {result['rank']}: {passage['source']}{_page_label(passage)}{note}]"
            blocks.append(f"{header}\n{passage['text']}")
    return "\n\n".join(blocks)
