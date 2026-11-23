import argparse
import io
import json
from pathlib import Path
import re

import numpy as np

if __package__:
    from .generate_embeddings import (
        CHUNKS_PATH, EMBEDDINGS_PATH, EMBEDDING_DIMENSION, index_manifest, load_model, manifest_path,
        validate_chunks, validate_vectors,
    )
else:
    from generate_embeddings import (
        CHUNKS_PATH, EMBEDDINGS_PATH, EMBEDDING_DIMENSION, index_manifest, load_model, manifest_path,
        validate_chunks, validate_vectors,
    )

# Keep reranking bounded and give cosine similarity most of the weight.
TITLE_BONUS = 0.06
REDUNDANCY_PENALTY = 0.18
MAX_CONTINUATION_SCORE_DROP = 0.15
TITLE_STOP_WORDS = {"aiu", "and", "for", "from", "of", "policy", "policies", "the", "with"}
MIN_CONTINUATION_WORDS = 12


def load_knowledge_base(chunks_path=CHUNKS_PATH, embeddings_path=EMBEDDINGS_PATH):
    """Load and verify a matching snapshot of chunk metadata and NumPy vectors."""
    try:
        manifest = json.loads(manifest_path(embeddings_path).read_bytes())
        chunk_bytes = chunks_path.read_bytes()
        vector_bytes = embeddings_path.read_bytes()
        if manifest != index_manifest(chunk_bytes, vector_bytes):
            raise ValueError("Chunk metadata and embeddings do not match the saved manifest.")
        chunks = json.loads(chunk_bytes)
        vectors = np.load(io.BytesIO(vector_bytes), allow_pickle=False)
        validate_chunks(chunks)
        if not isinstance(vectors, np.ndarray):
            raise ValueError("Expected a NumPy .npy array.")
        validate_vectors(vectors, len(chunks))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing knowledge-base file: {error.filename}. Rebuild the knowledge base.") from error
    except (OSError, ValueError, EOFError) as error:
        raise ValueError(f"Invalid knowledge base: {error} Run the rebuild command.") from error
    return chunks, vectors


def title_terms(text):
    words = re.findall(r"[^\W\d_]+", text.casefold())
    return {word.removesuffix("s")[:5] for word in words
            if len(word) >= 3 and word not in TITLE_STOP_WORDS}


def shingles(text):
    words = text.casefold().split()
    if len(words) < 5:
        return {tuple(words)}
    return set(zip(words, words[1:], words[2:], words[3:], words[4:]))


def overlap_fraction(first, second):
    return len(first & second) / min(len(first), len(second))


def select_passages(question, chunks, embeddings, scores, ranked, top_k):
    """Favor topical filenames and penalize overlap with evidence already selected."""
    pool = ranked[:max(12, 4 * top_k)]
    pool_vectors = embeddings[pool]
    query_terms = title_terms(question)
    relevance = scores[pool].copy()
    fingerprints = []
    for position, index in enumerate(pool):
        chunk = chunks[index]
        terms = title_terms(Path(chunk["source"]).stem)
        if terms:
            coverage = len(query_terms & terms) / len(terms)
            relevance[position] += TITLE_BONUS * coverage ** 2
        fingerprints.append(shingles(chunk["text"]))

    available = np.ones(len(pool), dtype=bool)
    redundancy = np.zeros(len(pool))
    selected = []
    for _ in range(min(top_k, len(pool))):
        position = np.argmax(np.where(available, relevance - REDUNDANCY_PENALTY * redundancy, -np.inf))
        selected.append(pool[position])
        available[position] = False
        semantic_overlap = np.clip((pool_vectors @ pool_vectors[position] - 0.8) / 0.2, 0, 1)
        text_overlap = [overlap_fraction(fingerprints[position], passage) for passage in fingerprints]
        redundancy = np.maximum(redundancy, np.maximum(semantic_overlap, text_overlap))
    return selected


def ends_mid_sentence(passage):
    return not passage["text"].rstrip().endswith((".", "!", "?"))


def add_continuation(chunks, scores, ranked, selected, results):
    """Attach at most one relevant continuation from the leading document."""
    ids = {chunk["chunk_id"]: index for index, chunk in enumerate(chunks)}
    used = set(selected)
    top_cosine = set(ranked[:len(selected)])
    first = results[0]
    source = first.get("source_path") or first["source"]
    # Prefer fixing an interrupted passage before restoring a suppressed top cosine hit.
    for index, result in sorted(zip(selected, results), key=lambda pair: not ends_mid_sentence(pair[1])):
        prefix, _, ordinal = result["chunk_id"].rpartition("#")
        if (result.get("source_path") or result["source"]) != source or not ordinal.isdecimal():
            continue
        neighbor = ids.get(f"{prefix}#{int(ordinal) + 1}")
        if neighbor is None or neighbor in used or scores[index] - scores[neighbor] > MAX_CONTINUATION_SCORE_DROP:
            continue
        if not ends_mid_sentence(result) and neighbor not in top_cosine:
            continue
        following = chunks[neighbor]
        if (following.get("source_path") or following["source"]) != source:
            continue
        words = result["text"].split()
        next_words = following["text"].split()
        # Require directional word overlap followed by new evidence.
        limit = min(len(words), len(next_words))
        overlap = next((size for size in range(limit, 7, -1) if words[-size:] == next_words[:size]), 0)
        if overlap and len(next_words) - overlap >= MIN_CONTINUATION_WORDS:
            result["context"] = [{**following, "score": float(scores[neighbor])}]
            return


def semantic_search(question, model, chunks, embeddings, top_k=2, *, diversify=True):
    """Return top-K passages with raw cosine scores and optional continuation context."""
    question = question.strip()
    if not question or top_k <= 0:
        return []
    if embeddings.shape != (len(chunks), EMBEDDING_DIMENSION):
        raise ValueError("Embedding dimensions or row count do not match the knowledge base.")
    query = np.asarray(model.encode(
        question, convert_to_numpy=True, normalize_embeddings=True,
    ))
    validate_vectors(query[np.newaxis, ...], 1)
    scores = embeddings @ query
    ranked = np.argsort(-scores, kind="stable")
    indices = select_passages(question, chunks, embeddings, scores, ranked, top_k) if diversify else ranked[:top_k]
    results = [{**chunks[index], "score": float(scores[index])} for index in indices]
    if diversify and results:
        add_continuation(chunks, scores, ranked, indices, results)
    return results


def print_passage(result):
    print("Source:", result["source"])
    if result.get("source_path"):
        print("Path:", result["source_path"])
    page = result.get("page_start")
    if page is not None:
        end = result.get("page_end") or page
        print(f"Page: {page}" if page == end else f"Pages: {page}-{end}")
    if result.get("section_headings"):
        print("Sections:", " | ".join(result["section_headings"]))
    print(f"Chunk: {result['chunk_id']} | Cosine similarity: {result['score']:.4f}")
    print(result["text"])


def print_results(question, model, chunks, embeddings, top_k, *, diversify=True):
    results = semantic_search(question, model, chunks, embeddings, top_k, diversify=diversify)
    if not results:
        print("Please enter a non-empty question.")
        return
    print(f"\nStudent question: {question}")
    for rank, result in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print_passage(result)
        for passage in result.get("context", []):
            print("\n--- Continuation ---")
            print_passage(passage)


def main():
    parser = argparse.ArgumentParser(description="Search UniGuide AI's document passages.")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--question", action="append", help="Question to search. Can be repeated.")
    parser.add_argument("--cosine-only", action="store_true", help="Use the original cosine ranking.")
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    try:
        chunks, embeddings = load_knowledge_base()
        model = load_model()
        if args.question:
            for question in args.question:
                print_results(question, model, chunks, embeddings, args.top_k, diversify=not args.cosine_only)
        else:
            print("Ask a question or type exit to quit.")
            while True:
                question = input("\nEnter your question: ").strip()
                if question.lower() == "exit":
                    break
                print_results(question, model, chunks, embeddings, args.top_k, diversify=not args.cosine_only)
    except (EOFError, KeyboardInterrupt):
        print()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Unable to search: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
