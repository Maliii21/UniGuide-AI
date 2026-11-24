
from .embeddings import load_model, embed_query

from .vector_store import (
    load_embedding_data,
    load_index,
    search
)

from .keyword_search import BM25Search
from .rank_fusion import reciprocal_rank_fusion
from .reranker import Reranker

from .relevance_filter import (
    check_relevance,
    DEFAULT_THRESHOLD
)


class RetrievalEngine:
    """Semantic, keyword, hybrid and reranked retrieval."""

    def __init__(self, device="cpu"):

        self.device = device

        # Load metadata and configuration
        _, self.metadata, self.config = load_embedding_data()

        # Load FAISS index
        self.index = load_index()

        # Load embedding model
        self.model = load_model(
            model_name=self.config["model_name"],
            device=device
        )

        # Validate model compatibility
        if self.model.get_embedding_dimension() != self.index.d:
            raise ValueError(
                "Model and index dimensions do not match."
            )

        # Initialize BM25
        self.keyword_engine = BM25Search(self.metadata)

        # Lazy loading for the reranking model
        self.reranker = None

    def semantic_search(self, query, top_k=5):
        """Retrieve chunks using semantic similarity."""

        query_embedding = embed_query(
            query,
            self.model
        )

        return search(
            index=self.index,
            query_embedding=query_embedding,
            metadata=self.metadata,
            top_k=top_k
        )

    def keyword_search(self, query, top_k=5):
        """Retrieve chunks using BM25."""

        return self.keyword_engine.search(
            query=query,
            top_k=top_k
        )

    def hybrid_search(
        self,
        query,
        top_k=5,
        candidate_k=20,
        rrf_k=60
    ):
        """Combine semantic and keyword search using RRF."""

        if type(candidate_k) is not int or candidate_k <= 0:
            raise ValueError(
                "candidate_k must be a positive integer."
            )

        if candidate_k < top_k:
            raise ValueError(
                "candidate_k must be greater than or equal to top_k."
            )

        semantic_results = self.semantic_search(
            query,
            top_k=candidate_k
        )

        keyword_results = self.keyword_search(
            query,
            top_k=candidate_k
        )

        return reciprocal_rank_fusion(
            semantic_results=semantic_results,
            keyword_results=keyword_results,
            top_k=top_k,
            k=rrf_k
        )

    def load_reranker(self):
        """Initialize the reranker only when needed."""

        if self.reranker is None:

            self.reranker = Reranker(
                device=self.device,
                use_title=False
            )

        return self.reranker

    def reranked_search(
        self,
        query,
        top_k=5,
        candidate_k=20,
        rrf_k=60
    ):
        """Retrieve hybrid candidates and rerank them."""

        if type(candidate_k) is not int or candidate_k <= 0:
            raise ValueError(
                "candidate_k must be a positive integer."
            )

        if candidate_k < top_k:
            raise ValueError(
                "candidate_k must be greater than or equal to top_k."
            )

        # Retrieve the candidate pool
        candidates = self.hybrid_search(
            query=query,
            top_k=candidate_k,
            candidate_k=candidate_k,
            rrf_k=rrf_k
        )

        if not candidates:
            return []

        # Load the Cross-Encoder when needed
        reranker = self.load_reranker()

        # Rerank and return the final top-k chunks
        return reranker.rerank(
            query=query,
            results=candidates,
            top_k=top_k
        )

    def retrieve(
        self,
        query,
        top_k=5,
        mode="hybrid",
        candidate_k=20,
        rrf_k=60
    ):
        """Retrieve chunks using the selected search mode."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "Query must be a non-empty string."
            )

        if type(top_k) is not int or top_k <= 0:
            raise ValueError(
                "top_k must be a positive integer."
            )

        valid_modes = (
            "semantic",
            "keyword",
            "hybrid",
            "reranked"
        )

        if mode not in valid_modes:
            raise ValueError(
                "Mode must be semantic, keyword, hybrid or reranked."
            )

        query = query.strip()

        if mode == "semantic":

            results = self.semantic_search(
                query,
                top_k=top_k
            )

        elif mode == "keyword":

            results = self.keyword_search(
                query,
                top_k=top_k
            )

        elif mode == "hybrid":

            results = self.hybrid_search(
                query,
                top_k=top_k,
                candidate_k=candidate_k,
                rrf_k=rrf_k
            )

        else:

            results = self.reranked_search(
                query,
                top_k=top_k,
                candidate_k=candidate_k,
                rrf_k=rrf_k
            )

        # Assign final ranks and retrieval mode
        for rank, result in enumerate(results, start=1):

            result["rank"] = rank
            result["retrieval_mode"] = mode

        return results

    def retrieve_with_relevance(
        self,
        query,
        top_k=5,
        candidate_k=20,
        rrf_k=60,
        threshold=DEFAULT_THRESHOLD
    ):
        """
        Retrieve, rerank and check context relevance.

        Passing the relevance gate does not guarantee
        that the retrieved context answers the question.
        """

        results = self.retrieve(
            query=query,
            top_k=top_k,
            mode="reranked",
            candidate_k=candidate_k,
            rrf_k=rrf_k
        )

        decision = check_relevance(
            results=results,
            threshold=threshold
        )

        context = self.build_context(
            decision["results"]
        )

        return {
            "query": query.strip(),
            "retrieval_mode": "reranked",
            "relevance_passed": decision["passed"],
            "max_score": decision["max_score"],
            "threshold": decision["threshold"],
            "reason": decision["reason"],
            "retrieved_count": len(results),
            "retained_count": len(decision["results"]),
            "results": decision["results"],
            "context": context
        }



    def build_context(self, results):
        """Prepare retrieved context for the LLM."""

        context_parts = []

        for result in results:

            context = (
                f"[Source {result['rank']}]\n"
                f"Document: {result['document_title']}\n"
                f"Chunk ID: {result['chunk_id']}\n"
                f"Page: {result['page']}\n"
                f"URL: {result['source_url']}\n"
                f"Document date: {result.get('document_date', '')}\n"
                f"Version notes: {result.get('version_notes', '')}\n"
                f"Review notes: {result.get('review_notes', '')}\n"
                f"Text:\n{result['text']}"
            )

            context_parts.append(context)

        return "\n\n".join(context_parts)


def main():

    import time

    print("Initializing Retrieval Engine...")

    engine = RetrievalEngine()

    print("Embedding model:", engine.config["model_name"])
    print("Indexed vectors:", engine.index.ntotal)
    print("BM25 chunks:", engine.keyword_engine.num_documents)

    # Verify lazy loading
    assert engine.reranker is None

    print("Loading reranker for warm-start testing...")

    start = time.perf_counter()

    engine.load_reranker()

    print(
        f"Reranker loading time: "
        f"{time.perf_counter() - start:.3f} seconds"
    )

    print("Retrieval Engine ready.")

    questions = [
        "What is the minimum GPA required for transfer students?",
        "How can I withdraw from a course?",
        "How can I register for courses?",
        "What is the weather in Alexandria today?"
    ]

    modes = [
        "semantic",
        "keyword",
        "hybrid",
        "reranked"
    ]

    score_keys = {
        "semantic": "similarity_score",
        "keyword": "bm25_score",
        "hybrid": "rrf_score",
        "reranked": "reranker_score"
    }

    for question in questions:

        print("\n" + "=" * 75)
        print("Question:", question)

        for mode in modes:

            print(f"\nMode: {mode.upper()}")

            start = time.perf_counter()

            results = engine.retrieve(
                query=question,
                top_k=5,
                mode=mode,
                candidate_k=20
            )

            elapsed = time.perf_counter() - start

            if not results:

                print("No results retrieved.")

            for result in results:

                score = result[score_keys[mode]]

                print(
                    f"Rank {result['rank']} | "
                    f"{result['chunk_id']} | "
                    f"Score: {score:.6f}"
                )

                print(
                    "Document:",
                    result["document_title"]
                )

                if mode == "reranked":

                    print(
                        "Original rank:",
                        result["original_rank"],
                        "| Truncated:",
                        result["reranker_truncated"]
                    )

            print(f"Retrieval time: {elapsed:.3f} seconds")

            # Validate result count
            assert len(results) <= 5

            # Validate unique chunk IDs
            assert len({
                result["chunk_id"]
                for result in results
            }) == len(results)

            # Validate final ranks
            assert [
                result["rank"]
                for result in results
            ] == list(range(1, len(results) + 1))

            # Validate retrieval mode
            assert all(
                result["retrieval_mode"] == mode
                for result in results
            )

            # Validate context generation
            context = engine.build_context(results)

            assert isinstance(context, str)

            assert all(
                result["chunk_id"] in context
                for result in results
            )

            if not results:
                assert context == ""

    # Verify the reranker is reused
    first_reranker = engine.load_reranker()
    second_reranker = engine.load_reranker()

    assert first_reranker is second_reranker

    print("\nReranked retrieval integration test completed successfully.")


if __name__ == "__main__":
    main()
