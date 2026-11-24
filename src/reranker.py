
import numpy as np
from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEVICE = "cpu"
MAX_LENGTH = 512
BATCH_SIZE = 8
WINDOW_OVERLAP = 64


class Reranker:
    """Rerank retrieved chunks using a Cross-Encoder."""

    def __init__(
        self,
        model_name=MODEL_NAME,
        device=DEVICE,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        window_overlap=WINDOW_OVERLAP,
        use_title=False,
        local_files_only=False
    ):

        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError(
                "batch_size must be a positive integer."
            )

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_title = use_title

        self.model = CrossEncoder(
            model_name,
            device=device,
            max_length=max_length,
            local_files_only=local_files_only
        )

    def build_windows(self, query, passage):
        """
        Construct valid query-passage windows without
        silently truncating oversized passages.
        """

        tokenizer = self.model.tokenizer

        query_ids = tokenizer.encode(
            query,
            add_special_tokens=False,
            truncation=False,
            verbose=False
        )

        passage_ids = tokenizer.encode(
            passage,
            add_special_tokens=False,
            truncation=False,
            verbose=False
        )

        special_tokens = tokenizer.num_special_tokens_to_add(
            pair=True
        )

        budget = (
            self.max_length
            - len(query_ids)
            - special_tokens
        )

        if budget <= 0:
            raise ValueError(
                "Query exceeds the reranker token budget."
            )

        original_token_count = len(
            tokenizer.encode(
                query,
                text_pair=passage,
                add_special_tokens=True,
                truncation=False,
                verbose=False
            )
        )

        # Preserve the original text when it already fits.
        if original_token_count <= self.max_length:

            return [passage], original_token_count

        overlap = min(WINDOW_OVERLAP, budget - 1)

        windows = []
        start = 0

        while start < len(passage_ids):

            end = min(
                start + budget,
                len(passage_ids)
            )

            while end > start:

                window_text = tokenizer.decode(
                    passage_ids[start:end],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False
                )

                pair_tokens = len(
                    tokenizer.encode(
                        query,
                        text_pair=window_text,
                        add_special_tokens=True,
                        truncation=False,
                        verbose=False
                    )
                )

                if pair_tokens <= self.max_length:
                    break

                end -= 1

            if end <= start:
                raise ValueError(
                    "Cannot construct a valid reranker window."
                )

            windows.append(window_text)

            if end == len(passage_ids):
                break

            start = max(start + 1, end - overlap)

        if not windows:
            raise ValueError("No reranker windows created.")

        return windows, original_token_count




    def rerank(self, query, results, top_k=5):

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "Query must be a non-empty string."
            )

        if type(top_k) is not int or top_k <= 0:
            raise ValueError(
                "top_k must be a positive integer."
            )

        if not results:
            return []

        query = query.strip()

        pairs = []
        window_ranges = []
        token_counts = []
        window_counts = []

        # Prepare all windows for batch prediction.
        for result in results:

            passage = result["text"]

            if (
                not isinstance(passage, str)
                or not passage.strip()
            ):
                raise ValueError(
                    "Chunk text must be a non-empty string."
                )

            if self.use_title:

                passage = (
                    result["document_title"]
                    + "\n"
                    + passage
                )

            windows, original_tokens = self.build_windows(
                query,
                passage
            )

            start_index = len(pairs)

            for window in windows:

                pairs.append((query, window))

            end_index = len(pairs)

            window_ranges.append(
                (start_index, end_index)
            )

            token_counts.append(original_tokens)
            window_counts.append(len(windows))

        # Score every window.
        window_scores = np.asarray(
            self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False
            )
        ).reshape(-1)

        if len(window_scores) != len(pairs):
            raise ValueError(
                "Reranking score count mismatch."
            )

        if not np.isfinite(window_scores).all():
            raise ValueError(
                "Reranking scores contain invalid values."
            )

        # Aggregate window scores per original chunk.
        chunk_scores = []
        best_windows = []

        for start, end in window_ranges:

            scores = window_scores[start:end]

            chunk_scores.append(
                float(np.max(scores))
            )

            best_windows.append(
                int(np.argmax(scores)) + 1
            )

        chunk_scores = np.asarray(chunk_scores)

        indices = np.argsort(
            -chunk_scores,
            kind="stable"
        )

        reranked_results = []

        for rank, index in enumerate(
            indices[:top_k],
            start=1
        ):

            result = results[index].copy()

            result["original_rank"] = result.get(
                "rank",
                int(index) + 1
            )

            result["reranker_score"] = float(
                chunk_scores[index]
            )

            result["reranker_input_tokens"] = (
                token_counts[index]
            )

            result["reranker_window_count"] = (
                window_counts[index]
            )

            result["reranker_windowed"] = (
                token_counts[index] > self.max_length
            )

            result["reranker_best_window"] = (
                best_windows[index]
            )

            result["reranker_truncated"] = False

            result["rank"] = rank

            reranked_results.append(result)

        return reranked_results




def main():

    import time
    from .retrieval import RetrievalEngine

    print("Loading retrieval engine...")
    engine = RetrievalEngine()

    print("Loading reranker...")

    reranker = Reranker(
        use_title=False,
        local_files_only=True
    )

    print("Reranker ready.")
    print("Model:", reranker.model_name)

    questions = [
        "What is the minimum GPA required for transfer students?",
        "How can I withdraw from a course?",
        "How can I register for courses?",
        "What is the weather in Alexandria today?"
    ]

    for question in questions:

        print("\n" + "=" * 75)
        print("Question:", question)

        candidates = engine.retrieve(
            query=question,
            top_k=20,
            mode="hybrid",
            candidate_k=20
        )

        start = time.perf_counter()

        results = reranker.rerank(
            query=question,
            results=candidates,
            top_k=20
        )

        elapsed = time.perf_counter() - start

        for result in results[:5]:

            print(
                f"Rank {result['rank']} | "
                f"{result['chunk_id']} | "
                f"Score: {result['reranker_score']:.6f}"
            )

            print(
                "Document:",
                result["document_title"]
            )

            print(
                "Original rank:",
                result["original_rank"]
            )

            print(
                "Input tokens:",
                result["reranker_input_tokens"],
                "| Truncated:",
                result["reranker_truncated"]
            )

        truncated_count = sum(
            result["reranker_truncated"]
            for result in results
        )

        print("Candidates:", len(candidates))
        print("Truncated candidates:", truncated_count)
        print(f"Reranking time: {elapsed:.3f} seconds")

        assert len(results) == len(candidates)

        assert len({
            result["chunk_id"]
            for result in results
        }) == len(results)

        assert all(
            "reranker_score" not in result
            for result in candidates
        )

        assert all(
            results[i]["reranker_score"]
            >= results[i + 1]["reranker_score"]
            for i in range(len(results) - 1)
        )

    assert reranker.rerank(
        "Test question",
        [],
        top_k=5
    ) == []

    print("\nReranker module test completed successfully.")


if __name__ == "__main__":
    main()
