
def reciprocal_rank_fusion(
    semantic_results,
    keyword_results,
    top_k=5,
    k=60
):
    """Combine semantic and keyword rankings using RRF."""

    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer.")

    fused_results = {}

    ranked_lists = [
        ("semantic", semantic_results),
        ("keyword", keyword_results)
    ]

    for method, results in ranked_lists:

        seen_ids = set()

        for rank, result in enumerate(results, start=1):

            chunk_id = result["chunk_id"]

            if chunk_id in seen_ids:
                continue

            seen_ids.add(chunk_id)

            if chunk_id not in fused_results:

                fused_results[chunk_id] = result.copy()

                fused_results[chunk_id].update({
                    "rrf_score": 0.0,
                    "semantic_rank": None,
                    "keyword_rank": None,
                    "similarity_score": None,
                    "bm25_score": None
                })

            entry = fused_results[chunk_id]

            # Add RRF contribution
            entry["rrf_score"] += 1.0 / (k + rank)

            # Preserve original ranks and scores
            if method == "semantic":

                entry["semantic_rank"] = rank

                entry["similarity_score"] = result.get(
                    "similarity_score"
                )

            else:

                entry["keyword_rank"] = rank

                entry["bm25_score"] = result.get(
                    "bm25_score"
                )

    # Sort by combined RRF score
    ranked_results = sorted(
        fused_results.values(),
        key=lambda result: (
            -result["rrf_score"],
            result["chunk_id"]
        )
    )

    ranked_results = ranked_results[:top_k]

    # Assign final ranks
    for rank, result in enumerate(
        ranked_results,
        start=1
    ):
        result["rank"] = rank

    return ranked_results


def main():

    semantic_results = [
        {
            "chunk_id": "A",
            "similarity_score": 0.90
        },
        {
            "chunk_id": "B",
            "similarity_score": 0.85
        },
        {
            "chunk_id": "C",
            "similarity_score": 0.80
        }
    ]

    keyword_results = [
        {
            "chunk_id": "B",
            "bm25_score": 10.0
        },
        {
            "chunk_id": "C",
            "bm25_score": 8.0
        },
        {
            "chunk_id": "D",
            "bm25_score": 6.0
        }
    ]

    results = reciprocal_rank_fusion(
        semantic_results,
        keyword_results,
        top_k=4
    )

    print("RRF Fusion Results:")

    for result in results:

        print(
            f"Rank {result['rank']} | "
            f"Chunk: {result['chunk_id']} | "
            f"RRF: {result['rrf_score']:.6f} | "
            f"Semantic rank: {result['semantic_rank']} | "
            f"Keyword rank: {result['keyword_rank']}"
        )

    assert [
        result["chunk_id"] for result in results
    ] == ["B", "C", "A", "D"]

    assert len(results) == 4

    assert len({
        result["chunk_id"] for result in results
    }) == 4

    assert reciprocal_rank_fusion([], []) == []

    print("\nRRF fusion test completed successfully.")


if __name__ == "__main__":
    main()
