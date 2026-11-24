
import argparse
import json
import time
import statistics

from pathlib import Path

import numpy as np

from sentence_transformers import SentenceTransformer

from evaluation.generate_comparative_embeddings import (
    BASE_DIR,
    CHUNKS_FILE,
    DEFAULT_OUTPUT_DIR,
    MODELS,
    load_dataset,
    count_tokens,
    validate_embeddings
)


# ==========================================
# 1. Development Questions
# ==========================================

QUESTIONS = [
    (
        "Q01",
        "What is the minimum GPA required for transfer students?",
        "AIU-001-P001-C001"
    ),
    (
        "Q02",
        "How can I withdraw from a course?",
        "AIU-020-P002-C002"
    ),
    (
        "Q03",
        "How can I register for courses?",
        "AIU-020-P003-C001"
    ),
    (
        "Q04",
        "How many credit hours can a student with a GPA of 3.0 register for?",
        "AIU-020-P002-C001"
    ),
    (
        "Q05",
        "Are transferred course grades included in the cumulative GPA?",
        "AIU-001-P002-C001"
    ),
    (
        "Q06",
        "What are the rules for class attendance?",
        "AIU-018-P001-C001"
    ),
    (
        "Q07",
        "Where can I find the course withdrawal request form?",
        "AIU-020-P006-C001"
    ),
    (
        "Q08",
        "Where can I find the incomplete course request form?",
        "AIU-021-P002-C001"
    )
]


# ==========================================
# 2. Load Experiment Data
# ==========================================

def load_experiment(experiment_dir, chunks_file):

    chunks, dataset_hash = load_dataset(chunks_file)

    with open(
        experiment_dir / "chunk_ids.json",
        encoding="utf-8-sig"
    ) as file:

        saved_ids = json.load(file)

    with open(
        experiment_dir / "comparison_summary.json",
        encoding="utf-8-sig"
    ) as file:

        summaries = json.load(file)

    current_ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    if saved_ids != current_ids:
        raise ValueError(
            "Chunk ID alignment mismatch."
        )

    summary_map = {
        item["alias"]: item
        for item in summaries
    }

    if len(summaries) != len(MODELS):
        raise ValueError(
            "Unexpected number of model summaries."
        )

    if set(summary_map) != {
        config["alias"] for config in MODELS
    }:
        raise ValueError("Model summary mismatch.")

    for config in MODELS:

        summary = summary_map[config["alias"]]

        if summary["dataset_sha256"] != dataset_hash:
            raise ValueError("Dataset hash mismatch.")

        if summary["model"] != config["model_name"]:
            raise ValueError("Model name mismatch.")

        for key in ("document_prefix", "query_prefix"):

            if summary[key] != config[key]:
                raise ValueError(
                    f"Model prefix mismatch: {key}"
                )

    for question_id, question, target_id in QUESTIONS:

        if target_id not in current_ids:
            raise ValueError(
                f"Missing target: {target_id}"
            )

    print("Dataset chunks:", len(chunks))
    print("Dataset SHA256:", dataset_hash)
    print("Chunk alignment: PASSED")
    print("Model configuration: PASSED")
    print("Development questions:", len(QUESTIONS))

    return chunks, dataset_hash, summary_map


# ==========================================
# 3. Query Encoding
# ==========================================

def encode_query(query, model, prefix):

    formatted_query = prefix + query.strip()

    token_count = count_tokens(
        formatted_query,
        model.tokenizer
    )

    if token_count > model.max_seq_length:
        raise ValueError("Query token limit exceeded.")

    start = time.perf_counter()

    embedding = model.encode(
        [formatted_query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    elapsed = time.perf_counter() - start

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    )

    validate_embeddings(
        embedding,
        expected_count=1,
        expected_dim=model.get_embedding_dimension()
    )

    return embedding[0], elapsed


# ==========================================
# 4. Evaluate One Model
# ==========================================

def evaluate_model(
    config,
    chunks,
    experiment_dir,
    generation_summary,
    device,
    cache_dir,
    local_files_only
):

    print("\n" + "=" * 75)
    print("Evaluating:", config["model_name"])

    embeddings = np.load(
        experiment_dir / f"{config['alias']}.npy",
        allow_pickle=False
    )

    validate_embeddings(
        embeddings,
        expected_count=len(chunks),
        expected_dim=768
    )

    start = time.perf_counter()

    model = SentenceTransformer(
        config["model_name"],
        device=device,
        cache_folder=cache_dir,
        local_files_only=local_files_only
    )

    model_loading_seconds = (
        time.perf_counter() - start
    )

    if model.get_embedding_dimension() != embeddings.shape[1]:
        raise ValueError("Model dimension mismatch.")

    # Warm-up query excluded from evaluation.

    encode_query(
        "University academic policies",
        model,
        config["query_prefix"]
    )

    chunk_ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    id_to_index = {
        chunk_id: index
        for index, chunk_id in enumerate(chunk_ids)
    }

    results = []

    for question_id, question, target_id in QUESTIONS:

        print("\n" + "-" * 70)
        print(question_id, question)
        print("Target:", target_id)

        query_embedding, encoding_seconds = encode_query(
            question,
            model,
            config["query_prefix"]
        )

        # Exact cosine similarity using normalized vectors.

        start = time.perf_counter()

        scores = embeddings @ query_embedding

        ranked_indices = np.argsort(
            -scores,
            kind="stable"
        )

        search_seconds = (
            time.perf_counter() - start
        )

        if not np.isfinite(scores).all():
            raise ValueError("Invalid similarity scores.")

        target_index = id_to_index[target_id]

        target_rank = int(
            np.where(
                ranked_indices == target_index
            )[0][0]
        ) + 1

        reciprocal_rank = (
            1.0 / target_rank
            if target_rank <= 10
            else 0.0
        )

        top_10_ids = [
            chunk_ids[index]
            for index in ranked_indices[:10]
        ]

        retrieval_seconds = (
            encoding_seconds + search_seconds
        )

        result = {
            "model": config["model_name"],
            "question_id": question_id,
            "question": question,
            "target_chunk_id": target_id,
            "target_rank": target_rank,
            "hit_at_1": int(target_rank <= 1),
            "hit_at_3": int(target_rank <= 3),
            "hit_at_5": int(target_rank <= 5),
            "hit_at_10": int(target_rank <= 10),
            "reciprocal_rank_at_10": reciprocal_rank,
            "query_encoding_seconds": encoding_seconds,
            "search_seconds": search_seconds,
            "retrieval_seconds": retrieval_seconds,
            "top_10_chunk_ids": top_10_ids
        }

        results.append(result)

        for rank, index in enumerate(
            ranked_indices[:5],
            start=1
        ):

            print(
                f"Rank {rank} | "
                f"{chunk_ids[index]} | "
                f"Score: {scores[index]:.6f}"
            )

            print(
                "Document:",
                chunks[index]["document_title"]
            )

        print("Target rank:", target_rank)
        print(
            "Retrieval seconds:",
            round(retrieval_seconds, 6)
        )

    summary = {
        "model": config["model_name"],
        "alias": config["alias"],
        "number_of_questions": len(results),
        "hit_rate_at_1": statistics.mean(
            r["hit_at_1"] for r in results
        ),
        "hit_rate_at_3": statistics.mean(
            r["hit_at_3"] for r in results
        ),
        "hit_rate_at_5": statistics.mean(
            r["hit_at_5"] for r in results
        ),
        "hit_rate_at_10": statistics.mean(
            r["hit_at_10"] for r in results
        ),
        "mrr_at_10": statistics.mean(
            r["reciprocal_rank_at_10"]
            for r in results
        ),
        "mean_target_rank": statistics.mean(
            r["target_rank"] for r in results
        ),
        "mean_query_encoding_seconds": statistics.mean(
            r["query_encoding_seconds"]
            for r in results
        ),
        "median_query_encoding_seconds": statistics.median(
            r["query_encoding_seconds"]
            for r in results
        ),
        "mean_search_seconds": statistics.mean(
            r["search_seconds"] for r in results
        ),
        "mean_retrieval_seconds": statistics.mean(
            r["retrieval_seconds"] for r in results
        ),
        "model_loading_seconds": model_loading_seconds,
        "document_encoding_seconds": (
            generation_summary["encoding_seconds"]
        ),
        "windowed_chunks": (
            generation_summary["windowed_chunks"]
        ),
        "target_ranks": [
            r["target_rank"] for r in results
        ]
    }

    return summary, results


# ==========================================
# 5. Main Evaluation
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Preliminary embedding model comparison "
            "for UniGuide-AI."
        )
    )

    parser.add_argument(
        "--chunks-file",
        type=Path,
        default=CHUNKS_FILE
    )

    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=None
    )

    parser.add_argument(
        "--device",
        default="cpu"
    )

    parser.add_argument(
        "--cache-dir",
        default=None
    )

    parser.add_argument(
        "--local-files-only",
        action="store_true"
    )

    args = parser.parse_args()

    output_file = args.output_file or (
        args.experiment_dir /
        "reproduced_retrieval_comparison.json"
    )

    if output_file.exists():
        raise FileExistsError(
            f"Output already exists: {output_file}"
        )

    if output_file.suffix.lower() != ".json":
        raise ValueError("Output must be a JSON file.")

    chunks, dataset_hash, generation_summaries = (
        load_experiment(
            args.experiment_dir,
            args.chunks_file
        )
    )

    summaries = []
    all_results = []

    for config in MODELS:

        summary, results = evaluate_model(
            config=config,
            chunks=chunks,
            experiment_dir=args.experiment_dir,
            generation_summary=generation_summaries[
                config["alias"]
            ],
            device=args.device,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only
        )

        summaries.append(summary)
        all_results.extend(results)

    _, final_hash = load_dataset(args.chunks_file)

    if final_hash != dataset_hash:
        raise ValueError("Dataset changed during evaluation.")

    expected_results = len(MODELS) * len(QUESTIONS)

    if len(all_results) != expected_results:
        raise ValueError("Evaluation result count mismatch.")

    report = {
        "dataset_sha256": dataset_hash,
        "evaluation_type": "preliminary_development",
        "relevance_type": "single_tracked_target",
        "search_method": "exact_numpy_cosine",
        "questions": [
            {
                "question_id": question_id,
                "question": question,
                "target_chunk_id": target_id
            }
            for question_id, question, target_id in QUESTIONS
        ],
        "summary": summaries,
        "results": all_results
    }

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_file,
        "x",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("\n" + "=" * 75)
    print("FINAL COMPARISON")

    for summary in summaries:

        print("\nModel:", summary["model"])

        for k in (1, 3, 5, 10):

            print(
                f"Hit Rate@{k}: "
                f"{summary[f'hit_rate_at_{k}']:.2%}"
            )

        print(
            "MRR@10:",
            round(summary["mrr_at_10"], 6)
        )

        print(
            "Mean target rank:",
            summary["mean_target_rank"]
        )

        print(
            "Mean retrieval seconds:",
            round(summary["mean_retrieval_seconds"], 6)
        )

        print("Target ranks:", summary["target_ranks"])

    print("\nCompleted evaluations:", len(all_results))
    print("Dataset integrity: PASSED")
    print("Results saved:", output_file)


if __name__ == "__main__":
    main()
