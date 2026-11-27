
"""Development-set embedding evaluation for UniGuide-AI."""

import argparse
import csv
import hashlib
import json
import statistics
import time

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from evaluation.generate_comparative_embeddings import (
    BASE_DIR,
    CHUNKS_FILE,
    MODELS,
    validate_embeddings,
)

from evaluation.evaluate_embedding_models import (
    load_experiment,
    encode_query,
)

from evaluation.metrics import (
    evaluate_retrieval,
    aggregate_metrics,
    DEFAULT_K_VALUES,
)


QUESTIONS_FILE = (
    BASE_DIR / "evaluation" / "test_questions.csv"
)

DEFAULT_EXPERIMENT_DIR = (
    BASE_DIR / "data" / "embeddings" / "comparative"
)

DEFAULT_OUTPUT_DIR = (
    BASE_DIR / "evaluation" / "final_results"
    / "embedding_development"
)


# ==========================================
# 1. Dataset Loading
# ==========================================

def load_questions(path, valid_chunk_ids):

    questions = []
    seen_ids = set()

    with open(path, encoding="utf-8-sig", newline="") as file:

        for row in csv.DictReader(file):

            # Never evaluate held-out questions during selection.
            if row["split"] != "development":
                continue

            question_id = row["question_id"]

            if question_id in seen_ids:
                raise ValueError(
                    f"Duplicate question: {question_id}"
                )

            seen_ids.add(question_id)

            if row["review_status"] != "verified":
                raise ValueError(
                    f"Unverified question: {question_id}"
                )

            relevant_ids = json.loads(
                row["relevant_chunk_ids"]
            )

            if row["answerable"] not in ("true", "false"):
                raise ValueError("Invalid answerability label.")

            answerable = row["answerable"] == "true"

            if answerable != bool(relevant_ids):
                raise ValueError(
                    f"Invalid ground truth: {question_id}"
                )

            if not set(relevant_ids).issubset(valid_chunk_ids):
                raise ValueError(
                    f"Unknown chunk IDs: {question_id}"
                )

            questions.append({
                "question_id": question_id,
                "question": row["question"],
                "language": row["language"],
                "question_type": row["question_type"],
                "difficulty": row["difficulty"],
                "answerable": answerable,
                "relevant_chunk_ids": relevant_ids,
            })

    if len(questions) != 28:
        raise ValueError("Expected 28 development questions.")

    if sum(q["answerable"] for q in questions) != 25:
        raise ValueError("Expected 25 answerable questions.")

    return questions


# ==========================================
# 2. Evaluate One Embedding Model
# ==========================================

def evaluate_model(
    config,
    chunks,
    questions,
    experiment_dir,
    generation_summary,
    device,
    cache_dir,
    local_files_only,
):

    alias = config["alias"]

    print("\n" + "=" * 70)
    print("Evaluating:", config["model_name"])
    print("=" * 70)

    embedding_path = experiment_dir / f"{alias}.npy"

    embeddings = np.load(
        embedding_path,
        allow_pickle=False
    )

    validate_embeddings(
        embeddings,
        expected_count=len(chunks),
        expected_dim=768,
    )

    start = time.perf_counter()

    model = SentenceTransformer(
        config["model_name"],
        device=device,
        cache_folder=cache_dir,
        local_files_only=local_files_only,
    )

    loading_seconds = time.perf_counter() - start

    if model.get_embedding_dimension() != embeddings.shape[1]:
        raise ValueError("Model dimension mismatch.")

    # Warm-up excluded from evaluation latency.
    encode_query(
        "University academic policies",
        model,
        config["query_prefix"],
    )

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    results = []

    for index, question in enumerate(questions, start=1):

        query_vector, encoding_seconds = encode_query(
            question["question"],
            model,
            config["query_prefix"],
        )

        start = time.perf_counter()

        scores = embeddings @ query_vector

        ranked_indices = np.argsort(
            -scores,
            kind="stable",
        )[:max(DEFAULT_K_VALUES)]

        search_seconds = time.perf_counter() - start

        if not np.isfinite(scores).all():
            raise ValueError("Invalid similarity scores.")

        retrieved_ids = [
            chunk_ids[i] for i in ranked_indices
        ]

        retrieved_scores = [
            float(scores[i]) for i in ranked_indices
        ]

        metrics = None

        if question["answerable"]:
            metrics = evaluate_retrieval(
                retrieved_ids,
                question["relevant_chunk_ids"],
                DEFAULT_K_VALUES,
            )

        result = {
            **question,
            "model": config["model_name"],
            "alias": alias,
            "retrieved_chunk_ids": retrieved_ids,
            "retrieved_scores": retrieved_scores,
            "top_similarity": retrieved_scores[0],
            "query_encoding_seconds": encoding_seconds,
            "search_seconds": search_seconds,
            "retrieval_seconds": (
                encoding_seconds + search_seconds
            ),
            "metrics": metrics,
        }

        results.append(result)

        if metrics is not None:
            message = (
                f"Recall@5={metrics['recall_at_5']:.3f} | "
                f"RR@10={metrics['rr_at_10']:.3f}"
            )
        else:
            message = "Unanswerable — metrics excluded"

        print(
            f"[{index:02d}/{len(questions)}] "
            f"{question['question_id']} | "
            f"{question['language']} | {message}"
        )

    summary = {
        "model": config["model_name"],
        "alias": alias,
        "model_loading_seconds": loading_seconds,
        "historical_document_encoding_seconds": (
            generation_summary["encoding_seconds"]
        ),
        "embedding_sha256": hashlib.sha256(
            embedding_path.read_bytes()
        ).hexdigest(),
        "embedding_shape": list(embeddings.shape),
        "model_max_seq_length": model.max_seq_length,
        "overall": summarize_group(results),
        "english": summarize_group([
            r for r in results if r["language"] == "en"
        ]),
        "arabic": summarize_group([
            r for r in results if r["language"] == "ar"
        ]),
        "by_question_type": {},
    }

    for question_type in sorted({
        r["question_type"] for r in results
    }):

        group = [
            r for r in results
            if r["question_type"] == question_type
        ]

        summary["by_question_type"][question_type] = (
            summarize_group(group)
        )

    del model

    return summary, results


# ==========================================
# 3. Aggregate Results
# ==========================================

def summarize_group(results):

    answerable = [
        r for r in results if r["answerable"]
    ]

    unanswerable = [
        r for r in results if not r["answerable"]
    ]

    summary = {
        "total_questions": len(results),
        "answerable_questions": len(answerable),
        "unanswerable_questions": len(unanswerable),
        "metrics": (
            aggregate_metrics([
                r["metrics"] for r in answerable
            ])
            if answerable else None
        ),
    }

    if results:

        latencies = [
            r["retrieval_seconds"] for r in results
        ]

        summary.update({
            "mean_retrieval_seconds": (
                statistics.mean(latencies)
            ),
            "median_retrieval_seconds": (
                statistics.median(latencies)
            ),
            "p95_retrieval_seconds": float(
                np.percentile(latencies, 95)
            ),
            "mean_query_encoding_seconds": statistics.mean(
                r["query_encoding_seconds"]
                for r in results
            ),
            "mean_search_seconds": statistics.mean(
                r["search_seconds"] for r in results
            ),
        })

    summary["unanswerable_similarity_scores"] = [
        {
            "question_id": r["question_id"],
            "top_similarity": r["top_similarity"],
        }
        for r in unanswerable
    ]

    return summary


# ==========================================
# 4. Save CSV and JSON Results
# ==========================================

def save_results(output_dir, report):

    output_dir.mkdir(parents=True, exist_ok=False)

    with open(
        output_dir / "embedding_results.json",
        "x",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    summary_rows = []

    for model_summary in report["summary"]:

        for group_name in ("overall", "english", "arabic"):

            group = model_summary[group_name]

            row = {
                "model": model_summary["model"],
                "group": group_name,
                "total_questions": group["total_questions"],
                "answerable_questions": (
                    group["answerable_questions"]
                ),
                "unanswerable_questions": (
                    group["unanswerable_questions"]
                ),
                "mean_retrieval_ms": (
                    group["mean_retrieval_seconds"] * 1000
                ),
                "median_retrieval_ms": (
                    group["median_retrieval_seconds"] * 1000
                ),
                "p95_retrieval_ms": (
                    group["p95_retrieval_seconds"] * 1000
                ),
                **(group["metrics"] or {}),
            }

            summary_rows.append(row)

    with open(
        output_dir / "embedding_summary.csv",
        "x",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(summary_rows[0]),
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    detailed_rows = []

    for result in report["results"]:

        row = {
            key: value
            for key, value in result.items()
            if key != "metrics"
        }

        for key in (
            "relevant_chunk_ids",
            "retrieved_chunk_ids",
            "retrieved_scores",
        ):
            row[key] = json.dumps(
                row[key],
                ensure_ascii=False,
            )

        row.update(result["metrics"] or {})

        detailed_rows.append(row)

    fields = list(dict.fromkeys(
        key for row in detailed_rows for key in row
    ))

    with open(
        output_dir / "embedding_question_results.csv",
        "x",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(detailed_rows)


# ==========================================
# 5. Print Comparison
# ==========================================

def print_comparison(summaries):

    print("\n" + "=" * 85)
    print("DEVELOPMENT EMBEDDING COMPARISON")
    print("=" * 85)

    print(
        f"{'Model':<20}"
        f"{'Group':<10}"
        f"{'Hit@5':>10}"
        f"{'Recall@5':>12}"
        f"{'MRR@10':>12}"
        f"{'nDCG@10':>12}"
        f"{'Latency':>12}"
    )

    for summary in summaries:

        for name in ("overall", "english", "arabic"):

            group = summary[name]
            metrics = group["metrics"]

            print(
                f"{summary['alias']:<20}"
                f"{name:<10}"
                f"{metrics['hit_rate_at_5']:>10.2%}"
                f"{metrics['recall_at_5']:>12.2%}"
                f"{metrics['mrr_at_10']:>12.4f}"
                f"{metrics['ndcg_at_10']:>12.4f}"
                f"{group['mean_retrieval_seconds'] * 1000:>9.2f} ms"
            )


# ==========================================
# 6. Main Experiment
# ==========================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir", default=None)

    parser.add_argument(
        "--local-files-only",
        action="store_true",
    )

    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {args.output_dir}"
        )

    questions_hash = hashlib.sha256(
        QUESTIONS_FILE.read_bytes()
    ).hexdigest()

    chunks, corpus_hash, generation_summaries = (
        load_experiment(
            args.experiment_dir,
            CHUNKS_FILE,
        )
    )

    questions = load_questions(
        QUESTIONS_FILE,
        {chunk["chunk_id"] for chunk in chunks},
    )

    print("Development questions:", len(questions))
    print("Answerable:", sum(q["answerable"] for q in questions))
    print("Unanswerable:", sum(not q["answerable"] for q in questions))
    print("Held-out test questions: NOT EVALUATED")

    summaries = []
    all_results = []

    for config in MODELS:

        summary, results = evaluate_model(
            config=config,
            chunks=chunks,
            questions=questions,
            experiment_dir=args.experiment_dir,
            generation_summary=generation_summaries[
                config["alias"]
            ],
            device=args.device,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )

        summaries.append(summary)
        all_results.extend(results)

    expected_count = len(MODELS) * len(questions)

    if len(all_results) != expected_count:
        raise ValueError("Evaluation result count mismatch.")

    if hashlib.sha256(
        QUESTIONS_FILE.read_bytes()
    ).hexdigest() != questions_hash:
        raise ValueError("Question dataset changed.")

    if hashlib.sha256(
        CHUNKS_FILE.read_bytes()
    ).hexdigest() != corpus_hash:
        raise ValueError("Corpus changed.")

    report = {
        "dataset_version": "1.0.0",
        "evaluation_type": "development_model_selection",
        "split": "development",
        "query_strategy": "original_language",
        "search_method": "exact_numpy_cosine",
        "relevance_type": "binary_multiple_ground_truth_chunks",
        "corpus_sha256": corpus_hash,
        "questions_sha256": questions_hash,
        "device": args.device,
        "k_values": list(DEFAULT_K_VALUES),
        "total_evaluations": len(all_results),
        "summary": summaries,
        "results": all_results,
    }

    save_results(args.output_dir, report)
    print_comparison(summaries)

    print("\nCompleted evaluations:", len(all_results))
    print("Dataset integrity: PASSED")
    print("Results saved:", args.output_dir)


if __name__ == "__main__":
    main()
