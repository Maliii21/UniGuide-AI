
"""Compare production retrieval strategies on development questions."""

import csv
import hashlib
import json
import math
import statistics
import time

from pathlib import Path

import numpy as np

from src.retrieval import RetrievalEngine
from src.reranker import Reranker
from src.llm import GeminiLLM
from src.query_translator import QueryTranslator

from evaluation.metrics import (
    evaluate_retrieval,
    aggregate_metrics,
    DEFAULT_K_VALUES,
)

from evaluation.evaluate_embeddings_final import (
    BASE_DIR,
    CHUNKS_FILE,
    QUESTIONS_FILE,
    load_questions,
)


RESULTS_DIR = BASE_DIR / "evaluation" / "final_results"

OUTPUT_DIR = RESULTS_DIR / "retrieval_development"

TRANSLATIONS_FILE = (
    RESULTS_DIR / "translation_development"
    / "translations.json"
)

BASELINE_FILE = (
    RESULTS_DIR / "embedding_development"
    / "embedding_results.json"
)

MODES = (
    "semantic",
    "keyword",
    "hybrid",
    "reranked",
)

SCORE_KEYS = {
    "semantic": "similarity_score",
    "keyword": "bm25_score",
    "hybrid": "rrf_score",
    "reranked": "reranker_score",
}

CANDIDATE_K = 20
RRF_K = 60
TOP_K = max(DEFAULT_K_VALUES)
DEVICE = "cpu"

EXPECTED_MODEL = "intfloat/multilingual-e5-base"


# ==========================================
# 1. Prepare Retrieval Queries
# ==========================================

def prepare_queries(questions):

    translations = json.loads(
        TRANSLATIONS_FILE.read_text(encoding="utf-8")
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cache_file = OUTPUT_DIR / "retrieval_queries.json"

    if cache_file.exists():

        cached = json.loads(
            cache_file.read_text(encoding="utf-8")
        )

        translations.update(cached)

    pending = [
        q for q in questions
        if q["language"] == "ar"
        and q["question_id"] not in translations
    ]

    if pending:

        llm = GeminiLLM()
        translator = QueryTranslator(llm)

        try:

            for question in pending:

                qid = question["question_id"]

                result = translator.translate(
                    question["question"]
                )

                if not result.success:
                    raise RuntimeError(
                        f"{qid}: {result.error_message}"
                    )

                translations[qid] = {
                    "original_question": question["question"],
                    "translated_question": result.retrieval_query,
                    "translation_seconds": result.latency_seconds,
                    "model": result.llm_response.model,
                    "total_tokens": result.llm_response.total_tokens,
                }

                cache_file.write_text(
                    json.dumps(
                        translations,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                print("New translation:", qid)
                print("Arabic:", question["question"])
                print("English:", result.retrieval_query)

        finally:
            llm.close()

    prepared = []

    for question in questions:

        if question["language"] == "ar":

            translation = translations[
                question["question_id"]
            ]

            if (
                translation["original_question"]
                != question["question"]
            ):
                raise ValueError(
                    "Cached translation mismatch."
                )

            query = translation["translated_question"]
            translation_seconds = translation[
                "translation_seconds"
            ]

        else:

            query = question["question"]
            translation_seconds = 0.0

        if not query.strip():
            raise ValueError("Empty retrieval query.")

        prepared.append({
            **question,
            "retrieval_query": query,
            "translation_seconds": translation_seconds,
        })

    return prepared


# ==========================================
# 2. Evaluate Retrieval Strategies
# ==========================================

def evaluate_question(engine, question, mode, valid_ids):

    query = question["retrieval_query"]

    start = time.perf_counter()

    retrieved = engine.retrieve(
        query=query,
        top_k=TOP_K,
        mode=mode,
        candidate_k=CANDIDATE_K,
        rrf_k=RRF_K,
    )

    elapsed = time.perf_counter() - start

    retrieved_ids = [
        item["chunk_id"] for item in retrieved
    ]

    if len(retrieved_ids) > TOP_K:
        raise ValueError("Too many retrieved chunks.")

    if len(retrieved_ids) != len(set(retrieved_ids)):
        raise ValueError("Duplicate retrieved chunks.")

    if not set(retrieved_ids).issubset(valid_ids):
        raise ValueError("Unknown retrieved chunk ID.")

    scores = [
        float(item[SCORE_KEYS[mode]])
        for item in retrieved
    ]

    if not all(math.isfinite(score) for score in scores):
        raise ValueError("Invalid retrieval score.")

    metrics = None

    if question["answerable"]:

        metrics = evaluate_retrieval(
            retrieved_ids,
            question["relevant_chunk_ids"],
            DEFAULT_K_VALUES,
        )

    return {
        **question,
        "mode": mode,
        "retrieved_chunk_ids": retrieved_ids,
        "retrieved_scores": scores,
        "score_type": SCORE_KEYS[mode],
        "retrieved_count": len(retrieved_ids),
        "top_score": max(scores) if scores else None,
        "retrieval_seconds": elapsed,
        "metrics": metrics,
    }


# ==========================================
# 3. Aggregate Metrics
# ==========================================

def summarize_group(results):

    answerable = [
        r for r in results if r["answerable"]
    ]

    latencies = [
        r["retrieval_seconds"] * 1000
        for r in results
    ]

    return {
        "total_questions": len(results),
        "answerable_questions": len(answerable),
        "unanswerable_questions": (
            len(results) - len(answerable)
        ),
        "metrics": (
            aggregate_metrics([
                r["metrics"] for r in answerable
            ])
            if answerable else None
        ),
        "mean_retrieval_ms": (
            statistics.mean(latencies)
            if latencies else None
        ),
        "median_retrieval_ms": (
            statistics.median(latencies)
            if latencies else None
        ),
        "p95_retrieval_ms": (
            float(np.percentile(latencies, 95))
            if latencies else None
        ),
    }


def summarize_modes(results):

    summaries = []

    for mode in MODES:

        group = [
            r for r in results if r["mode"] == mode
        ]

        summary = {
            "mode": mode,
            "overall": summarize_group(group),
            "english": summarize_group([
                r for r in group
                if r["language"] == "en"
            ]),
            "arabic": summarize_group([
                r for r in group
                if r["language"] == "ar"
            ]),
            "by_question_type": {},
        }

        for question_type in sorted({
            r["question_type"] for r in group
        }):

            summary["by_question_type"][question_type] = (
                summarize_group([
                    r for r in group
                    if r["question_type"] == question_type
                ])
            )

        summaries.append(summary)

    return summaries


# ==========================================
# 4. Save Results
# ==========================================

def save_results(report):

    summary_rows = []

    for summary in report["summary"]:

        for name in ("overall", "english", "arabic"):

            group = summary[name]

            summary_rows.append({
                "mode": summary["mode"],
                "group": name,
                "total_questions": group["total_questions"],
                "answerable_questions": (
                    group["answerable_questions"]
                ),
                "unanswerable_questions": (
                    group["unanswerable_questions"]
                ),
                "mean_retrieval_ms": group["mean_retrieval_ms"],
                "median_retrieval_ms": group["median_retrieval_ms"],
                "p95_retrieval_ms": group["p95_retrieval_ms"],
                **(group["metrics"] or {}),
            })

    with (
        OUTPUT_DIR / "retrieval_summary.csv"
    ).open("w", encoding="utf-8", newline="") as file:

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

    with (
        OUTPUT_DIR / "retrieval_question_results.csv"
    ).open("w", encoding="utf-8", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(detailed_rows)

    # Write the complete report last.
    with (
        OUTPUT_DIR / "retrieval_results.json"
    ).open("x", encoding="utf-8") as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )


# ==========================================
# 5. Print Comparison
# ==========================================

def print_comparison(summaries):

    print("\n" + "=" * 95)
    print("RETRIEVAL STRATEGY COMPARISON")
    print("=" * 95)

    print(
        f"{'Mode':<12}"
        f"{'Group':<10}"
        f"{'Hit@5':>10}"
        f"{'Recall@5':>12}"
        f"{'MRR@5':>10}"
        f"{'nDCG@5':>10}"
        f"{'Mean ms':>12}"
        f"{'P95 ms':>12}"
    )

    for summary in summaries:

        for name in ("overall", "english", "arabic"):

            group = summary[name]
            metrics = group["metrics"]

            print(
                f"{summary['mode']:<12}"
                f"{name:<10}"
                f"{metrics['hit_rate_at_5']:>10.2%}"
                f"{metrics['recall_at_5']:>12.2%}"
                f"{metrics['mrr_at_5']:>10.4f}"
                f"{metrics['ndcg_at_5']:>10.4f}"
                f"{group['mean_retrieval_ms']:>12.2f}"
                f"{group['p95_retrieval_ms']:>12.2f}"
            )


# ==========================================
# 6. Main Experiment
# ==========================================

def main():

    if (OUTPUT_DIR / "retrieval_results.json").exists():
        raise FileExistsError(
            "Retrieval evaluation already completed."
        )

    corpus_hash = hashlib.sha256(
        CHUNKS_FILE.read_bytes()
    ).hexdigest()

    questions_hash = hashlib.sha256(
        QUESTIONS_FILE.read_bytes()
    ).hexdigest()

    baseline = json.loads(
        BASELINE_FILE.read_text(encoding="utf-8")
    )

    if corpus_hash != baseline["corpus_sha256"]:
        raise ValueError("Corpus mismatch.")

    if questions_hash != baseline["questions_sha256"]:
        raise ValueError("Question dataset mismatch.")

    chunks = json.loads(
        CHUNKS_FILE.read_text(encoding="utf-8-sig")
    )

    valid_ids = {
        c["chunk_id"] for c in chunks
    }

    questions = load_questions(
        QUESTIONS_FILE,
        valid_ids,
    )

    questions = prepare_queries(questions)

    print("\nInitializing production retrieval engine...")

    engine = RetrievalEngine(device=DEVICE)

    if engine.config["model_name"] != EXPECTED_MODEL:
        raise ValueError("Unexpected embedding model.")

    if engine.index.ntotal != len(chunks):
        raise ValueError("FAISS index size mismatch.")

    if [
        (c["chunk_id"], c["text"])
        for c in engine.metadata
    ] != [
        (c["chunk_id"], c["text"])
        for c in chunks
    ]:
        raise ValueError("Production metadata mismatch.")

    print("Loading Cross-Encoder...")

    engine.reranker = Reranker(
        device=DEVICE,
        use_title=False,
        local_files_only=True,
    )

    # Warm-up excluded from measured latency.
    for mode in MODES:

        engine.retrieve(
            query="University academic policies",
            top_k=TOP_K,
            mode=mode,
            candidate_k=CANDIDATE_K,
            rrf_k=RRF_K,
        )

    print("\nDevelopment questions:", len(questions))
    print("Retrieval strategies:", len(MODES))
    print("Expected evaluations:", len(questions) * len(MODES))
    print("Relevance threshold: DISABLED")
    print("Held-out questions: NOT EVALUATED")

    results = []

    for index, question in enumerate(questions, start=1):

        print(
            f"\n[{index:02d}/{len(questions)}] "
            f"{question['question_id']} | "
            f"{question['language']}"
        )

        # Rotate execution order to reduce fixed-order bias.
        offset = (index - 1) % len(MODES)
        ordered_modes = MODES[offset:] + MODES[:offset]

        for mode in ordered_modes:

            result = evaluate_question(
                engine,
                question,
                mode,
                valid_ids,
            )

            results.append(result)

            if result["metrics"] is not None:

                recall = result["metrics"]["recall_at_5"]
                rr = result["metrics"]["rr_at_5"]

                message = (
                    f"Recall@5={recall:.3f} | "
                    f"RR@5={rr:.3f}"
                )

            else:
                message = "Unanswerable"

            print(
                f"  {mode:<10} | "
                f"{message} | "
                f"{result['retrieval_seconds'] * 1000:.2f} ms"
            )

    expected_pairs = {
        (q["question_id"], mode)
        for q in questions
        for mode in MODES
    }

    actual_pairs = {
        (r["question_id"], r["mode"])
        for r in results
    }

    if (
        actual_pairs != expected_pairs
        or len(results) != len(expected_pairs)
    ):
        raise ValueError("Evaluation coverage mismatch.")

    if hashlib.sha256(
        QUESTIONS_FILE.read_bytes()
    ).hexdigest() != questions_hash:
        raise ValueError("Question dataset changed.")

    if hashlib.sha256(
        CHUNKS_FILE.read_bytes()
    ).hexdigest() != corpus_hash:
        raise ValueError("Corpus changed.")

    summaries = summarize_modes(results)

    report = {
        "dataset_version": "1.0.0",
        "evaluation_type": "retrieval_strategy_selection",
        "split": "development",
        "embedding_model": EXPECTED_MODEL,
        "reranker_model": engine.reranker.model_name,
        "query_strategy": "english_or_translated_arabic",
        "corpus_sha256": corpus_hash,
        "questions_sha256": questions_hash,
        "candidate_k": CANDIDATE_K,
        "rrf_k": RRF_K,
        "top_k": TOP_K,
        "k_values": list(DEFAULT_K_VALUES),
        "device": DEVICE,
        "relevance_threshold_enabled": False,
        "total_evaluations": len(results),
        "summary": summaries,
        "results": results,
    }

    save_results(report)
    print_comparison(summaries)

    print("\nCompleted evaluations:", len(results))
    print("Dataset integrity: PASSED")
    print("Results saved:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
