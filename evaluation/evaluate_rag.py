
"""Final held-out evaluation of the production RAG pipeline."""

import argparse
import csv
import hashlib
import json
import statistics
import time

from dataclasses import asdict
from pathlib import Path

from src.rag_pipeline import RAGPipeline
from src.llm import GeminiLLM, load_config, SYSTEM_INSTRUCTION
from src.query_translator import TRANSLATION_SYSTEM_INSTRUCTION


BASE_DIR = Path(__file__).resolve().parents[1]

QUESTIONS_FILE = BASE_DIR / "evaluation" / "test_questions.csv"
CHUNKS_FILE = BASE_DIR / "data" / "processed" / "chunks.json"

OUTPUT_DIR = (
    BASE_DIR / "evaluation" / "final_results" / "rag_test"
)

CONFIG_FILE = OUTPUT_DIR / "evaluation_config.json"
RESULTS_FILE = OUTPUT_DIR / "rag_results.json"

EXPECTED_MODEL = "gemini-3.1-flash-lite"
EXPECTED_EMBEDDING = "intfloat/multilingual-e5-base"
EXPECTED_RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"

TOP_K = 5
CANDIDATE_K = 20
RRF_K = 60
THRESHOLD = -8.0
DEVICE = "cpu"

EXPECTED_QUESTIONS_HASH = (
    "489af6af97a3a6a48e7ca0e05cf652ab"
    "785f87fb291cae4c43c30d8e6ec7adaf"
)

EXPECTED_CORPUS_HASH = (
    "eeec7a6aa2b333d5ceb5f4284f5f8872"
    "deb334e9ed1a26c9c675faae0435b09c"
)


# ==========================================
# 1. Utility Functions
# ==========================================

def sha256_file(path):

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def sha256_text(text):

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def save_json(path, data):

    temporary = path.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    temporary.replace(path)


# ==========================================
# 2. Load Held-out Questions
# ==========================================

def load_test_questions():

    if sha256_file(QUESTIONS_FILE) != EXPECTED_QUESTIONS_HASH:
        raise ValueError("Frozen question dataset changed.")

    if sha256_file(CHUNKS_FILE) != EXPECTED_CORPUS_HASH:
        raise ValueError("Frozen corpus changed.")

    chunks = json.loads(
        CHUNKS_FILE.read_text(encoding="utf-8-sig")
    )

    valid_ids = {
        chunk["chunk_id"] for chunk in chunks
    }

    questions = []

    with QUESTIONS_FILE.open(
        encoding="utf-8-sig",
        newline="",
    ) as file:

        for row in csv.DictReader(file):

            if row["split"] != "test":
                continue

            if row["review_status"] != "verified":
                raise ValueError("Unverified question.")

            if row["answerable"].lower() not in ("true", "false"):
                raise ValueError("Invalid answerability label.")

            row["answerable"] = (
                row["answerable"].lower() == "true"
            )

            row["relevant_chunk_ids"] = json.loads(
                row["relevant_chunk_ids"]
            )

            row["source_document_ids"] = json.loads(
                row["source_document_ids"]
            )

            if not set(
                row["relevant_chunk_ids"]
            ).issubset(valid_ids):
                raise ValueError("Unknown ground-truth chunk.")

            if bool(row["relevant_chunk_ids"]) != row["answerable"]:
                raise ValueError("Invalid evidence annotation.")

            questions.append(row)

    if len(questions) != 42:
        raise ValueError("Expected 42 held-out questions.")

    if len({
        q["question_id"] for q in questions
    }) != 42:
        raise ValueError("Duplicate question IDs.")

    if sum(q["answerable"] for q in questions) != 38:
        raise ValueError("Unexpected answerability distribution.")

    if sum(q["language"] == "en" for q in questions) != 30:
        raise ValueError("Unexpected English question count.")

    if sum(q["language"] == "ar" for q in questions) != 12:
        raise ValueError("Unexpected Arabic question count.")

    return questions


# ==========================================
# 3. Freeze Evaluation Configuration
# ==========================================

def build_configuration(llm_config):

    source_files = sorted(
        (BASE_DIR / "src").rglob("*.py")
    )

    artifact_files = sorted(
        path
        for path in (BASE_DIR / "data" / "embeddings").iterdir()
        if path.is_file()
        and path.suffix in (".json", ".npy", ".faiss")
    )

    return {
        "dataset_version": "1.0.0",
        "split": "test",
        "question_count": 42,
        "questions_sha256": sha256_file(QUESTIONS_FILE),
        "corpus_sha256": sha256_file(CHUNKS_FILE),
        "embedding_model": EXPECTED_EMBEDDING,
        "reranker_model": EXPECTED_RERANKER,
        "llm_model": llm_config.model,
        "llm_timeout_ms": llm_config.timeout_ms,
        "retrieval_mode": "reranked",
        "top_k": TOP_K,
        "candidate_k": CANDIDATE_K,
        "rrf_k": RRF_K,
        "threshold": THRESHOLD,
        "device": DEVICE,
        "translation_enabled": True,
        "citation_validation_enabled": True,
        "generation_prompt_sha256": sha256_text(
            SYSTEM_INSTRUCTION
        ),
        "translation_prompt_sha256": sha256_text(
            TRANSLATION_SYSTEM_INSTRUCTION
        ),
        "source_hashes": {
            path.relative_to(BASE_DIR).as_posix():
                sha256_file(path)
            for path in source_files
        },
        "embedding_artifact_hashes": {
            path.relative_to(BASE_DIR).as_posix():
                sha256_file(path)
            for path in artifact_files
        },
        "evaluation_script_sha256": sha256_file(
            Path(__file__).resolve()
        ),
    }


def freeze_configuration(configuration):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if CONFIG_FILE.exists():

        existing = json.loads(
            CONFIG_FILE.read_text(encoding="utf-8")
        )

        if existing != configuration:
            raise ValueError(
                "Evaluation configuration changed. "
                "Do not mix results from different configurations."
            )

        print("Frozen configuration verified.")

    else:

        save_json(CONFIG_FILE, configuration)

        print("Evaluation configuration frozen.")


# ==========================================
# 4. Load Checkpoint
# ==========================================

def load_checkpoint(configuration):

    if RESULTS_FILE.exists():

        report = json.loads(
            RESULTS_FILE.read_text(encoding="utf-8")
        )

        if report["configuration"] != configuration:
            raise ValueError("Checkpoint configuration mismatch.")

        return report

    return {
        "evaluation_type": "final_end_to_end_rag",
        "configuration": configuration,
        "results": {},
    }


# ==========================================
# 5. Collect Response Metrics
# ==========================================

def collect_metrics(question, response):

    translation = (
        response.translation_result.llm_response
        if response.translation_result is not None
        else None
    )

    generation = response.llm_response

    api_responses = [
        item
        for item in (translation, generation)
        if item is not None
    ]

    token_values = [
        item.total_tokens for item in api_responses
    ]

    known_tokens = sum(
        value
        for value in token_values
        if value is not None
    )

    complete_tokens = all(
        value is not None
        for value in token_values
    )

    technical_error = response.status not in (
        "success",
        "insufficient_context",
    )

    if technical_error:
        outcome = "technical_error"

    elif response.status == "insufficient_context":

        outcome = (
            "false_abstention"
            if question["answerable"]
            else "correct_abstention"
        )

    else:

        outcome = (
            "answer_generated"
            if question["answerable"]
            else "unanswerable_answer_generated"
        )

    retrieved_ids = {
        chunk["chunk_id"]
        for chunk in response.retrieved_chunks
    }

    relevant_ids = set(
        question["relevant_chunk_ids"]
    )

    retained_recall = (
        len(retrieved_ids & relevant_ids) / len(relevant_ids)
        if relevant_ids
        else None
    )

    validation = response.citation_validation

    return {
        "outcome": outcome,
        "technical_error": technical_error,
        "correct_abstention": (
            outcome == "correct_abstention"
        ),
        "false_abstention": (
            outcome == "false_abstention"
        ),
        "retained_evidence_recall_at_5": retained_recall,
        "translation_seconds": (
            response.translation_latency_seconds
        ),
        "retrieval_seconds": (
            response.retrieval_latency_seconds
        ),
        "generation_seconds": (
            generation.total_elapsed_seconds
            if generation is not None
            else 0.0
        ),
        "total_latency_seconds": (
            response.total_latency_seconds
        ),
        "logical_api_requests": len(api_responses),
        "api_attempts": sum(
            item.attempt_count for item in api_responses
        ),
        "api_retries": sum(
            item.retry_count for item in api_responses
        ),
        "translation_tokens": (
            translation.total_tokens
            if translation is not None
            else 0
        ),
        "generation_tokens": (
            generation.total_tokens
            if generation is not None
            else 0
        ),
        "known_total_tokens": known_tokens,
        "total_tokens": (
            known_tokens if complete_tokens else None
        ),
        "token_metadata_complete": complete_tokens,
        "citation_count": (
            validation.citation_count
            if validation is not None
            else None
        ),
        "valid_citation_count": (
            validation.valid_citation_count
            if validation is not None
            else None
        ),
        "citation_validity": (
            validation.citation_validity
            if validation is not None
            else None
        ),
    }


# ==========================================
# 6. Run Evaluation
# ==========================================

def run_evaluation(
    questions,
    configuration,
    llm_config,
    report,
    limit=None,
):

    pending = [
        question
        for question in questions
        if question["question_id"] not in report["results"]
    ]

    if limit is not None:
        pending = pending[:limit]

    if not pending:
        print("No pending questions.")
        return

    print("Initializing production RAG pipeline...")

    with GeminiLLM(config=llm_config) as llm:

        with RAGPipeline(
            llm=llm,
            device=DEVICE,
        ) as pipeline:

            engine = pipeline.retrieval_engine

            if engine.config["model_name"] != EXPECTED_EMBEDDING:
                raise ValueError("Embedding model mismatch.")

            reranker = engine.load_reranker()

            if reranker.model_name != EXPECTED_RERANKER:
                raise ValueError("Reranker model mismatch.")

            # Local retrieval warm-up; no API request.
            engine.retrieve(
                query="University academic policies",
                top_k=TOP_K,
                mode="reranked",
                candidate_k=CANDIDATE_K,
                rrf_k=RRF_K,
            )

            for question in pending:

                qid = question["question_id"]

                print("\n" + "=" * 70)

                print(
                    f"[{len(report['results']) + 1}/42] "
                    f"{qid} | {question['language']}"
                )

                start = time.perf_counter()

                try:

                    response = pipeline.ask(
                        question=question["question"],
                        top_k=TOP_K,
                        candidate_k=CANDIDATE_K,
                        rrf_k=RRF_K,
                        threshold=THRESHOLD,
                    )

                    record = {
                        "ground_truth": question,
                        "response": asdict(response),
                        "metrics": collect_metrics(
                            question,
                            response,
                        ),
                    }

                except Exception as error:

                    record = {
                        "ground_truth": question,
                        "response": {
                            "status": "pipeline_exception",
                            "error_type": type(error).__name__,
                        },
                        "metrics": {
                            "outcome": "technical_error",
                            "technical_error": True,
                            "total_latency_seconds": (
                                time.perf_counter() - start
                            ),
                        },
                    }

                report["results"][qid] = record

                # Checkpoint after every question.
                save_json(RESULTS_FILE, report)

                print(
                    "Status:",
                    record["response"]["status"],
                )

                print(
                    "Outcome:",
                    record["metrics"]["outcome"],
                )

                print(
                    "Latency:",
                    record["metrics"]["total_latency_seconds"],
                    "seconds",
                )

                print(
                    "Tokens:",
                    record["metrics"].get("total_tokens"),
                )

                print(
                    "API attempts:",
                    record["metrics"].get("api_attempts"),
                )

                # Preserve failures and pause for investigation.
                if record["metrics"]["technical_error"]:

                    print(
                        "\nTechnical failure recorded. "
                        "Evaluation paused."
                    )

                    break


# ==========================================
# 7. Export Results and Review Sheet
# ==========================================

def export_results(report):

    records = list(
        report["results"].values()
    )

    if not records:
        return

    rows = []
    review_rows = []

    for record in records:

        question = record["ground_truth"]
        response = record["response"]
        metrics = record["metrics"]

        rows.append({
            "question_id": question["question_id"],
            "language": question["language"],
            "question_type": question["question_type"],
            "difficulty": question["difficulty"],
            "answerable": question["answerable"],
            "status": response["status"],
            "relevance_passed": response.get("relevance_passed"),
            "max_relevance_score": response.get(
                "max_relevance_score"
            ),
            **metrics,
        })

        validation = response.get("citation_validation") or {}

        review_rows.append({
            "question_id": question["question_id"],
            "language": question["language"],
            "question": question["question"],
            "answerable": question["answerable"],
            "expected_answer": question["expected_answer"],
            "relevant_chunk_ids": json.dumps(
                question["relevant_chunk_ids"]
            ),
            "generated_answer": response.get("answer"),
            "status": response["status"],
            "raw_generated_answer": (
                response.get("llm_response") or {}
            ).get("answer"),
            "retrieved_chunk_ids": json.dumps([
                chunk["chunk_id"]
                for chunk in response.get("retrieved_chunks", [])
            ]),
            "cited_ids": json.dumps(
                validation.get("cited_ids", [])
            ),
            "retrieved_context": response.get("context"),
            "correctness": "",
            "completeness": "",
            "faithfulness": "",
            "citation_faithfulness": "",
            "unsupported_claims": "",
            "reviewer": "",
            "review_notes": "",
            "review_status": "pending",
        })

    fields = list(dict.fromkeys(
        key for row in rows for key in row
    ))

    with (
        OUTPUT_DIR / "rag_question_results.csv"
    ).open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)

    review_file = OUTPUT_DIR / "rag_answer_review.csv"

    # Never overwrite a review that may already be annotated.
    if not review_file.exists() and len(records) == 42:

        with review_file.open(
            "x",
            encoding="utf-8-sig",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(review_rows[0]),
            )

            writer.writeheader()
            writer.writerows(review_rows)


# ==========================================
# 8. Preliminary Automatic Summary
# ==========================================

def summarize(report):

    records = list(
        report["results"].values()
    )

    metrics = [
        record["metrics"]
        for record in records
    ]

    outcomes = {}

    for row in metrics:

        outcome = row["outcome"]

        outcomes[outcome] = (
            outcomes.get(outcome, 0) + 1
        )

    latencies = [
        row["total_latency_seconds"]
        for row in metrics
    ]

    known_attempts = [
        row["api_attempts"]
        for row in metrics
        if row.get("api_attempts") is not None
    ]

    known_tokens = [
        row["known_total_tokens"]
        for row in metrics
        if row.get("known_total_tokens") is not None
    ]

    summary = {
        "evaluated_questions": len(records),
        "expected_questions": 42,
        "evaluation_complete": len(records) == 42,
        "outcomes": outcomes,
        "mean_latency_seconds": (
            statistics.mean(latencies)
            if latencies else None
        ),
        "median_latency_seconds": (
            statistics.median(latencies)
            if latencies else None
        ),
        "known_api_attempts": sum(known_attempts),
        "known_total_tokens": sum(known_tokens),
        "questions_with_complete_token_metadata": sum(
            row.get("token_metadata_complete", False)
            for row in metrics
        ),
        "semantic_answer_review_complete": False,
    }

    save_json(
        OUTPUT_DIR / "rag_automatic_summary.json",
        summary,
    )

    print("\nAUTOMATIC EVALUATION SUMMARY")
    print("=" * 60)

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )


# ==========================================
# 9. Main
# ==========================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--freeze-only",
        action="store_true",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")

    questions = load_test_questions()

    llm_config = load_config()

    if llm_config.model != EXPECTED_MODEL:
        raise ValueError(
            f"Unexpected Gemini model: {llm_config.model}"
        )

    configuration = build_configuration(llm_config)

    freeze_configuration(configuration)

    print("LLM model:", llm_config.model)
    print("Threshold:", THRESHOLD)
    print("Held-out questions:", len(questions))
    print("Dataset integrity: PASSED")

    if args.freeze_only:

        print("Configuration saved:", CONFIG_FILE)
        print("API requests made: 0")

        return

    report = load_checkpoint(configuration)

    valid_ids = {
        question["question_id"]
        for question in questions
    }

    if not set(report["results"]).issubset(valid_ids):
        raise ValueError("Unexpected checkpoint question IDs.")

    run_evaluation(
        questions,
        configuration,
        llm_config,
        report,
        limit=args.limit,
    )

    if build_configuration(llm_config) != configuration:
        raise ValueError(
            "Configuration changed during evaluation."
        )

    export_results(report)
    summarize(report)

    print("\nResults saved:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
