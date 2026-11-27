
"""Validate the selected relevance threshold on critical dev questions."""

import json
from dataclasses import asdict

from src.rag_pipeline import RAGPipeline

from evaluation.evaluate_embeddings_final import (
    BASE_DIR,
    QUESTIONS_FILE,
    CHUNKS_FILE,
    load_questions,
)


OUTPUT_FILE = (
    BASE_DIR
    / "evaluation"
    / "final_results"
    / "relevance_development"
    / "threshold_e2e_validation.json"
)

QUESTION_IDS = (
    "AR-006",
    "AR-008",
    "EN-019",
)

THRESHOLD = -8.0


def main():

    chunks = json.loads(
        CHUNKS_FILE.read_text(encoding="utf-8-sig")
    )

    valid_ids = {
        chunk["chunk_id"]
        for chunk in chunks
    }

    questions = load_questions(
        QUESTIONS_FILE,
        valid_ids,
    )

    question_map = {
        q["question_id"]: q
        for q in questions
    }

    if OUTPUT_FILE.exists():

        report = json.loads(
            OUTPUT_FILE.read_text(encoding="utf-8")
        )

        if report["threshold"] != THRESHOLD:
            raise ValueError("Threshold mismatch.")

    else:

        report = {
            "evaluation_type": "targeted_threshold_validation",
            "split": "development",
            "threshold": THRESHOLD,
            "top_k": 5,
            "candidate_k": 20,
            "rrf_k": 60,
            "results": {},
        }

    pending = [
        qid for qid in QUESTION_IDS
        if qid not in report["results"]
    ]

    if not pending:
        print("All validation questions already completed.")
        return

    print("Initializing production RAG pipeline...")

    with RAGPipeline(device="cpu") as pipeline:

        for qid in pending:

            question = question_map[qid]

            print("\n" + "=" * 70)
            print("Question ID:", qid)
            print("Question:", question["question"])
            print("Answerable:", question["answerable"])

            response = pipeline.ask(
                question=question["question"],
                top_k=5,
                candidate_k=20,
                rrf_k=60,
                threshold=THRESHOLD,
            )

            print("Status:", response.status)
            print("Retrieval query:", response.retrieval_query)
            print("Maximum score:", response.max_relevance_score)
            print("Relevance passed:", response.relevance_passed)

            print("\nGenerated answer:")
            print(response.answer)

            print("\nRetrieved chunks:")

            for chunk in response.retrieved_chunks:
                print(
                    chunk["chunk_id"],
                    chunk.get("reranker_score"),
                )

            print(
                "\nTotal latency:",
                response.total_latency_seconds,
                "seconds",
            )

            if response.status not in (
                "success",
                "insufficient_context",
            ):
                raise RuntimeError(
                    f"{qid}: {response.status} | "
                    f"{response.error_message}"
                )

            report["results"][qid] = {
                "ground_truth": question,
                "response": asdict(response),
            }

            # Save each completed question immediately.
            temporary_file = OUTPUT_FILE.with_suffix(".tmp")

            temporary_file.write_text(
                json.dumps(
                    report,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )

            temporary_file.replace(OUTPUT_FILE)

    print("\nCompleted questions:", len(report["results"]))
    print("Results saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
