
import json
import csv
import time
from pathlib import Path
from dataclasses import asdict

from src.llm import GeminiLLM, load_config


BASE_DIR = Path(__file__).resolve().parents[1]

RAG_RESULTS_FILE = (
    BASE_DIR / "evaluation/final_results/rag_test/rag_results.json"
)

OUTPUT_DIR = (
    BASE_DIR / "evaluation/final_results/without_rag_test"
)

RESULTS_FILE = OUTPUT_DIR / "without_rag_results.json"
REVIEW_FILE = OUTPUT_DIR / "without_rag_answer_review.csv"


SYSTEM_INSTRUCTION = """
You are UniGuide-AI, a university assistant for
Alamein International University (AIU).

Answer the student's question using your existing knowledge only.

No university documents or retrieved context are provided.

Rules:
- Answer in the same language as the question.
- Be accurate, clear, concise, and relevant.
- Do not invent university-specific policies or procedures.
- Preserve important conditions, deadlines, and exceptions.
- Do not fabricate citations or references.
- If you cannot reliably answer, return exactly:
  INSUFFICIENT_CONTEXT
"""


def save_results(data):
    temporary_file = RESULTS_FILE.with_suffix(".tmp")

    temporary_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    temporary_file.replace(RESULTS_FILE)


def export_review(data):
    fields = [
        "question_id",
        "language",
        "question",
        "answerable",
        "expected_answer",
        "generated_answer",
        "status",
        "outcome",
        "correctness",
        "relevance",
        "completeness",
        "unsupported_claims",
        "reviewer",
        "review_notes",
        "review_status",
    ]

    if REVIEW_FILE.exists():
        return

    with REVIEW_FILE.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:

        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for question_id, record in data["results"].items():
            ground_truth = record["ground_truth"]

            writer.writerow({
                "question_id": question_id,
                "language": ground_truth["language"],
                "question": ground_truth["question"],
                "answerable": ground_truth["answerable"],
                "expected_answer": ground_truth["expected_answer"],
                "generated_answer": record["answer"],
                "status": record["status"],
                "outcome": record["outcome"],
                "review_status": "pending",
            })


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rag_data = json.loads(
        RAG_RESULTS_FILE.read_text(encoding="utf-8")
    )

    questions = {
        question_id: record["ground_truth"]
        for question_id, record in rag_data["results"].items()
    }

    if len(questions) != 42:
        raise ValueError("Expected exactly 42 test questions.")

    config = load_config()

    expected_model = rag_data["configuration"]["llm_model"]

    if config.model != expected_model:
        raise ValueError(
            f"Model mismatch: expected {expected_model}"
        )

    baseline_config = {
        "dataset_version": rag_data["configuration"]["dataset_version"],
        "questions_sha256": rag_data["configuration"]["questions_sha256"],
        "model": expected_model,
        "question_count": len(questions),
        "retrieval_enabled": False,
        "translation_enabled": False,
        "context_enabled": False,
        "system_instruction": SYSTEM_INSTRUCTION,
    }

    if RESULTS_FILE.exists():
        data = json.loads(
            RESULTS_FILE.read_text(encoding="utf-8")
        )

        if data["configuration"] != baseline_config:
            raise ValueError("Saved configuration does not match.")

        if not set(data["results"]).issubset(questions):
            raise ValueError("Unexpected checkpoint question IDs.")

    else:
        data = {
            "evaluation_type": "without_rag",
            "configuration": baseline_config,
            "results": {},
        }

    print(f"Model: {expected_model}")
    print(f"Total questions: {len(questions)}")
    print(f"Already completed: {len(data['results'])}")

    with GeminiLLM(config=config) as llm:

        for question_id, question in questions.items():

            if question_id in data["results"]:
                continue

            print("\n" + "=" * 60)
            print(
                f"[{len(data['results']) + 1}/42] "
                f"{question_id} | {question['language']}"
            )

            start = time.perf_counter()

            response = llm._generate(
                prompt=question["question"],
                system_instruction=SYSTEM_INSTRUCTION,
            )

            latency = time.perf_counter() - start

            answer = response.answer or ""

            if not response.success:
                status = "technical_error"
                outcome = "technical_error"

            elif answer.strip() == "INSUFFICIENT_CONTEXT":
                status = "insufficient_context"

                outcome = (
                    "false_abstention"
                    if question["answerable"]
                    else "correct_abstention"
                )

            else:
                status = "success"

                outcome = (
                    "answer_generated"
                    if question["answerable"]
                    else "unanswerable_answer_generated"
                )

            data["results"][question_id] = {
                "ground_truth": question,
                "answer": answer,
                "status": status,
                "outcome": outcome,
                "latency_seconds": round(latency, 4),
                "llm_response": asdict(response),
            }

            save_results(data)

            print(f"Status: {status}")
            print(f"Outcome: {outcome}")
            print(f"Latency: {latency:.2f} seconds")
            print(f"Tokens: {response.total_tokens}")

            if status == "technical_error":
                print(f"Error: {response.error_type}")
                print("Evaluation paused. Checkpoint saved.")
                break

            # Reduce the risk of exceeding API rate limits.
            time.sleep(5)

    completed = len(data["results"])

    print("\n" + "=" * 60)
    print(f"Completed: {completed}/42")
    print(f"Results saved: {RESULTS_FILE}")

    if completed == len(questions):
        export_review(data)
        print(f"Review file saved: {REVIEW_FILE}")
        print("Without-RAG evaluation completed.")


if __name__ == "__main__":
    main()
