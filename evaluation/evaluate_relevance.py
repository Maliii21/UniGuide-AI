
"""Evaluate relevance thresholds using saved reranker scores."""

import csv
import json
import math
import hashlib

from pathlib import Path

from src.relevance_filter import (
    check_relevance,
    DEFAULT_THRESHOLD,
)

from evaluation.evaluate_embeddings_final import (
    BASE_DIR,
    QUESTIONS_FILE,
    CHUNKS_FILE,
)


RESULTS_DIR = BASE_DIR / "evaluation" / "final_results"

INPUT_FILE = (
    RESULTS_DIR
    / "retrieval_development"
    / "retrieval_results.json"
)

OUTPUT_DIR = RESULTS_DIR / "relevance_development"

TOP_K = 5


def safe_divide(numerator, denominator):

    return numerator / denominator if denominator else 0.0


# ==========================================
# 1. Load Saved Retrieval Results
# ==========================================

def load_results():

    report = json.loads(
        INPUT_FILE.read_text(encoding="utf-8")
    )

    if report["split"] != "development":
        raise ValueError("Expected development results.")

    if report["relevance_threshold_enabled"]:
        raise ValueError("Expected unfiltered retrieval results.")

    if report["candidate_k"] != 20 or report["rrf_k"] != 60:
        raise ValueError("Unexpected retrieval configuration.")

    if report["top_k"] < TOP_K:
        raise ValueError("Insufficient saved retrieval results.")

    if report["questions_sha256"] != hashlib.sha256(
        QUESTIONS_FILE.read_bytes()
    ).hexdigest():
        raise ValueError("Question dataset mismatch.")

    if report["corpus_sha256"] != hashlib.sha256(
        CHUNKS_FILE.read_bytes()
    ).hexdigest():
        raise ValueError("Corpus mismatch.")

    rows = [
        row for row in report["results"]
        if row["mode"] == "reranked"
    ]

    if len(rows) != 28:
        raise ValueError("Expected 28 reranked results.")

    if len({r["question_id"] for r in rows}) != 28:
        raise ValueError("Duplicate questions.")

    if sum(r["answerable"] for r in rows) != 25:
        raise ValueError("Unexpected answerability distribution.")

    prepared = []

    for row in rows:

        if row["score_type"] != "reranker_score":
            raise ValueError("Unexpected score type.")

        ids = row["retrieved_chunk_ids"][:TOP_K]
        scores = row["retrieved_scores"][:TOP_K]

        if len(ids) != len(scores):
            raise ValueError("Score alignment mismatch.")

        results = [
            {
                "chunk_id": cid,
                "reranker_score": score,
            }
            for cid, score in zip(ids, scores)
        ]

        decision = check_relevance(
            results,
            threshold=DEFAULT_THRESHOLD,
        )

        prepared.append({
            "question_id": row["question_id"],
            "question": row["question"],
            "language": row["language"],
            "answerable": row["answerable"],
            "max_score": decision["max_score"],
            "results": results,
        })

    return prepared, report


# ==========================================
# 2. Classification Metrics
# ==========================================

def calculate_metrics(tp, tn, fp, fn):

    accuracy = safe_divide(
        tp + tn,
        tp + tn + fp + fn,
    )

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)

    f1 = safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": (
            recall + specificity
        ) / 2,
        "false_rejection_rate": safe_divide(
            fn, tp + fn
        ),
        "false_acceptance_rate": safe_divide(
            fp, tn + fp
        ),
    }


# ==========================================
# 3. Evaluate One Threshold
# ==========================================

def evaluate_threshold(questions, threshold):

    tp = tn = fp = fn = 0

    details = []

    for question in questions:

        decision = check_relevance(
            question["results"],
            threshold=threshold,
        )

        actual = question["answerable"]
        predicted = decision["passed"]

        if actual and predicted:
            tp += 1
            outcome = "TP"

        elif not actual and not predicted:
            tn += 1
            outcome = "TN"

        elif not actual and predicted:
            fp += 1
            outcome = "FP"

        else:
            fn += 1
            outcome = "FN"

        details.append({
            "question_id": question["question_id"],
            "question": question["question"],
            "language": question["language"],
            "answerable": actual,
            "max_score": decision["max_score"],
            "passed": predicted,
            "outcome": outcome,
            "retained_count": len(decision["results"]),
        })

    return {
        "threshold": float(threshold),
        **calculate_metrics(tp, tn, fp, fn),
        "details": details,
    }


# ==========================================
# 4. Threshold Candidates
# ==========================================

def build_thresholds(questions):

    scores = sorted({
        q["max_score"]
        for q in questions
        if q["max_score"] is not None
    })

    if not scores:
        raise ValueError("No reranker scores available.")

    # Fixed grid plus score boundaries.
    thresholds = {
        i / 2 for i in range(-24, 25)
    }

    thresholds.add(float(DEFAULT_THRESHOLD))

    thresholds.add(
        math.nextafter(scores[0], -math.inf)
    )

    thresholds.add(
        math.nextafter(scores[-1], math.inf)
    )

    for left, right in zip(scores, scores[1:]):

        thresholds.add(
            (left + right) / 2
        )

    return sorted(thresholds)


# ==========================================
# 5. Conservative Threshold Selection
# ==========================================

def select_threshold(results):

    # Prioritize avoiding rejection of answerable questions.
    # Among equally safe candidates, maximize specificity.
    # Prefer the existing threshold when performance ties.

    return max(
        results,
        key=lambda row: (
            row["recall"],
            row["specificity"],
            -abs(
                row["threshold"] - DEFAULT_THRESHOLD
            ),
            -row["threshold"],
        ),
    )


# ==========================================
# 6. Save Results
# ==========================================

def save_results(report):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    with (
        OUTPUT_DIR / "threshold_results.json"
    ).open("x", encoding="utf-8") as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    rows = [
        {
            key: value
            for key, value in row.items()
            if key != "details"
        }
        for row in report["threshold_sweep"]
    ]

    with (
        OUTPUT_DIR / "threshold_sweep.csv"
    ).open("x", encoding="utf-8", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)

    details = report["selected"]["details"]

    with (
        OUTPUT_DIR / "selected_threshold_details.csv"
    ).open("x", encoding="utf-8", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(details[0]),
        )

        writer.writeheader()
        writer.writerows(details)


# ==========================================
# 7. Print Results
# ==========================================

def print_result(label, result):

    print("\n" + label)
    print("-" * 55)

    print("Threshold:", result["threshold"])

    for key in ("tp", "tn", "fp", "fn"):
        print(f"{key.upper():<22}: {result[key]}")

    for key in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "balanced_accuracy",
        "false_rejection_rate",
        "false_acceptance_rate",
    ):

        print(
            f"{key:<22}: {result[key]:.2%}"
        )


# ==========================================
# 8. Main Experiment
# ==========================================

def main():

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            "Relevance evaluation directory already exists."
        )

    questions, source_report = load_results()

    print("Development questions:", len(questions))
    print("Answerable:", sum(q["answerable"] for q in questions))
    print("Unanswerable:", sum(not q["answerable"] for q in questions))
    print("Model inference: NOT REQUIRED")
    print("API calls: NOT REQUIRED")
    print("Held-out questions: NOT EVALUATED")

    print("\nQUESTION-LEVEL MAXIMUM SCORES")
    print("=" * 65)

    for question in sorted(
        questions,
        key=lambda q: (
            q["max_score"]
            if q["max_score"] is not None
            else -math.inf
        ),
    ):

        print(
            f"{question['question_id']:<10} | "
            f"Answerable={str(question['answerable']):<5} | "
            f"Max score={question['max_score']}"
        )

    thresholds = build_thresholds(questions)

    results = [
        evaluate_threshold(questions, threshold)
        for threshold in thresholds
    ]

    baseline = evaluate_threshold(
        questions,
        DEFAULT_THRESHOLD,
    )

    selected = select_threshold(results)

    best_f1 = max(
        results,
        key=lambda r: (
            r["f1"],
            r["recall"],
            -abs(r["threshold"] - DEFAULT_THRESHOLD),
        ),
    )

    report = {
        "evaluation_type": "relevance_threshold_selection",
        "split": "development",
        "target_label": "answerability_proxy",
        "dataset_version": "1.0.0",
        "corpus_sha256": source_report["corpus_sha256"],
        "questions_sha256": source_report["questions_sha256"],
        "retrieval_results_sha256": hashlib.sha256(
            INPUT_FILE.read_bytes()
        ).hexdigest(),
        "embedding_model": source_report["embedding_model"],
        "reranker_model": source_report["reranker_model"],
        "candidate_k": source_report["candidate_k"],
        "rrf_k": source_report["rrf_k"],
        "top_k": TOP_K,
        "selection_policy": (
            "maximize_recall_then_specificity_"
            "then_prefer_existing_threshold"
        ),
        "total_questions": len(questions),
        "thresholds_evaluated": len(thresholds),
        "baseline": baseline,
        "selected": selected,
        "best_f1": best_f1,
        "threshold_sweep": results,
    }

    save_results(report)

    print("\nThresholds evaluated:", len(thresholds))

    print_result(
        "BASELINE PRODUCTION THRESHOLD",
        baseline,
    )

    print_result(
        "SELECTED CONSERVATIVE THRESHOLD",
        selected,
    )

    print_result(
        "BEST F1 THRESHOLD",
        best_f1,
    )

    print("\nSELECTED THRESHOLD ERRORS")
    print("=" * 55)

    errors = [
        row for row in selected["details"]
        if row["outcome"] in ("FP", "FN")
    ]

    if not errors:
        print("No development classification errors.")

    for row in errors:

        print(
            f"{row['question_id']} | "
            f"{row['outcome']} | "
            f"Score={row['max_score']}"
        )

        print(row["question"])

    print("\nResults saved:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
