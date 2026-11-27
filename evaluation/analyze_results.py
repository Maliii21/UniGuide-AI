
import csv
from pathlib import Path
from statistics import mean
from collections import Counter

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    BASE_DIR / "evaluation/final_results/comparison"
)

REVIEW_FILE = OUTPUT_DIR / "paired_answer_review.csv"
SUMMARY_FILE = OUTPUT_DIR / "comparison_summary.csv"
FIGURE_FILE = OUTPUT_DIR / "answer_quality_comparison.png"


def load_reviews():
    with REVIEW_FILE.open(
        encoding="utf-8-sig", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    if len(rows) != 84:
        raise ValueError("Expected 84 evaluation records.")

    for row in rows:
        expected_status = (
            "technical_error"
            if row["outcome"] == "technical_error"
            else "reviewed"
    )

    if row["review_status"] != expected_status:
        raise ValueError(
            f"Invalid review status for "
            f"{row['question_id']} ({row['mode']})"
        )

    with_rag = [
        row for row in rows if row["mode"] == "with_rag"
    ]

    without_rag = [
        row for row in rows if row["mode"] == "without_rag"
    ]

    if len(with_rag) != 42 or len(without_rag) != 42:
        raise ValueError("Expected 42 questions per mode.")

    rag_ids = {row["question_id"] for row in with_rag}
    baseline_ids = {row["question_id"] for row in without_rag}

    if len(rag_ids) != 42 or rag_ids != baseline_ids:
        raise ValueError("Question IDs do not match.")

    return with_rag, without_rag


def calculate_metrics(rows, mode):
    outcomes = Counter(row["outcome"] for row in rows)

    scored = [
        row for row in rows
        if row["outcome"] != "technical_error"
    ]

    correctness = [
        float(row["correctness"]) for row in scored
    ]

    relevance = [
        float(row["relevance"]) for row in scored
    ]

    strict_success = sum(
        float(row["correctness"]) == 2
        and float(row["relevance"]) == 2
        for row in scored
    )

    return {
        "mode": mode,
        "questions": len(rows),
        "generated_answers": (
            outcomes["answer_generated"]
            + outcomes["unanswerable_answer_generated"]
        ),
        "correct_abstentions": outcomes["correct_abstention"],
        "false_abstentions": outcomes["false_abstention"],
        "technical_errors": outcomes["technical_error"],
        "semantic_scored": len(scored),
        "correctness_pct": mean(correctness) / 2 * 100,
        "relevance_pct": mean(relevance) / 2 * 100,
        "strict_accuracy_pct": strict_success / len(rows) * 100,
        "mean_latency_seconds": mean(
            float(row["latency_seconds"]) for row in rows
        ),
    }


def save_summary(results):
    with SUMMARY_FILE.open(
        "w", encoding="utf-8", newline=""
    ) as file:

        writer = csv.DictWriter(
            file, fieldnames=list(results[0].keys())
        )

        writer.writeheader()
        writer.writerows(results)


def create_chart(results):
    metrics = [
        "correctness_pct",
        "relevance_pct",
        "strict_accuracy_pct",
    ]

    labels = [
        "Correctness",
        "Response Relevance",
        "Strict Accuracy",
    ]

    rag = [results[0][key] for key in metrics]
    baseline = [results[1][key] for key in metrics]

    x = range(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(
        [i - width / 2 for i in x],
        rag,
        width,
        label="With RAG",
    )

    bars2 = ax.bar(
        [i + width / 2 for i in x],
        baseline,
        width,
        label="Without RAG",
    )

    ax.bar_label(bars1, fmt="%.2f%%", padding=3)
    ax.bar_label(bars2, fmt="%.2f%%", padding=3)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 110)
    ax.set_title("UniGuide-AI: With RAG vs Without RAG")
    ax.legend(loc="lower center")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(FIGURE_FILE, dpi=200)
    plt.close(fig)


def main():
    with_rag, without_rag = load_reviews()

    results = [
        calculate_metrics(with_rag, "with_rag"),
        calculate_metrics(without_rag, "without_rag"),
    ]

    save_summary(results)
    create_chart(results)

    print("\nFINAL EVALUATION RESULTS")
    print("=" * 65)

    print(
        f"{'Metric':<30}"
        f"{'With RAG':>16}"
        f"{'Without RAG':>19}"
    )

    for key, label in [
        ("correctness_pct", "Correctness (%)"),
        ("relevance_pct", "Response relevance (%)"),
        ("strict_accuracy_pct", "Strict accuracy (%)"),
        ("mean_latency_seconds", "Average latency (s)"),
        ("generated_answers", "Generated answers"),
        ("correct_abstentions", "Correct abstentions"),
        ("false_abstentions", "False abstentions"),
        ("technical_errors", "Technical errors"),
    ]:
        print(
            f"{label:<30}"
            f"{results[0][key]:>16.2f}"
            f"{results[1][key]:>19.2f}"
        )

    print("=" * 65)
    print(f"Summary saved: {SUMMARY_FILE}")
    print(f"Chart saved: {FIGURE_FILE}")
    print("Evaluation analysis completed.")


if __name__ == "__main__":
    main()
