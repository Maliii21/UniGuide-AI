
"""Validate the UniGuide-AI evaluation dataset."""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = ROOT / "evaluation" / "test_questions.csv"
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.json"

REQUIRED_COLUMNS = [
    "question_id",
    "question",
    "language",
    "expected_answer",
    "relevant_chunk_ids",
    "source_document_ids",
    "category",
    "question_type",
    "difficulty",
    "answerable",
    "split",
    "fact_group_id",
    "annotation_notes",
    "review_status",
]

EXPECTED_TYPES = {
    "direct_factual": 21,
    "paraphrased": 21,
    "procedural": 14,
    "complex_reasoning": 7,
    "unanswerable": 7,
}

ALLOWED_VALUES = {
    "language": {"en", "ar"},
    "question_type": set(EXPECTED_TYPES),
    "difficulty": {"easy", "medium", "hard"},
    "answerable": {"true", "false"},
    "split": {"development", "test"},
    "review_status": {"pending", "verified"},
}


def validate_dataset():
    errors = []
    warnings = []

    with DATASET_PATH.open(
        encoding="utf-8", newline=""
    ) as file:
        reader = csv.DictReader(file, strict=True)

        if reader.fieldnames != REQUIRED_COLUMNS:
            print("ERROR: Incorrect CSV header.")
            return False

        rows = list(reader)

    with CHUNKS_PATH.open(encoding="utf-8") as file:
        chunks = json.load(file)

    chunk_map = {
        chunk["chunk_id"]: chunk["document_id"]
        for chunk in chunks
    }

    question_ids = set()
    question_texts = set()
    fact_groups = defaultdict(set)

    for row_number, row in enumerate(rows, start=2):

        if None in row or None in row.values():
            errors.append(f"Malformed row: {row_number}")
            continue

        qid = row["question_id"]

        # Required fields
        for column in REQUIRED_COLUMNS:
            if not row[column].strip():
                errors.append(f"{qid}: Empty {column}")

        # Unique question IDs
        if qid in question_ids:
            errors.append(f"Duplicate question ID: {qid}")

        question_ids.add(qid)

        # Duplicate question text
        normalized = " ".join(
            row["question"].casefold().split()
        )

        if normalized in question_texts:
            errors.append(f"{qid}: Duplicate question text")

        question_texts.add(normalized)

        # Allowed categorical values
        for field, allowed in ALLOWED_VALUES.items():
            if row[field] not in allowed:
                errors.append(
                    f"{qid}: Invalid {field}: {row[field]}"
                )

        # Evidence arrays
        try:
            chunk_ids = json.loads(row["relevant_chunk_ids"])
            document_ids = json.loads(row["source_document_ids"])

            if not all(
                isinstance(value, list)
                and all(isinstance(x, str) for x in value)
                for value in (chunk_ids, document_ids)
            ):
                raise ValueError("Expected JSON string arrays")

        except (ValueError, TypeError) as exc:
            errors.append(f"{qid}: Invalid evidence arrays: {exc}")
            continue

        if len(chunk_ids) != len(set(chunk_ids)):
            errors.append(f"{qid}: Duplicate evidence chunks")

        if len(document_ids) != len(set(document_ids)):
            errors.append(f"{qid}: Duplicate source documents")

        # Chunk existence
        for cid in chunk_ids:
            if cid not in chunk_map:
                errors.append(f"{qid}: Unknown chunk {cid}")

        # Chunk-document consistency
        actual_documents = {
            chunk_map[cid]
            for cid in chunk_ids
            if cid in chunk_map
        }

        if actual_documents != set(document_ids):
            errors.append(f"{qid}: Source document mismatch")

        # Answerability consistency
        if row["answerable"] == "true":
            if not chunk_ids or not document_ids:
                errors.append(f"{qid}: Missing supporting evidence")

            if row["question_type"] == "unanswerable":
                errors.append(f"{qid}: Incorrect question type")

        elif row["answerable"] == "false":
            if chunk_ids or document_ids:
                errors.append(f"{qid}: Unexpected supporting evidence")

            if row["question_type"] != "unanswerable":
                errors.append(f"{qid}: Incorrect question type")

        # Fact-group leakage
        fact_groups[row["fact_group_id"]].add(row["split"])

    for group, splits in fact_groups.items():
        if len(splits) > 1:
            errors.append(
                f"Fact-group leakage: {group} appears in {splits}"
            )

    # Dataset distribution
    distributions = {
        "language": {"en": 50, "ar": 20},
        "split": {"development": 28, "test": 42},
        "answerable": {"true": 63, "false": 7},
        "question_type": EXPECTED_TYPES,
    }

    if len(rows) != 70:
        errors.append(f"Expected 70 questions, found {len(rows)}")

    for field, expected in distributions.items():
        actual = Counter(row[field] for row in rows)

        if actual != expected:
            errors.append(
                f"{field} distribution mismatch: {dict(actual)}"
            )

    expected_split_languages = {
        ("development", "en"): 20,
        ("development", "ar"): 8,
        ("test", "en"): 30,
        ("test", "ar"): 12,
    }

    actual_split_languages = Counter(
        (row["split"], row["language"])
        for row in rows
    )

    if actual_split_languages != expected_split_languages:
        errors.append("Language distribution across splits mismatch")

    expected_split_answerability = {
        ("development", "true"): 25,
        ("development", "false"): 3,
        ("test", "true"): 38,
        ("test", "false"): 4,
    }

    actual_split_answerability = Counter(
        (row["split"], row["answerable"])
        for row in rows
    )

    if actual_split_answerability != expected_split_answerability:
        errors.append("Answerability distribution across splits mismatch")

    pending = sum(
        row["review_status"] == "pending"
        for row in rows
    )

    if pending:
        warnings.append(
            f"{pending} questions still require human verification."
        )

    # Report
    print("=" * 60)
    print("UNIGUIDE-AI EVALUATION DATASET VALIDATION")
    print("=" * 60)

    print(f"Total questions: {len(rows)}")
    print(f"Available corpus chunks: {len(chunk_map)}")
    print(f"Unique question IDs: {len(question_ids)}")
    print(f"Unique fact groups: {len(fact_groups)}")

    for field in distributions:
        print(f"{field}: {dict(Counter(r[field] for r in rows))}")

    print(f"\nErrors: {len(errors)}")
    for error in errors:
        print(f"  ERROR: {error}")

    print(f"\nWarnings: {len(warnings)}")
    for warning in warnings:
        print(f"  WARNING: {warning}")

    print("\n" + "=" * 60)

    if errors:
        print("VALIDATION FAILED")
    elif warnings:
        print("STRUCTURAL VALIDATION PASSED - REVIEW PENDING")
    else:
        print("STRUCTURAL VALIDATION PASSED")

    print("=" * 60)

    return not errors


if __name__ == "__main__":
    try:
        sys.exit(0 if validate_dataset() else 1)
    except (OSError, ValueError, csv.Error, KeyError) as exc:
        print(f"VALIDATION FAILED: {exc}")
        sys.exit(1)
