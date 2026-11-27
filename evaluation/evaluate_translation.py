
"""Evaluate the effect of Arabic-to-English query translation."""

import argparse
import json
from pathlib import Path

from src.llm import GeminiLLM
from src.query_translator import QueryTranslator

from evaluation.evaluate_embeddings_final import (
    BASE_DIR,
    CHUNKS_FILE,
    QUESTIONS_FILE,
    MODELS,
    load_experiment,
    load_questions,
    evaluate_model,
    summarize_group,
)


RESULTS_DIR = (
    BASE_DIR / "evaluation" / "final_results"
)

BASELINE_FILE = (
    RESULTS_DIR / "embedding_development"
    / "embedding_results.json"
)

OUTPUT_DIR = RESULTS_DIR / "translation_development"
TRANSLATIONS_FILE = OUTPUT_DIR / "translations.json"


def translate_questions(questions):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if TRANSLATIONS_FILE.exists():
        translations = json.loads(
            TRANSLATIONS_FILE.read_text(encoding="utf-8")
        )
    else:
        translations = {}

    pending = [
        q for q in questions
        if q["question_id"] not in translations
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
                    "total_tokens": (
                        result.llm_response.total_tokens
                    ),
                }

                TRANSLATIONS_FILE.write_text(
                    json.dumps(
                        translations,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                print(f"\n{qid}")
                print("Arabic:", question["question"])
                print("English:", result.retrieval_query)

        finally:
            llm.close()

    translated_questions = []

    for question in questions:

        translation = translations[
            question["question_id"]
        ]

        if (
            translation["original_question"]
            != question["question"]
        ):
            raise ValueError("Cached translation mismatch.")

        translated_questions.append({
            **question,
            "question": translation["translated_question"],
            "original_question": question["question"],
            "translation_seconds": (
                translation["translation_seconds"]
            ),
        })

    return translated_questions, translations


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--cache-dir",
        default=None,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--local-files-only",
        action="store_true",
    )

    args = parser.parse_args()

    output_file = OUTPUT_DIR / "translation_results.json"

    if output_file.exists():
        raise FileExistsError(
            "Translation evaluation already completed."
        )

    baseline = json.loads(
        BASELINE_FILE.read_text(encoding="utf-8")
    )

    chunks, corpus_hash, generation_summaries = (
        load_experiment(
            args.experiment_dir,
            CHUNKS_FILE,
        )
    )

    if baseline["corpus_sha256"] != corpus_hash:
        raise ValueError("Baseline corpus mismatch.")

    questions = load_questions(
        QUESTIONS_FILE,
        {c["chunk_id"] for c in chunks},
    )

    arabic_questions = [
        q for q in questions
        if q["language"] == "ar" and q["answerable"]
    ]

    if len(arabic_questions) != 7:
        raise ValueError(
            "Expected 7 answerable Arabic questions."
        )

    translated_questions, translations = (
        translate_questions(arabic_questions)
    )

    comparisons = []
    all_results = []

    for config in MODELS:

        alias = config["alias"]

        _, translated_results = evaluate_model(
            config=config,
            chunks=chunks,
            questions=translated_questions,
            experiment_dir=args.experiment_dir,
            generation_summary=generation_summaries[alias],
            device=args.device,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )

        original_results = [
            r for r in baseline["results"]
            if r["alias"] == alias
            and r["language"] == "ar"
            and r["answerable"]
        ]

        if {
            r["question_id"] for r in original_results
        } != {
            r["question_id"] for r in translated_results
        }:
            raise ValueError("Question alignment mismatch.")

        english_results = [
            r for r in baseline["results"]
            if r["alias"] == alias
            and r["language"] == "en"
            and r["answerable"]
        ]

        before = summarize_group(original_results)
        after = summarize_group(translated_results)

        combined = summarize_group(
            english_results + translated_results
        )

        comparisons.append({
            "model": config["model_name"],
            "alias": alias,
            "arabic_before_translation": before,
            "arabic_after_translation": after,
            "combined_answerable_performance": combined,
            "recall_at_5_improvement": (
                after["metrics"]["recall_at_5"]
                - before["metrics"]["recall_at_5"]
            ),
        })

        all_results.extend(translated_results)

    if len(all_results) != 21:
        raise ValueError("Expected 21 evaluations.")

    report = {
        "evaluation_type": "translation_ablation",
        "split": "development",
        "corpus_sha256": corpus_hash,
        "arabic_questions": len(arabic_questions),
        "total_evaluations": len(all_results),
        "translations": translations,
        "comparisons": comparisons,
        "results": all_results,
    }

    with output_file.open("x", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print("\n" + "=" * 75)
    print("ARABIC TRANSLATION COMPARISON")
    print("=" * 75)

    print(
        f"{'Model':<20}"
        f"{'Before R@5':>15}"
        f"{'After R@5':>15}"
        f"{'After MRR@10':>15}"
        f"{'Combined R@5':>15}"
    )

    for row in comparisons:

        before = row["arabic_before_translation"]["metrics"]
        after = row["arabic_after_translation"]["metrics"]
        combined = row["combined_answerable_performance"]["metrics"]

        print(
            f"{row['alias']:<20}"
            f"{before['recall_at_5']:>15.2%}"
            f"{after['recall_at_5']:>15.2%}"
            f"{after['mrr_at_10']:>15.4f}"
            f"{combined['recall_at_5']:>15.2%}"
        )

    print("\nCompleted evaluations:", len(all_results))
    print("Results saved:", output_file)


if __name__ == "__main__":
    main()
