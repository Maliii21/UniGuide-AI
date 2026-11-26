from collections import Counter
import argparse
import hashlib
import json
import time

from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values
from google import genai
from google.genai import types


# ==================================================
# 1. Configuration
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]

EXPERIMENT_DIR = BASE_DIR / "evaluation" / "llm_experiments"

QUESTIONS_FILE = EXPERIMENT_DIR / "development_questions.json"
PROMPT_FILE = EXPERIMENT_DIR / "evaluation_prompt.txt"

CHUNKS_FILE = BASE_DIR / "data" / "processed" / "chunks.json"

CONTEXTS_FILE = EXPERIMENT_DIR / "evaluation_contexts.json"

RESULTS_FILE = EXPERIMENT_DIR / "comparison_results_v3.json"

BASELINE_RESULTS_FILE = (
    EXPERIMENT_DIR / "comparison_results_v2.json"
)

BASELINE_MODEL = "gemini-3.5-flash-lite"

REQUEST_TIMEOUT_MS = 120000
SDK_MAX_ATTEMPTS = 1

MODELS = {
    "gemini-3.5-flash-lite": 5,
    "gemini-3.1-flash-lite": 5
}

MAX_ATTEMPTS = 3
RETRY_DELAY = 10


# ==================================================
# 2. Utility functions
# ==================================================

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8-sig")
    )


def file_hash(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def experiment_fingerprint():

    config = {
        "questions_hash": file_hash(QUESTIONS_FILE),
        "prompt_hash": file_hash(PROMPT_FILE),
        "chunks_hash": file_hash(CHUNKS_FILE),
        "contexts_hash": file_hash(CONTEXTS_FILE),
        "script_hash": file_hash(Path(__file__))
    }

    return config


def save_results(data):

    temporary_file = RESULTS_FILE.with_suffix(".tmp")

    temporary_file.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    temporary_file.replace(RESULTS_FILE)


# ==================================================
# 3. Dataset and context preparation
# ==================================================


def load_experiment():

    questions = load_json(QUESTIONS_FILE)
    chunks = load_json(CHUNKS_FILE)
    contexts = load_json(CONTEXTS_FILE)

    template = PROMPT_FILE.read_text(
        encoding="utf-8-sig"
    )

    original_chunk_map = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    if len(questions) != 12:
        raise ValueError(
            "Expected 12 development questions."
        )

    if len(original_chunk_map) != len(chunks):
        raise ValueError(
            "Duplicate chunk IDs detected."
        )

    question_ids = [
        q["question_id"] for q in questions
    ]

    if len(set(question_ids)) != len(question_ids):
        raise ValueError(
            "Duplicate question IDs detected."
        )

    required_context_ids = {
        cid
        for question in questions
        for cid in question["context_chunk_ids"]
    }

    if set(contexts) != required_context_ids:
        raise ValueError(
            "Context IDs do not match development questions."
        )

    for cid, context in contexts.items():

        original = original_chunk_map[cid]

        if context["chunk_id"] != cid:
            raise ValueError(
                f"Invalid chunk ID: {cid}"
            )

        if not context["text"].strip():
            raise ValueError(
                f"Empty context: {cid}"
            )

        if context["text"] not in original["text"]:
            raise ValueError(
                f"Context differs from original source: {cid}"
            )

        if context["page"] != original["page"]:
            raise ValueError(
                f"Page mismatch: {cid}"
            )

        if context["document_title"] != original["document_title"]:
            raise ValueError(
                f"Document title mismatch: {cid}"
            )

    for question in questions:

        relevant_ids = set(
            question["relevant_chunk_ids"]
        )

        context_ids = set(
            question["context_chunk_ids"]
        )

        if not relevant_ids.issubset(context_ids):
            raise ValueError(
                f"Missing relevant context: "
                f"{question['question_id']}"
            )

    print("Minimized context validation: PASSED")

    return questions, contexts, template



def build_context(question, chunk_map):

    context_parts = []

    for chunk_id in question["context_chunk_ids"]:

        chunk = chunk_map[chunk_id]

        context_parts.append(
            f"Source ID: {chunk_id}\n"
            f"Document: {chunk['document_title']}\n"
            f"Page: {chunk['page']}\n"
            f"Content:\n{chunk['text']}"
        )

    return "\n\n---\n\n".join(context_parts)


def build_prompt(question, chunk_map, template):

    context = build_context(
        question,
        chunk_map
    )

    return template.format(
        context=context,
        question=question["question"]
    )


def import_baseline_results(questions, chunk_map, template):

    if not BASELINE_RESULTS_FILE.exists():
        raise FileNotFoundError(
            "V2 baseline results file not found."
        )

    baseline_data = load_json(BASELINE_RESULTS_FILE)

    current_fingerprint = experiment_fingerprint()

    # Verify identical experimental inputs.
    shared_keys = [
        "questions_hash",
        "prompt_hash",
        "chunks_hash",
        "contexts_hash"
    ]

    for key in shared_keys:

        if baseline_data["fingerprint"][key] != current_fingerprint[key]:
            raise ValueError(
                f"Baseline fingerprint mismatch: {key}"
            )

    # Verify the original V2 script.
    v2_script = BASE_DIR / "evaluation" / "evaluate_llm_models.py"

    if file_hash(v2_script) != baseline_data["fingerprint"]["script_hash"]:
        raise ValueError("Original V2 script has changed.")

    records = [
        record
        for record in baseline_data["results"]
        if record["model"] == BASELINE_MODEL
        and record["status"] == "success"
    ]

    question_map = {
        q["question_id"]: q for q in questions
    }

    counts = Counter(
        record["question_id"] for record in records
    )

    if len(records) != len(questions):
        raise ValueError("Incomplete baseline results.")

    if set(counts) != set(question_map):
        raise ValueError("Baseline question IDs mismatch.")

    if any(count != 1 for count in counts.values()):
        raise ValueError("Duplicate baseline results.")

    imported_records = []

    for record in records:

        question_id = record["question_id"]
        question = question_map[question_id]

        prompt = build_prompt(
            question,
            chunk_map,
            template
        )

        prompt_hash = hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest()

        if record["prompt_sha256"] != prompt_hash:
            raise ValueError(
                f"Baseline prompt mismatch: {question_id}"
            )

        if record["prompt_characters"] != len(prompt):
            raise ValueError(
                f"Baseline prompt length mismatch: {question_id}"
            )

        if not (record.get("answer") or "").strip():
            raise ValueError(
                f"Empty baseline answer: {question_id}"
            )

        metadata_fields = [
            "question",
            "question_type",
            "difficulty",
            "answerable",
            "expected_answer",
            "expected_language",
            "relevant_chunk_ids",
            "context_chunk_ids"
        ]

        for field in metadata_fields:

            if record[field] != question[field]:
                raise ValueError(
                    f"Baseline metadata mismatch: "
                    f"{question_id} | {field}"
                )

        imported_record = record.copy()

        imported_record["result_origin"] = "imported_v2"

        imported_records.append(imported_record)

    print("Baseline integrity validation: PASSED")
    print("Verified baseline responses:", len(imported_records))

    return imported_records

# ==================================================
# 4. Results initialization and resume support
# ==================================================


def initialize_results(questions, chunk_map, template):

    fingerprint = experiment_fingerprint()

    baseline_records = import_baseline_results(
        questions,
        chunk_map,
        template
    )

    baseline_source = {
        "file": BASELINE_RESULTS_FILE.name,
        "sha256": file_hash(BASELINE_RESULTS_FILE),
        "model": BASELINE_MODEL,
        "imported_count": len(baseline_records)
    }

    if RESULTS_FILE.exists():

        data = load_json(RESULTS_FILE)

        if data["fingerprint"] != fingerprint:
            raise ValueError(
                "Experiment files have changed. "
                "Archive previous results before "
                "starting a new experiment."
            )

        if data.get("baseline_source") != baseline_source:
            raise ValueError(
                "Baseline source has changed."
            )

        if data.get("models") != list(MODELS):
            raise ValueError("Experiment models mismatch.")

        saved_baseline = [
            record
            for record in data["results"]
            if record["model"] == BASELINE_MODEL
        ]

        if saved_baseline != baseline_records:
            raise ValueError(
                "Imported baseline records have changed."
            )

        print("Existing V3 experiment loaded.")

        return data

    print("Initializing new V3 experiment.")
    print("Imported baseline responses:", len(baseline_records))

    return {
        "experiment_name": "UniGuide-AI LLM Comparison V3",
        "evaluation_type": "fixed_context_development",
        "created_at": utc_now(),
        "fingerprint": fingerprint,
        "baseline_source": baseline_source,
        "models": list(MODELS),
        "results": baseline_records
    }

def already_completed(data, model_name, question_id):

    return any(
        result["model"] == model_name
        and result["question_id"] == question_id
        and result["status"] == "success"
        for result in data["results"]
    )

# ==================================================
# 5. API generation and retry handling
# ==================================================

def generate_answer(client, model_name, prompt):

    attempts = []

    total_start = time.perf_counter()

    for attempt in range(1, MAX_ATTEMPTS + 1):

        print(f"Attempt: {attempt}")

        start = time.perf_counter()

        try:

            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            latency = time.perf_counter() - start

            answer = response.text or ""

            usage = response.usage_metadata

            attempts.append({
                "attempt": attempt,
                "status": (
                    "success" if answer.strip()
                    else "empty_response"
                ),
                "latency_seconds": round(latency, 4)
            })

            return {
                "status": (
                    "success" if answer.strip()
                    else "empty_response"
                ),
                "answer": answer.strip(),
                "latency_seconds": round(latency, 4),
                "total_elapsed_seconds": round(
                    time.perf_counter() - total_start,
                    4
                ),
                "attempt_count": attempt,
                "retry_count": attempt - 1,
                "attempts": attempts,
                "input_tokens": getattr(
                    usage, "prompt_token_count", None
                ),
                "output_tokens": getattr(
                    usage, "candidates_token_count", None
                ),
                "thinking_tokens": getattr(
                    usage, "thoughts_token_count", None
                ),
                "total_tokens": getattr(
                    usage, "total_token_count", None
                ),
                "error_code": None,
                "error_type": None
            }

        except Exception as error:

            latency = time.perf_counter() - start

            code = getattr(error, "code", None)

            attempts.append({
                "attempt": attempt,
                "status": "error",
                "error_code": code,
                "error_type": type(error).__name__,
                "latency_seconds": round(latency, 4)
            })

            print("Error type:", type(error).__name__)
            print("Error code:", code)

            if code == 503 and attempt < MAX_ATTEMPTS:

                wait_time = RETRY_DELAY * (
                    2 ** (attempt - 1)
                )

                print(
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

                continue

            return {
                "status": (
                    "rate_limited" if code == 429
                    else "error"
                ),
                "answer": None,
                "latency_seconds": None,
                "total_elapsed_seconds": round(
                    time.perf_counter() - total_start,
                    4
                ),
                "attempt_count": attempt,
                "retry_count": attempt - 1,
                "attempts": attempts,
                "input_tokens": None,
                "output_tokens": None,
                "thinking_tokens": None,
                "total_tokens": None,
                "error_code": code,
                "error_type": type(error).__name__
            }


# ==================================================
# 6. Evaluation execution
# ==================================================

def run_evaluation(
    models,
    questions,
    chunk_map,
    template,
    limit=None,
    dry_run=False
):

    selected_questions = (
        questions[:limit]
        if limit is not None
        else questions
    )

    print("\nSelected models:", len(models))
    print("Selected questions:", len(selected_questions))

    
    data = initialize_results(
        questions,
        chunk_map,
        template
    )

    if dry_run:

        print("\nDRY RUN MODE")
        print("No API requests will be sent.")

        planned_requests = 0
        skipped_requests = 0

        for model_name in models:

            for question in selected_questions:

                question_id = question["question_id"]

                prompt = build_prompt(
                    question,
                    chunk_map,
                    template
                )

                completed = already_completed(
                    data,
                    model_name,
                    question_id
                )

                if completed:
                    skipped_requests += 1
                    status = "SKIPPED - EXISTING SUCCESS"

                else:
                    planned_requests += 1
                    status = "PENDING API REQUEST"

                print(
                    model_name,
                    "|", question_id,
                    "| Prompt characters:", len(prompt),
                    "|", status
                )

        print("\nExisting successful responses:", skipped_requests)
        print("Planned initial requests:", planned_requests)
        print("API requests sent: 0")
        print("Dry run: PASSED")

        return

    pending_requests = any(
        not already_completed(
            data,
            model_name,
            question["question_id"]
        )
        for model_name in models
        for question in selected_questions
    )

    if not pending_requests:

        if not RESULTS_FILE.exists():
            save_results(data)

        print("All selected responses already completed.")
        print("No API requests needed.")

        return

    if not RESULTS_FILE.exists():
        save_results(data)

    api_key = dotenv_values(
        BASE_DIR / ".env"
    ).get("GEMINI_API_KEY")

    if not api_key or not api_key.strip():
        raise ValueError("GEMINI_API_KEY is missing.")

    client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(
        timeout=REQUEST_TIMEOUT_MS,
        retry_options=types.HttpRetryOptions(
            attempts=SDK_MAX_ATTEMPTS
        )
    )
)

    try:

        for model_name in models:

            print("\n" + "=" * 60)
            print("MODEL:", model_name)
            print("=" * 60)

            for question in selected_questions:

                question_id = question["question_id"]

                if already_completed(
                    data,
                    model_name,
                    question_id
                ):

                    print(question_id, "| SKIPPED")
                    continue

                print("\nQuestion ID:", question_id)
                print("Question:", question["question"])

                prompt = build_prompt(
                    question,
                    chunk_map,
                    template
                )

                generation = generate_answer(
                    client,
                    model_name,
                    prompt
                )

                result = {
                    "model": model_name,
                    "result_origin": "generated_v3",
                    "question_id": question_id,
                    "question": question["question"],
                    "question_type": question["question_type"],
                    "difficulty": question["difficulty"],
                    "answerable": question["answerable"],
                    "expected_answer": question["expected_answer"],
                    "expected_language": question["expected_language"],
                    "relevant_chunk_ids": question["relevant_chunk_ids"],
                    "context_chunk_ids": question["context_chunk_ids"],
                    "prompt_characters": len(prompt),
                    "prompt_sha256": hashlib.sha256(
                        prompt.encode("utf-8")
                    ).hexdigest(),
                    "timestamp": utc_now(),
                    **generation,
                    "quality_scores": None
                }

                data["results"].append(result)

                save_results(data)

                print("Status:", result["status"])
                print("Answer:", result["answer"])
                print(
                    "Latency:",
                    result["latency_seconds"]
                )
                print(
                    "Total elapsed:",
                    result["total_elapsed_seconds"]
                )
                print(
                    "Total tokens:",
                    result["total_tokens"]
                )

                print("Result saved.")

                if result["status"] != "success":

                    print(
                        "\nEvaluation paused after "
                        "an unsuccessful response."
                    )

                    print(
                        "Successful results have been saved."
                    )

                    return

                time.sleep(MODELS[model_name])

    finally:
        client.close()

    print("\nEvaluation completed.")
    print("Results file:", RESULTS_FILE)


# ==================================================
# 7. Command-line interface
# ==================================================

def main():

    parser = argparse.ArgumentParser(
        description="Evaluate UniGuide-AI LLM candidates."
    )

    parser.add_argument(
        "--model",
        choices=["all", *MODELS.keys()],
        default="all"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None
    )

    parser.add_argument(
        "--dry-run",
        action="store_true"
    )

    args = parser.parse_args()

    if args.limit is not None:

        if not 1 <= args.limit <= 12:
            parser.error("--limit must be between 1 and 12.")

    models = (
        list(MODELS)
        if args.model == "all"
        else [args.model]
    )

    questions, chunk_map, template = load_experiment()

    run_evaluation(
        models=models,
        questions=questions,
        chunk_map=chunk_map,
        template=template,
        limit=args.limit,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
