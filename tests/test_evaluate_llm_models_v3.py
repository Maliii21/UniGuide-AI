
import copy
import hashlib
import json

import pytest

from unittest.mock import Mock

from evaluation import evaluate_llm_models_v3 as v3

# ==================================================
# 1. Test fixtures
# ==================================================

@pytest.fixture
def experiment(monkeypatch, tmp_path):

    questions, contexts, template = v3.load_experiment()

    original_baseline = v3.load_json(
        v3.BASELINE_RESULTS_FILE
    )

    baseline_path = tmp_path / "comparison_results_v2.json"
    results_path = tmp_path / "comparison_results_v3.json"

    baseline_path.write_text(
        json.dumps(
            original_baseline,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    monkeypatch.setattr(
        v3,
        "BASELINE_RESULTS_FILE",
        baseline_path
    )

    monkeypatch.setattr(
        v3,
        "RESULTS_FILE",
        results_path
    )

    # Prevent accidental API calls during testing.
    client_mock = Mock(
        side_effect=AssertionError(
            "Real API client must not be created."
        )
    )

    monkeypatch.setattr(
        v3.genai,
        "Client",
        client_mock
    )

    return {
        "questions": questions,
        "contexts": contexts,
        "template": template,
        "baseline": original_baseline,
        "baseline_path": baseline_path,
        "results_path": results_path,
        "client_mock": client_mock
    }

def get_args(experiment):

    return (
        experiment["questions"],
        experiment["contexts"],
        experiment["template"]
    )

def write_baseline(experiment, data):

    experiment["baseline_path"].write_text(
        json.dumps(data, ensure_ascii=False),
        encoding="utf-8"
    )

def baseline_records(data):

    return [
        record
        for record in data["results"]
        if record["model"] == v3.BASELINE_MODEL
        and record["status"] == "success"
    ]

# ==================================================
# 2. Baseline import tests
# ==================================================

def test_imports_all_baseline_responses(experiment):

    records = v3.import_baseline_results(
        *get_args(experiment)
    )

    assert len(records) == 12

    assert len({
        r["question_id"] for r in records
    }) == 12

    assert all(
        r["model"] == v3.BASELINE_MODEL
        for r in records
    )

    assert all(
        r["result_origin"] == "imported_v2"
        for r in records
    )

def test_preserves_original_response_data(experiment):

    imported = v3.import_baseline_results(
        *get_args(experiment)
    )

    original = baseline_records(
        experiment["baseline"]
    )

    for imported_record, original_record in zip(
        imported, original
    ):

        record = imported_record.copy()

        assert record.pop(
            "result_origin"
        ) == "imported_v2"

        assert record == original_record

def test_baseline_source_not_modified(experiment):

    before = v3.file_hash(
        experiment["baseline_path"]
    )

    v3.import_baseline_results(
        *get_args(experiment)
    )

    after = v3.file_hash(
        experiment["baseline_path"]
    )

    assert before == after

def test_missing_baseline_rejected(experiment):

    experiment["baseline_path"].unlink()

    with pytest.raises(FileNotFoundError):

        v3.import_baseline_results(
            *get_args(experiment)
        )

@pytest.mark.parametrize(
    "key",
    [
        "questions_hash",
        "prompt_hash",
        "chunks_hash",
        "contexts_hash",
        "script_hash"
    ]
)
def test_invalid_fingerprint_rejected(experiment, key):

    data = copy.deepcopy(experiment["baseline"])

    data["fingerprint"][key] = "invalid_hash"

    write_baseline(experiment, data)

    with pytest.raises(ValueError):

        v3.import_baseline_results(
            *get_args(experiment)
        )

def test_incomplete_baseline_rejected(experiment):

    data = copy.deepcopy(experiment["baseline"])

    record = baseline_records(data)[0]

    data["results"].remove(record)

    write_baseline(experiment, data)

    with pytest.raises(
        ValueError,
        match="Incomplete baseline"
    ):

        v3.import_baseline_results(
            *get_args(experiment)
        )

def test_duplicate_baseline_rejected(experiment):

    data = copy.deepcopy(experiment["baseline"])

    records = baseline_records(data)

    data["results"].remove(records[1])

    data["results"].append(
        copy.deepcopy(records[0])
    )

    write_baseline(experiment, data)

    with pytest.raises(ValueError):

        v3.import_baseline_results(
            *get_args(experiment)
        )

@pytest.mark.parametrize(
    "field, value",
    [
        ("prompt_sha256", "invalid_hash"),
        ("prompt_characters", -1),
        ("answer", ""),
        ("answer", "   "),
        ("expected_answer", "incorrect metadata"),
        ("expected_language", "invalid_language"),
        ("context_chunk_ids", []),
        ("question", "Modified question")
    ]
)
def test_invalid_baseline_record_rejected(
    experiment, field, value
):

    data = copy.deepcopy(experiment["baseline"])

    record = baseline_records(data)[0]

    record[field] = value

    write_baseline(experiment, data)

    with pytest.raises(ValueError):

        v3.import_baseline_results(
            *get_args(experiment)
        )

# ==================================================
# 3. Initialization and resume tests
# ==================================================

def test_initialize_new_experiment(experiment):

    data = v3.initialize_results(
        *get_args(experiment)
    )

    assert len(data["results"]) == 12

    assert data["models"] == list(v3.MODELS)

    assert data["baseline_source"][
        "imported_count"
    ] == 12

    assert data["baseline_source"][
        "sha256"
    ] == v3.file_hash(
        experiment["baseline_path"]
    )

    assert not experiment["results_path"].exists()

def test_save_and_resume(experiment):

    data = v3.initialize_results(
        *get_args(experiment)
    )

    v3.save_results(data)

    resumed = v3.initialize_results(
        *get_args(experiment)
    )

    assert resumed == data

    assert len(resumed["results"]) == 12

def test_modified_imported_answer_rejected(experiment):

    data = v3.initialize_results(
        *get_args(experiment)
    )

    data["results"][0]["answer"] = "Modified answer"

    v3.save_results(data)

    with pytest.raises(
        ValueError,
        match="Imported baseline records have changed"
    ):

        v3.initialize_results(
            *get_args(experiment)
        )

def test_modified_v3_fingerprint_rejected(experiment):

    data = v3.initialize_results(
        *get_args(experiment)
    )

    data["fingerprint"]["prompt_hash"] = "invalid"

    v3.save_results(data)

    with pytest.raises(
        ValueError,
        match="Experiment files have changed"
    ):

        v3.initialize_results(
            *get_args(experiment)
        )

def test_changed_baseline_source_rejected(experiment):

    data = v3.initialize_results(
        *get_args(experiment)
    )

    v3.save_results(data)

    # Change file bytes without changing JSON contents.
    path = experiment["baseline_path"]

    path.write_bytes(
        path.read_bytes() + b"\n"
    )

    with pytest.raises(
        ValueError,
        match="Baseline source has changed"
    ):

        v3.initialize_results(
            *get_args(experiment)
        )

def test_resume_skips_successful_response(experiment):

    data = v3.initialize_results(
        *get_args(experiment)
    )

    assert v3.already_completed(
        data,
        v3.BASELINE_MODEL,
        "DEV-01"
    )

def test_resume_does_not_skip_failed_response():

    data = {
        "results": [
            {
                "model": "gemini-3.1-flash-lite",
                "question_id": "DEV-01",
                "status": "error"
            }
        ]
    }

    assert not v3.already_completed(
        data,
        "gemini-3.1-flash-lite",
        "DEV-01"
    )

# ==================================================
# 4. Dry-run and API protection tests
# ==================================================

@pytest.mark.parametrize(
    "models, skipped, planned",
    [
        (list(v3.MODELS), 12, 12),
        (["gemini-3.5-flash-lite"], 12, 0),
        (["gemini-3.1-flash-lite"], 0, 12)
    ]
)
def test_dry_run_request_counts(
    experiment,
    capsys,
    models,
    skipped,
    planned
):

    v3.run_evaluation(
        models,
        *get_args(experiment),
        dry_run=True
    )

    output = capsys.readouterr().out

    assert (
        f"Existing successful responses: {skipped}"
        in output
    )

    assert (
        f"Planned initial requests: {planned}"
        in output
    )

    assert "API requests sent: 0" in output

    assert not experiment["results_path"].exists()

    experiment["client_mock"].assert_not_called()

def test_baseline_only_execution_needs_no_api(experiment):

    v3.run_evaluation(
        [v3.BASELINE_MODEL],
        *get_args(experiment)
    )

    assert experiment["results_path"].exists()

    data = v3.load_json(
        experiment["results_path"]
    )

    assert len(data["results"]) == 12

    experiment["client_mock"].assert_not_called()

def test_baseline_resume_preserves_results(experiment):

    v3.run_evaluation(
        [v3.BASELINE_MODEL],
        *get_args(experiment)
    )

    before = v3.file_hash(
        experiment["results_path"]
    )

    v3.run_evaluation(
        [v3.BASELINE_MODEL],
        *get_args(experiment)
    )

    after = v3.file_hash(
        experiment["results_path"]
    )

    assert before == after

    experiment["client_mock"].assert_not_called()

# ==================================================
# 5. Mocked candidate generation
# ==================================================

def test_candidate_generation_and_resume(
    experiment,
    monkeypatch
):

    candidate = "gemini-3.1-flash-lite"

    client = Mock()

    client_factory = Mock(return_value=client)

    monkeypatch.setattr(
        v3.genai,
        "Client",
        client_factory
    )

    monkeypatch.setattr(
        v3,
        "dotenv_values",
        lambda *args: {
            "GEMINI_API_KEY": "mock_test_key"
        }
    )

    monkeypatch.setattr(
        v3.time,
        "sleep",
        lambda seconds: None
    )

    generation = Mock(return_value={
        "status": "success",
        "answer": "Mocked candidate answer.",
        "latency_seconds": 1.0,
        "total_elapsed_seconds": 1.0,
        "attempt_count": 1,
        "retry_count": 0,
        "attempts": [],
        "input_tokens": 100,
        "output_tokens": 10,
        "thinking_tokens": None,
        "total_tokens": 110,
        "error_code": None,
        "error_type": None
    })

    monkeypatch.setattr(
        v3,
        "generate_answer",
        generation
    )

    v3.run_evaluation(
        list(v3.MODELS),
        *get_args(experiment),
        limit=1
    )

    generation.assert_called_once()

    assert generation.call_args.args[1] == candidate

    data = v3.load_json(
        experiment["results_path"]
    )

    assert len(data["results"]) == 13

    record = data["results"][-1]

    assert record["model"] == candidate

    assert record["result_origin"] == "generated_v3"

    assert record["question_id"] == "DEV-01"

    prompt = v3.build_prompt(
        experiment["questions"][0],
        experiment["contexts"],
        experiment["template"]
    )

    assert record["prompt_sha256"] == hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest()

    client.close.assert_called_once()

    # Resume: the successful candidate must be skipped.
    generation.reset_mock()
    client_factory.reset_mock()

    v3.run_evaluation(
        list(v3.MODELS),
        *get_args(experiment),
        limit=1
    )

    generation.assert_not_called()
    client_factory.assert_not_called()