
import json

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from evaluation import evaluate_llm_models as evaluator


# ==================================================
# Mock API helpers
# ==================================================

class FakeAPIError(Exception):

    def __init__(self, code):
        super().__init__(f"Mock API error: {code}")
        self.code = code


def fake_response(answer="Test answer [AIU-001-P001-C001]."):

    return SimpleNamespace(
        text=answer,
        usage_metadata=SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=20,
            thoughts_token_count=10,
            total_token_count=130
        )
    )


@pytest.fixture
def client():
    return Mock()


@pytest.fixture
def mock_sleep(monkeypatch):

    sleep = Mock()

    monkeypatch.setattr(
        evaluator.time,
        "sleep",
        sleep
    )

    return sleep


# ==================================================
# Generation tests
# ==================================================

def test_successful_generation(client, mock_sleep):

    client.models.generate_content.return_value = (
        fake_response()
    )

    result = evaluator.generate_answer(
        client,
        "test-model",
        "Test prompt"
    )

    assert result["status"] == "success"
    assert result["attempt_count"] == 1
    assert result["retry_count"] == 0
    assert result["input_tokens"] == 100
    assert result["output_tokens"] == 20
    assert result["thinking_tokens"] == 10
    assert result["total_tokens"] == 130

    assert result["latency_seconds"] >= 0
    assert result["total_elapsed_seconds"] >= 0

    client.models.generate_content.assert_called_once_with(
        model="test-model",
        contents="Test prompt"
    )

    mock_sleep.assert_not_called()


def test_503_then_success(client, mock_sleep):

    client.models.generate_content.side_effect = [
        FakeAPIError(503),
        fake_response()
    ]

    result = evaluator.generate_answer(
        client,
        "test-model",
        "Test prompt"
    )

    assert result["status"] == "success"
    assert result["attempt_count"] == 2
    assert result["retry_count"] == 1

    assert result["attempts"][0]["error_code"] == 503
    assert result["attempts"][1]["status"] == "success"

    assert client.models.generate_content.call_count == 2

    mock_sleep.assert_called_once_with(10)


def test_two_503_errors_then_success(client, mock_sleep):

    client.models.generate_content.side_effect = [
        FakeAPIError(503),
        FakeAPIError(503),
        fake_response()
    ]

    result = evaluator.generate_answer(
        client,
        "test-model",
        "Test prompt"
    )

    assert result["status"] == "success"
    assert result["attempt_count"] == 3
    assert result["retry_count"] == 2

    assert client.models.generate_content.call_count == 3

    delays = [
        call.args[0]
        for call in mock_sleep.call_args_list
    ]

    assert delays == [10, 20]


def test_persistent_503(client, mock_sleep):

    client.models.generate_content.side_effect = (
        FakeAPIError(503)
    )

    result = evaluator.generate_answer(
        client,
        "test-model",
        "Test prompt"
    )

    assert result["status"] == "error"
    assert result["error_code"] == 503
    assert result["attempt_count"] == 3
    assert result["retry_count"] == 2
    assert result["answer"] is None

    assert client.models.generate_content.call_count == 3
    assert mock_sleep.call_count == 2


@pytest.mark.parametrize(
    "error_code, expected_status",
    [
        (429, "rate_limited"),
        (400, "error"),
        (401, "error"),
        (403, "error"),
        (500, "error")
    ]
)
def test_non_retryable_errors(
    client,
    mock_sleep,
    error_code,
    expected_status
):

    client.models.generate_content.side_effect = (
        FakeAPIError(error_code)
    )

    result = evaluator.generate_answer(
        client,
        "test-model",
        "Test prompt"
    )

    assert result["status"] == expected_status
    assert result["error_code"] == error_code
    assert result["attempt_count"] == 1
    assert result["retry_count"] == 0

    assert client.models.generate_content.call_count == 1

    mock_sleep.assert_not_called()


@pytest.mark.parametrize("answer", ["", "   ", None])
def test_empty_response(client, mock_sleep, answer):

    client.models.generate_content.return_value = (
        fake_response(answer)
    )

    result = evaluator.generate_answer(
        client,
        "test-model",
        "Test prompt"
    )

    assert result["status"] == "empty_response"
    assert result["attempt_count"] == 1
    assert result["retry_count"] == 0

    mock_sleep.assert_not_called()


# ==================================================
# SDK configuration tests
# ==================================================

def test_sdk_configuration(monkeypatch, client):

    client_factory = Mock(return_value=client)

    monkeypatch.setattr(
        evaluator.genai,
        "Client",
        client_factory
    )

    monkeypatch.setattr(
        evaluator,
        "dotenv_values",
        lambda *args, **kwargs: {
            "GEMINI_API_KEY": "fake-test-key"
        }
    )

    monkeypatch.setattr(
        evaluator,
        "initialize_results",
        lambda: {"results": []}
    )

    evaluator.run_evaluation(
        models=["test-model"],
        questions=[],
        chunk_map={},
        template=""
    )

    client_factory.assert_called_once()

    options = client_factory.call_args.kwargs[
        "http_options"
    ]

    assert options.timeout == 120000
    assert options.retry_options.attempts == 1

    client.close.assert_called_once()

    client.models.generate_content.assert_not_called()


# ==================================================
# Result saving and resume tests
# ==================================================

def test_save_results(tmp_path, monkeypatch):

    results_file = tmp_path / "results.json"

    monkeypatch.setattr(
        evaluator,
        "RESULTS_FILE",
        results_file
    )

    data = {
        "results": [
            {
                "model": "test-model",
                "question_id": "DEV-01",
                "status": "success",
                "answer": "إجابة تجريبية"
            }
        ]
    }

    evaluator.save_results(data)

    assert results_file.exists()

    saved = json.loads(
        results_file.read_text(encoding="utf-8")
    )

    assert saved == data

    assert not results_file.with_suffix(".tmp").exists()


def test_resume_skips_successful_results():

    data = {
        "results": [
            {
                "model": "test-model",
                "question_id": "DEV-01",
                "status": "success"
            }
        ]
    }

    assert evaluator.already_completed(
        data, "test-model", "DEV-01"
    )

    assert not evaluator.already_completed(
        data, "test-model", "DEV-02"
    )

    assert not evaluator.already_completed(
        data, "another-model", "DEV-01"
    )


def test_resume_does_not_skip_failed_results():

    data = {
        "results": [
            {
                "model": "test-model",
                "question_id": "DEV-01",
                "status": "error"
            }
        ]
    }

    assert not evaluator.already_completed(
        data, "test-model", "DEV-01"
    )


def test_fingerprint_mismatch(tmp_path, monkeypatch):

    results_file = tmp_path / "results.json"

    results_file.write_text(
        json.dumps({
            "fingerprint": {"script_hash": "old"},
            "results": []
        }),
        encoding="utf-8"
    )

    monkeypatch.setattr(
        evaluator,
        "RESULTS_FILE",
        results_file
    )

    monkeypatch.setattr(
        evaluator,
        "experiment_fingerprint",
        lambda: {"script_hash": "new"}
    )

    with pytest.raises(
        ValueError,
        match="Experiment files have changed"
    ):
        evaluator.initialize_results()


def test_matching_fingerprint(tmp_path, monkeypatch):

    results_file = tmp_path / "results.json"

    fingerprint = {
        "script_hash": "same",
        "contexts_hash": "same"
    }

    data = {
        "fingerprint": fingerprint,
        "results": []
    }

    results_file.write_text(
        json.dumps(data),
        encoding="utf-8"
    )

    monkeypatch.setattr(
        evaluator,
        "RESULTS_FILE",
        results_file
    )

    monkeypatch.setattr(
        evaluator,
        "experiment_fingerprint",
        lambda: fingerprint
    )

    assert evaluator.initialize_results() == data
