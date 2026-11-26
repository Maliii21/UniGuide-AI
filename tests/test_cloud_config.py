
"""Tests for local and cloud Gemini configuration."""

import os
from unittest.mock import patch

import pytest

from src.llm import (
    load_config,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_MS,
)


@pytest.fixture(autouse=True)
def isolate_environment():
    """Prevent real credentials from affecting tests."""

    with patch.dict(os.environ, {}, clear=True):
        yield


def test_cloud_configuration_without_env_file(tmp_path):

    os.environ["GEMINI_API_KEY"] = "test-cloud-key"
    os.environ["GEMINI_MODEL"] = "test-cloud-model"
    os.environ["GEMINI_TIMEOUT_MS"] = "60000"

    config = load_config(tmp_path / "missing.env")

    assert config.api_key == "test-cloud-key"
    assert config.model == "test-cloud-model"
    assert config.timeout_ms == 60000


def test_local_env_configuration(tmp_path):

    env_file = tmp_path / ".env"

    env_file.write_text(
        "GEMINI_API_KEY=test-local-key\n"
        "GEMINI_MODEL=test-local-model\n"
        "GEMINI_TIMEOUT_MS=90000\n",
        encoding="utf-8"
    )

    config = load_config(env_file)

    assert config.api_key == "test-local-key"
    assert config.model == "test-local-model"
    assert config.timeout_ms == 90000


def test_environment_variables_override_env_file(tmp_path):

    env_file = tmp_path / ".env"

    env_file.write_text(
        "GEMINI_API_KEY=local-key\n"
        "GEMINI_MODEL=local-model\n"
        "GEMINI_TIMEOUT_MS=90000\n",
        encoding="utf-8"
    )

    os.environ["GEMINI_API_KEY"] = "cloud-key"
    os.environ["GEMINI_MODEL"] = "cloud-model"
    os.environ["GEMINI_TIMEOUT_MS"] = "60000"

    config = load_config(env_file)

    assert config.api_key == "cloud-key"
    assert config.model == "cloud-model"
    assert config.timeout_ms == 60000


def test_mixed_configuration_sources(tmp_path):

    env_file = tmp_path / ".env"

    env_file.write_text(
        "GEMINI_MODEL=local-model\n"
        "GEMINI_TIMEOUT_MS=90000\n",
        encoding="utf-8"
    )

    os.environ["GEMINI_API_KEY"] = "cloud-key"

    config = load_config(env_file)

    assert config.api_key == "cloud-key"
    assert config.model == "local-model"
    assert config.timeout_ms == 90000


def test_cloud_configuration_defaults(tmp_path):

    os.environ["GEMINI_API_KEY"] = "test-key"

    config = load_config(tmp_path / "missing.env")

    assert config.model == DEFAULT_MODEL
    assert config.timeout_ms == DEFAULT_TIMEOUT_MS


def test_missing_api_key(tmp_path):

    with pytest.raises(ValueError, match="API key"):

        load_config(tmp_path / "missing.env")


@pytest.mark.parametrize("key", ["", " ", "   "])
def test_empty_cloud_api_key(tmp_path, key):

    os.environ["GEMINI_API_KEY"] = key

    with pytest.raises(ValueError, match="API key"):

        load_config(tmp_path / "missing.env")


@pytest.mark.parametrize(
    "timeout",
    ["invalid", "0", "-100", "1.5", ""]
)
def test_invalid_cloud_timeout(tmp_path, timeout):

    os.environ["GEMINI_API_KEY"] = "test-key"
    os.environ["GEMINI_TIMEOUT_MS"] = timeout

    with pytest.raises(ValueError):

        load_config(tmp_path / "missing.env")


def test_empty_cloud_model(tmp_path):

    os.environ["GEMINI_API_KEY"] = "test-key"
    os.environ["GEMINI_MODEL"] = " "

    with pytest.raises(ValueError, match="model"):

        load_config(tmp_path / "missing.env")


def test_api_key_not_exposed_in_repr(tmp_path):

    os.environ["GEMINI_API_KEY"] = "private-test-key"

    config = load_config(tmp_path / "missing.env")

    assert "private-test-key" not in repr(config)
