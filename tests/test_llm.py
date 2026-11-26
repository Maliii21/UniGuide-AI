
"""
Unit tests for the UniGuide-AI production LLM module.

All Gemini API interactions are mocked.
No real API requests are performed.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.llm import (
    LLMConfig,
    LLMResponse,
    GeminiLLM,
    load_config,
    build_prompt,
    classify_error,
    SYSTEM_INSTRUCTION,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_MS
)


@pytest.fixture
def config():

    return LLMConfig(
        api_key="test-api-key",
        model="gemini-3.1-flash-lite",
        timeout_ms=120000
    )


@pytest.fixture
def mock_client():

    return Mock()


@pytest.fixture
def llm(config, mock_client):

    return GeminiLLM(
        config=config,
        client=mock_client
    )


def make_response(text="Test answer."):

    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=20,
            thoughts_token_count=5,
            total_token_count=125
        )
    )


class FakeAPIError(Exception):

    def __init__(self, code):

        super().__init__("Simulated API error")
        self.code = code


# ==========================================
# Configuration tests
# ==========================================

class TestLLMConfig:

    def test_valid_configuration(self, config):

        assert config.api_key == "test-api-key"
        assert config.model == "gemini-3.1-flash-lite"
        assert config.timeout_ms == 120000

    def test_default_configuration(self):

        config = LLMConfig(api_key="test-key")

        assert config.model == DEFAULT_MODEL
        assert config.timeout_ms == DEFAULT_TIMEOUT_MS

    @pytest.mark.parametrize("api_key", ["", " "])
    def test_invalid_api_key(self, api_key):

        with pytest.raises(ValueError):

            LLMConfig(api_key=api_key)

    @pytest.mark.parametrize("model", ["", " "])
    def test_invalid_model(self, model):

        with pytest.raises(ValueError):

            LLMConfig(
                api_key="test-key",
                model=model
            )

    @pytest.mark.parametrize("timeout", [0, -1, 1.5, True])
    def test_invalid_timeout(self, timeout):

        with pytest.raises(ValueError):

            LLMConfig(
                api_key="test-key",
                timeout_ms=timeout
            )

    def test_api_key_hidden_from_repr(self, config):

        assert config.api_key not in repr(config)

    def test_load_config_from_env_file(self, tmp_path):

        env_file = tmp_path / ".env"

        env_file.write_text(
            "GEMINI_API_KEY=test-key\n"
            "GEMINI_MODEL=gemini-test-model\n"
            "GEMINI_TIMEOUT_MS=60000\n",
            encoding="utf-8"
        )

        config = load_config(env_file)

        assert config.api_key == "test-key"
        assert config.model == "gemini-test-model"
        assert config.timeout_ms == 60000

    def test_load_config_with_defaults(self, tmp_path):

        env_file = tmp_path / ".env"

        env_file.write_text(
            "GEMINI_API_KEY=test-key\n",
            encoding="utf-8"
        )

        config = load_config(env_file)

        assert config.model == DEFAULT_MODEL
        assert config.timeout_ms == DEFAULT_TIMEOUT_MS

    def test_missing_api_key(self, tmp_path):

        env_file = tmp_path / ".env"

        env_file.write_text(
            "GEMINI_MODEL=test-model\n",
            encoding="utf-8"
        )

        with pytest.raises(ValueError):

            load_config(env_file)

    def test_invalid_timeout_in_env(self, tmp_path):

        env_file = tmp_path / ".env"

        env_file.write_text(
            "GEMINI_API_KEY=test-key\n"
            "GEMINI_TIMEOUT_MS=invalid\n",
            encoding="utf-8"
        )

        with pytest.raises(ValueError):

            load_config(env_file)


# ==========================================
# Prompt construction tests
# ==========================================

class TestPromptConstruction:

    def test_prompt_contains_question_and_context(self):

        question = "Is summer registration mandatory?"
        context = "Summer registration is optional."

        prompt = build_prompt(question, context)

        assert question in prompt
        assert context in prompt

        assert "UNIVERSITY POLICY CONTEXT:" in prompt
        assert "STUDENT QUESTION:" in prompt
        assert "ANSWER:" in prompt

    @pytest.mark.parametrize("question", ["", " ", "\n"])
    def test_empty_question(self, question):

        with pytest.raises(ValueError):

            build_prompt(
                question,
                "Valid context"
            )

    @pytest.mark.parametrize("context", ["", " ", "\n"])
    def test_empty_context(self, context):

        with pytest.raises(ValueError):

            build_prompt(
                "Valid question",
                context
            )

    def test_abstention_instruction_exists(self):

        assert "INSUFFICIENT_CONTEXT" in SYSTEM_INSTRUCTION

    def test_citation_instruction_exists(self):

        assert "[AIU-001-P001-C001]" in SYSTEM_INSTRUCTION


# ==========================================
# Generation tests
# ==========================================

class TestLLMGeneration:

    def test_successful_generation(self, llm, mock_client):

        mock_client.models.generate_content.return_value = (
            make_response(
                "Summer registration is optional. "
                "[AIU-003-P002-C002]"
            )
        )

        result = llm.generate_answer(
            question="Is summer registration mandatory?",
            context="Summer registration is optional."
        )

        assert isinstance(result, LLMResponse)
        assert result.success
        assert result.status == "success"

        assert result.answer == (
            "Summer registration is optional. "
            "[AIU-003-P002-C002]"
        )

        assert result.model == "gemini-3.1-flash-lite"
        assert result.attempt_count == 1
        assert result.retry_count == 0
        assert result.error_message is None

        mock_client.models.generate_content.assert_called_once()

    def test_token_usage(self, llm, mock_client):

        mock_client.models.generate_content.return_value = (
            make_response()
        )

        result = llm.generate_answer(
            "Question",
            "Context"
        )

        assert result.input_tokens == 100
        assert result.output_tokens == 20
        assert result.thinking_tokens == 5
        assert result.total_tokens == 125

    def test_latency_recorded(self, llm, mock_client):

        mock_client.models.generate_content.return_value = (
            make_response()
        )

        result = llm.generate_answer(
            "Question",
            "Context"
        )

        assert result.latency_seconds is not None
        assert result.latency_seconds >= 0
        assert result.total_elapsed_seconds >= 0

    def test_correct_model_and_prompt_sent(
        self, llm, mock_client
    ):

        mock_client.models.generate_content.return_value = (
            make_response()
        )

        llm.generate_answer(
            "Test question",
            "Test context"
        )

        kwargs = (
            mock_client.models.generate_content
            .call_args.kwargs
        )

        assert kwargs["model"] == "gemini-3.1-flash-lite"

        assert "Test question" in kwargs["contents"]
        assert "Test context" in kwargs["contents"]

        assert (
            kwargs["config"].system_instruction
            == SYSTEM_INSTRUCTION
        )

    @pytest.mark.parametrize("text", ["", " ", None])
    def test_empty_model_response(
        self, text, llm, mock_client
    ):

        mock_client.models.generate_content.return_value = (
            make_response(text)
        )

        result = llm.generate_answer(
            "Question",
            "Context"
        )

        assert not result.success
        assert result.status == "empty_response"
        assert result.answer is None

    def test_missing_usage_metadata(
        self, llm, mock_client
    ):

        mock_client.models.generate_content.return_value = (
            SimpleNamespace(
                text="Valid answer.",
                usage_metadata=None
            )
        )

        result = llm.generate_answer(
            "Question",
            "Context"
        )

        assert result.success
        assert result.input_tokens is None
        assert result.output_tokens is None
        assert result.total_tokens is None

    def test_abstention_response(
        self, llm, mock_client
    ):

        mock_client.models.generate_content.return_value = (
            make_response("INSUFFICIENT_CONTEXT")
        )

        result = llm.generate_answer(
            "Question",
            "Context"
        )

        assert result.success
        assert result.answer == "INSUFFICIENT_CONTEXT"


# ==========================================
# API error and retry tests
# ==========================================

class TestLLMErrorHandling:

    @pytest.mark.parametrize(
        "code",
        [401, 403, 404, 429, 504]
    )
    def test_non_retryable_errors(
        self, code, llm, mock_client
    ):

        mock_client.models.generate_content.side_effect = (
            FakeAPIError(code)
        )

        result = llm.generate_answer(
            "Question",
            "Context"
        )

        assert not result.success
        assert result.answer is None
        assert result.error_code == code
        assert result.error_message
        assert result.attempt_count == 1
        assert result.retry_count == 0

        mock_client.models.generate_content.assert_called_once()

    def test_retry_after_service_unavailable(
        self, llm, mock_client
    ):

        mock_client.models.generate_content.side_effect = [
            FakeAPIError(503),
            make_response("Recovered answer.")
        ]

        with patch("src.llm.time.sleep") as mock_sleep:

            result = llm.generate_answer(
                "Question",
                "Context"
            )

        assert result.success
        assert result.answer == "Recovered answer."
        assert result.attempt_count == 2
        assert result.retry_count == 1

        assert (
            mock_client.models.generate_content.call_count
            == 2
        )

        mock_sleep.assert_called_once_with(10)

    def test_retry_limit_exhausted(
        self, llm, mock_client
    ):

        mock_client.models.generate_content.side_effect = (
            FakeAPIError(503)
        )

        with patch("src.llm.time.sleep") as mock_sleep:

            result = llm.generate_answer(
                "Question",
                "Context"
            )

        assert not result.success
        assert result.error_code == 503
        assert result.attempt_count == 3
        assert result.retry_count == 2

        assert (
            mock_client.models.generate_content.call_count
            == 3
        )

        assert mock_sleep.call_count == 2

        delays = [
            call.args[0]
            for call in mock_sleep.call_args_list
        ]

        assert delays == [10, 20]

    def test_timeout_error(self, llm, mock_client):

        mock_client.models.generate_content.side_effect = (
            TimeoutError("Simulated timeout")
        )

        result = llm.generate_answer(
            "Question",
            "Context"
        )

        assert not result.success
        assert result.attempt_count == 1
        assert result.error_message

    def test_unexpected_error(self, llm, mock_client):

        mock_client.models.generate_content.side_effect = (
            RuntimeError("Internal error details")
        )

        result = llm.generate_answer(
            "Question",
            "Context"
        )

        assert not result.success
        assert result.answer is None
        assert result.error_message

        assert (
            "Internal error details"
            not in result.error_message
        )


# ==========================================
# Client lifecycle tests
# ==========================================

class TestClientLifecycle:

    def test_injected_client_not_closed(
        self, llm, mock_client
    ):

        llm.close()

        mock_client.close.assert_not_called()

    def test_owned_client_closed_once(self, config):

        with patch("src.llm.genai.Client") as factory:

            client = factory.return_value

            llm = GeminiLLM(config=config)

            llm.close()
            llm.close()

            client.close.assert_called_once()

    def test_generation_after_close(
        self, llm, mock_client
    ):

        llm.close()

        with pytest.raises(RuntimeError):

            llm.generate_answer(
                "Question",
                "Context"
            )

        mock_client.models.generate_content.assert_not_called()

    def test_context_manager(self, config):

        with patch("src.llm.genai.Client") as factory:

            client = factory.return_value

            with GeminiLLM(config=config) as llm:

                assert isinstance(llm, GeminiLLM)

            client.close.assert_called_once()


# ==========================================
# Query translation API tests
# ==========================================

class TestLLMTranslation:

    ARABIC_QUESTION = "هل التسجيل الصيفي إجباري؟"

    ENGLISH_TRANSLATION = (
        "Is summer registration mandatory?"
    )

    def test_successful_translation(
        self, llm, mock_client
    ):

        mock_client.models.generate_content.return_value = (
            make_response(self.ENGLISH_TRANSLATION)
        )

        result = llm.translate_query(
            self.ARABIC_QUESTION
        )

        assert isinstance(result, LLMResponse)
        assert result.success
        assert result.status == "success"
        assert result.answer == self.ENGLISH_TRANSLATION
        assert result.model == "gemini-3.1-flash-lite"
        assert result.attempt_count == 1
        assert result.retry_count == 0
        assert result.error_message is None

        mock_client.models.generate_content.assert_called_once()

    def test_correct_translation_prompt(
        self, llm, mock_client
    ):

        from src.query_translator import (
            TRANSLATION_SYSTEM_INSTRUCTION
        )

        mock_client.models.generate_content.return_value = (
            make_response(self.ENGLISH_TRANSLATION)
        )

        llm.translate_query(self.ARABIC_QUESTION)

        kwargs = (
            mock_client.models.generate_content
            .call_args.kwargs
        )

        assert kwargs["model"] == "gemini-3.1-flash-lite"

        assert kwargs["contents"] == (
            "STUDENT QUESTION:\n"
            f"{self.ARABIC_QUESTION}"
        )

        assert (
            kwargs["config"].system_instruction
            == TRANSLATION_SYSTEM_INSTRUCTION
        )

        assert (
            kwargs["config"].system_instruction
            != SYSTEM_INSTRUCTION
        )

    def test_translation_token_usage(
        self, llm, mock_client
    ):

        mock_client.models.generate_content.return_value = (
            make_response(self.ENGLISH_TRANSLATION)
        )

        result = llm.translate_query(
            self.ARABIC_QUESTION
        )

        assert result.input_tokens == 100
        assert result.output_tokens == 20
        assert result.thinking_tokens == 5
        assert result.total_tokens == 125

    def test_translation_latency(
        self, llm, mock_client
    ):

        mock_client.models.generate_content.return_value = (
            make_response(self.ENGLISH_TRANSLATION)
        )

        result = llm.translate_query(
            self.ARABIC_QUESTION
        )

        assert result.latency_seconds is not None
        assert result.latency_seconds >= 0
        assert result.total_elapsed_seconds >= 0

    @pytest.mark.parametrize(
        "question",
        [None, "", " ", "\n", 123]
    )
    def test_invalid_translation_question(
        self, llm, mock_client, question
    ):

        with pytest.raises(ValueError):

            llm.translate_query(question)

        mock_client.models.generate_content.assert_not_called()

    @pytest.mark.parametrize(
        "text",
        ["", " ", None]
    )
    def test_empty_translation_response(
        self, llm, mock_client, text
    ):

        mock_client.models.generate_content.return_value = (
            make_response(text)
        )

        result = llm.translate_query(
            self.ARABIC_QUESTION
        )

        assert not result.success
        assert result.status == "empty_response"
        assert result.answer is None

    @pytest.mark.parametrize(
        "code",
        [401, 403, 404, 429, 504]
    )
    def test_translation_non_retryable_errors(
        self, llm, mock_client, code
    ):

        mock_client.models.generate_content.side_effect = (
            FakeAPIError(code)
        )

        result = llm.translate_query(
            self.ARABIC_QUESTION
        )

        assert not result.success
        assert result.answer is None
        assert result.error_code == code
        assert result.attempt_count == 1
        assert result.retry_count == 0
        assert result.error_message

        mock_client.models.generate_content.assert_called_once()

    def test_translation_retry_success(
        self, llm, mock_client
    ):

        mock_client.models.generate_content.side_effect = [
            FakeAPIError(503),
            make_response(self.ENGLISH_TRANSLATION)
        ]

        with patch("src.llm.time.sleep") as mock_sleep:

            result = llm.translate_query(
                self.ARABIC_QUESTION
            )

        assert result.success
        assert result.answer == self.ENGLISH_TRANSLATION
        assert result.attempt_count == 2
        assert result.retry_count == 1

        assert (
            mock_client.models.generate_content.call_count
            == 2
        )

        mock_sleep.assert_called_once_with(10)

    def test_translation_retry_exhausted(
        self, llm, mock_client
    ):

        mock_client.models.generate_content.side_effect = (
            FakeAPIError(503)
        )

        with patch("src.llm.time.sleep") as mock_sleep:

            result = llm.translate_query(
                self.ARABIC_QUESTION
            )

        assert not result.success
        assert result.error_code == 503
        assert result.attempt_count == 3
        assert result.retry_count == 2

        assert (
            mock_client.models.generate_content.call_count
            == 3
        )

        delays = [
            call.args[0]
            for call in mock_sleep.call_args_list
        ]

        assert delays == [10, 20]

    def test_translation_after_close(
        self, llm, mock_client
    ):

        llm.close()

        with pytest.raises(RuntimeError):

            llm.translate_query(
                self.ARABIC_QUESTION
            )

        mock_client.models.generate_content.assert_not_called()
