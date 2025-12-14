"""Unit tests for inference modules."""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import asdict

from largeforge.inference.base import (
    GenerationRequest, GenerationResponse, StreamingResponse,
    InferenceBackend,
)
from largeforge.inference.generator import TextGenerator


class TestGenerationRequest:
    """Tests for GenerationRequest."""

    def test_default_values(self):
        """Test default request values."""
        request = GenerationRequest(prompt="Hello")

        assert request.prompt == "Hello"
        assert request.max_tokens == 256
        assert request.temperature == 0.7
        assert request.stream is False

    def test_custom_values(self):
        """Test custom request values."""
        request = GenerationRequest(
            prompt="Hello",
            max_tokens=100,
            temperature=0.5,
            stop=["END"],
        )

        assert request.max_tokens == 100
        assert request.temperature == 0.5
        assert request.stop == ["END"]

    def test_from_config(self):
        """Test creating request from config."""
        from largeforge.config import GenerationConfig

        config = GenerationConfig(max_tokens=512, temperature=0.9)
        request = GenerationRequest.from_config("Test prompt", config)

        assert request.prompt == "Test prompt"
        assert request.max_tokens == 512
        assert request.temperature == 0.9


class TestGenerationResponse:
    """Tests for GenerationResponse."""

    def test_default_values(self):
        """Test default response values."""
        response = GenerationResponse(text="Hello!", prompt="Hi")

        assert response.text == "Hello!"
        assert response.prompt == "Hi"
        assert response.finish_reason == "stop"

    def test_usage_properties(self):
        """Test usage property accessors."""
        response = GenerationResponse(
            text="Hello!",
            prompt="Hi",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        )

        assert response.prompt_tokens == 10
        assert response.completion_tokens == 20
        assert response.total_tokens == 30

    def test_empty_usage(self):
        """Test usage with empty dict."""
        response = GenerationResponse(text="test", prompt="test")

        assert response.prompt_tokens == 0
        assert response.completion_tokens == 0


class TestStreamingResponse:
    """Tests for StreamingResponse."""

    def test_default_values(self):
        """Test default streaming response values."""
        response = StreamingResponse(delta="Hi")

        assert response.delta == "Hi"
        assert response.finish_reason is None
        assert response.is_final is False

    def test_final_response(self):
        """Test final streaming response."""
        response = StreamingResponse(
            delta="",
            finish_reason="stop",
            is_final=True,
            accumulated_text="Hello world",
            token_count=5,
        )

        assert response.is_final is True
        assert response.finish_reason == "stop"
        assert response.accumulated_text == "Hello world"


class MockBackend(InferenceBackend):
    """Mock backend for testing."""

    def load(self):
        self._is_loaded = True
        self.tokenizer = MagicMock()
        self.tokenizer.encode.return_value = [1, 2, 3]

    def unload(self):
        self._is_loaded = False

    def generate(self, request):
        return GenerationResponse(
            text="Generated text",
            prompt=request.prompt,
            finish_reason="stop",
            usage={"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        )

    def generate_stream(self, request):
        yield StreamingResponse(delta="Hello", accumulated_text="Hello")
        yield StreamingResponse(delta=" world", accumulated_text="Hello world")
        yield StreamingResponse(delta="", is_final=True, accumulated_text="Hello world")


class TestInferenceBackend:
    """Tests for InferenceBackend (via MockBackend)."""

    def test_initialization(self):
        """Test backend initialization."""
        backend = MockBackend("test-model")

        assert backend.model_path == "test-model"
        assert backend.is_loaded is False

    def test_load_unload(self):
        """Test load and unload."""
        backend = MockBackend("test-model")

        backend.load()
        assert backend.is_loaded is True

        backend.unload()
        assert backend.is_loaded is False

    def test_generate(self):
        """Test generation."""
        backend = MockBackend("test-model")
        backend.load()

        request = GenerationRequest(prompt="Hello")
        response = backend.generate(request)

        assert response.text == "Generated text"
        assert response.prompt == "Hello"

    def test_generate_stream(self):
        """Test streaming generation."""
        backend = MockBackend("test-model")
        backend.load()

        request = GenerationRequest(prompt="Hello", stream=True)
        chunks = list(backend.generate_stream(request))

        assert len(chunks) == 3
        assert chunks[0].delta == "Hello"
        assert chunks[-1].is_final is True

    def test_count_tokens(self):
        """Test token counting."""
        backend = MockBackend("test-model")
        backend.load()

        count = backend.count_tokens("Hello world")
        assert count == 3

    def test_get_model_info(self):
        """Test getting model info."""
        backend = MockBackend("test-model", device="cuda", dtype="float16")
        info = backend.get_model_info()

        assert info["model_path"] == "test-model"
        assert info["device"] == "cuda"
        assert info["dtype"] == "float16"


class TestTextGenerator:
    """Tests for TextGenerator."""

    @patch("largeforge.inference.generator.TextGenerator._create_backend")
    def test_initialization(self, mock_create):
        """Test generator initialization."""
        mock_backend = MockBackend("test-model")
        mock_create.return_value = mock_backend

        generator = TextGenerator("test-model", backend="transformers")

        assert generator.model_path == "test-model"
        assert generator.backend_name == "transformers"

    @patch("largeforge.inference.generator.TextGenerator._create_backend")
    def test_load_unload(self, mock_create):
        """Test load and unload."""
        mock_backend = MockBackend("test-model")
        mock_create.return_value = mock_backend

        generator = TextGenerator("test-model")
        generator.load()

        assert mock_backend.is_loaded is True

        generator.unload()
        assert mock_backend.is_loaded is False

    @patch("largeforge.inference.generator.TextGenerator._create_backend")
    def test_generate(self, mock_create):
        """Test text generation."""
        mock_backend = MockBackend("test-model")
        mock_backend.load()
        mock_create.return_value = mock_backend

        generator = TextGenerator("test-model")
        generator.load()

        text = generator.generate("Hello")

        assert text == "Generated text"

    @patch("largeforge.inference.generator.TextGenerator._create_backend")
    def test_generate_with_info(self, mock_create):
        """Test generation with full response."""
        mock_backend = MockBackend("test-model")
        mock_backend.load()
        mock_create.return_value = mock_backend

        generator = TextGenerator("test-model")
        generator.load()

        response = generator.generate_with_info("Hello")

        assert isinstance(response, GenerationResponse)
        assert response.text == "Generated text"
        assert response.total_tokens == 15

    @patch("largeforge.inference.generator.TextGenerator._create_backend")
    def test_generate_stream(self, mock_create):
        """Test streaming generation."""
        mock_backend = MockBackend("test-model")
        mock_backend.load()
        mock_create.return_value = mock_backend

        generator = TextGenerator("test-model")
        generator.load()

        chunks = list(generator.generate_stream("Hello"))

        # Only non-empty deltas
        assert len(chunks) == 2
        assert chunks[0] == "Hello"
        assert chunks[1] == " world"

    @patch("largeforge.inference.generator.TextGenerator._create_backend")
    def test_context_manager(self, mock_create):
        """Test context manager usage."""
        mock_backend = MockBackend("test-model")
        mock_create.return_value = mock_backend

        with TextGenerator("test-model") as generator:
            assert mock_backend.is_loaded is True
            text = generator.generate("Hello")

        assert mock_backend.is_loaded is False

    @patch("largeforge.inference.generator.TextGenerator._create_backend")
    def test_get_info(self, mock_create):
        """Test getting generator info."""
        mock_backend = MockBackend("test-model")
        mock_create.return_value = mock_backend

        generator = TextGenerator("test-model", backend="transformers")
        info = generator.get_info()

        assert info["model_path"] == "test-model"
        assert info["backend"] == "transformers"

    def test_auto_backend_selection(self):
        """Test auto backend selection falls back to transformers."""
        # This should not raise even without vllm installed
        generator = TextGenerator("test-model", backend="auto")

        # Backend is lazily created, so just check initialization works
        assert generator.backend_name == "auto"


class TestTransformersBackend:
    """Tests for TransformersBackend (mocked)."""

    def test_load(self):
        """Test backend loading with mocked transformers."""
        from largeforge.inference.transformers_backend import TransformersBackend

        with patch("transformers.AutoTokenizer") as mock_tokenizer_cls, \
             patch("transformers.AutoModelForCausalLM") as mock_model_cls:
            # Setup mocks
            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = None
            mock_tokenizer.eos_token = "</s>"
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

            mock_model = MagicMock()
            mock_model.device = "cpu"
            mock_model_cls.from_pretrained.return_value = mock_model

            backend = TransformersBackend("test-model", device="cpu")
            backend.load()

            assert backend.is_loaded is True
            mock_tokenizer_cls.from_pretrained.assert_called_once()
            mock_model_cls.from_pretrained.assert_called_once()


class TestVLLMBackend:
    """Tests for VLLMBackend (mocked)."""

    def test_initialization(self):
        """Test backend initialization without loading."""
        from largeforge.inference.vllm_backend import VLLMBackend

        backend = VLLMBackend(
            "test-model",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.8,
        )

        assert backend.model_path == "test-model"
        assert backend.tensor_parallel_size == 2
        assert backend.gpu_memory_utilization == 0.8

    def test_get_model_info(self):
        """Test getting model info."""
        from largeforge.inference.vllm_backend import VLLMBackend

        backend = VLLMBackend(
            "test-model",
            tensor_parallel_size=4,
            quantization="awq",
        )

        info = backend.get_model_info()

        assert info["backend"] == "vllm"
        assert info["tensor_parallel_size"] == 4
        assert info["quantization"] == "awq"


class TestInferenceConvenienceFunctions:
    """Tests for convenience functions."""

    @patch("largeforge.inference.generator.TextGenerator")
    def test_generate_function(self, mock_generator_cls):
        """Test generate() convenience function."""
        from largeforge.inference.generator import generate

        mock_generator = MagicMock()
        mock_generator.generate.return_value = "Generated text"
        mock_generator.__enter__ = MagicMock(return_value=mock_generator)
        mock_generator.__exit__ = MagicMock(return_value=None)
        mock_generator_cls.return_value = mock_generator

        result = generate("Hello", model_path="test-model")

        assert result == "Generated text"

    @patch("largeforge.inference.generator.TextGenerator")
    def test_generate_batch_function(self, mock_generator_cls):
        """Test generate_batch() convenience function."""
        from largeforge.inference.generator import generate_batch

        mock_generator = MagicMock()
        mock_generator.generate_batch.return_value = ["Text 1", "Text 2"]
        mock_generator.__enter__ = MagicMock(return_value=mock_generator)
        mock_generator.__exit__ = MagicMock(return_value=None)
        mock_generator_cls.return_value = mock_generator

        result = generate_batch(["Prompt 1", "Prompt 2"], model_path="test-model")

        assert result == ["Text 1", "Text 2"]
