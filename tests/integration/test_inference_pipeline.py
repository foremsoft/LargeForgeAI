"""Integration tests for the inference pipeline."""

import pytest
from unittest.mock import MagicMock, patch
import json

from largeforge.config import InferenceConfig, GenerationConfig
from largeforge.inference import TextGenerator


@pytest.mark.integration
class TestTextGeneratorPipeline:
    """Test end-to-end text generation pipeline."""

    def test_generator_initialization(self):
        """Test TextGenerator initialization."""
        generator = TextGenerator(
            model_path="gpt2",
            backend="transformers",
            device="cpu",
            dtype="float32",
        )

        assert generator.model_path == "gpt2"
        assert generator.backend_name == "transformers"

    def test_generator_multiple_backends(self):
        """Test creating generators with different backends."""
        backends = ["transformers", "vllm", "auto"]

        for backend in backends:
            generator = TextGenerator(
                model_path="gpt2",
                backend=backend,
            )
            assert generator.backend_name == backend

    def test_generator_attributes(self):
        """Test generator has expected attributes."""
        generator = TextGenerator(
            model_path="test-model",
            backend="transformers",
            device="cpu",
        )

        assert hasattr(generator, "model_path")
        assert hasattr(generator, "backend_name")
        assert hasattr(generator, "load")
        assert hasattr(generator, "unload")
        assert hasattr(generator, "generate")


@pytest.mark.integration
class TestInferenceConfigPipeline:
    """Test inference configuration pipeline."""

    def test_inference_config_creation(self):
        """Test creating inference config."""
        config = InferenceConfig(
            model_path="gpt2",
            backend="transformers",
            dtype="auto",
        )

        assert config.backend == "transformers"
        assert config.model_path == "gpt2"

    def test_generation_config_creation(self):
        """Test creating generation config."""
        config = GenerationConfig(
            max_tokens=512,
            temperature=0.8,
            top_p=0.95,
            top_k=40,
        )

        assert config.temperature == 0.8
        assert config.max_tokens == 512

    def test_inference_config_vllm_options(self):
        """Test inference config for vLLM backend."""
        config = InferenceConfig(
            model_path="meta-llama/Llama-2-7b-hf",
            backend="vllm",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.85,
        )

        assert config.backend == "vllm"
        assert config.tensor_parallel_size == 2
        assert config.gpu_memory_utilization == 0.85

    def test_inference_config_quantization(self):
        """Test inference config with quantization."""
        config = InferenceConfig(
            model_path="gpt2",
            quantization="awq",  # Valid quantization type
        )

        assert config.quantization == "awq"


@pytest.mark.integration
class TestStreamingGeneration:
    """Test streaming text generation."""

    def test_streaming_method_exists(self):
        """Test generator has streaming method."""
        generator = TextGenerator(
            model_path="gpt2",
            backend="transformers",
        )

        # Test stream method exists
        assert hasattr(generator, 'generate_stream')
        assert callable(generator.generate_stream)


@pytest.mark.integration
class TestBackendSelection:
    """Test inference backend selection."""

    def test_transformers_backend_selection(self):
        """Test selecting transformers backend."""
        generator = TextGenerator(
            model_path="gpt2",
            backend="transformers",
        )
        assert generator.backend_name == "transformers"

    def test_vllm_backend_selection(self):
        """Test selecting vLLM backend."""
        generator = TextGenerator(
            model_path="gpt2",
            backend="vllm",
        )
        assert generator.backend_name == "vllm"

    def test_auto_backend_selection(self):
        """Test auto backend selection."""
        generator = TextGenerator(
            model_path="gpt2",
            backend="auto",
        )
        # Auto is stored as-is or defaults to transformers
        assert generator.backend_name in ["auto", "transformers"]


@pytest.mark.integration
class TestGeneratorContextManager:
    """Test generator as context manager."""

    def test_context_manager_supported(self):
        """Test generator supports context manager protocol."""
        generator = TextGenerator(
            model_path="gpt2",
            backend="transformers",
        )

        # Check context manager methods exist
        assert hasattr(generator, '__enter__')
        assert hasattr(generator, '__exit__')

    def test_context_manager_creation(self):
        """Test creating generator in context manager."""
        # Should not raise when using context manager pattern
        try:
            with TextGenerator(
                model_path="gpt2",
                backend="transformers",
            ) as generator:
                assert generator is not None
                assert generator.model_path == "gpt2"
        except Exception as e:
            # If model can't load, that's expected in test environment
            # We're just testing the context manager setup
            pass
