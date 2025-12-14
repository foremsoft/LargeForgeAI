"""End-to-end integration tests for complete workflows."""

import pytest
from unittest.mock import MagicMock, patch
import json

from largeforge.data import load_dataset, validate_dataset, detect_format
from largeforge.data.converters import alpaca_to_sharegpt, to_chat_format
from largeforge.config import SFTConfig, InferenceConfig, GenerationConfig, RouterConfig
from largeforge.router import HybridRouter, Expert


@pytest.mark.integration
class TestDataToTrainingWorkflow:
    """Test complete data to training workflow."""

    def test_load_validate_prepare_workflow(self, integration_data_dir):
        """Test loading data, validating, and preparing for training."""
        # Step 1: Load data
        data_path = integration_data_dir / "alpaca.jsonl"
        data = load_dataset(str(data_path))
        assert len(data) == 3

        # Step 2: Auto-detect format
        detected_format = detect_format(data)
        assert detected_format == "alpaca"

        # Step 3: Validate
        results = validate_dataset(data, format_type=detected_format)
        valid_data = [data[i] for i, r in enumerate(results) if r["valid"]]
        assert len(valid_data) == 3

        # Step 4: Convert to chat format for training
        chat_data = to_chat_format(valid_data, format="alpaca")
        assert len(chat_data) == len(valid_data)

        # Verify chat format structure - returns list of message lists
        for messages in chat_data:
            roles = [m["role"] for m in messages]
            assert "user" in roles
            assert "assistant" in roles

    def test_data_to_training_config_workflow(
        self, integration_data_dir, tmp_path
    ):
        """Test complete workflow from data validation to training config."""
        # Load and validate data
        data_path = integration_data_dir / "alpaca.jsonl"
        data = load_dataset(str(data_path))
        results = validate_dataset(data, format_type="alpaca")

        valid_count = sum(1 for r in results if r["valid"])
        assert valid_count == len(data)

        # Create training config
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        config = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            learning_rate=5e-5,
        )

        assert config is not None
        assert config.num_train_epochs == 1
        assert config.learning_rate == 5e-5


@pytest.mark.integration
class TestRoutingToInferenceWorkflow:
    """Test complete routing to inference workflow."""

    def test_router_selection_to_config_workflow(self):
        """Test routing query to selecting appropriate inference config."""
        # Setup router with expert configurations
        router = HybridRouter(config=RouterConfig(
            neural_weight=0.4,
            keyword_weight=0.6,
        ))

        # Add experts with inference metadata
        router.add_expert(Expert(
            name="code",
            description="Code generation expert",
            keywords=["code", "python", "javascript", "function", "class"],
            metadata={
                "model": "codellama/CodeLlama-7b-hf",
                "temperature": 0.2,
                "max_tokens": 1024,
            },
        ))
        router.add_expert(Expert(
            name="chat",
            description="General conversation expert",
            keywords=["chat", "help", "question", "explain"],
            metadata={
                "model": "meta-llama/Llama-2-7b-chat-hf",
                "temperature": 0.7,
                "max_tokens": 512,
            },
        ))

        # Route a code query
        result = router.route("Write a Python function to sort a list")
        assert result.expert.name == "code"

        # Get inference config from expert metadata
        expert_config = result.expert.metadata
        gen_config = GenerationConfig(
            max_tokens=expert_config.get("max_tokens", 256),
            temperature=expert_config.get("temperature", 0.7),
        )

        assert gen_config.temperature == 0.2
        assert gen_config.max_tokens == 1024

    def test_router_to_generator_config_workflow(self):
        """Test routing query and configuring generator."""
        from largeforge.inference import TextGenerator

        # Setup router
        router = HybridRouter()
        router.add_expert(Expert(
            name="code",
            keywords=["code", "python", "function"],
            metadata={"temperature": 0.2},
        ))
        router.add_expert(Expert(
            name="chat",
            keywords=["chat", "help"],
            metadata={"temperature": 0.7},
        ))

        # Route query
        query = "Write a Python function to sort a list"
        result = router.route(query)

        # Use expert metadata for generation config
        temp = result.expert.metadata.get("temperature", 0.7)
        assert temp == 0.2  # Code expert should be selected

        # Create generator (don't load - just test config flow)
        generator = TextGenerator(
            model_path="gpt2",
            backend="transformers",
        )

        assert generator is not None
        assert generator.model_path == "gpt2"

        # Verify the routing result provides correct metadata
        assert result.expert.name == "code"
        assert result.expert.metadata["temperature"] == 0.2


@pytest.mark.integration
class TestConfigurationWorkflow:
    """Test configuration management workflows."""

    def test_config_cascade_workflow(self, tmp_path):
        """Test loading and merging configurations."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create base training config
        base_config = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=3,
            learning_rate=2e-5,
        )

        # Override with specific values
        override_config = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=5,  # Override
            learning_rate=1e-4,  # Override
        )

        assert override_config.learning_rate == 1e-4
        assert override_config.num_train_epochs == 5

    def test_multi_config_workflow(self, tmp_path):
        """Test using multiple configs together."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Training config
        train_config = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=3,
        )

        # Generation config
        gen_config = GenerationConfig(
            temperature=0.7,
            max_tokens=256,
        )

        # Router config
        router_config = RouterConfig(
            neural_weight=0.5,
            keyword_weight=0.5,
        )

        # All configs should be valid
        assert train_config.num_train_epochs == 3
        assert gen_config.temperature == 0.7
        assert router_config.neural_weight == 0.5


@pytest.mark.integration
class TestErrorRecoveryWorkflow:
    """Test error handling and recovery workflows."""

    def test_data_validation_error_recovery(self, tmp_path):
        """Test recovering from data validation errors."""
        from largeforge.utils import save_jsonl

        # Create mixed valid/invalid data
        mixed_data = [
            {"instruction": "Valid 1", "output": "Output 1"},
            {"bad": "Invalid record"},  # Invalid
            {"instruction": "Valid 2", "output": "Output 2"},
            {"instruction": "Missing output"},  # Invalid
            {"instruction": "Valid 3", "output": "Output 3"},
        ]

        data_path = tmp_path / "mixed.jsonl"
        save_jsonl(mixed_data, str(data_path))

        # Load data
        data = load_dataset(str(data_path))
        assert len(data) == 5

        # Validate and filter
        results = validate_dataset(data, format_type="alpaca")
        valid_data = [data[i] for i, r in enumerate(results) if r["valid"]]
        invalid_indices = [i for i, r in enumerate(results) if not r["valid"]]

        assert len(valid_data) == 3
        assert len(invalid_indices) == 2
        assert 1 in invalid_indices
        assert 3 in invalid_indices

        # Verify filtered data is correct
        assert all("instruction" in d and "output" in d for d in valid_data)

    def test_router_fallback_workflow(self):
        """Test router fallback when no good match found."""
        router = HybridRouter(config=RouterConfig(
            confidence_threshold=0.9,  # High threshold
            neural_weight=0.5,
            keyword_weight=0.5,
        ))

        router.add_expert(Expert(
            name="specific",
            keywords=["very_specific_keyword"],
        ))

        # Query that won't match well
        result = router.route("Random unrelated query")

        # Should still return a result (best available)
        assert result is not None
        # Score might be low but routing should still work
        assert result.expert is not None


@pytest.mark.integration
class TestFullPipelineWorkflow:
    """Test complete end-to-end pipeline."""

    def test_data_train_config_workflow(
        self, integration_data_dir, tmp_path
    ):
        """Test complete data -> train config workflow."""
        # Step 1: Load and validate data
        data = load_dataset(str(integration_data_dir / "alpaca.jsonl"))
        results = validate_dataset(data, format_type="alpaca")
        valid_count = sum(1 for r in results if r["valid"])
        assert valid_count == 3

        # Step 2: Create training config
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        train_config = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=1,
        )
        assert train_config is not None
        assert train_config.num_train_epochs == 1

        # Step 3: Create generation config
        gen_config = GenerationConfig(
            max_tokens=128,
            temperature=0.7,
        )
        assert gen_config.max_tokens == 128
        assert gen_config.temperature == 0.7

    def test_router_to_inference_config_workflow(self):
        """Test routing to inference configuration workflow."""
        from largeforge.inference import TextGenerator

        # Setup router with experts
        router = HybridRouter()
        router.add_expert(Expert(
            name="assistant",
            keywords=["help", "question"],
            metadata={"model": "gpt2", "temperature": 0.7},
        ))
        router.add_expert(Expert(
            name="code",
            keywords=["code", "program"],
            metadata={"model": "codellama", "temperature": 0.2},
        ))

        # Route a help query
        result = router.route("I need help with something")
        assert result.expert.name == "assistant"
        assert result.expert.metadata["temperature"] == 0.7

        # Route a code query
        result = router.route("Write some code for me")
        assert result.expert.name == "code"
        assert result.expert.metadata["temperature"] == 0.2

        # Create generator config based on routing
        gen_config = GenerationConfig(
            max_tokens=256,
            temperature=result.expert.metadata["temperature"],
        )
        assert gen_config.temperature == 0.2

        # Create generator (don't load - testing config flow)
        generator = TextGenerator(
            model_path=result.expert.metadata.get("model", "gpt2"),
            backend="transformers",
        )
        assert generator.model_path == "codellama"
