"""Unit tests for configuration modules."""

import pytest
import tempfile
from pathlib import Path

from largeforge.config.base import BaseConfig, ModelConfig
from largeforge.config.training import TrainingConfig, LoRAConfig, SFTConfig, DPOConfig
from largeforge.config.inference import GenerationConfig, InferenceConfig, RouterConfig


class TestBaseConfig:
    """Tests for BaseConfig."""

    def test_to_dict(self):
        """Test config serialization to dict."""
        config = ModelConfig(name="test-model")
        data = config.to_dict()
        assert data["name"] == "test-model"
        assert "torch_dtype" in data

    def test_from_dict(self):
        """Test config deserialization from dict."""
        data = {"name": "test-model", "revision": "v1"}
        config = ModelConfig.from_dict(data)
        assert config.name == "test-model"
        assert config.revision == "v1"

    def test_yaml_roundtrip(self, tmp_path):
        """Test YAML save and load."""
        config = ModelConfig(name="test-model", torch_dtype="float16")
        yaml_path = tmp_path / "config.yaml"
        config.to_yaml(str(yaml_path))

        loaded = ModelConfig.from_yaml(str(yaml_path))
        assert loaded.name == config.name
        assert loaded.torch_dtype == config.torch_dtype


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ModelConfig(name="meta-llama/Llama-2-7b-hf")
        assert config.torch_dtype == "bfloat16"
        assert config.device_map == "auto"
        assert config.revision == "main"

    def test_invalid_torch_dtype(self):
        """Test validation of invalid torch_dtype."""
        with pytest.raises(ValueError):
            ModelConfig(name="test", torch_dtype="invalid")

    def test_valid_torch_dtypes(self):
        """Test all valid torch dtypes."""
        for dtype in ["float16", "bfloat16", "float32"]:
            config = ModelConfig(name="test", torch_dtype=dtype)
            assert config.torch_dtype == dtype

    def test_cuda_device_map(self):
        """Test cuda:N device map format."""
        config = ModelConfig(name="test", device_map="cuda:0")
        assert config.device_map == "cuda:0"


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self):
        """Test default training configuration."""
        config = TrainingConfig(output_dir="./output")
        assert config.num_train_epochs == 3
        assert config.learning_rate == 2e-5
        assert config.bf16 is True

    def test_fp16_bf16_conflict(self):
        """Test that fp16 and bf16 cannot both be True."""
        with pytest.raises(ValueError, match="Cannot enable both"):
            TrainingConfig(output_dir="./out", fp16=True, bf16=True)

    def test_valid_fp16_only(self):
        """Test fp16 without bf16."""
        config = TrainingConfig(output_dir="./out", fp16=True, bf16=False)
        assert config.fp16 is True
        assert config.bf16 is False

    def test_field_constraints(self):
        """Test field value constraints."""
        with pytest.raises(ValueError):
            TrainingConfig(output_dir="./out", num_train_epochs=0)

        with pytest.raises(ValueError):
            TrainingConfig(output_dir="./out", learning_rate=-1)


class TestLoRAConfig:
    """Tests for LoRAConfig."""

    def test_default_values(self):
        """Test default LoRA configuration."""
        config = LoRAConfig()
        assert config.r == 8
        assert config.lora_alpha == 16
        assert "q_proj" in config.target_modules

    def test_invalid_bias(self):
        """Test validation of invalid bias value."""
        with pytest.raises(ValueError):
            LoRAConfig(bias="invalid")

    def test_valid_bias_values(self):
        """Test all valid bias values."""
        for bias in ["none", "all", "lora_only"]:
            config = LoRAConfig(bias=bias)
            assert config.bias == bias

    def test_r_constraints(self):
        """Test LoRA rank constraints."""
        with pytest.raises(ValueError):
            LoRAConfig(r=0)

        with pytest.raises(ValueError):
            LoRAConfig(r=300)


class TestSFTConfig:
    """Tests for SFTConfig."""

    def test_inherits_training_config(self):
        """Test SFTConfig inherits from TrainingConfig."""
        config = SFTConfig(output_dir="./out")
        assert hasattr(config, "num_train_epochs")
        assert hasattr(config, "max_seq_length")

    def test_default_values(self):
        """Test SFT-specific defaults."""
        config = SFTConfig(output_dir="./out")
        assert config.max_seq_length == 2048
        assert config.packing is False


class TestDPOConfig:
    """Tests for DPOConfig."""

    def test_default_values(self):
        """Test DPO-specific defaults."""
        config = DPOConfig(output_dir="./out")
        assert config.beta == 0.1
        assert config.loss_type == "sigmoid"

    def test_invalid_loss_type(self):
        """Test validation of invalid loss type."""
        with pytest.raises(ValueError):
            DPOConfig(output_dir="./out", loss_type="invalid")

    def test_valid_loss_types(self):
        """Test all valid loss types."""
        for loss_type in ["sigmoid", "hinge", "ipo", "kto_pair"]:
            config = DPOConfig(output_dir="./out", loss_type=loss_type)
            assert config.loss_type == loss_type

    def test_length_validation(self):
        """Test that max_prompt_length < max_length."""
        with pytest.raises(ValueError):
            DPOConfig(output_dir="./out", max_length=512, max_prompt_length=512)


class TestGenerationConfig:
    """Tests for GenerationConfig."""

    def test_default_values(self):
        """Test generation defaults."""
        config = GenerationConfig()
        assert config.max_tokens == 256
        assert config.temperature == 0.7
        assert config.stream is False

    def test_temperature_bounds(self):
        """Test temperature value bounds."""
        with pytest.raises(ValueError):
            GenerationConfig(temperature=-0.1)

        with pytest.raises(ValueError):
            GenerationConfig(temperature=2.5)

    def test_zero_temperature_disables_sampling(self):
        """Test that temperature=0 sets do_sample=False."""
        config = GenerationConfig(temperature=0)
        assert config.do_sample is False


class TestInferenceConfig:
    """Tests for InferenceConfig."""

    def test_default_values(self):
        """Test inference defaults."""
        config = InferenceConfig(model_path="./model")
        assert config.backend == "auto"
        assert config.port == 8000

    def test_invalid_backend(self):
        """Test validation of invalid backend."""
        with pytest.raises(ValueError):
            InferenceConfig(model_path="./model", backend="invalid")

    def test_quantization_validation(self):
        """Test quantization method validation."""
        config = InferenceConfig(model_path="./model", quantization="awq")
        assert config.quantization == "awq"

        # Test case insensitivity
        config = InferenceConfig(model_path="./model", quantization="AWQ")
        assert config.quantization == "awq"


class TestRouterConfig:
    """Tests for RouterConfig."""

    def test_default_values(self):
        """Test router defaults."""
        config = RouterConfig()
        assert config.classifier_type == "hybrid"
        assert config.confidence_threshold == 0.6

    def test_weight_validation(self):
        """Test that weights sum to 1.0."""
        with pytest.raises(ValueError, match="must equal 1.0"):
            RouterConfig(keyword_weight=0.5, neural_weight=0.6)

    def test_valid_weights(self):
        """Test valid weight combinations."""
        config = RouterConfig(keyword_weight=0.4, neural_weight=0.6)
        assert config.keyword_weight == 0.4
        assert config.neural_weight == 0.6

    def test_non_hybrid_weights(self):
        """Test that weights are not validated for non-hybrid classifiers."""
        # Should not raise even with invalid weights for keyword-only
        config = RouterConfig(
            classifier_type="keyword",
            keyword_weight=0.5,
            neural_weight=0.6
        )
        assert config.classifier_type == "keyword"
