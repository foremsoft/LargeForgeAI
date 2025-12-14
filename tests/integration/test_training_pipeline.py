"""Integration tests for the training pipeline."""

import pytest
from unittest.mock import MagicMock, patch
import json

from largeforge.config import TrainingConfig, LoRAConfig, SFTConfig, DPOConfig


@pytest.mark.integration
class TestSFTTrainingPipeline:
    """Test end-to-end SFT training pipeline."""

    def test_sft_config_creation(self, tmp_path):
        """Test SFT config can be created with valid values."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        config = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=3,
            per_device_train_batch_size=4,
            learning_rate=2e-5,
            max_seq_length=2048,
        )

        assert config.output_dir == str(output_dir)
        assert config.num_train_epochs == 3
        assert config.max_seq_length == 2048

    def test_sft_config_with_lora(self, tmp_path):
        """Test SFT config alongside LoRA config."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        sft_config = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=1,
            per_device_train_batch_size=2,
        )

        lora_config = LoRAConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
        )

        # Both configs should be valid
        assert sft_config.num_train_epochs == 1
        assert lora_config.r == 16


@pytest.mark.integration
class TestDPOTrainingPipeline:
    """Test end-to-end DPO training pipeline."""

    def test_dpo_config_creation(self, tmp_path):
        """Test DPO config creation."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        config = DPOConfig(
            output_dir=str(output_dir),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            beta=0.1,
            max_length=1024,
            max_prompt_length=512,
        )

        assert config.beta == 0.1
        assert config.max_length == 1024

    def test_dpo_loss_types(self, tmp_path):
        """Test DPO config with different loss types."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        for loss_type in ["sigmoid", "hinge", "ipo", "kto_pair"]:
            config = DPOConfig(
                output_dir=str(output_dir),
                loss_type=loss_type,
            )
            assert config.loss_type == loss_type


@pytest.mark.integration
class TestTrainingConfigIntegration:
    """Test training configuration integration."""

    def test_training_config_defaults(self, tmp_path):
        """Test training config default values."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        config = TrainingConfig(output_dir=str(output_dir))

        assert config.num_train_epochs == 3
        assert config.per_device_train_batch_size == 4
        assert config.learning_rate == 2e-5
        assert config.warmup_ratio == 0.1

    def test_lora_config_creation(self):
        """Test LoRA config creation."""
        lora_config = LoRAConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

        assert lora_config.r == 16
        assert lora_config.lora_alpha == 32
        assert len(lora_config.target_modules) == 4

    def test_config_precision_validation(self, tmp_path):
        """Test config validation for precision settings."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Valid: only bf16
        config = TrainingConfig(output_dir=str(output_dir), bf16=True, fp16=False)
        assert config.bf16 is True

        # Valid: only fp16
        config = TrainingConfig(output_dir=str(output_dir), bf16=False, fp16=True)
        assert config.fp16 is True

        # Invalid: both
        with pytest.raises(Exception):  # Pydantic ValidationError
            TrainingConfig(output_dir=str(output_dir), bf16=True, fp16=True)

    def test_sft_inherits_training_config(self, tmp_path):
        """Test SFT config inherits from training config."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        config = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=5,
            learning_rate=1e-4,
            max_seq_length=4096,
        )

        # Should have both training and SFT specific fields
        assert config.num_train_epochs == 5
        assert config.max_seq_length == 4096

    def test_config_yaml_roundtrip(self, tmp_path):
        """Test config can be saved and loaded as YAML."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        config = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=5,
            learning_rate=1e-4,
        )

        # Save to YAML
        yaml_path = tmp_path / "config.yaml"
        config.to_yaml(yaml_path)

        # Load from YAML
        loaded = SFTConfig.from_yaml(yaml_path)

        assert loaded.num_train_epochs == config.num_train_epochs
        assert loaded.learning_rate == config.learning_rate
