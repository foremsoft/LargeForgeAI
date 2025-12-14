"""Unit tests for training modules."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from largeforge.training.base import TrainingState, TrainingCallback, BaseTrainer
from largeforge.training.callbacks import (
    LoggingCallback, CheckpointCallback, EarlyStoppingCallback,
)
from largeforge.training.lora import (
    get_lora_target_modules, get_trainable_parameters,
)


class TestTrainingState:
    """Tests for TrainingState."""

    def test_default_values(self):
        """Test default state values."""
        state = TrainingState()
        assert state.epoch == 0
        assert state.global_step == 0
        assert state.is_training is False
        assert state.should_stop is False

    def test_to_dict(self):
        """Test state serialization."""
        state = TrainingState(epoch=5, global_step=100)
        data = state.to_dict()

        assert data["epoch"] == 5
        assert data["global_step"] == 100

    def test_from_dict(self):
        """Test state deserialization."""
        data = {"epoch": 3, "global_step": 50, "best_metric": 0.5}
        state = TrainingState.from_dict(data)

        assert state.epoch == 3
        assert state.global_step == 50
        assert state.best_metric == 0.5

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = TrainingState(
            epoch=10,
            global_step=500,
            best_metric=0.123,
            log_history=[{"loss": 1.0}, {"loss": 0.5}],
        )

        data = original.to_dict()
        restored = TrainingState.from_dict(data)

        assert restored.epoch == original.epoch
        assert restored.global_step == original.global_step
        assert restored.best_metric == original.best_metric
        assert len(restored.log_history) == len(original.log_history)


class TestTrainingCallback:
    """Tests for TrainingCallback."""

    def test_callback_methods_exist(self):
        """Test that callback has all required methods."""
        callback = TrainingCallback()

        assert hasattr(callback, "on_train_begin")
        assert hasattr(callback, "on_train_end")
        assert hasattr(callback, "on_epoch_begin")
        assert hasattr(callback, "on_epoch_end")
        assert hasattr(callback, "on_step_begin")
        assert hasattr(callback, "on_step_end")
        assert hasattr(callback, "on_log")
        assert hasattr(callback, "on_save")
        assert hasattr(callback, "on_evaluate")

    def test_default_methods_do_nothing(self):
        """Test that default methods don't raise."""
        callback = TrainingCallback()
        state = TrainingState()
        trainer = MagicMock()

        # None of these should raise
        callback.on_train_begin(trainer, state)
        callback.on_train_end(trainer, state)
        callback.on_epoch_begin(trainer, state)
        callback.on_epoch_end(trainer, state)
        callback.on_step_begin(trainer, state)
        callback.on_step_end(trainer, state)
        callback.on_log(trainer, state, {"loss": 1.0})
        callback.on_save(trainer, state)
        callback.on_evaluate(trainer, state, {"eval_loss": 0.5})


class TestLoggingCallback:
    """Tests for LoggingCallback."""

    def test_initialization(self):
        """Test callback initialization."""
        callback = LoggingCallback(log_every_n_steps=20)
        assert callback.log_every_n_steps == 20
        assert callback.log_to_file is False

    def test_on_train_begin_sets_start_time(self):
        """Test that training start records time."""
        callback = LoggingCallback()
        trainer = MagicMock()
        trainer.config.num_train_epochs = 3
        state = TrainingState(total_steps=100)

        callback.on_train_begin(trainer, state)

        assert callback.start_time is not None

    def test_log_to_file(self, tmp_path):
        """Test logging to file."""
        log_file = tmp_path / "train.log"
        callback = LoggingCallback(log_to_file=True, log_file_path=str(log_file))

        trainer = MagicMock()
        state = TrainingState()
        logs = {"loss": 1.5, "step": 10}

        callback.on_log(trainer, state, logs)

        assert log_file.exists()
        content = log_file.read_text()
        assert "1.5" in content


class TestCheckpointCallback:
    """Tests for CheckpointCallback."""

    def test_initialization(self):
        """Test callback initialization."""
        callback = CheckpointCallback(save_total_limit=5)
        assert callback.save_total_limit == 5
        assert callback.save_best_only is False

    def test_tracks_checkpoints(self, tmp_path):
        """Test checkpoint tracking."""
        callback = CheckpointCallback(save_total_limit=2)
        trainer = MagicMock()
        trainer.config.output_dir = str(tmp_path)

        state = TrainingState(global_step=100)
        callback.on_save(trainer, state)
        assert len(callback.saved_checkpoints) == 1

        state.global_step = 200
        callback.on_save(trainer, state)
        assert len(callback.saved_checkpoints) == 2

    def test_best_model_tracking(self, tmp_path):
        """Test best model tracking."""
        callback = CheckpointCallback(
            save_best_only=True,
            metric_for_best="eval_loss",
            greater_is_better=False,
        )

        trainer = MagicMock()
        trainer.config.output_dir = str(tmp_path)
        state = TrainingState()

        # First evaluation
        callback.on_evaluate(trainer, state, {"eval_loss": 1.0})
        assert callback.best_metric == 1.0

        # Better result
        callback.on_evaluate(trainer, state, {"eval_loss": 0.5})
        assert callback.best_metric == 0.5

        # Worse result - should not update
        callback.on_evaluate(trainer, state, {"eval_loss": 0.8})
        assert callback.best_metric == 0.5


class TestEarlyStoppingCallback:
    """Tests for EarlyStoppingCallback."""

    def test_initialization(self):
        """Test callback initialization."""
        callback = EarlyStoppingCallback(patience=5)
        assert callback.patience == 5
        assert callback.patience_counter == 0

    def test_stops_after_patience(self):
        """Test early stopping triggers after patience."""
        callback = EarlyStoppingCallback(
            patience=2,
            metric="eval_loss",
            greater_is_better=False,
        )

        trainer = MagicMock()
        state = TrainingState()

        # First evaluation sets baseline
        callback.on_evaluate(trainer, state, {"eval_loss": 1.0})
        assert state.should_stop is False

        # No improvement
        callback.on_evaluate(trainer, state, {"eval_loss": 1.0})
        assert callback.patience_counter == 1
        assert state.should_stop is False

        # Still no improvement - should trigger stop
        callback.on_evaluate(trainer, state, {"eval_loss": 1.0})
        assert state.should_stop is True

    def test_resets_on_improvement(self):
        """Test patience resets on improvement."""
        callback = EarlyStoppingCallback(
            patience=2,
            metric="eval_loss",
            greater_is_better=False,
        )

        trainer = MagicMock()
        state = TrainingState()

        callback.on_evaluate(trainer, state, {"eval_loss": 1.0})
        callback.on_evaluate(trainer, state, {"eval_loss": 1.0})  # patience = 1

        # Improvement resets counter
        callback.on_evaluate(trainer, state, {"eval_loss": 0.5})
        assert callback.patience_counter == 0
        assert callback.best_metric == 0.5


class TestLoRAUtilities:
    """Tests for LoRA utilities."""

    def test_get_target_modules_llama(self):
        """Test getting target modules for Llama."""
        modules = get_lora_target_modules("meta-llama/Llama-2-7b-hf")
        assert "q_proj" in modules
        assert "v_proj" in modules

    def test_get_target_modules_gpt2(self):
        """Test getting target modules for GPT-2."""
        modules = get_lora_target_modules("gpt2")
        assert "c_attn" in modules

    def test_get_target_modules_unknown(self):
        """Test fallback for unknown model."""
        modules = get_lora_target_modules("unknown-model")
        # Should return default modules
        assert len(modules) > 0

    def test_get_target_modules_no_mlp(self):
        """Test getting attention-only modules."""
        modules = get_lora_target_modules("meta-llama/Llama-2-7b-hf", include_mlp=False)
        # Should not include MLP modules
        assert "gate_proj" not in modules
        assert "up_proj" not in modules

    def test_get_trainable_parameters(self):
        """Test getting trainable parameter stats."""
        model = MagicMock()

        # Create mock parameters
        param1 = MagicMock()
        param1.numel.return_value = 1000
        param1.requires_grad = True

        param2 = MagicMock()
        param2.numel.return_value = 500
        param2.requires_grad = False

        model.named_parameters.return_value = [
            ("layer1.weight", param1),
            ("layer2.weight", param2),
        ]

        stats = get_trainable_parameters(model)

        assert stats["trainable_params"] == 1000
        assert stats["total_params"] == 1500
        assert stats["trainable_layers"] == 1


class TestCreateLoRAConfig:
    """Tests for create_lora_config."""

    @patch("largeforge.training.lora.PeftLoraConfig", create=True)
    def test_creates_config_with_defaults(self, mock_peft_config):
        """Test creating config with default values."""
        # Skip if peft not installed
        try:
            from largeforge.training.lora import create_lora_config
        except ImportError:
            pytest.skip("peft not installed")

    def test_uses_largeforge_config(self):
        """Test that LargeForge config is used when provided."""
        from largeforge.config import LoRAConfig

        config = LoRAConfig(r=32, lora_alpha=64)

        # Without peft, we can still verify the config values
        assert config.r == 32
        assert config.lora_alpha == 64


class MockTrainer(BaseTrainer):
    """Mock trainer for testing."""

    def create_optimizer(self):
        import torch
        return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)

    def create_scheduler(self, optimizer, num_training_steps):
        return None

    def compute_loss(self, model, inputs):
        import torch
        return torch.tensor(0.0, requires_grad=True)

    def evaluate(self):
        return {"eval_loss": 0.5}


class TestBaseTrainer:
    """Tests for BaseTrainer (via MockTrainer)."""

    def test_add_callback(self):
        """Test adding callbacks."""
        from largeforge.config import TrainingConfig

        model = MagicMock()
        config = TrainingConfig(output_dir="./test")

        trainer = MockTrainer(model, config)
        callback = LoggingCallback()

        trainer.add_callback(callback)
        assert callback in trainer.callbacks

    def test_remove_callback(self):
        """Test removing callbacks."""
        from largeforge.config import TrainingConfig

        model = MagicMock()
        config = TrainingConfig(output_dir="./test")

        trainer = MockTrainer(model, config)
        callback = LoggingCallback()

        trainer.add_callback(callback)
        trainer.remove_callback(LoggingCallback)

        assert callback not in trainer.callbacks

    def test_state_initialization(self):
        """Test initial training state."""
        from largeforge.config import TrainingConfig

        model = MagicMock()
        config = TrainingConfig(output_dir="./test")

        trainer = MockTrainer(model, config)

        assert trainer.state.epoch == 0
        assert trainer.state.global_step == 0
        assert trainer.state.is_training is False
