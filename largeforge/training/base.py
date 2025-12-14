"""Base trainer class for LargeForgeAI."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset

from largeforge.config import TrainingConfig
from largeforge.utils import get_logger, ensure_dir

logger = get_logger(__name__)


@dataclass
class TrainingState:
    """Tracks the state of training."""

    epoch: int = 0
    global_step: int = 0
    total_steps: int = 0
    best_metric: Optional[float] = None
    best_model_checkpoint: Optional[str] = None
    log_history: List[Dict[str, Any]] = field(default_factory=list)
    is_training: bool = False
    should_stop: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "total_steps": self.total_steps,
            "best_metric": self.best_metric,
            "best_model_checkpoint": self.best_model_checkpoint,
            "log_history": self.log_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingState":
        """Create state from dictionary."""
        return cls(
            epoch=data.get("epoch", 0),
            global_step=data.get("global_step", 0),
            total_steps=data.get("total_steps", 0),
            best_metric=data.get("best_metric"),
            best_model_checkpoint=data.get("best_model_checkpoint"),
            log_history=data.get("log_history", []),
        )


class TrainingCallback(ABC):
    """Base class for training callbacks."""

    def on_train_begin(
        self, trainer: "BaseTrainer", state: TrainingState, **kwargs
    ) -> None:
        """Called at the beginning of training."""
        pass

    def on_train_end(
        self, trainer: "BaseTrainer", state: TrainingState, **kwargs
    ) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(
        self, trainer: "BaseTrainer", state: TrainingState, **kwargs
    ) -> None:
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(
        self, trainer: "BaseTrainer", state: TrainingState, **kwargs
    ) -> None:
        """Called at the end of each epoch."""
        pass

    def on_step_begin(
        self, trainer: "BaseTrainer", state: TrainingState, **kwargs
    ) -> None:
        """Called at the beginning of each training step."""
        pass

    def on_step_end(
        self, trainer: "BaseTrainer", state: TrainingState, **kwargs
    ) -> None:
        """Called at the end of each training step."""
        pass

    def on_log(
        self, trainer: "BaseTrainer", state: TrainingState, logs: Dict[str, Any], **kwargs
    ) -> None:
        """Called when logging metrics."""
        pass

    def on_save(
        self, trainer: "BaseTrainer", state: TrainingState, **kwargs
    ) -> None:
        """Called when saving a checkpoint."""
        pass

    def on_evaluate(
        self, trainer: "BaseTrainer", state: TrainingState, metrics: Dict[str, float], **kwargs
    ) -> None:
        """Called after evaluation."""
        pass


class BaseTrainer(ABC):
    """Abstract base class for trainers."""

    def __init__(
        self,
        model,
        config: TrainingConfig,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer=None,
        callbacks: Optional[List[TrainingCallback]] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: The model to train
            config: Training configuration
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            tokenizer: Tokenizer for the model
            callbacks: List of training callbacks
        """
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.callbacks = callbacks or []

        self.state = TrainingState()
        self.optimizer = None
        self.scheduler = None

        # Ensure output directory exists
        ensure_dir(config.output_dir)

    def add_callback(self, callback: TrainingCallback) -> None:
        """Add a callback to the trainer."""
        self.callbacks.append(callback)

    def remove_callback(self, callback_class: type) -> None:
        """Remove callbacks of a specific type."""
        self.callbacks = [
            cb for cb in self.callbacks if not isinstance(cb, callback_class)
        ]

    def _call_callbacks(self, event: str, **kwargs) -> None:
        """Call all callbacks for a specific event."""
        for callback in self.callbacks:
            method = getattr(callback, event, None)
            if method:
                method(self, self.state, **kwargs)

    @abstractmethod
    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create the optimizer."""
        pass

    @abstractmethod
    def create_scheduler(self, optimizer: torch.optim.Optimizer, num_training_steps: int):
        """Create the learning rate scheduler."""
        pass

    @abstractmethod
    def compute_loss(self, model, inputs: Dict[str, Any]) -> torch.Tensor:
        """Compute the training loss."""
        pass

    def get_train_dataloader(self) -> DataLoader:
        """Create the training dataloader."""
        if self.train_dataset is None:
            raise ValueError("No training dataset provided")

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.dataloader_pin_memory,
            drop_last=self.config.dataloader_drop_last,
        )

    def get_eval_dataloader(self) -> Optional[DataLoader]:
        """Create the evaluation dataloader."""
        if self.eval_dataset is None:
            return None

        return DataLoader(
            self.eval_dataset,
            batch_size=self.config.per_device_eval_batch_size,
            shuffle=False,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.dataloader_pin_memory,
        )

    def train(self) -> TrainingState:
        """
        Run the training loop.

        Returns:
            Final training state
        """
        logger.info("Starting training...")
        self.state.is_training = True
        self._call_callbacks("on_train_begin")

        train_dataloader = self.get_train_dataloader()
        num_training_steps = len(train_dataloader) * self.config.num_train_epochs
        self.state.total_steps = num_training_steps

        # Create optimizer and scheduler
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler(self.optimizer, num_training_steps)

        # Training loop
        for epoch in range(self.config.num_train_epochs):
            if self.state.should_stop:
                break

            self.state.epoch = epoch
            self._call_callbacks("on_epoch_begin")

            self._train_epoch(train_dataloader)

            self._call_callbacks("on_epoch_end")

            # Evaluation
            if self.eval_dataset is not None:
                metrics = self.evaluate()
                self._call_callbacks("on_evaluate", metrics=metrics)

            # Save checkpoint
            if self.config.save_strategy == "epoch":
                self.save_checkpoint()

        self.state.is_training = False
        self._call_callbacks("on_train_end")

        logger.info(f"Training complete. Total steps: {self.state.global_step}")
        return self.state

    def _train_epoch(self, dataloader: DataLoader) -> None:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for step, batch in enumerate(dataloader):
            if self.state.should_stop:
                break

            self._call_callbacks("on_step_begin")

            # Move batch to device
            batch = self._prepare_inputs(batch)

            # Forward pass
            loss = self.compute_loss(self.model, batch)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )

            # Optimizer step
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()

            # Update state
            self.state.global_step += 1
            total_loss += loss.item()

            self._call_callbacks("on_step_end")

            # Logging
            if self.state.global_step % self.config.logging_steps == 0:
                avg_loss = total_loss / (step + 1)
                logs = {
                    "loss": avg_loss,
                    "learning_rate": self._get_learning_rate(),
                    "epoch": self.state.epoch,
                    "step": self.state.global_step,
                }
                self._log(logs)

            # Save checkpoint
            if (
                self.config.save_strategy == "steps"
                and self.state.global_step % self.config.save_steps == 0
            ):
                self.save_checkpoint()

    def _prepare_inputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to the correct device."""
        device = next(self.model.parameters()).device
        prepared = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(device)
            else:
                prepared[key] = value
        return prepared

    def _get_learning_rate(self) -> float:
        """Get the current learning rate."""
        if self.scheduler is not None:
            return self.scheduler.get_last_lr()[0]
        return self.config.learning_rate

    def _log(self, logs: Dict[str, Any]) -> None:
        """Log metrics."""
        self.state.log_history.append(logs)
        self._call_callbacks("on_log", logs=logs)
        logger.info(f"Step {self.state.global_step}: {logs}")

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation and return metrics."""
        pass

    def save_checkpoint(self, output_dir: Optional[str] = None) -> str:
        """
        Save a checkpoint.

        Args:
            output_dir: Optional specific output directory

        Returns:
            Path to saved checkpoint
        """
        if output_dir is None:
            output_dir = Path(self.config.output_dir) / f"checkpoint-{self.state.global_step}"

        output_dir = Path(output_dir)
        ensure_dir(output_dir)

        logger.info(f"Saving checkpoint to {output_dir}")

        # Save model
        self.model.save_pretrained(output_dir)

        # Save tokenizer if available
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Save training state
        state_path = output_dir / "trainer_state.json"
        import json
        with open(state_path, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)

        self._call_callbacks("on_save")

        return str(output_dir)

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """
        Load a checkpoint.

        Args:
            checkpoint_dir: Path to checkpoint directory
        """
        checkpoint_dir = Path(checkpoint_dir)

        logger.info(f"Loading checkpoint from {checkpoint_dir}")

        # Load training state
        state_path = checkpoint_dir / "trainer_state.json"
        if state_path.exists():
            import json
            with open(state_path, "r") as f:
                state_dict = json.load(f)
            self.state = TrainingState.from_dict(state_dict)
