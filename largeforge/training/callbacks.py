"""Training callbacks for LargeForgeAI."""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from largeforge.training.base import BaseTrainer, TrainingCallback, TrainingState
from largeforge.utils import get_logger, ensure_dir

logger = get_logger(__name__)


class LoggingCallback(TrainingCallback):
    """Callback for logging training progress."""

    def __init__(
        self,
        log_every_n_steps: int = 10,
        log_to_file: bool = False,
        log_file_path: Optional[str] = None,
    ):
        """
        Initialize the logging callback.

        Args:
            log_every_n_steps: Log every N steps
            log_to_file: Whether to write logs to file
            log_file_path: Path to log file
        """
        self.log_every_n_steps = log_every_n_steps
        self.log_to_file = log_to_file
        self.log_file_path = log_file_path
        self.start_time = None

    def on_train_begin(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        """Called at the beginning of training."""
        self.start_time = time.time()
        logger.info("=" * 60)
        logger.info("Training started")
        logger.info(f"Total steps: {state.total_steps}")
        logger.info(f"Epochs: {trainer.config.num_train_epochs}")
        logger.info("=" * 60)

    def on_train_end(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        """Called at the end of training."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        logger.info("=" * 60)
        logger.info("Training complete")
        logger.info(f"Total time: {elapsed:.2f}s")
        logger.info(f"Final step: {state.global_step}")
        logger.info("=" * 60)

    def on_log(
        self, trainer: BaseTrainer, state: TrainingState, logs: Dict[str, Any], **kwargs
    ) -> None:
        """Called when logging metrics."""
        if self.log_to_file and self.log_file_path:
            ensure_dir(Path(self.log_file_path).parent)
            with open(self.log_file_path, "a") as f:
                f.write(json.dumps(logs) + "\n")

    def on_epoch_end(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        """Called at the end of each epoch."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        logger.info(f"Epoch {state.epoch + 1} complete. Total time: {elapsed:.2f}s")


class CheckpointCallback(TrainingCallback):
    """Callback for saving checkpoints."""

    def __init__(
        self,
        save_total_limit: int = 3,
        save_best_only: bool = False,
        metric_for_best: str = "eval_loss",
        greater_is_better: bool = False,
    ):
        """
        Initialize the checkpoint callback.

        Args:
            save_total_limit: Maximum number of checkpoints to keep
            save_best_only: Only save when metric improves
            metric_for_best: Metric to use for best model selection
            greater_is_better: Whether higher metric values are better
        """
        self.save_total_limit = save_total_limit
        self.save_best_only = save_best_only
        self.metric_for_best = metric_for_best
        self.greater_is_better = greater_is_better
        self.saved_checkpoints = []
        self.best_metric = None

    def on_save(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        """Called when saving a checkpoint."""
        output_dir = Path(trainer.config.output_dir)
        checkpoint_dir = output_dir / f"checkpoint-{state.global_step}"

        self.saved_checkpoints.append(str(checkpoint_dir))

        # Remove old checkpoints if over limit
        if len(self.saved_checkpoints) > self.save_total_limit:
            old_checkpoint = self.saved_checkpoints.pop(0)
            self._remove_checkpoint(old_checkpoint)

    def on_evaluate(
        self, trainer: BaseTrainer, state: TrainingState, metrics: Dict[str, float], **kwargs
    ) -> None:
        """Called after evaluation."""
        if not self.save_best_only:
            return

        current_metric = metrics.get(self.metric_for_best)
        if current_metric is None:
            return

        is_better = False
        if self.best_metric is None:
            is_better = True
        elif self.greater_is_better:
            is_better = current_metric > self.best_metric
        else:
            is_better = current_metric < self.best_metric

        if is_better:
            self.best_metric = current_metric
            state.best_metric = current_metric

            # Save best model
            best_path = Path(trainer.config.output_dir) / "best_model"
            trainer.save_checkpoint(str(best_path))
            state.best_model_checkpoint = str(best_path)

            logger.info(
                f"New best model! {self.metric_for_best}={current_metric:.4f}"
            )

    def _remove_checkpoint(self, checkpoint_path: str) -> None:
        """Remove a checkpoint directory."""
        import shutil
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path)
            logger.info(f"Removed old checkpoint: {checkpoint_path}")


class EarlyStoppingCallback(TrainingCallback):
    """Callback for early stopping based on validation metrics."""

    def __init__(
        self,
        patience: int = 3,
        metric: str = "eval_loss",
        greater_is_better: bool = False,
        min_delta: float = 0.0,
    ):
        """
        Initialize early stopping callback.

        Args:
            patience: Number of evaluations to wait for improvement
            metric: Metric to monitor
            greater_is_better: Whether higher metric values are better
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.metric = metric
        self.greater_is_better = greater_is_better
        self.min_delta = min_delta

        self.best_metric = None
        self.patience_counter = 0

    def on_evaluate(
        self, trainer: BaseTrainer, state: TrainingState, metrics: Dict[str, float], **kwargs
    ) -> None:
        """Check if training should stop."""
        current_metric = metrics.get(self.metric)
        if current_metric is None:
            return

        is_better = False
        if self.best_metric is None:
            is_better = True
        elif self.greater_is_better:
            is_better = current_metric > self.best_metric + self.min_delta
        else:
            is_better = current_metric < self.best_metric - self.min_delta

        if is_better:
            self.best_metric = current_metric
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            logger.info(
                f"No improvement in {self.metric}. "
                f"Patience: {self.patience_counter}/{self.patience}"
            )

        if self.patience_counter >= self.patience:
            logger.info(f"Early stopping triggered after {self.patience} evaluations")
            state.should_stop = True


class WandBCallback(TrainingCallback):
    """Callback for Weights & Biases logging."""

    def __init__(
        self,
        project: str = "largeforge",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        log_model: bool = False,
    ):
        """
        Initialize W&B callback.

        Args:
            project: W&B project name
            name: Run name
            config: Configuration to log
            log_model: Whether to log model checkpoints
        """
        self.project = project
        self.name = name
        self.config = config
        self.log_model = log_model
        self.run = None

    def on_train_begin(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        """Initialize W&B run."""
        try:
            import wandb
        except ImportError:
            logger.warning("wandb not installed. Skipping W&B logging.")
            return

        config = self.config or {}
        config.update(trainer.config.to_dict())

        self.run = wandb.init(
            project=self.project,
            name=self.name,
            config=config,
            resume="allow",
        )

        logger.info(f"W&B run initialized: {wandb.run.url}")

    def on_train_end(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        """Finish W&B run."""
        if self.run is not None:
            try:
                import wandb
                wandb.finish()
            except ImportError:
                pass

    def on_log(
        self, trainer: BaseTrainer, state: TrainingState, logs: Dict[str, Any], **kwargs
    ) -> None:
        """Log metrics to W&B."""
        if self.run is None:
            return

        try:
            import wandb
            wandb.log(logs, step=state.global_step)
        except ImportError:
            pass

    def on_save(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        """Log model artifact to W&B."""
        if not self.log_model or self.run is None:
            return

        try:
            import wandb

            artifact = wandb.Artifact(
                name=f"model-{state.global_step}",
                type="model",
            )
            output_dir = Path(trainer.config.output_dir) / f"checkpoint-{state.global_step}"
            artifact.add_dir(str(output_dir))
            wandb.log_artifact(artifact)
        except ImportError:
            pass


class TensorBoardCallback(TrainingCallback):
    """Callback for TensorBoard logging."""

    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize TensorBoard callback.

        Args:
            log_dir: Directory for TensorBoard logs
        """
        self.log_dir = log_dir
        self.writer = None

    def on_train_begin(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            logger.warning("tensorboard not installed. Skipping TB logging.")
            return

        log_dir = self.log_dir or Path(trainer.config.output_dir) / "runs"
        ensure_dir(log_dir)

        self.writer = SummaryWriter(log_dir=str(log_dir))
        logger.info(f"TensorBoard logging to {log_dir}")

    def on_train_end(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()

    def on_log(
        self, trainer: BaseTrainer, state: TrainingState, logs: Dict[str, Any], **kwargs
    ) -> None:
        """Log metrics to TensorBoard."""
        if self.writer is None:
            return

        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, state.global_step)


class GradientAccumulationCallback(TrainingCallback):
    """Callback for gradient accumulation status logging."""

    def __init__(self, accumulation_steps: int = 1):
        """
        Initialize gradient accumulation callback.

        Args:
            accumulation_steps: Number of steps to accumulate
        """
        self.accumulation_steps = accumulation_steps
        self.accumulated_loss = 0.0
        self.accumulation_count = 0

    def on_step_end(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        """Track accumulated gradients."""
        self.accumulation_count += 1

        if self.accumulation_count >= self.accumulation_steps:
            effective_batch_size = (
                trainer.config.per_device_train_batch_size * self.accumulation_steps
            )
            logger.debug(f"Gradient step with effective batch size: {effective_batch_size}")
            self.accumulation_count = 0
