"""Training callback for experiment tracking."""

from typing import Any, Dict, Optional

from largeforge.experiments.models import ExperimentStatus
from largeforge.experiments.tracker import ExperimentTracker
from largeforge.training.callbacks import TrainingCallback
from largeforge.utils import get_logger

logger = get_logger(__name__)


class ExperimentCallback(TrainingCallback):
    """
    Training callback that logs metrics and artifacts to the experiment tracker.

    This callback integrates with the training loop to automatically track:
    - Training metrics (loss, learning rate, etc.)
    - Evaluation metrics
    - Checkpoints as artifacts
    - Training completion status
    """

    def __init__(
        self,
        tracker: ExperimentTracker,
        experiment_id: str,
        log_interval: int = 10,
        log_lr: bool = True,
    ):
        """
        Initialize the experiment callback.

        Args:
            tracker: Experiment tracker instance
            experiment_id: ID of the experiment to track
            log_interval: Log metrics every N steps
            log_lr: Whether to log learning rate
        """
        self.tracker = tracker
        self.experiment_id = experiment_id
        self.log_interval = log_interval
        self.log_lr = log_lr
        self._step = 0

    def on_train_begin(self, trainer: Any, state: Any, **kwargs) -> None:
        """Called at the beginning of training."""
        logger.info(f"Starting experiment tracking for {self.experiment_id}")

        # Log initial configuration
        if hasattr(trainer, "args"):
            args = trainer.args
            config = {
                "learning_rate": args.learning_rate,
                "num_epochs": args.num_train_epochs,
                "batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "warmup_ratio": args.warmup_ratio,
                "fp16": args.fp16,
                "bf16": args.bf16,
            }

            exp = self.tracker.get(self.experiment_id)
            if exp:
                exp.config.update(config)
                self.tracker._save_experiment(exp)

    def on_step_end(self, trainer: Any, state: Any, **kwargs) -> None:
        """Called at the end of each training step."""
        self._step = state.global_step

        # Only log at intervals
        if self._step % self.log_interval != 0:
            return

        # Collect metrics
        metrics: Dict[str, float] = {}

        # Get loss from state
        if hasattr(state, "log_history") and state.log_history:
            last_log = state.log_history[-1]
            if "loss" in last_log:
                metrics["loss"] = last_log["loss"]
            if self.log_lr and "learning_rate" in last_log:
                metrics["learning_rate"] = last_log["learning_rate"]

        # Log metrics
        if metrics:
            self.tracker.log_metrics(self.experiment_id, metrics, self._step)

    def on_epoch_end(self, trainer: Any, state: Any, **kwargs) -> None:
        """Called at the end of each epoch."""
        epoch = state.epoch

        # Log epoch-level metrics
        if hasattr(state, "log_history") and state.log_history:
            # Find metrics from this epoch
            epoch_metrics = {}
            for log in reversed(state.log_history):
                if log.get("epoch", 0) <= epoch:
                    if "eval_loss" in log:
                        epoch_metrics["eval_loss"] = log["eval_loss"]
                    if "train_loss" in log:
                        epoch_metrics["train_loss"] = log["train_loss"]
                    break

            if epoch_metrics:
                self.tracker.log_metrics(
                    self.experiment_id,
                    {f"epoch_{k}": v for k, v in epoch_metrics.items()},
                    self._step,
                )

    def on_evaluate(self, trainer: Any, state: Any, metrics: Optional[Dict] = None, **kwargs) -> None:
        """Called after evaluation."""
        if not metrics:
            return

        eval_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                # Clean up metric names
                clean_key = key.replace("eval_", "")
                eval_metrics[f"eval_{clean_key}"] = value

        if eval_metrics:
            self.tracker.log_metrics(self.experiment_id, eval_metrics, self._step)

    def on_save(self, trainer: Any, state: Any, **kwargs) -> None:
        """Called when a checkpoint is saved."""
        checkpoint_dir = kwargs.get("checkpoint_folder") or kwargs.get("output_dir")

        if checkpoint_dir:
            self.tracker.log_artifact(
                self.experiment_id,
                name=f"checkpoint-{state.global_step}",
                path=str(checkpoint_dir),
                artifact_type="checkpoint",
                metadata={
                    "step": state.global_step,
                    "epoch": state.epoch,
                },
            )

    def on_log(self, trainer: Any, state: Any, logs: Optional[Dict] = None, **kwargs) -> None:
        """Called when logs are updated."""
        if not logs:
            return

        # Log any additional metrics
        extra_metrics = {}
        for key, value in logs.items():
            if isinstance(value, (int, float)) and key not in ("loss", "learning_rate", "epoch"):
                extra_metrics[key] = value

        if extra_metrics:
            self.tracker.log_metrics(self.experiment_id, extra_metrics, self._step)

    def on_train_end(self, trainer: Any, state: Any, **kwargs) -> None:
        """Called at the end of training."""
        logger.info(f"Training completed for experiment {self.experiment_id}")

        # Mark experiment as completed
        self.tracker.update_status(self.experiment_id, ExperimentStatus.COMPLETED)

        # Log final model as artifact
        output_dir = getattr(trainer.args, "output_dir", None)
        if output_dir:
            self.tracker.log_artifact(
                self.experiment_id,
                name="final_model",
                path=str(output_dir),
                artifact_type="model",
                metadata={
                    "final_step": state.global_step,
                    "total_epochs": state.epoch,
                },
            )

    def on_error(self, trainer: Any, state: Any, error: Exception, **kwargs) -> None:
        """Called when training encounters an error."""
        logger.error(f"Training error in experiment {self.experiment_id}: {error}")
        self.tracker.update_status(
            self.experiment_id,
            ExperimentStatus.FAILED,
            error=str(error),
        )


def create_experiment_callback(
    experiment_name: str,
    tracker: Optional[ExperimentTracker] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
    job_id: Optional[str] = None,
    log_interval: int = 10,
) -> tuple:
    """
    Create an experiment and its tracking callback.

    Args:
        experiment_name: Name for the experiment
        tracker: Experiment tracker (uses global if None)
        config: Training configuration
        tags: Tags for filtering
        job_id: Associated job ID
        log_interval: Logging interval

    Returns:
        Tuple of (experiment, callback)
    """
    from largeforge.experiments.tracker import get_tracker

    if tracker is None:
        tracker = get_tracker()

    experiment = tracker.create(
        name=experiment_name,
        config=config,
        tags=tags,
        job_id=job_id,
    )

    callback = ExperimentCallback(
        tracker=tracker,
        experiment_id=experiment.id,
        log_interval=log_interval,
    )

    return experiment, callback
