"""Enhanced progress tracking for training jobs."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from largeforge.training.base import BaseTrainer, TrainingCallback, TrainingState
from largeforge.utils import get_logger, ensure_dir

logger = get_logger(__name__)


@dataclass
class ProgressTracker:
    """Tracks detailed training progress."""

    epoch: int = 0
    total_epochs: int = 1
    step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    start_time: float = field(default_factory=time.time)
    metrics: Dict[str, float] = field(default_factory=dict)
    _last_step_time: float = field(default=0.0, repr=False)
    _steps_per_second: float = field(default=0.0, repr=False)

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time

    @property
    def eta_seconds(self) -> float:
        """Estimate time remaining in seconds."""
        if self.step == 0 or self.total_steps == 0:
            return 0.0

        elapsed = self.elapsed_seconds
        steps_remaining = self.total_steps - self.step
        steps_per_second = self.step / elapsed if elapsed > 0 else 0

        if steps_per_second > 0:
            return steps_remaining / steps_per_second
        return 0.0

    @property
    def progress_pct(self) -> float:
        """Get progress percentage (0-100)."""
        if self.total_steps == 0:
            return 0.0
        return (self.step / self.total_steps) * 100

    @property
    def epoch_progress_pct(self) -> float:
        """Get current epoch progress percentage."""
        if self.total_epochs == 0:
            return 0.0
        steps_per_epoch = self.total_steps / self.total_epochs
        if steps_per_epoch == 0:
            return 0.0
        epoch_step = self.step % steps_per_epoch
        return (epoch_step / steps_per_epoch) * 100

    @property
    def steps_per_second(self) -> float:
        """Get average steps per second."""
        elapsed = self.elapsed_seconds
        if elapsed > 0 and self.step > 0:
            return self.step / elapsed
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert tracker to dictionary."""
        return {
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "step": self.step,
            "total_steps": self.total_steps,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "elapsed_seconds": self.elapsed_seconds,
            "eta_seconds": self.eta_seconds,
            "progress_pct": self.progress_pct,
            "steps_per_second": self.steps_per_second,
            "metrics": self.metrics,
        }

    def update_step(
        self,
        step: int,
        loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update progress with step information."""
        self.step = step
        if loss is not None:
            self.loss = loss
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if metrics:
            self.metrics.update(metrics)

    def update_epoch(self, epoch: int) -> None:
        """Update epoch information."""
        self.epoch = epoch


class ProgressCallback(TrainingCallback):
    """Callback for enhanced progress tracking with optional external notification."""

    def __init__(
        self,
        callback_fn: Optional[Callable[[ProgressTracker], None]] = None,
        update_interval: int = 1,
    ):
        """
        Initialize progress callback.

        Args:
            callback_fn: Optional function called with tracker on updates
            update_interval: Update callback every N steps
        """
        self.callback_fn = callback_fn
        self.update_interval = update_interval
        self.tracker = ProgressTracker()

    def get_progress(self) -> ProgressTracker:
        """Get current progress tracker."""
        return self.tracker

    def on_train_begin(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        """Initialize tracker at training start."""
        self.tracker = ProgressTracker(
            epoch=0,
            total_epochs=trainer.config.num_train_epochs,
            step=0,
            total_steps=state.total_steps,
            start_time=time.time(),
        )
        logger.debug(f"Progress tracking started. Total steps: {state.total_steps}")

    def on_train_end(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        """Finalize tracking at training end."""
        self.tracker.step = state.global_step
        logger.debug(
            f"Training complete. Elapsed: {self.tracker.elapsed_seconds:.2f}s"
        )
        self._notify()

    def on_step_end(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        """Update tracker after each step."""
        self.tracker.step = state.global_step
        self.tracker.learning_rate = trainer._get_learning_rate()

        if state.global_step % self.update_interval == 0:
            self._notify()

    def on_epoch_end(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        """Update epoch information."""
        self.tracker.epoch = state.epoch + 1
        self._notify()

    def on_log(
        self, trainer: BaseTrainer, state: TrainingState, logs: Dict[str, Any], **kwargs
    ) -> None:
        """Capture logged metrics."""
        if "loss" in logs:
            self.tracker.loss = logs["loss"]
        if "learning_rate" in logs:
            self.tracker.learning_rate = logs["learning_rate"]

        # Capture additional metrics
        for key, value in logs.items():
            if key not in ("loss", "learning_rate", "epoch", "step"):
                if isinstance(value, (int, float)):
                    self.tracker.metrics[key] = value

        self._notify()

    def _notify(self) -> None:
        """Call the callback function if set."""
        if self.callback_fn is not None:
            try:
                self.callback_fn(self.tracker)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")


class FileProgressCallback(ProgressCallback):
    """Progress callback that writes to a JSON file."""

    def __init__(
        self,
        output_path: str,
        update_interval: int = 10,
        callback_fn: Optional[Callable[[ProgressTracker], None]] = None,
    ):
        """
        Initialize file progress callback.

        Args:
            output_path: Path to output directory
            update_interval: Update file every N steps
            callback_fn: Optional additional callback
        """
        super().__init__(callback_fn=callback_fn, update_interval=update_interval)
        self.output_path = Path(output_path)
        self.progress_file = self.output_path / "progress.json"

    def on_train_begin(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        """Initialize and create output directory."""
        super().on_train_begin(trainer, state, **kwargs)
        ensure_dir(self.output_path)
        self._write_progress()

    def _notify(self) -> None:
        """Write progress to file and call callback."""
        self._write_progress()
        super()._notify()

    def _write_progress(self) -> None:
        """Write current progress to JSON file."""
        try:
            with open(self.progress_file, "w") as f:
                json.dump(self.tracker.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write progress file: {e}")


class AsyncProgressCallback(ProgressCallback):
    """Progress callback with async notification support."""

    def __init__(
        self,
        async_callback_fn: Optional[Callable[[ProgressTracker], None]] = None,
        update_interval: int = 1,
    ):
        """
        Initialize async progress callback.

        Args:
            async_callback_fn: Async function called with tracker on updates
            update_interval: Update callback every N steps
        """
        super().__init__(callback_fn=None, update_interval=update_interval)
        self.async_callback_fn = async_callback_fn
        self._loop = None

    def _notify(self) -> None:
        """Notify via async callback."""
        if self.async_callback_fn is None:
            return

        try:
            import asyncio

            # Get or create event loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Schedule the async callback
            if asyncio.iscoroutinefunction(self.async_callback_fn):
                loop.create_task(self.async_callback_fn(self.tracker))
            else:
                self.async_callback_fn(self.tracker)

        except Exception as e:
            logger.warning(f"Async progress callback error: {e}")
