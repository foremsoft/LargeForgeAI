"""Callbacks for integrating training with web layer."""

import asyncio
from datetime import datetime
from typing import Any, Dict, Optional

from largeforge.training.base import BaseTrainer, TrainingCallback, TrainingState
from largeforge.training.events import EventEmitter, EventType, TrainingEvent, emit_event
from largeforge.training.progress import ProgressTracker
from largeforge.utils import get_logger
from largeforge.web.schemas import TrainingProgress, WebSocketMessage
from largeforge.web.state import JobStateManager

logger = get_logger(__name__)


class WebSocketBroadcastCallback(TrainingCallback):
    """Callback that broadcasts training progress via WebSocket."""

    def __init__(
        self,
        connection_manager,
        job_id: str,
        broadcast_interval: int = 10,
    ):
        """
        Initialize WebSocket broadcast callback.

        Args:
            connection_manager: WebSocket ConnectionManager
            job_id: Training job ID
            broadcast_interval: Broadcast every N steps
        """
        self.connection_manager = connection_manager
        self.job_id = job_id
        self.broadcast_interval = broadcast_interval
        self._start_time: Optional[float] = None
        self._last_loss: float = 0.0

    def on_train_begin(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        """Broadcast training started."""
        import time
        self._start_time = time.time()

        self._broadcast_status("running")
        logger.debug(f"WebSocket broadcast: training started for job {self.job_id}")

    def on_train_end(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        """Broadcast training completed."""
        status = "cancelled" if state.should_stop else "completed"
        self._broadcast_status(status)

    def on_step_end(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        """Broadcast progress after each step."""
        if state.global_step % self.broadcast_interval != 0:
            return

        import time
        elapsed = time.time() - self._start_time if self._start_time else 0

        progress = TrainingProgress(
            epoch=state.epoch,
            total_epochs=trainer.config.num_train_epochs,
            step=state.global_step,
            total_steps=state.total_steps,
            loss=self._last_loss,
            learning_rate=trainer._get_learning_rate(),
            elapsed_seconds=elapsed,
            eta_seconds=self._calculate_eta(state, elapsed),
        )

        self._broadcast_progress(progress)

    def on_epoch_end(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        """Broadcast epoch completion."""
        import time
        elapsed = time.time() - self._start_time if self._start_time else 0

        progress = TrainingProgress(
            epoch=state.epoch + 1,
            total_epochs=trainer.config.num_train_epochs,
            step=state.global_step,
            total_steps=state.total_steps,
            loss=self._last_loss,
            learning_rate=trainer._get_learning_rate(),
            elapsed_seconds=elapsed,
            eta_seconds=self._calculate_eta(state, elapsed),
        )

        self._broadcast_progress(progress)
        logger.debug(f"Epoch {state.epoch + 1} completed for job {self.job_id}")

    def on_log(
        self, trainer: BaseTrainer, state: TrainingState, logs: Dict[str, Any], **kwargs
    ) -> None:
        """Capture and broadcast log entries."""
        if "loss" in logs:
            self._last_loss = logs["loss"]

        # Broadcast log line
        log_line = f"Step {state.global_step}: loss={logs.get('loss', 0):.4f}, lr={logs.get('learning_rate', 0):.2e}"
        self._broadcast_log(log_line, "info")

    def on_save(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        """Broadcast checkpoint saved."""
        self._broadcast_log(f"Checkpoint saved at step {state.global_step}", "info")

    def _calculate_eta(self, state: TrainingState, elapsed: float) -> float:
        """Calculate estimated time remaining."""
        if state.global_step == 0 or state.total_steps == 0:
            return 0.0
        steps_remaining = state.total_steps - state.global_step
        steps_per_second = state.global_step / elapsed if elapsed > 0 else 0
        return steps_remaining / steps_per_second if steps_per_second > 0 else 0.0

    def _broadcast_progress(self, progress: TrainingProgress) -> None:
        """Broadcast progress via WebSocket."""
        if not self.connection_manager:
            return

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(
                    self.connection_manager.broadcast_progress(self.job_id, progress)
                )
            else:
                loop.run_until_complete(
                    self.connection_manager.broadcast_progress(self.job_id, progress)
                )
        except RuntimeError:
            # No event loop available
            pass
        except Exception as e:
            logger.warning(f"Failed to broadcast progress: {e}")

    def _broadcast_status(self, status: str, error: Optional[str] = None) -> None:
        """Broadcast status change via WebSocket."""
        if not self.connection_manager:
            return

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(
                    self.connection_manager.broadcast_status(self.job_id, status, error)
                )
            else:
                loop.run_until_complete(
                    self.connection_manager.broadcast_status(self.job_id, status, error)
                )
        except RuntimeError:
            pass
        except Exception as e:
            logger.warning(f"Failed to broadcast status: {e}")

    def _broadcast_log(self, log_line: str, level: str = "info") -> None:
        """Broadcast log line via WebSocket."""
        if not self.connection_manager:
            return

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(
                    self.connection_manager.broadcast_log(self.job_id, log_line, level)
                )
            else:
                loop.run_until_complete(
                    self.connection_manager.broadcast_log(self.job_id, log_line, level)
                )
        except RuntimeError:
            pass
        except Exception as e:
            logger.warning(f"Failed to broadcast log: {e}")


class StateUpdateCallback(TrainingCallback):
    """Callback that updates job state in the state manager."""

    def __init__(
        self,
        state_manager: JobStateManager,
        job_id: str,
        update_interval: int = 10,
    ):
        """
        Initialize state update callback.

        Args:
            state_manager: Job state manager
            job_id: Training job ID
            update_interval: Update state every N steps
        """
        self.state_manager = state_manager
        self.job_id = job_id
        self.update_interval = update_interval
        self._start_time: Optional[float] = None
        self._last_loss: float = 0.0
        self._metrics: Dict[str, float] = {}

    def on_train_begin(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        """Mark job as running."""
        import time
        self._start_time = time.time()
        self.state_manager.set_running(self.job_id)

    def on_train_end(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        """Mark job as completed or cancelled."""
        if state.should_stop:
            self.state_manager.cancel_job(self.job_id)
        # Note: completion is handled by orchestrator after saving

    def on_step_end(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        """Update progress in state."""
        if state.global_step % self.update_interval != 0:
            return

        self._update_progress(trainer, state)

    def on_log(
        self, trainer: BaseTrainer, state: TrainingState, logs: Dict[str, Any], **kwargs
    ) -> None:
        """Capture metrics from logs."""
        if "loss" in logs:
            self._last_loss = logs["loss"]

        for key, value in logs.items():
            if isinstance(value, (int, float)) and key not in ("step", "epoch"):
                self._metrics[key] = value

        self._update_progress(trainer, state)

    def on_save(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        """Record checkpoint saved."""
        self._update_progress(trainer, state)

    def _update_progress(self, trainer: BaseTrainer, state: TrainingState) -> None:
        """Update progress in state manager."""
        import time
        elapsed = time.time() - self._start_time if self._start_time else 0

        progress = TrainingProgress(
            epoch=state.epoch,
            total_epochs=trainer.config.num_train_epochs,
            step=state.global_step,
            total_steps=state.total_steps,
            loss=self._last_loss,
            learning_rate=trainer._get_learning_rate(),
            elapsed_seconds=elapsed,
            eta_seconds=self._calculate_eta(state, elapsed),
            metrics=self._metrics.copy(),
        )

        self.state_manager.update_progress(self.job_id, progress)

    def _calculate_eta(self, state: TrainingState, elapsed: float) -> float:
        """Calculate estimated time remaining."""
        if state.global_step == 0 or state.total_steps == 0:
            return 0.0
        steps_remaining = state.total_steps - state.global_step
        steps_per_second = state.global_step / elapsed if elapsed > 0 else 0
        return steps_remaining / steps_per_second if steps_per_second > 0 else 0.0


class EventEmitterCallback(TrainingCallback):
    """Callback that emits training events for pub/sub."""

    def __init__(self, job_id: str, emit_step_events: bool = False):
        """
        Initialize event emitter callback.

        Args:
            job_id: Training job ID
            emit_step_events: Whether to emit events for every step
        """
        self.job_id = job_id
        self.emit_step_events = emit_step_events

    def on_train_begin(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        """Emit job started event."""
        emit_event(
            EventType.JOB_STARTED,
            self.job_id,
            {
                "total_steps": state.total_steps,
                "epochs": trainer.config.num_train_epochs,
            }
        )

    def on_train_end(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        """Emit job completed/cancelled event."""
        if state.should_stop:
            emit_event(EventType.JOB_CANCELLED, self.job_id)
        else:
            emit_event(
                EventType.JOB_COMPLETED,
                self.job_id,
                {"final_step": state.global_step}
            )

    def on_epoch_begin(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        """Emit epoch start event."""
        emit_event(
            EventType.EPOCH_START,
            self.job_id,
            {"epoch": state.epoch, "total_epochs": trainer.config.num_train_epochs}
        )

    def on_epoch_end(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        """Emit epoch end event."""
        emit_event(
            EventType.EPOCH_END,
            self.job_id,
            {"epoch": state.epoch + 1, "total_epochs": trainer.config.num_train_epochs}
        )

    def on_step_end(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        """Emit step complete event (if enabled)."""
        if not self.emit_step_events:
            return

        emit_event(
            EventType.STEP_COMPLETE,
            self.job_id,
            {
                "step": state.global_step,
                "total_steps": state.total_steps,
            }
        )

    def on_log(
        self, trainer: BaseTrainer, state: TrainingState, logs: Dict[str, Any], **kwargs
    ) -> None:
        """Emit metrics logged event."""
        emit_event(
            EventType.METRICS_LOGGED,
            self.job_id,
            {"metrics": logs}
        )

    def on_save(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        """Emit checkpoint saved event."""
        emit_event(
            EventType.CHECKPOINT_SAVED,
            self.job_id,
            {"step": state.global_step}
        )

    def on_evaluate(
        self, trainer: BaseTrainer, state: TrainingState, metrics: Dict[str, float], **kwargs
    ) -> None:
        """Emit evaluation complete event."""
        emit_event(
            EventType.EVALUATION_COMPLETE,
            self.job_id,
            {"step": state.global_step, "metrics": metrics}
        )


class CompositeCallback(TrainingCallback):
    """Combines multiple callbacks into one for easier management."""

    def __init__(self, callbacks: list):
        """
        Initialize composite callback.

        Args:
            callbacks: List of callbacks to combine
        """
        self.callbacks = callbacks

    def add_callback(self, callback: TrainingCallback) -> None:
        """Add a callback."""
        self.callbacks.append(callback)

    def on_train_begin(self, trainer: BaseTrainer, state: TrainingState, **kwargs) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(trainer, state, **kwargs)

    def on_train_end(self, trainer: BaseTrainer, state: TrainingState, **kwargs) -> None:
        for cb in self.callbacks:
            cb.on_train_end(trainer, state, **kwargs)

    def on_epoch_begin(self, trainer: BaseTrainer, state: TrainingState, **kwargs) -> None:
        for cb in self.callbacks:
            cb.on_epoch_begin(trainer, state, **kwargs)

    def on_epoch_end(self, trainer: BaseTrainer, state: TrainingState, **kwargs) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(trainer, state, **kwargs)

    def on_step_begin(self, trainer: BaseTrainer, state: TrainingState, **kwargs) -> None:
        for cb in self.callbacks:
            cb.on_step_begin(trainer, state, **kwargs)

    def on_step_end(self, trainer: BaseTrainer, state: TrainingState, **kwargs) -> None:
        for cb in self.callbacks:
            cb.on_step_end(trainer, state, **kwargs)

    def on_log(self, trainer: BaseTrainer, state: TrainingState, logs: Dict[str, Any], **kwargs) -> None:
        for cb in self.callbacks:
            cb.on_log(trainer, state, logs, **kwargs)

    def on_save(self, trainer: BaseTrainer, state: TrainingState, **kwargs) -> None:
        for cb in self.callbacks:
            cb.on_save(trainer, state, **kwargs)

    def on_evaluate(self, trainer: BaseTrainer, state: TrainingState, metrics: Dict[str, float], **kwargs) -> None:
        for cb in self.callbacks:
            cb.on_evaluate(trainer, state, metrics, **kwargs)


def create_web_callbacks(
    job_id: str,
    state_manager: JobStateManager,
    connection_manager=None,
    update_interval: int = 10,
) -> CompositeCallback:
    """
    Create standard web integration callbacks.

    Args:
        job_id: Training job ID
        state_manager: Job state manager
        connection_manager: Optional WebSocket connection manager
        update_interval: Update interval for progress

    Returns:
        CompositeCallback with all web callbacks
    """
    callbacks = [
        StateUpdateCallback(state_manager, job_id, update_interval),
        EventEmitterCallback(job_id),
    ]

    if connection_manager:
        callbacks.append(
            WebSocketBroadcastCallback(connection_manager, job_id, update_interval)
        )

    return CompositeCallback(callbacks)
