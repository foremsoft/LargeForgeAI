"""Training callback for cost tracking."""

from typing import Any, Optional

from largeforge.costs.tracker import CostTracker
from largeforge.training.callbacks import TrainingCallback
from largeforge.utils import get_logger

logger = get_logger(__name__)


class CostTrackingCallback(TrainingCallback):
    """
    Training callback that tracks GPU usage and costs.

    This callback integrates with the training loop to:
    - Start cost tracking when training begins
    - Periodically sample GPU usage
    - Stop tracking and calculate final costs when training ends
    """

    def __init__(
        self,
        tracker: CostTracker,
        job_id: str,
    ):
        """
        Initialize the cost tracking callback.

        Args:
            tracker: Cost tracker instance
            job_id: ID of the training job
        """
        self.tracker = tracker
        self.job_id = job_id
        self._started = False

    def on_train_begin(self, trainer: Any, state: Any, **kwargs) -> None:
        """Called at the beginning of training."""
        if not self._started:
            self.tracker.start_tracking(self.job_id)
            self._started = True
            logger.info(f"Started cost tracking for job {self.job_id}")

    def on_train_end(self, trainer: Any, state: Any, **kwargs) -> None:
        """Called at the end of training."""
        if self._started:
            entry = self.tracker.stop_tracking(self.job_id)
            self._started = False

            if entry:
                logger.info(
                    f"Cost tracking complete for job {self.job_id}: "
                    f"{entry.gpu_hours:.2f} GPU-hours, "
                    f"${entry.estimated_cost_usd:.2f}"
                )

    def on_error(self, trainer: Any, state: Any, error: Exception, **kwargs) -> None:
        """Called when training encounters an error."""
        if self._started:
            entry = self.tracker.stop_tracking(self.job_id)
            self._started = False

            if entry:
                logger.warning(
                    f"Cost tracking stopped due to error for job {self.job_id}: "
                    f"{entry.gpu_hours:.2f} GPU-hours, "
                    f"${entry.estimated_cost_usd:.2f}"
                )


def create_cost_callback(
    job_id: str,
    tracker: Optional[CostTracker] = None,
) -> CostTrackingCallback:
    """
    Create a cost tracking callback.

    Args:
        job_id: ID of the training job
        tracker: Cost tracker (uses global if None)

    Returns:
        Cost tracking callback
    """
    from largeforge.costs.tracker import get_cost_tracker

    if tracker is None:
        tracker = get_cost_tracker()

    return CostTrackingCallback(tracker=tracker, job_id=job_id)
