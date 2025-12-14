"""Cost tracking module for LargeForgeAI.

This module provides GPU usage monitoring and cost estimation for training runs.

Example:
    >>> from largeforge.costs import get_cost_tracker, create_cost_callback
    >>>
    >>> # Create a cost tracking callback
    >>> callback = create_cost_callback(job_id="job-123")
    >>>
    >>> # Use callback in training
    >>> trainer.train(callbacks=[callback])
    >>>
    >>> # Get cost summary
    >>> tracker = get_cost_tracker()
    >>> summary = tracker.get_monthly_summary(2024, 1)
    >>> print(f"Total cost: ${summary.total_cost_usd:.2f}")
"""

from largeforge.costs.models import (
    CostConfig,
    CostEntry,
    CostSummary,
    GPUUsageEntry,
)
from largeforge.costs.tracker import (
    CostTracker,
    get_cost_tracker,
    get_gpu_info,
)
from largeforge.costs.callback import (
    CostTrackingCallback,
    create_cost_callback,
)

__all__ = [
    # Models
    "CostConfig",
    "CostEntry",
    "CostSummary",
    "GPUUsageEntry",
    # Tracker
    "CostTracker",
    "get_cost_tracker",
    "get_gpu_info",
    # Callback
    "CostTrackingCallback",
    "create_cost_callback",
]
