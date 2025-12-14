"""Experiment tracking module for LargeForgeAI.

This module provides experiment tracking capabilities for training runs,
including metrics logging, artifact management, and experiment comparison.

Example:
    >>> from largeforge.experiments import get_tracker, create_experiment_callback
    >>>
    >>> # Create an experiment with callback
    >>> experiment, callback = create_experiment_callback(
    ...     experiment_name="qwen-sft-v1",
    ...     config={"learning_rate": 2e-5},
    ...     tags=["sft", "qwen"],
    ... )
    >>>
    >>> # Use callback in training
    >>> trainer.train(callbacks=[callback])
    >>>
    >>> # Compare experiments
    >>> tracker = get_tracker()
    >>> comparison = tracker.compare([exp1.id, exp2.id], ["loss", "eval_loss"])
"""

from largeforge.experiments.models import (
    Artifact,
    Experiment,
    ExperimentComparison,
    ExperimentStatus,
    MetricEntry,
)
from largeforge.experiments.tracker import (
    ExperimentTracker,
    get_tracker,
)
from largeforge.experiments.callback import (
    ExperimentCallback,
    create_experiment_callback,
)

__all__ = [
    # Models
    "Artifact",
    "Experiment",
    "ExperimentComparison",
    "ExperimentStatus",
    "MetricEntry",
    # Tracker
    "ExperimentTracker",
    "get_tracker",
    # Callback
    "ExperimentCallback",
    "create_experiment_callback",
]
