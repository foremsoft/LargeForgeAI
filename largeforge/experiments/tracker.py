"""Experiment tracking service."""

import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from largeforge.experiments.models import (
    Artifact,
    Experiment,
    ExperimentComparison,
    ExperimentStatus,
    MetricEntry,
)
from largeforge.utils import get_logger, ensure_dir

logger = get_logger(__name__)


class ExperimentTracker:
    """Service for tracking experiments and their metrics."""

    def __init__(self, storage_path: str = ".largeforge/experiments"):
        """
        Initialize the experiment tracker.

        Args:
            storage_path: Path to store experiment data
        """
        self.storage_path = Path(storage_path)
        ensure_dir(self.storage_path)
        self._lock = threading.Lock()
        self._experiments: Dict[str, Experiment] = {}
        self._load_experiments()

    def _load_experiments(self) -> None:
        """Load existing experiments from storage."""
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                exp = Experiment.from_dict(data)
                self._experiments[exp.id] = exp
            except Exception as e:
                logger.warning(f"Failed to load experiment {file_path}: {e}")

    def _save_experiment(self, experiment: Experiment) -> None:
        """Save an experiment to storage."""
        file_path = self.storage_path / f"{experiment.id}.json"
        with open(file_path, "w") as f:
            json.dump(experiment.to_dict(), f, indent=2)

    def create(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        job_id: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Experiment:
        """
        Create a new experiment.

        Args:
            name: Name of the experiment
            config: Configuration dictionary
            tags: List of tags for filtering
            description: Optional description
            job_id: Associated training job ID
            params: Training parameters to track

        Returns:
            Created experiment
        """
        experiment = Experiment(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            config=config or {},
            tags=tags or [],
            job_id=job_id,
            params=params or {},
            status=ExperimentStatus.RUNNING,
            started_at=datetime.utcnow(),
        )

        with self._lock:
            self._experiments[experiment.id] = experiment
            self._save_experiment(experiment)

        logger.info(f"Created experiment: {experiment.id} ({name})")
        return experiment

    def get(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID."""
        return self._experiments.get(experiment_id)

    def list(
        self,
        tags: Optional[List[str]] = None,
        status: Optional[ExperimentStatus] = None,
        limit: int = 100,
    ) -> List[Experiment]:
        """
        List experiments with optional filtering.

        Args:
            tags: Filter by tags (any match)
            status: Filter by status
            limit: Maximum number of results

        Returns:
            List of matching experiments
        """
        experiments = list(self._experiments.values())

        # Filter by tags
        if tags:
            experiments = [
                e for e in experiments
                if any(t in e.tags for t in tags)
            ]

        # Filter by status
        if status:
            experiments = [e for e in experiments if e.status == status]

        # Sort by creation time (newest first)
        experiments.sort(key=lambda e: e.created_at, reverse=True)

        return experiments[:limit]

    def delete(self, experiment_id: str) -> bool:
        """
        Delete an experiment.

        Args:
            experiment_id: ID of experiment to delete

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if experiment_id not in self._experiments:
                return False

            del self._experiments[experiment_id]

            file_path = self.storage_path / f"{experiment_id}.json"
            if file_path.exists():
                file_path.unlink()

        logger.info(f"Deleted experiment: {experiment_id}")
        return True

    def update_status(
        self,
        experiment_id: str,
        status: ExperimentStatus,
        error: Optional[str] = None,
    ) -> Optional[Experiment]:
        """
        Update experiment status.

        Args:
            experiment_id: ID of experiment
            status: New status
            error: Optional error message for failed status

        Returns:
            Updated experiment or None if not found
        """
        with self._lock:
            exp = self._experiments.get(experiment_id)
            if not exp:
                return None

            exp.status = status
            exp.updated_at = datetime.utcnow()

            if status in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.CANCELLED):
                exp.ended_at = datetime.utcnow()

            if error and status == ExperimentStatus.FAILED:
                exp.notes = f"Error: {error}"

            self._save_experiment(exp)

        return exp

    def log_metric(
        self,
        experiment_id: str,
        name: str,
        value: float,
        step: int,
    ) -> Optional[MetricEntry]:
        """
        Log a metric value.

        Args:
            experiment_id: ID of experiment
            name: Metric name
            value: Metric value
            step: Training step

        Returns:
            Created metric entry or None if experiment not found
        """
        with self._lock:
            exp = self._experiments.get(experiment_id)
            if not exp:
                return None

            entry = MetricEntry(name=name, value=value, step=step)
            exp.metrics.append(entry)
            exp.updated_at = datetime.utcnow()
            self._save_experiment(exp)

        return entry

    def log_metrics(
        self,
        experiment_id: str,
        metrics: Dict[str, float],
        step: int,
    ) -> int:
        """
        Log multiple metrics at once.

        Args:
            experiment_id: ID of experiment
            metrics: Dictionary of metric names to values
            step: Training step

        Returns:
            Number of metrics logged
        """
        with self._lock:
            exp = self._experiments.get(experiment_id)
            if not exp:
                return 0

            for name, value in metrics.items():
                entry = MetricEntry(name=name, value=value, step=step)
                exp.metrics.append(entry)

            exp.updated_at = datetime.utcnow()
            self._save_experiment(exp)

        return len(metrics)

    def get_metrics(
        self,
        experiment_id: str,
        metric_name: Optional[str] = None,
    ) -> List[MetricEntry]:
        """
        Get metrics for an experiment.

        Args:
            experiment_id: ID of experiment
            metric_name: Optional filter by metric name

        Returns:
            List of metric entries
        """
        exp = self._experiments.get(experiment_id)
        if not exp:
            return []

        if metric_name:
            return [m for m in exp.metrics if m.name == metric_name]
        return exp.metrics

    def log_artifact(
        self,
        experiment_id: str,
        name: str,
        path: str,
        artifact_type: str = "file",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Artifact]:
        """
        Log an artifact.

        Args:
            experiment_id: ID of experiment
            name: Artifact name
            path: Path to artifact
            artifact_type: Type of artifact (file, model, checkpoint, etc.)
            metadata: Additional metadata

        Returns:
            Created artifact or None if experiment not found
        """
        with self._lock:
            exp = self._experiments.get(experiment_id)
            if not exp:
                return None

            # Get file size if it exists
            size_bytes = 0
            artifact_path = Path(path)
            if artifact_path.exists():
                if artifact_path.is_file():
                    size_bytes = artifact_path.stat().st_size
                else:
                    # Sum up directory contents
                    size_bytes = sum(f.stat().st_size for f in artifact_path.rglob("*") if f.is_file())

            artifact = Artifact(
                name=name,
                path=str(path),
                artifact_type=artifact_type,
                size_bytes=size_bytes,
                metadata=metadata or {},
            )
            exp.artifacts.append(artifact)
            exp.updated_at = datetime.utcnow()
            self._save_experiment(exp)

        return artifact

    def get_artifacts(self, experiment_id: str) -> List[Artifact]:
        """Get all artifacts for an experiment."""
        exp = self._experiments.get(experiment_id)
        return exp.artifacts if exp else []

    def add_tags(self, experiment_id: str, tags: List[str]) -> Optional[Experiment]:
        """Add tags to an experiment."""
        with self._lock:
            exp = self._experiments.get(experiment_id)
            if not exp:
                return None

            for tag in tags:
                if tag not in exp.tags:
                    exp.tags.append(tag)

            exp.updated_at = datetime.utcnow()
            self._save_experiment(exp)

        return exp

    def set_notes(self, experiment_id: str, notes: str) -> Optional[Experiment]:
        """Set notes for an experiment."""
        with self._lock:
            exp = self._experiments.get(experiment_id)
            if not exp:
                return None

            exp.notes = notes
            exp.updated_at = datetime.utcnow()
            self._save_experiment(exp)

        return exp

    def compare(
        self,
        experiment_ids: List[str],
        metric_names: Optional[List[str]] = None,
    ) -> Optional[ExperimentComparison]:
        """
        Compare multiple experiments.

        Args:
            experiment_ids: List of experiment IDs to compare
            metric_names: Metrics to compare (all if None)

        Returns:
            Comparison results or None if experiments not found
        """
        experiments = [self._experiments.get(eid) for eid in experiment_ids]
        if not all(experiments):
            return None

        # Collect all metric names if not specified
        if not metric_names:
            all_names = set()
            for exp in experiments:
                for m in exp.metrics:
                    all_names.add(m.name)
            metric_names = list(all_names)

        # Build comparison data
        data: Dict[str, Dict[str, List[float]]] = {}
        for exp in experiments:
            data[exp.id] = {}
            for name in metric_names:
                data[exp.id][name] = exp.get_metric_values(name)

        # Build summary
        summary: Dict[str, Dict[str, Any]] = {}
        for name in metric_names:
            all_values = []
            best_value = float("inf")
            best_exp_id = None

            for exp_id, metrics in data.items():
                values = metrics.get(name, [])
                if values:
                    all_values.extend(values)
                    min_val = min(values)
                    if min_val < best_value:
                        best_value = min_val
                        best_exp_id = exp_id

            if all_values:
                summary[name] = {
                    "min": min(all_values),
                    "max": max(all_values),
                    "mean": sum(all_values) / len(all_values),
                    "best_exp_id": best_exp_id,
                }

        return ExperimentComparison(
            experiment_ids=experiment_ids,
            metric_names=metric_names,
            data=data,
            summary=summary,
        )


# Global instance
_tracker: Optional[ExperimentTracker] = None


def get_tracker(storage_path: str = ".largeforge/experiments") -> ExperimentTracker:
    """Get or create the global experiment tracker."""
    global _tracker
    if _tracker is None:
        _tracker = ExperimentTracker(storage_path)
    return _tracker
