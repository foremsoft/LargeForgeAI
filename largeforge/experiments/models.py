"""Data models for experiment tracking."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ExperimentStatus(str, Enum):
    """Status of an experiment."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class MetricEntry:
    """A single metric entry recorded during training."""

    name: str
    value: float
    step: int
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "step": self.step,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricEntry":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            value=data["value"],
            step=data["step"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


@dataclass
class Artifact:
    """An artifact associated with an experiment."""

    name: str
    path: str
    artifact_type: str
    size_bytes: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "path": self.path,
            "artifact_type": self.artifact_type,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Artifact":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            path=data["path"],
            artifact_type=data["artifact_type"],
            size_bytes=data.get("size_bytes", 0),
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Experiment:
    """An experiment tracking a training run."""

    id: str
    name: str
    description: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    status: ExperimentStatus = ExperimentStatus.RUNNING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    job_id: Optional[str] = None
    metrics: List[MetricEntry] = field(default_factory=list)
    artifacts: List[Artifact] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "config": self.config,
            "tags": self.tags,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "job_id": self.job_id,
            "metrics": [m.to_dict() for m in self.metrics],
            "artifacts": [a.to_dict() for a in self.artifacts],
            "params": self.params,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experiment":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description"),
            config=data.get("config", {}),
            tags=data.get("tags", []),
            status=ExperimentStatus(data.get("status", "running")),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            ended_at=datetime.fromisoformat(data["ended_at"]) if data.get("ended_at") else None,
            job_id=data.get("job_id"),
            metrics=[MetricEntry.from_dict(m) for m in data.get("metrics", [])],
            artifacts=[Artifact.from_dict(a) for a in data.get("artifacts", [])],
            params=data.get("params", {}),
            notes=data.get("notes"),
        )

    def get_metric_values(self, metric_name: str) -> List[float]:
        """Get all values for a specific metric."""
        return [m.value for m in self.metrics if m.name == metric_name]

    def get_metric_history(self, metric_name: str) -> List[tuple]:
        """Get step-value pairs for a metric."""
        return [(m.step, m.value) for m in self.metrics if m.name == metric_name]

    def get_latest_metrics(self) -> Dict[str, float]:
        """Get the latest value for each metric."""
        latest: Dict[str, tuple] = {}  # name -> (step, value)
        for m in self.metrics:
            if m.name not in latest or m.step > latest[m.name][0]:
                latest[m.name] = (m.step, m.value)
        return {name: val for name, (_, val) in latest.items()}


@dataclass
class ExperimentComparison:
    """Comparison results between multiple experiments."""

    experiment_ids: List[str]
    metric_names: List[str]
    data: Dict[str, Dict[str, List[float]]]  # exp_id -> metric_name -> values
    summary: Dict[str, Dict[str, float]]  # metric_name -> {min, max, mean, best_exp_id}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_ids": self.experiment_ids,
            "metric_names": self.metric_names,
            "data": self.data,
            "summary": self.summary,
        }
