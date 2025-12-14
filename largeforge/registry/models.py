"""Data models for model registry."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ModelStage(str, Enum):
    """Stage of a model in the registry."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelVersion:
    """A specific version of a registered model."""

    version: str
    path: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    training_job_id: Optional[str] = None
    experiment_id: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    stage: ModelStage = ModelStage.DEVELOPMENT
    description: Optional[str] = None
    size_bytes: int = 0
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "path": self.path,
            "created_at": self.created_at.isoformat(),
            "training_job_id": self.training_job_id,
            "experiment_id": self.experiment_id,
            "metrics": self.metrics,
            "tags": self.tags,
            "stage": self.stage.value,
            "description": self.description,
            "size_bytes": self.size_bytes,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        """Create from dictionary."""
        return cls(
            version=data["version"],
            path=data["path"],
            created_at=datetime.fromisoformat(data["created_at"]),
            training_job_id=data.get("training_job_id"),
            experiment_id=data.get("experiment_id"),
            metrics=data.get("metrics", {}),
            tags=data.get("tags", []),
            stage=ModelStage(data.get("stage", "development")),
            description=data.get("description"),
            size_bytes=data.get("size_bytes", 0),
            config=data.get("config", {}),
        )


@dataclass
class RegisteredModel:
    """A registered model with multiple versions."""

    name: str
    description: Optional[str] = None
    base_model: str = ""
    versions: List[ModelVersion] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    current_version: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    owner: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "base_model": self.base_model,
            "versions": [v.to_dict() for v in self.versions],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "current_version": self.current_version,
            "tags": self.tags,
            "owner": self.owner,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegisteredModel":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description"),
            base_model=data.get("base_model", ""),
            versions=[ModelVersion.from_dict(v) for v in data.get("versions", [])],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            current_version=data.get("current_version"),
            tags=data.get("tags", []),
            owner=data.get("owner"),
            metadata=data.get("metadata", {}),
        )

    def get_version(self, version: str) -> Optional[ModelVersion]:
        """Get a specific version."""
        for v in self.versions:
            if v.version == version:
                return v
        return None

    def get_current_version(self) -> Optional[ModelVersion]:
        """Get the current production version."""
        if not self.current_version:
            return None
        return self.get_version(self.current_version)

    def get_latest_version(self) -> Optional[ModelVersion]:
        """Get the most recent version."""
        if not self.versions:
            return None
        return max(self.versions, key=lambda v: v.created_at)

    def get_production_version(self) -> Optional[ModelVersion]:
        """Get the production stage version."""
        for v in self.versions:
            if v.stage == ModelStage.PRODUCTION:
                return v
        return None


@dataclass
class ModelDeployment:
    """A deployment of a registered model."""

    deployment_id: str
    model_name: str
    version: str
    endpoint_url: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "active"
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "deployment_id": self.deployment_id,
            "model_name": self.model_name,
            "version": self.version,
            "endpoint_url": self.endpoint_url,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "config": self.config,
        }
