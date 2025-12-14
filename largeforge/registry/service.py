"""Model registry service."""

import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from largeforge.registry.models import (
    ModelStage,
    ModelVersion,
    RegisteredModel,
)
from largeforge.utils import get_logger, ensure_dir

logger = get_logger(__name__)


def parse_version(version: str) -> tuple:
    """Parse a semver string into a tuple for comparison."""
    match = re.match(r"v?(\d+)\.(\d+)\.(\d+)", version)
    if match:
        return tuple(int(x) for x in match.groups())
    return (0, 0, 0)


def increment_version(version: str, part: str = "patch") -> str:
    """Increment a version number."""
    major, minor, patch = parse_version(version)

    if part == "major":
        return f"v{major + 1}.0.0"
    elif part == "minor":
        return f"v{major}.{minor + 1}.0"
    else:
        return f"v{major}.{minor}.{patch + 1}"


class ModelRegistry:
    """Service for managing model versions and deployments."""

    def __init__(self, storage_path: str = ".largeforge/registry"):
        """
        Initialize the model registry.

        Args:
            storage_path: Path to store registry data
        """
        self.storage_path = Path(storage_path)
        ensure_dir(self.storage_path)
        self._lock = threading.Lock()
        self._models: Dict[str, RegisteredModel] = {}
        self._load_models()

    def _load_models(self) -> None:
        """Load existing models from storage."""
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                model = RegisteredModel.from_dict(data)
                self._models[model.name] = model
            except Exception as e:
                logger.warning(f"Failed to load model {file_path}: {e}")

    def _save_model(self, model: RegisteredModel) -> None:
        """Save a model to storage."""
        # Sanitize name for filename
        safe_name = re.sub(r"[^\w\-]", "_", model.name)
        file_path = self.storage_path / f"{safe_name}.json"
        with open(file_path, "w") as f:
            json.dump(model.to_dict(), f, indent=2)

    def register(
        self,
        name: str,
        path: str,
        base_model: str = "",
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metrics: Optional[Dict[str, float]] = None,
        training_job_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        owner: Optional[str] = None,
    ) -> RegisteredModel:
        """
        Register a new model or add a version to an existing model.

        Args:
            name: Name of the model
            path: Path to the model files
            base_model: Name of the base model
            description: Model description
            tags: Tags for filtering
            metrics: Training/evaluation metrics
            training_job_id: Associated training job ID
            experiment_id: Associated experiment ID
            owner: Owner username

        Returns:
            Registered model
        """
        with self._lock:
            # Check if model already exists
            if name in self._models:
                model = self._models[name]
                # Add a new version
                version = self._next_version(model)
                model_version = ModelVersion(
                    version=version,
                    path=path,
                    training_job_id=training_job_id,
                    experiment_id=experiment_id,
                    metrics=metrics or {},
                    tags=tags or [],
                )
                model.versions.append(model_version)
                model.updated_at = datetime.utcnow()
            else:
                # Create new model
                model_version = ModelVersion(
                    version="v1.0.0",
                    path=path,
                    training_job_id=training_job_id,
                    experiment_id=experiment_id,
                    metrics=metrics or {},
                    tags=tags or [],
                )
                model = RegisteredModel(
                    name=name,
                    description=description,
                    base_model=base_model,
                    versions=[model_version],
                    current_version="v1.0.0",
                    tags=tags or [],
                    owner=owner,
                )
                self._models[name] = model

            # Calculate size
            model_path = Path(path)
            if model_path.exists():
                if model_path.is_file():
                    model_version.size_bytes = model_path.stat().st_size
                else:
                    model_version.size_bytes = sum(
                        f.stat().st_size for f in model_path.rglob("*") if f.is_file()
                    )

            self._save_model(model)

        logger.info(f"Registered model: {name} (version {model_version.version})")
        return model

    def _next_version(self, model: RegisteredModel) -> str:
        """Get the next version number for a model."""
        if not model.versions:
            return "v1.0.0"

        latest = model.get_latest_version()
        if latest:
            return increment_version(latest.version)
        return "v1.0.0"

    def get(self, name: str) -> Optional[RegisteredModel]:
        """Get a registered model by name."""
        return self._models.get(name)

    def list(
        self,
        tags: Optional[List[str]] = None,
        owner: Optional[str] = None,
    ) -> List[RegisteredModel]:
        """
        List registered models with optional filtering.

        Args:
            tags: Filter by tags
            owner: Filter by owner

        Returns:
            List of matching models
        """
        models = list(self._models.values())

        if tags:
            models = [m for m in models if any(t in m.tags for t in tags)]

        if owner:
            models = [m for m in models if m.owner == owner]

        return sorted(models, key=lambda m: m.updated_at, reverse=True)

    def delete(self, name: str) -> bool:
        """
        Delete a registered model.

        Args:
            name: Name of model to delete

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if name not in self._models:
                return False

            del self._models[name]

            # Delete file
            safe_name = re.sub(r"[^\w\-]", "_", name)
            file_path = self.storage_path / f"{safe_name}.json"
            if file_path.exists():
                file_path.unlink()

        logger.info(f"Deleted model: {name}")
        return True

    def update(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[RegisteredModel]:
        """
        Update model metadata.

        Args:
            name: Model name
            description: New description
            tags: New tags (replaces existing)
            metadata: Additional metadata

        Returns:
            Updated model or None if not found
        """
        with self._lock:
            model = self._models.get(name)
            if not model:
                return None

            if description is not None:
                model.description = description
            if tags is not None:
                model.tags = tags
            if metadata is not None:
                model.metadata.update(metadata)

            model.updated_at = datetime.utcnow()
            self._save_model(model)

        return model

    def add_version(
        self,
        name: str,
        path: str,
        version: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        training_job_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
    ) -> Optional[ModelVersion]:
        """
        Add a new version to an existing model.

        Args:
            name: Model name
            path: Path to model files
            version: Version string (auto-generated if None)
            metrics: Training/evaluation metrics
            tags: Version-specific tags
            description: Version description
            training_job_id: Associated training job ID
            experiment_id: Associated experiment ID

        Returns:
            Created version or None if model not found
        """
        with self._lock:
            model = self._models.get(name)
            if not model:
                return None

            # Determine version
            if version is None:
                version = self._next_version(model)

            # Check for duplicate version
            if model.get_version(version):
                logger.warning(f"Version {version} already exists for model {name}")
                return None

            model_version = ModelVersion(
                version=version,
                path=path,
                training_job_id=training_job_id,
                experiment_id=experiment_id,
                metrics=metrics or {},
                tags=tags or [],
                description=description,
            )

            # Calculate size
            model_path = Path(path)
            if model_path.exists():
                if model_path.is_file():
                    model_version.size_bytes = model_path.stat().st_size
                else:
                    model_version.size_bytes = sum(
                        f.stat().st_size for f in model_path.rglob("*") if f.is_file()
                    )

            model.versions.append(model_version)
            model.updated_at = datetime.utcnow()
            self._save_model(model)

        logger.info(f"Added version {version} to model {name}")
        return model_version

    def get_version(self, name: str, version: str) -> Optional[ModelVersion]:
        """Get a specific version of a model."""
        model = self._models.get(name)
        if not model:
            return None
        return model.get_version(version)

    def set_current_version(self, name: str, version: str) -> Optional[RegisteredModel]:
        """
        Set the current (default) version for a model.

        Args:
            name: Model name
            version: Version to set as current

        Returns:
            Updated model or None if not found
        """
        with self._lock:
            model = self._models.get(name)
            if not model:
                return None

            if not model.get_version(version):
                logger.warning(f"Version {version} not found for model {name}")
                return None

            model.current_version = version
            model.updated_at = datetime.utcnow()
            self._save_model(model)

        logger.info(f"Set current version of {name} to {version}")
        return model

    def transition_stage(
        self,
        name: str,
        version: str,
        stage: ModelStage,
    ) -> Optional[ModelVersion]:
        """
        Transition a model version to a new stage.

        Args:
            name: Model name
            version: Version to transition
            stage: Target stage

        Returns:
            Updated version or None if not found
        """
        with self._lock:
            model = self._models.get(name)
            if not model:
                return None

            model_version = model.get_version(version)
            if not model_version:
                return None

            # If transitioning to production, demote other production versions
            if stage == ModelStage.PRODUCTION:
                for v in model.versions:
                    if v.stage == ModelStage.PRODUCTION:
                        v.stage = ModelStage.STAGING

            model_version.stage = stage
            model.updated_at = datetime.utcnow()
            self._save_model(model)

        logger.info(f"Transitioned {name}:{version} to {stage.value}")
        return model_version

    def list_versions(self, name: str) -> List[ModelVersion]:
        """Get all versions of a model."""
        model = self._models.get(name)
        return model.versions if model else []


# Global instance
_registry: Optional[ModelRegistry] = None


def get_registry(storage_path: str = ".largeforge/registry") -> ModelRegistry:
    """Get or create the global model registry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry(storage_path)
    return _registry
