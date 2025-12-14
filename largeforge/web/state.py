"""Training job state management."""

import json
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from largeforge.utils import get_logger
from largeforge.web.schemas import TrainingJobCreate, TrainingProgress

logger = get_logger(__name__)


class JobStatus(str, Enum):
    """Training job status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingJob:
    """Training job data."""

    job_id: str
    status: JobStatus
    training_type: str
    model_path: str
    dataset_path: str
    created_at: datetime
    config: Dict[str, Any] = field(default_factory=dict)
    job_name: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: Optional[TrainingProgress] = None
    error: Optional[str] = None
    output_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {
            "job_id": self.job_id,
            "status": self.status.value,
            "training_type": self.training_type,
            "model_path": self.model_path,
            "dataset_path": self.dataset_path,
            "created_at": self.created_at.isoformat(),
            "config": self.config,
            "job_name": self.job_name,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress.model_dump() if self.progress else None,
            "error": self.error,
            "output_path": self.output_path,
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingJob":
        """Create from dictionary."""
        return cls(
            job_id=data["job_id"],
            status=JobStatus(data["status"]),
            training_type=data["training_type"],
            model_path=data["model_path"],
            dataset_path=data["dataset_path"],
            created_at=datetime.fromisoformat(data["created_at"]),
            config=data.get("config", {}),
            job_name=data.get("job_name"),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            progress=TrainingProgress(**data["progress"]) if data.get("progress") else None,
            error=data.get("error"),
            output_path=data.get("output_path"),
        )


class JobStateManager:
    """Manages training job state with file-based persistence."""

    def __init__(self, storage_path: str = ".largeforge/jobs"):
        """
        Initialize job state manager.

        Args:
            storage_path: Directory for storing job state files
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs_cache: Dict[str, TrainingJob] = {}
        self._load_all_jobs()

    def _load_all_jobs(self) -> None:
        """Load all jobs from storage into cache."""
        for job_file in self.storage_path.glob("*.json"):
            try:
                job = self._load_job_file(job_file)
                if job:
                    self._jobs_cache[job.job_id] = job
            except Exception as e:
                logger.warning(f"Failed to load job file {job_file}: {e}")

    def _load_job_file(self, path: Path) -> Optional[TrainingJob]:
        """Load a single job from file."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return TrainingJob.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading job from {path}: {e}")
            return None

    def _save_job(self, job: TrainingJob) -> None:
        """Save job to file."""
        job_file = self.storage_path / f"{job.job_id}.json"
        try:
            with open(job_file, "w") as f:
                json.dump(job.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving job {job.job_id}: {e}")

    def create_job(self, request: TrainingJobCreate) -> TrainingJob:
        """
        Create a new training job.

        Args:
            request: Job creation request

        Returns:
            Created TrainingJob
        """
        with self._lock:
            job_id = str(uuid.uuid4())

            # Build config from request
            config = request.model_dump(exclude={"model_path", "dataset_path", "training_type", "job_name"})

            job = TrainingJob(
                job_id=job_id,
                status=JobStatus.PENDING,
                training_type=request.training_type,
                model_path=request.model_path,
                dataset_path=request.dataset_path,
                created_at=datetime.utcnow(),
                config=config,
                job_name=request.job_name,
            )

            self._jobs_cache[job_id] = job
            self._save_job(job)

            logger.info(f"Created job {job_id}")
            return job

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """
        Get a job by ID.

        Args:
            job_id: Job ID

        Returns:
            TrainingJob or None if not found
        """
        with self._lock:
            return self._jobs_cache.get(job_id)

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[TrainingJob]:
        """
        List jobs with optional filtering.

        Args:
            status: Filter by status
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip

        Returns:
            List of TrainingJob
        """
        with self._lock:
            jobs = list(self._jobs_cache.values())

            # Filter by status
            if status:
                jobs = [j for j in jobs if j.status == status]

            # Sort by created_at descending
            jobs.sort(key=lambda j: j.created_at, reverse=True)

            # Apply pagination
            return jobs[offset:offset + limit]

    def count_jobs(self, status: Optional[JobStatus] = None) -> int:
        """Count jobs with optional status filter."""
        with self._lock:
            if status:
                return sum(1 for j in self._jobs_cache.values() if j.status == status)
            return len(self._jobs_cache)

    def update_progress(self, job_id: str, progress: TrainingProgress) -> None:
        """
        Update job progress.

        Args:
            job_id: Job ID
            progress: New progress data
        """
        with self._lock:
            job = self._jobs_cache.get(job_id)
            if job:
                job.progress = progress
                self._save_job(job)

    def set_running(self, job_id: str) -> None:
        """Mark job as running."""
        with self._lock:
            job = self._jobs_cache.get(job_id)
            if job and job.status == JobStatus.PENDING:
                job.status = JobStatus.RUNNING
                job.started_at = datetime.utcnow()
                self._save_job(job)
                logger.info(f"Job {job_id} started")

    def complete_job(self, job_id: str, output_path: str) -> None:
        """
        Mark job as completed.

        Args:
            job_id: Job ID
            output_path: Path to output directory
        """
        with self._lock:
            job = self._jobs_cache.get(job_id)
            if job:
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.utcnow()
                job.output_path = output_path
                self._save_job(job)
                logger.info(f"Job {job_id} completed")

    def fail_job(self, job_id: str, error: str) -> None:
        """
        Mark job as failed.

        Args:
            job_id: Job ID
            error: Error message
        """
        with self._lock:
            job = self._jobs_cache.get(job_id)
            if job:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.utcnow()
                job.error = error
                self._save_job(job)
                logger.error(f"Job {job_id} failed: {error}")

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job.

        Args:
            job_id: Job ID

        Returns:
            True if job was cancelled, False if not cancellable
        """
        with self._lock:
            job = self._jobs_cache.get(job_id)
            if job and job.status in (JobStatus.PENDING, JobStatus.RUNNING):
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.utcnow()
                self._save_job(job)
                logger.info(f"Job {job_id} cancelled")
                return True
            return False

    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job.

        Args:
            job_id: Job ID

        Returns:
            True if job was deleted
        """
        with self._lock:
            job = self._jobs_cache.get(job_id)
            if job:
                # Only delete if not running
                if job.status == JobStatus.RUNNING:
                    return False

                # Remove from cache
                del self._jobs_cache[job_id]

                # Remove file
                job_file = self.storage_path / f"{job_id}.json"
                if job_file.exists():
                    job_file.unlink()

                logger.info(f"Job {job_id} deleted")
                return True
            return False

    def get_running_jobs(self) -> List[TrainingJob]:
        """Get all running jobs."""
        return self.list_jobs(status=JobStatus.RUNNING)

    def get_pending_jobs(self) -> List[TrainingJob]:
        """Get all pending jobs."""
        return self.list_jobs(status=JobStatus.PENDING)
