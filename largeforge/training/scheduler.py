"""Job queue and scheduling system for training jobs."""

import collections
import threading
import time
from typing import Callable, List, Optional

from largeforge.training.orchestrator import TrainingOrchestrator
from largeforge.training.events import EventEmitter, EventType
from largeforge.utils import get_logger
from largeforge.web.state import JobStateManager, JobStatus

logger = get_logger(__name__)


class JobQueue:
    """Thread-safe job queue for pending training jobs."""

    def __init__(self):
        """Initialize job queue."""
        self._queue: collections.deque[str] = collections.deque()
        self._lock = threading.Lock()

    def add(self, job_id: str) -> int:
        """
        Add a job to the queue.

        Args:
            job_id: Job ID to add

        Returns:
            Position in queue (1-based)
        """
        with self._lock:
            if job_id not in self._queue:
                self._queue.append(job_id)
            return self.get_position(job_id)

    def remove(self, job_id: str) -> bool:
        """
        Remove a job from the queue.

        Args:
            job_id: Job ID to remove

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            try:
                self._queue.remove(job_id)
                return True
            except ValueError:
                return False

    def get_next(self) -> Optional[str]:
        """
        Get and remove the next job from the queue.

        Returns:
            Job ID or None if queue is empty
        """
        with self._lock:
            if self._queue:
                return self._queue.popleft()
            return None

    def peek(self) -> Optional[str]:
        """
        Peek at the next job without removing it.

        Returns:
            Job ID or None if queue is empty
        """
        with self._lock:
            if self._queue:
                return self._queue[0]
            return None

    def get_position(self, job_id: str) -> Optional[int]:
        """
        Get position of a job in the queue.

        Args:
            job_id: Job ID to find

        Returns:
            Position (1-based) or None if not found
        """
        with self._lock:
            try:
                return list(self._queue).index(job_id) + 1
            except ValueError:
                return None

    def get_all(self) -> List[str]:
        """Get all job IDs in queue order."""
        with self._lock:
            return list(self._queue)

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        with self._lock:
            return len(self._queue) == 0

    def size(self) -> int:
        """Get queue size."""
        with self._lock:
            return len(self._queue)

    def clear(self) -> None:
        """Clear all jobs from queue."""
        with self._lock:
            self._queue.clear()


class Scheduler:
    """
    Job scheduler that manages the job queue and starts jobs when resources are available.
    """

    def __init__(
        self,
        orchestrator: TrainingOrchestrator,
        state_manager: JobStateManager,
        check_interval: float = 5.0,
        gpu_memory_threshold: float = 0.8,
    ):
        """
        Initialize scheduler.

        Args:
            orchestrator: Training orchestrator
            state_manager: Job state manager
            check_interval: Interval between queue checks (seconds)
            gpu_memory_threshold: GPU memory threshold (0-1) for starting new jobs
        """
        self.orchestrator = orchestrator
        self.state_manager = state_manager
        self.check_interval = check_interval
        self.gpu_memory_threshold = gpu_memory_threshold

        self.queue = JobQueue()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        logger.info("Scheduler initialized")

    def start(self) -> None:
        """Start the scheduler loop in a background thread."""
        with self._lock:
            if self._running:
                logger.warning("Scheduler is already running")
                return

            self._running = True
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._scheduler_loop,
                name="job-scheduler",
                daemon=True,
            )
            self._thread.start()
            logger.info("Scheduler started")

    def stop(self, wait: bool = True) -> None:
        """
        Stop the scheduler.

        Args:
            wait: Wait for scheduler thread to finish
        """
        with self._lock:
            if not self._running:
                return

            self._running = False
            self._stop_event.set()

        if wait and self._thread:
            self._thread.join(timeout=self.check_interval * 2)

        logger.info("Scheduler stopped")

    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running

    def submit_job(self, job_id: str) -> int:
        """
        Submit a job to the scheduler queue.

        Args:
            job_id: Job ID to submit

        Returns:
            Position in queue
        """
        position = self.queue.add(job_id)
        logger.info(f"Job {job_id} added to queue at position {position}")
        return position

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job (remove from queue or stop if running).

        Args:
            job_id: Job ID to cancel

        Returns:
            True if cancelled successfully
        """
        # Try to remove from queue first
        if self.queue.remove(job_id):
            self.state_manager.cancel_job(job_id)
            logger.info(f"Job {job_id} removed from queue")
            return True

        # Try to stop running job
        if self.orchestrator.stop_job(job_id):
            return True

        return False

    def get_queue_position(self, job_id: str) -> Optional[int]:
        """
        Get queue position for a job.

        Args:
            job_id: Job ID

        Returns:
            Position (1-based) or None if not queued
        """
        return self.queue.get_position(job_id)

    def get_queue_status(self) -> dict:
        """Get current queue status."""
        return {
            "queue_size": self.queue.size(),
            "queued_jobs": self.queue.get_all(),
            "running_jobs": self.orchestrator.get_running_jobs(),
            "running_count": self.orchestrator.get_running_count(),
            "can_start_new": self._can_start_job(),
        }

    def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        logger.info("Scheduler loop started")

        while not self._stop_event.is_set():
            try:
                self._process_queue()
            except Exception as e:
                logger.error(f"Scheduler error: {e}")

            # Wait for next check
            self._stop_event.wait(self.check_interval)

        logger.info("Scheduler loop ended")

    def _process_queue(self) -> None:
        """Process the job queue and start jobs if possible."""
        if self.queue.is_empty():
            return

        if not self._can_start_job():
            return

        # Get next job from queue
        job_id = self.queue.peek()
        if not job_id:
            return

        # Get job from state manager
        job = self.state_manager.get_job(job_id)
        if not job:
            # Job no longer exists, remove from queue
            self.queue.remove(job_id)
            return

        # Check job status
        if job.status != JobStatus.PENDING:
            # Job is no longer pending, remove from queue
            self.queue.remove(job_id)
            return

        # Check resources
        if not self._check_resources():
            logger.debug("Resources not available, waiting...")
            return

        # Start the job
        self.queue.get_next()  # Remove from queue
        success = self.orchestrator.start_job(job)

        if not success:
            # Failed to start, re-add to queue
            self.queue.add(job_id)
            logger.warning(f"Failed to start job {job_id}, re-queued")

    def _can_start_job(self) -> bool:
        """Check if a new job can be started."""
        return self.orchestrator.can_start_job()

    def _check_resources(self) -> bool:
        """
        Check if system resources are available for a new job.

        Returns:
            True if resources are available
        """
        try:
            import torch

            if not torch.cuda.is_available():
                # No GPU, always allow (will use CPU)
                return True

            # Check GPU memory
            for i in range(torch.cuda.device_count()):
                memory_total = torch.cuda.get_device_properties(i).total_memory
                memory_reserved = torch.cuda.memory_reserved(i)
                memory_ratio = memory_reserved / memory_total

                if memory_ratio < self.gpu_memory_threshold:
                    return True

            # All GPUs are above threshold
            logger.debug(
                f"GPU memory above threshold ({self.gpu_memory_threshold * 100}%)"
            )
            return False

        except Exception as e:
            logger.warning(f"Resource check failed: {e}")
            # Allow job to start on error
            return True

    def _get_gpu_memory_usage(self) -> dict:
        """Get current GPU memory usage."""
        try:
            import torch

            if not torch.cuda.is_available():
                return {"available": False}

            gpus = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                reserved = torch.cuda.memory_reserved(i)
                allocated = torch.cuda.memory_allocated(i)
                free = props.total_memory - reserved

                gpus.append({
                    "index": i,
                    "name": props.name,
                    "total_gb": props.total_memory / (1024**3),
                    "reserved_gb": reserved / (1024**3),
                    "allocated_gb": allocated / (1024**3),
                    "free_gb": free / (1024**3),
                    "utilization": reserved / props.total_memory,
                })

            return {"available": True, "gpus": gpus}

        except Exception as e:
            return {"available": False, "error": str(e)}


class AutoScheduler:
    """
    Scheduler that automatically picks up pending jobs from the state manager.
    """

    def __init__(
        self,
        orchestrator: TrainingOrchestrator,
        state_manager: JobStateManager,
        check_interval: float = 10.0,
    ):
        """
        Initialize auto scheduler.

        Args:
            orchestrator: Training orchestrator
            state_manager: Job state manager
            check_interval: Interval between scans (seconds)
        """
        self.scheduler = Scheduler(orchestrator, state_manager, check_interval)
        self.state_manager = state_manager
        self._auto_running = False
        self._auto_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start auto scheduler."""
        # Load pending jobs into queue
        self._load_pending_jobs()

        # Start base scheduler
        self.scheduler.start()

        # Start auto-scan thread
        self._auto_running = True
        self._stop_event.clear()
        self._auto_thread = threading.Thread(
            target=self._auto_scan_loop,
            name="auto-scheduler",
            daemon=True,
        )
        self._auto_thread.start()

        logger.info("AutoScheduler started")

    def stop(self, wait: bool = True) -> None:
        """Stop auto scheduler."""
        self._auto_running = False
        self._stop_event.set()
        self.scheduler.stop(wait)

        if wait and self._auto_thread:
            self._auto_thread.join(timeout=5.0)

        logger.info("AutoScheduler stopped")

    def _load_pending_jobs(self) -> None:
        """Load existing pending jobs into queue."""
        pending_jobs = self.state_manager.get_pending_jobs()
        for job in pending_jobs:
            self.scheduler.submit_job(job.job_id)
        logger.info(f"Loaded {len(pending_jobs)} pending jobs into queue")

    def _auto_scan_loop(self) -> None:
        """Periodically scan for new pending jobs."""
        while not self._stop_event.is_set():
            try:
                # Get pending jobs not in queue
                pending = self.state_manager.get_pending_jobs()
                queued = set(self.scheduler.queue.get_all())

                for job in pending:
                    if job.job_id not in queued:
                        self.scheduler.submit_job(job.job_id)

            except Exception as e:
                logger.error(f"Auto-scan error: {e}")

            self._stop_event.wait(self.scheduler.check_interval)
