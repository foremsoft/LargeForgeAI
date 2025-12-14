"""Web UI and API module for LargeForgeAI.

This module provides a web-based interface for:
- Training job management (create, monitor, cancel)
- Real-time training progress via WebSocket
- Model management and verification
- System monitoring (GPU, memory)

Example:
    >>> from largeforge.web import create_app, run_server
    >>> app = create_app()
    >>> run_server(host="0.0.0.0", port=7860)
"""

from largeforge.web.app import create_app, run_server
from largeforge.web.state import JobStateManager, TrainingJob, JobStatus
from largeforge.web.schemas import (
    TrainingJobCreate,
    TrainingJobResponse,
    TrainingProgress,
    JobListResponse,
    ModelInfo,
    SystemInfo,
)

__all__ = [
    # App
    "create_app",
    "run_server",
    # State
    "JobStateManager",
    "TrainingJob",
    "JobStatus",
    # Schemas
    "TrainingJobCreate",
    "TrainingJobResponse",
    "TrainingProgress",
    "JobListResponse",
    "ModelInfo",
    "SystemInfo",
]
