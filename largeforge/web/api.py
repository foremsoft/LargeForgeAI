"""REST API endpoints for the web interface."""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import psutil
from fastapi import APIRouter, Depends, HTTPException, Query, status

from largeforge.utils import get_logger
from largeforge.utils.device import get_device, get_device_count, get_device_memory, get_device_name
from largeforge.version import __version__
from largeforge.web.schemas import (
    ErrorResponse,
    GPUInfo,
    HealthResponse,
    JobListResponse,
    ModelInfo,
    ModelListResponse,
    SystemInfo,
    TrainingJobCreate,
    TrainingJobResponse,
    TrainingProgress,
    VerificationRequest,
    VerificationResponse,
)
from largeforge.web.state import JobStateManager, JobStatus, TrainingJob

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["api"])

# Global state manager (initialized in app.py)
_state_manager: Optional[JobStateManager] = None
_start_time: float = time.time()


def get_state_manager() -> JobStateManager:
    """Dependency to get the job state manager."""
    if _state_manager is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="State manager not initialized"
        )
    return _state_manager


def set_state_manager(manager: JobStateManager) -> None:
    """Set the global state manager."""
    global _state_manager
    _state_manager = manager


def _job_to_response(job: TrainingJob) -> TrainingJobResponse:
    """Convert TrainingJob to response schema."""
    return TrainingJobResponse(
        job_id=job.job_id,
        job_name=job.job_name,
        status=job.status.value,
        training_type=job.training_type,
        model_path=job.model_path,
        dataset_path=job.dataset_path,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        progress=job.progress,
        error=job.error,
        output_path=job.output_path,
        config=job.config,
    )


# ============== Health & System ==============

@router.get("/health", response_model=HealthResponse)
async def health_check(
    state_manager: JobStateManager = Depends(get_state_manager),
) -> HealthResponse:
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        version=__version__,
        uptime_seconds=time.time() - _start_time,
        active_jobs=state_manager.count_jobs(status=JobStatus.RUNNING),
        gpu_available=get_device() == "cuda",
    )


@router.get("/system/info", response_model=SystemInfo)
async def get_system_info() -> SystemInfo:
    """Get system information including CPU, memory, disk, and GPU."""
    # CPU info
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=0.1)

    # Memory info
    memory = psutil.virtual_memory()
    memory_total_gb = memory.total / (1024**3)
    memory_used_gb = memory.used / (1024**3)

    # Disk info
    disk = psutil.disk_usage("/")
    disk_total_gb = disk.total / (1024**3)
    disk_used_gb = disk.used / (1024**3)

    # GPU info
    gpus = []
    gpu_count = get_device_count()
    for i in range(gpu_count):
        mem = get_device_memory(i)
        gpus.append(GPUInfo(
            index=i,
            name=get_device_name(i),
            total_memory_gb=mem.get("total_gb", 0) or 0,
            used_memory_gb=mem.get("allocated_gb", 0) or 0,
            free_memory_gb=mem.get("free_gb", 0) or 0,
            utilization_percent=0.0,  # Would need pynvml for this
        ))

    # PyTorch/CUDA info
    cuda_available = get_device() == "cuda"
    cuda_version = None
    torch_version = None

    try:
        import torch
        torch_version = torch.__version__
        if cuda_available:
            cuda_version = torch.version.cuda
    except ImportError:
        pass

    return SystemInfo(
        cpu_count=cpu_count,
        cpu_percent=cpu_percent,
        memory_total_gb=round(memory_total_gb, 2),
        memory_used_gb=round(memory_used_gb, 2),
        memory_percent=memory.percent,
        disk_total_gb=round(disk_total_gb, 2),
        disk_used_gb=round(disk_used_gb, 2),
        disk_percent=disk.percent,
        gpus=gpus,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        torch_version=torch_version,
    )


@router.get("/system/gpu", response_model=List[GPUInfo])
async def get_gpu_info() -> List[GPUInfo]:
    """Get detailed GPU information."""
    gpus = []
    gpu_count = get_device_count()

    for i in range(gpu_count):
        mem = get_device_memory(i)
        gpus.append(GPUInfo(
            index=i,
            name=get_device_name(i),
            total_memory_gb=mem.get("total_gb", 0) or 0,
            used_memory_gb=mem.get("allocated_gb", 0) or 0,
            free_memory_gb=mem.get("free_gb", 0) or 0,
            utilization_percent=0.0,
        ))

    return gpus


# ============== Training Jobs ==============

@router.post("/jobs", response_model=TrainingJobResponse, status_code=status.HTTP_201_CREATED)
async def create_job(
    request: TrainingJobCreate,
    state_manager: JobStateManager = Depends(get_state_manager),
) -> TrainingJobResponse:
    """Create a new training job."""
    # Validate paths
    if not os.path.exists(request.dataset_path):
        # Allow HuggingFace datasets
        if not request.dataset_path.startswith("hf://"):
            logger.warning(f"Dataset path may not exist: {request.dataset_path}")

    job = state_manager.create_job(request)
    logger.info(f"Created training job: {job.job_id}")

    return _job_to_response(job)


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    state_manager: JobStateManager = Depends(get_state_manager),
) -> JobListResponse:
    """List all training jobs with optional filtering."""
    job_status = None
    if status:
        try:
            job_status = JobStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status}"
            )

    offset = (page - 1) * page_size
    jobs = state_manager.list_jobs(status=job_status, limit=page_size, offset=offset)
    total = state_manager.count_jobs(status=job_status)

    return JobListResponse(
        jobs=[_job_to_response(j) for j in jobs],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_job(
    job_id: str,
    state_manager: JobStateManager = Depends(get_state_manager),
) -> TrainingJobResponse:
    """Get details of a specific job."""
    job = state_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}"
        )

    return _job_to_response(job)


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_job(
    job_id: str,
    state_manager: JobStateManager = Depends(get_state_manager),
) -> None:
    """Delete a job (only if not running)."""
    job = state_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}"
        )

    if job.status == JobStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot delete running job. Cancel it first."
        )

    state_manager.delete_job(job_id)


@router.post("/jobs/{job_id}/cancel", response_model=TrainingJobResponse)
async def cancel_job(
    job_id: str,
    state_manager: JobStateManager = Depends(get_state_manager),
) -> TrainingJobResponse:
    """Cancel a pending or running job."""
    job = state_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}"
        )

    if job.status not in (JobStatus.PENDING, JobStatus.RUNNING):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot cancel job with status: {job.status.value}"
        )

    # TODO: Actually stop the training if running
    state_manager.cancel_job(job_id)

    # Refresh job
    job = state_manager.get_job(job_id)
    return _job_to_response(job)


# ============== Models ==============

@router.get("/models", response_model=ModelListResponse)
async def list_models(
    path: Optional[str] = Query(None, description="Search path for local models"),
    state_manager: JobStateManager = Depends(get_state_manager),
) -> ModelListResponse:
    """List available models (local and from completed jobs)."""
    models = []

    # Get models from completed training jobs
    completed_jobs = state_manager.list_jobs(status=JobStatus.COMPLETED, limit=100)
    for job in completed_jobs:
        if job.output_path and os.path.exists(job.output_path):
            models.append(ModelInfo(
                path=job.output_path,
                name=job.job_name or f"Job {job.job_id[:8]}",
                model_type=job.training_type,
                is_local=True,
                is_lora=job.config.get("lora_enabled", False),
                base_model=job.model_path,
                training_job_id=job.job_id,
                created_at=job.completed_at,
            ))

    # Scan local path if provided
    if path and os.path.isdir(path):
        for item in Path(path).iterdir():
            if item.is_dir():
                # Check if it's a model directory
                if (item / "config.json").exists() or (item / "adapter_config.json").exists():
                    is_lora = (item / "adapter_config.json").exists()
                    models.append(ModelInfo(
                        path=str(item),
                        name=item.name,
                        is_local=True,
                        is_lora=is_lora,
                    ))

    return ModelListResponse(
        models=models,
        total=len(models),
    )


@router.get("/models/{model_id:path}/info", response_model=ModelInfo)
async def get_model_info(model_id: str) -> ModelInfo:
    """Get information about a specific model."""
    model_path = Path(model_id)

    if not model_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}"
        )

    is_lora = (model_path / "adapter_config.json").exists()

    # Calculate size
    size_gb = None
    try:
        total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
        size_gb = total_size / (1024**3)
    except Exception:
        pass

    return ModelInfo(
        path=str(model_path),
        name=model_path.name,
        size_gb=round(size_gb, 2) if size_gb else None,
        is_local=True,
        is_lora=is_lora,
    )


@router.post("/models/{model_id:path}/verify", response_model=VerificationResponse)
async def verify_model(
    model_id: str,
    request: Optional[VerificationRequest] = None,
) -> VerificationResponse:
    """Run verification on a model."""
    from largeforge.verification import ModelValidator, ValidationConfig, ValidationLevel

    model_path = model_id
    if request:
        model_path = request.model_path

    level = ValidationLevel.STANDARD
    if request and request.level:
        level = ValidationLevel(request.level)

    config = ValidationConfig(level=level)
    validator = ModelValidator(model_path, config=config)

    result = validator.validate()

    return VerificationResponse(
        passed=result.passed,
        level=result.level.value,
        model_path=result.model_path,
        summary=result.summary,
        recommendations=result.recommendations,
        validated_at=result.validated_at,
        validation_time_seconds=result.validation_time_seconds,
    )
