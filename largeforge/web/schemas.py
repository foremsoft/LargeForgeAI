"""Pydantic schemas for the Web API."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class TrainingProgress(BaseModel):
    """Real-time training progress information."""

    model_config = ConfigDict(extra="forbid")

    epoch: int = 0
    total_epochs: int = 1
    step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    elapsed_seconds: float = 0.0
    eta_seconds: float = 0.0
    metrics: Dict[str, float] = Field(default_factory=dict)

    @property
    def progress_percent(self) -> float:
        """Calculate overall progress percentage."""
        if self.total_steps == 0:
            return 0.0
        return (self.step / self.total_steps) * 100


class TrainingJobCreate(BaseModel):
    """Request schema for creating a training job."""

    model_config = ConfigDict(extra="forbid")

    # Required fields
    model_path: str = Field(..., description="Path to base model or HuggingFace model ID")
    dataset_path: str = Field(..., description="Path to training dataset")

    # Training type
    training_type: Literal["sft", "dpo"] = Field(
        default="sft",
        description="Type of training: sft (supervised fine-tuning) or dpo (direct preference optimization)"
    )

    # Basic training parameters
    epochs: int = Field(default=3, ge=1, le=100, description="Number of training epochs")
    learning_rate: float = Field(default=2e-5, gt=0, le=1, description="Learning rate")
    batch_size: int = Field(default=4, ge=1, le=128, description="Training batch size per device")
    max_seq_length: int = Field(default=2048, ge=128, le=32768, description="Maximum sequence length")

    # LoRA configuration
    lora_enabled: bool = Field(default=True, description="Enable LoRA training")
    lora_r: int = Field(default=16, ge=1, le=256, description="LoRA rank")
    lora_alpha: int = Field(default=32, ge=1, description="LoRA alpha scaling")
    lora_dropout: float = Field(default=0.05, ge=0, le=0.5, description="LoRA dropout")

    # Advanced training parameters
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=128)
    warmup_ratio: float = Field(default=0.1, ge=0, le=0.5)
    weight_decay: float = Field(default=0.01, ge=0, le=1)
    max_grad_norm: float = Field(default=1.0, gt=0)

    # Precision
    fp16: bool = Field(default=False, description="Use FP16 mixed precision")
    bf16: bool = Field(default=True, description="Use BF16 mixed precision")

    # DPO-specific
    dpo_beta: float = Field(default=0.1, ge=0, le=1, description="DPO beta parameter")

    # Output
    output_dir: Optional[str] = Field(default=None, description="Output directory (auto-generated if None)")
    job_name: Optional[str] = Field(default=None, description="Optional job name")

    # Callbacks
    save_steps: int = Field(default=500, ge=1)
    eval_steps: int = Field(default=500, ge=1)
    logging_steps: int = Field(default=10, ge=1)


class TrainingJobResponse(BaseModel):
    """Response schema for training job."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    job_name: Optional[str] = None
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    training_type: str
    model_path: str
    dataset_path: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: Optional[TrainingProgress] = None
    error: Optional[str] = None
    output_path: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)


class JobListResponse(BaseModel):
    """Response schema for listing jobs."""

    model_config = ConfigDict(extra="forbid")

    jobs: List[TrainingJobResponse]
    total: int
    page: int = 1
    page_size: int = 20


class ModelInfo(BaseModel):
    """Information about an available model."""

    model_config = ConfigDict(extra="forbid")

    path: str
    name: str
    size_gb: Optional[float] = None
    created_at: Optional[datetime] = None
    model_type: Optional[str] = None
    is_local: bool = True
    is_lora: bool = False
    base_model: Optional[str] = None
    training_job_id: Optional[str] = None


class ModelListResponse(BaseModel):
    """Response schema for listing models."""

    model_config = ConfigDict(extra="forbid")

    models: List[ModelInfo]
    total: int


class GPUInfo(BaseModel):
    """GPU device information."""

    model_config = ConfigDict(extra="forbid")

    index: int
    name: str
    total_memory_gb: float
    used_memory_gb: float
    free_memory_gb: float
    utilization_percent: float = 0.0
    temperature_celsius: Optional[float] = None


class SystemInfo(BaseModel):
    """System information."""

    model_config = ConfigDict(extra="forbid")

    cpu_count: int
    cpu_percent: float
    memory_total_gb: float
    memory_used_gb: float
    memory_percent: float
    disk_total_gb: float
    disk_used_gb: float
    disk_percent: float
    gpus: List[GPUInfo] = Field(default_factory=list)
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    torch_version: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["healthy", "unhealthy"]
    version: str
    uptime_seconds: float
    active_jobs: int = 0
    gpu_available: bool = False


class ErrorResponse(BaseModel):
    """Error response schema."""

    model_config = ConfigDict(extra="forbid")

    error: str
    detail: Optional[str] = None
    code: Optional[str] = None


class VerificationRequest(BaseModel):
    """Request to verify a model."""

    model_config = ConfigDict(extra="forbid")

    model_path: str
    level: Literal["quick", "standard", "thorough"] = "standard"


class VerificationResponse(BaseModel):
    """Response from model verification."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    level: str
    model_path: str
    summary: str
    recommendations: List[str]
    validated_at: datetime
    validation_time_seconds: float


class WebSocketMessage(BaseModel):
    """WebSocket message format."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["progress", "status", "log", "error", "ping", "pong"]
    job_id: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
