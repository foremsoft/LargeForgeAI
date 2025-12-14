"""Data models for cost tracking."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class GPUUsageEntry:
    """A single GPU usage snapshot."""

    timestamp: datetime
    gpu_id: int
    memory_used_gb: float
    memory_total_gb: float
    utilization_percent: float
    temperature_celsius: float = 0.0
    power_watts: float = 0.0
    job_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "gpu_id": self.gpu_id,
            "memory_used_gb": self.memory_used_gb,
            "memory_total_gb": self.memory_total_gb,
            "utilization_percent": self.utilization_percent,
            "temperature_celsius": self.temperature_celsius,
            "power_watts": self.power_watts,
            "job_id": self.job_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GPUUsageEntry":
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            gpu_id=data["gpu_id"],
            memory_used_gb=data["memory_used_gb"],
            memory_total_gb=data["memory_total_gb"],
            utilization_percent=data["utilization_percent"],
            temperature_celsius=data.get("temperature_celsius", 0.0),
            power_watts=data.get("power_watts", 0.0),
            job_id=data.get("job_id"),
        )


@dataclass
class CostEntry:
    """Cost tracking entry for a job or time period."""

    job_id: str
    gpu_hours: float
    estimated_cost_usd: float
    start_time: datetime
    end_time: Optional[datetime] = None
    gpu_type: str = "unknown"
    gpu_count: int = 1
    usage_entries: List[GPUUsageEntry] = field(default_factory=list)
    peak_memory_gb: float = 0.0
    avg_utilization_percent: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "gpu_hours": self.gpu_hours,
            "estimated_cost_usd": self.estimated_cost_usd,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "gpu_type": self.gpu_type,
            "gpu_count": self.gpu_count,
            "usage_entries": [e.to_dict() for e in self.usage_entries],
            "peak_memory_gb": self.peak_memory_gb,
            "avg_utilization_percent": self.avg_utilization_percent,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CostEntry":
        """Create from dictionary."""
        return cls(
            job_id=data["job_id"],
            gpu_hours=data["gpu_hours"],
            estimated_cost_usd=data["estimated_cost_usd"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            gpu_type=data.get("gpu_type", "unknown"),
            gpu_count=data.get("gpu_count", 1),
            usage_entries=[GPUUsageEntry.from_dict(e) for e in data.get("usage_entries", [])],
            peak_memory_gb=data.get("peak_memory_gb", 0.0),
            avg_utilization_percent=data.get("avg_utilization_percent", 0.0),
        )

    def calculate_metrics(self) -> None:
        """Calculate peak memory and average utilization from usage entries."""
        if not self.usage_entries:
            return

        self.peak_memory_gb = max(e.memory_used_gb for e in self.usage_entries)
        self.avg_utilization_percent = sum(e.utilization_percent for e in self.usage_entries) / len(self.usage_entries)


@dataclass
class CostConfig:
    """Configuration for cost calculations."""

    # GPU cost per hour in USD (can be customized per GPU type)
    gpu_costs_per_hour: Dict[str, float] = field(default_factory=lambda: {
        # NVIDIA Consumer GPUs
        "NVIDIA GeForce RTX 4090": 0.50,
        "NVIDIA GeForce RTX 4080": 0.40,
        "NVIDIA GeForce RTX 4070": 0.30,
        "NVIDIA GeForce RTX 3090": 0.35,
        "NVIDIA GeForce RTX 3080": 0.25,

        # NVIDIA Data Center GPUs
        "NVIDIA A100": 2.50,
        "NVIDIA A100-80GB": 3.00,
        "NVIDIA H100": 4.00,
        "NVIDIA A10": 1.00,
        "NVIDIA V100": 1.50,
        "NVIDIA T4": 0.50,

        # NVIDIA Professional GPUs
        "NVIDIA RTX A6000": 1.20,
        "NVIDIA RTX A5000": 0.80,

        # Default fallback
        "default": 1.00,
    })

    # Electricity cost per kWh in USD
    electricity_cost_per_kwh: float = 0.12

    # Include electricity cost in estimates
    include_electricity: bool = True

    def get_gpu_cost(self, gpu_name: str) -> float:
        """Get hourly cost for a GPU type."""
        # Try exact match first
        if gpu_name in self.gpu_costs_per_hour:
            return self.gpu_costs_per_hour[gpu_name]

        # Try partial match
        for key, cost in self.gpu_costs_per_hour.items():
            if key.lower() in gpu_name.lower() or gpu_name.lower() in key.lower():
                return cost

        return self.gpu_costs_per_hour.get("default", 1.00)


@dataclass
class CostSummary:
    """Summary of costs over a period."""

    period_start: datetime
    period_end: datetime
    total_gpu_hours: float
    total_cost_usd: float
    job_count: int
    entries: List[CostEntry]
    by_gpu_type: Dict[str, float] = field(default_factory=dict)
    by_job: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_gpu_hours": self.total_gpu_hours,
            "total_cost_usd": self.total_cost_usd,
            "job_count": self.job_count,
            "by_gpu_type": self.by_gpu_type,
            "by_job": self.by_job,
        }
