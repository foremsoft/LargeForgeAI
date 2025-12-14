"""Cost tracking service."""

import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from largeforge.costs.models import (
    CostConfig,
    CostEntry,
    CostSummary,
    GPUUsageEntry,
)
from largeforge.utils import get_logger, ensure_dir

logger = get_logger(__name__)


def get_gpu_info() -> List[Dict[str, Any]]:
    """Get current GPU information using pynvml or torch."""
    gpus = []

    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_allocated = torch.cuda.memory_allocated(i) / 1e9
                memory_total = props.total_memory / 1e9

                gpus.append({
                    "id": i,
                    "name": props.name,
                    "memory_used_gb": memory_allocated,
                    "memory_total_gb": memory_total,
                    "utilization_percent": 0.0,  # Not available from torch
                    "temperature_celsius": 0.0,
                    "power_watts": 0.0,
                })
    except ImportError:
        pass

    # Try pynvml for more detailed info
    try:
        import pynvml
        pynvml.nvmlInit()

        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

            try:
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temperature = 0

            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to watts
            except:
                power = 0

            # Update or add GPU info
            if i < len(gpus):
                gpus[i].update({
                    "name": name,
                    "memory_used_gb": memory_info.used / 1e9,
                    "memory_total_gb": memory_info.total / 1e9,
                    "utilization_percent": utilization.gpu,
                    "temperature_celsius": temperature,
                    "power_watts": power,
                })
            else:
                gpus.append({
                    "id": i,
                    "name": name,
                    "memory_used_gb": memory_info.used / 1e9,
                    "memory_total_gb": memory_info.total / 1e9,
                    "utilization_percent": utilization.gpu,
                    "temperature_celsius": temperature,
                    "power_watts": power,
                })

        pynvml.nvmlShutdown()
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to get GPU info via pynvml: {e}")

    return gpus


class CostTracker:
    """Service for tracking GPU usage and costs."""

    def __init__(
        self,
        storage_path: str = ".largeforge/costs",
        config: Optional[CostConfig] = None,
        sample_interval: float = 60.0,
    ):
        """
        Initialize cost tracker.

        Args:
            storage_path: Path to store cost data
            config: Cost configuration
            sample_interval: Interval in seconds between GPU samples
        """
        self.storage_path = Path(storage_path)
        ensure_dir(self.storage_path)
        self.config = config or CostConfig()
        self.sample_interval = sample_interval

        self._lock = threading.Lock()
        self._active_jobs: Dict[str, CostEntry] = {}
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._entries: Dict[str, CostEntry] = {}
        self._load_entries()

    def _load_entries(self) -> None:
        """Load existing cost entries from storage."""
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                entry = CostEntry.from_dict(data)
                self._entries[entry.job_id] = entry
            except Exception as e:
                logger.warning(f"Failed to load cost entry {file_path}: {e}")

    def _save_entry(self, entry: CostEntry) -> None:
        """Save a cost entry to storage."""
        file_path = self.storage_path / f"{entry.job_id}.json"
        with open(file_path, "w") as f:
            json.dump(entry.to_dict(), f, indent=2)

    def start_tracking(self, job_id: str) -> CostEntry:
        """
        Start tracking costs for a job.

        Args:
            job_id: ID of the training job

        Returns:
            Created cost entry
        """
        gpu_info = get_gpu_info()
        gpu_type = gpu_info[0]["name"] if gpu_info else "unknown"
        gpu_count = len(gpu_info) if gpu_info else 1

        entry = CostEntry(
            job_id=job_id,
            gpu_hours=0.0,
            estimated_cost_usd=0.0,
            start_time=datetime.utcnow(),
            gpu_type=gpu_type,
            gpu_count=gpu_count,
        )

        with self._lock:
            self._active_jobs[job_id] = entry
            self._entries[job_id] = entry

        # Start monitoring if not already running
        self._start_monitoring()

        logger.info(f"Started cost tracking for job {job_id} ({gpu_type})")
        return entry

    def stop_tracking(self, job_id: str) -> Optional[CostEntry]:
        """
        Stop tracking costs for a job and calculate final costs.

        Args:
            job_id: ID of the training job

        Returns:
            Final cost entry or None if not found
        """
        with self._lock:
            entry = self._active_jobs.pop(job_id, None)

            if not entry:
                return self._entries.get(job_id)

            # Calculate final metrics
            entry.end_time = datetime.utcnow()
            duration_hours = (entry.end_time - entry.start_time).total_seconds() / 3600
            entry.gpu_hours = duration_hours * entry.gpu_count

            # Calculate cost
            hourly_rate = self.config.get_gpu_cost(entry.gpu_type)
            entry.estimated_cost_usd = entry.gpu_hours * hourly_rate

            # Add electricity cost if enabled
            if self.config.include_electricity and entry.usage_entries:
                avg_power_watts = sum(e.power_watts for e in entry.usage_entries) / len(entry.usage_entries)
                kwh = (avg_power_watts / 1000) * duration_hours * entry.gpu_count
                entry.estimated_cost_usd += kwh * self.config.electricity_cost_per_kwh

            # Calculate metrics from usage entries
            entry.calculate_metrics()

            self._entries[job_id] = entry
            self._save_entry(entry)

        logger.info(
            f"Stopped cost tracking for job {job_id}: "
            f"{entry.gpu_hours:.2f} GPU-hours, ${entry.estimated_cost_usd:.2f}"
        )
        return entry

    def _start_monitoring(self) -> None:
        """Start the background monitoring thread."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def _stop_monitoring(self) -> None:
        """Stop the background monitoring thread."""
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None

    def _monitor_loop(self) -> None:
        """Background loop that samples GPU usage."""
        while not self._stop_event.is_set():
            try:
                self._sample_usage()
            except Exception as e:
                logger.warning(f"Error sampling GPU usage: {e}")

            # Sleep in small intervals to allow quick shutdown
            for _ in range(int(self.sample_interval)):
                if self._stop_event.is_set():
                    break
                time.sleep(1.0)

    def _sample_usage(self) -> None:
        """Sample current GPU usage for all active jobs."""
        gpu_info = get_gpu_info()
        if not gpu_info:
            return

        timestamp = datetime.utcnow()

        with self._lock:
            for job_id, entry in self._active_jobs.items():
                for gpu in gpu_info:
                    usage = GPUUsageEntry(
                        timestamp=timestamp,
                        gpu_id=gpu["id"],
                        memory_used_gb=gpu["memory_used_gb"],
                        memory_total_gb=gpu["memory_total_gb"],
                        utilization_percent=gpu["utilization_percent"],
                        temperature_celsius=gpu["temperature_celsius"],
                        power_watts=gpu["power_watts"],
                        job_id=job_id,
                    )
                    entry.usage_entries.append(usage)

    def get_current_usage(self) -> List[GPUUsageEntry]:
        """Get current GPU usage snapshot."""
        gpu_info = get_gpu_info()
        timestamp = datetime.utcnow()

        return [
            GPUUsageEntry(
                timestamp=timestamp,
                gpu_id=gpu["id"],
                memory_used_gb=gpu["memory_used_gb"],
                memory_total_gb=gpu["memory_total_gb"],
                utilization_percent=gpu["utilization_percent"],
                temperature_celsius=gpu["temperature_celsius"],
                power_watts=gpu["power_watts"],
            )
            for gpu in gpu_info
        ]

    def get_job_costs(self, job_id: str) -> Optional[CostEntry]:
        """Get cost entry for a job."""
        return self._entries.get(job_id)

    def get_total_costs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> float:
        """
        Get total costs for a time period.

        Args:
            start_date: Start of period (default: beginning of time)
            end_date: End of period (default: now)

        Returns:
            Total cost in USD
        """
        total = 0.0

        for entry in self._entries.values():
            if start_date and entry.start_time < start_date:
                continue
            if end_date and entry.start_time > end_date:
                continue
            total += entry.estimated_cost_usd

        return total

    def get_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> CostSummary:
        """
        Get cost summary for a time period.

        Args:
            start_date: Start of period
            end_date: End of period

        Returns:
            Cost summary
        """
        start_date = start_date or datetime.min
        end_date = end_date or datetime.utcnow()

        entries = []
        total_hours = 0.0
        total_cost = 0.0
        by_gpu_type: Dict[str, float] = {}
        by_job: Dict[str, float] = {}

        for entry in self._entries.values():
            if entry.start_time < start_date or entry.start_time > end_date:
                continue

            entries.append(entry)
            total_hours += entry.gpu_hours
            total_cost += entry.estimated_cost_usd

            by_gpu_type[entry.gpu_type] = by_gpu_type.get(entry.gpu_type, 0.0) + entry.estimated_cost_usd
            by_job[entry.job_id] = entry.estimated_cost_usd

        return CostSummary(
            period_start=start_date,
            period_end=end_date,
            total_gpu_hours=total_hours,
            total_cost_usd=total_cost,
            job_count=len(entries),
            entries=entries,
            by_gpu_type=by_gpu_type,
            by_job=by_job,
        )

    def get_daily_summary(self, date: Optional[datetime] = None) -> CostSummary:
        """Get cost summary for a specific day."""
        date = date or datetime.utcnow()
        start = datetime(date.year, date.month, date.day)
        end = start + timedelta(days=1)
        return self.get_summary(start, end)

    def get_monthly_summary(self, year: int, month: int) -> CostSummary:
        """Get cost summary for a specific month."""
        start = datetime(year, month, 1)
        if month == 12:
            end = datetime(year + 1, 1, 1)
        else:
            end = datetime(year, month + 1, 1)
        return self.get_summary(start, end)


# Global instance
_tracker: Optional[CostTracker] = None


def get_cost_tracker(
    storage_path: str = ".largeforge/costs",
    config: Optional[CostConfig] = None,
) -> CostTracker:
    """Get or create the global cost tracker."""
    global _tracker
    if _tracker is None:
        _tracker = CostTracker(storage_path, config)
    return _tracker
