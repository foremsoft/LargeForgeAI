"""Device management utilities for LargeForgeAI."""

from typing import Optional

import torch


def get_device() -> str:
    """
    Get the best available device.

    Returns:
        "cuda" if GPU available, otherwise "cpu"
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_device_count() -> int:
    """
    Get the number of available GPUs.

    Returns:
        Number of CUDA devices, 0 if none available
    """
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def get_device_name(index: int = 0) -> str:
    """
    Get the name of a GPU device.

    Args:
        index: GPU device index

    Returns:
        Device name or "CPU" if no GPU
    """
    if torch.cuda.is_available() and index < torch.cuda.device_count():
        return torch.cuda.get_device_name(index)
    return "CPU"


def get_device_memory(index: int = 0) -> dict:
    """
    Get GPU memory information.

    Args:
        index: GPU device index

    Returns:
        Dict with total_gb and free_gb, or None values if no GPU
    """
    if torch.cuda.is_available() and index < torch.cuda.device_count():
        total = torch.cuda.get_device_properties(index).total_memory
        allocated = torch.cuda.memory_allocated(index)
        free = total - allocated
        return {
            "total_gb": total / (1024**3),
            "free_gb": free / (1024**3),
            "allocated_gb": allocated / (1024**3),
        }
    return {"total_gb": None, "free_gb": None, "allocated_gb": None}


def select_device(preference: str = "auto") -> torch.device:
    """
    Select a torch device based on preference.

    Args:
        preference: "auto", "cuda", "cpu", or "cuda:N"

    Returns:
        torch.device instance
    """
    if preference == "auto":
        device_str = get_device()
    elif preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    else:
        device_str = preference

    return torch.device(device_str)


def is_bf16_supported() -> bool:
    """
    Check if BF16 is supported on the current GPU.

    Returns:
        True if BF16 is supported, False otherwise
    """
    if not torch.cuda.is_available():
        return False

    # BF16 requires compute capability >= 8.0 (Ampere+)
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def get_optimal_dtype() -> torch.dtype:
    """
    Get the optimal dtype for the current hardware.

    Returns:
        torch.bfloat16 if supported, otherwise torch.float16
    """
    if is_bf16_supported():
        return torch.bfloat16
    elif torch.cuda.is_available():
        return torch.float16
    return torch.float32


def empty_cache() -> None:
    """Clear CUDA cache if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def synchronize() -> None:
    """Synchronize CUDA if available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
