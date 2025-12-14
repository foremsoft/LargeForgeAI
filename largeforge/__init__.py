"""LargeForgeAI - A comprehensive LLM training and inference framework."""

from largeforge.version import __version__, get_version, get_version_info

# Config exports
from largeforge.config import (
    BaseConfig,
    ModelConfig,
    TrainingConfig,
    LoRAConfig,
    SFTConfig,
    DPOConfig,
    GenerationConfig,
    InferenceConfig,
    RouterConfig,
)

# Utils exports
from largeforge.utils import (
    get_logger,
    timed_operation,
    set_log_level,
    get_device,
    get_device_count,
    is_bf16_supported,
    get_optimal_dtype,
    select_device,
    load_json,
    save_json,
    load_jsonl,
    save_jsonl,
    atomic_write,
    ensure_dir,
)

__all__ = [
    # Version
    "__version__",
    "get_version",
    "get_version_info",
    # Config
    "BaseConfig",
    "ModelConfig",
    "TrainingConfig",
    "LoRAConfig",
    "SFTConfig",
    "DPOConfig",
    "GenerationConfig",
    "InferenceConfig",
    "RouterConfig",
    # Utils - Logging
    "get_logger",
    "timed_operation",
    "set_log_level",
    # Utils - Device
    "get_device",
    "get_device_count",
    "is_bf16_supported",
    "get_optimal_dtype",
    "select_device",
    # Utils - IO
    "load_json",
    "save_json",
    "load_jsonl",
    "save_jsonl",
    "atomic_write",
    "ensure_dir",
]
