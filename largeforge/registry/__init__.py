"""Model registry module for LargeForgeAI.

This module provides model version management and registry capabilities.

Example:
    >>> from largeforge.registry import get_registry, ModelStage
    >>>
    >>> # Register a model
    >>> registry = get_registry()
    >>> model = registry.register(
    ...     name="my-chat-model",
    ...     path="./output/model",
    ...     base_model="Qwen/Qwen2.5-7B-Instruct",
    ...     description="Fine-tuned chat model",
    ...     tags=["chat", "production"],
    ...     metrics={"eval_loss": 0.5, "perplexity": 8.2},
    ... )
    >>>
    >>> # Add a new version
    >>> registry.add_version(
    ...     name="my-chat-model",
    ...     path="./output/model-v2",
    ...     metrics={"eval_loss": 0.4},
    ... )
    >>>
    >>> # Promote to production
    >>> registry.transition_stage(
    ...     name="my-chat-model",
    ...     version="v1.0.1",
    ...     stage=ModelStage.PRODUCTION,
    ... )
"""

from largeforge.registry.models import (
    ModelDeployment,
    ModelStage,
    ModelVersion,
    RegisteredModel,
)
from largeforge.registry.service import (
    ModelRegistry,
    get_registry,
    increment_version,
    parse_version,
)

__all__ = [
    # Models
    "ModelDeployment",
    "ModelStage",
    "ModelVersion",
    "RegisteredModel",
    # Service
    "ModelRegistry",
    "get_registry",
    "increment_version",
    "parse_version",
]
