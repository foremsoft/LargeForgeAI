"""Deployment generation module for LargeForgeAI.

Provides tools for generating Docker deployment artifacts:
- Dockerfile generation for vLLM and transformers backends
- Docker Compose configuration
- Complete deployment bundles with all necessary files

Example:
    >>> from largeforge.deployment import DeploymentBundle, BundleConfig
    >>> config = BundleConfig(model_path="./output/my-model", output_dir="./deploy")
    >>> bundle = DeploymentBundle(config)
    >>> bundle.generate()
"""

from largeforge.deployment.dockerfile import DockerConfig, DockerGenerator
from largeforge.deployment.compose import ComposeConfig, ComposeGenerator
from largeforge.deployment.bundle import BundleConfig, DeploymentBundle

__all__ = [
    # Dockerfile
    "DockerConfig",
    "DockerGenerator",
    # Docker Compose
    "ComposeConfig",
    "ComposeGenerator",
    # Bundle
    "BundleConfig",
    "DeploymentBundle",
]
