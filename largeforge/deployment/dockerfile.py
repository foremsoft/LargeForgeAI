"""Dockerfile generator for LargeForgeAI deployments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

from jinja2 import Template

from largeforge.config.base import BaseConfig
from largeforge.deployment.templates import (
    DOCKERFILE_TRANSFORMERS,
    DOCKERFILE_CUDA,
    DOCKERFILE_VLLM,
    DOCKERIGNORE_TEMPLATE,
)
from largeforge.utils import get_logger, ensure_dir

logger = get_logger(__name__)


@dataclass
class DockerConfig(BaseConfig):
    """Configuration for Dockerfile generation."""

    # Base images
    base_image: str = "python:3.10-slim"
    cuda_image: str = "nvidia/cuda:12.1.0-runtime-ubuntu22.04"
    vllm_version: str = "latest"

    # Backend selection
    backend: Literal["auto", "transformers", "vllm"] = "auto"

    # Model configuration
    model_path: str = ""
    model_copy_mode: Literal["copy", "mount"] = "mount"

    # Server configuration
    port: int = 8000
    healthcheck: bool = True

    # Build options
    multi_stage: bool = True
    cuda_version: Optional[str] = "12.1"
    cuda_devices: str = "0"

    # vLLM options
    tensor_parallel: int = 1
    max_model_len: Optional[int] = None

    # Additional packages
    extra_requirements: List[str] = field(default_factory=list)

    # Quantization
    quantization: Optional[Literal["awq", "gptq"]] = None


class DockerGenerator:
    """Generates Dockerfiles for LargeForgeAI inference servers."""

    def __init__(self, config: DockerConfig):
        """
        Initialize Dockerfile generator.

        Args:
            config: Docker configuration
        """
        self.config = config

    def generate(self) -> str:
        """
        Generate Dockerfile content.

        Returns:
            Dockerfile content as string
        """
        backend = self._select_backend()
        template = self._get_template(backend)

        context = self._build_context(backend)
        dockerfile = Template(template).render(**context)

        logger.info(f"Generated Dockerfile for {backend} backend")
        return dockerfile

    def generate_dockerignore(self) -> str:
        """
        Generate .dockerignore content.

        Returns:
            .dockerignore content as string
        """
        return DOCKERIGNORE_TEMPLATE

    def save(self, output_dir: str) -> None:
        """
        Save Dockerfile and .dockerignore to directory.

        Args:
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        ensure_dir(output_path)

        # Write Dockerfile
        dockerfile_path = output_path / "Dockerfile"
        with open(dockerfile_path, "w") as f:
            f.write(self.generate())
        logger.info(f"Saved Dockerfile to {dockerfile_path}")

        # Write .dockerignore
        dockerignore_path = output_path / ".dockerignore"
        with open(dockerignore_path, "w") as f:
            f.write(self.generate_dockerignore())
        logger.info(f"Saved .dockerignore to {dockerignore_path}")

    def _select_backend(self) -> str:
        """
        Select the appropriate backend.

        Returns:
            Backend name: "transformers", "vllm", or "cuda"
        """
        if self.config.backend != "auto":
            return self.config.backend

        # Auto-detect based on configuration
        if self.config.tensor_parallel > 1:
            return "vllm"

        if self.config.cuda_version:
            return "cuda"

        return "transformers"

    def _get_template(self, backend: str) -> str:
        """
        Get the appropriate template for backend.

        Args:
            backend: Backend name

        Returns:
            Template string
        """
        templates = {
            "transformers": DOCKERFILE_TRANSFORMERS,
            "cuda": DOCKERFILE_CUDA,
            "vllm": DOCKERFILE_VLLM,
        }
        return templates.get(backend, DOCKERFILE_TRANSFORMERS)

    def _build_context(self, backend: str) -> dict:
        """
        Build template context.

        Args:
            backend: Selected backend

        Returns:
            Template context dictionary
        """
        # Determine model path for environment variable
        if self.config.model_copy_mode == "copy":
            model_path_env = "/app/model"
            copy_model = True
        else:
            model_path_env = self.config.model_path
            copy_model = False

        context = {
            # Base images
            "base_image": self.config.base_image,
            "cuda_image": self.config.cuda_image,
            "vllm_version": self.config.vllm_version,
            # Model
            "model_path": self.config.model_path,
            "model_path_env": model_path_env,
            "copy_model": copy_model,
            # Server
            "port": self.config.port,
            "healthcheck": self.config.healthcheck,
            # CUDA
            "cuda_devices": self.config.cuda_devices,
            # vLLM
            "tensor_parallel": self.config.tensor_parallel,
            "max_model_len": self.config.max_model_len,
        }

        return context

    def get_requirements(self) -> List[str]:
        """
        Get list of Python requirements.

        Returns:
            List of requirement strings
        """
        requirements = [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "pydantic>=2.5.0",
            "torch>=2.1.0",
            "transformers>=4.36.0",
            "accelerate>=0.25.0",
            "peft>=0.7.0",
            "python-multipart>=0.0.6",
            "httpx>=0.25.0",
            "pyyaml>=6.0.1",
        ]

        backend = self._select_backend()
        if backend == "vllm":
            requirements.append("vllm>=0.2.0")

        if self.config.quantization == "awq":
            requirements.append("autoawq>=0.1.6")
        elif self.config.quantization == "gptq":
            requirements.append("auto-gptq>=0.5.0")

        requirements.extend(self.config.extra_requirements)

        return requirements


def generate_dockerfile(
    model_path: str,
    output_dir: str,
    backend: str = "auto",
    port: int = 8000,
    cuda: bool = True,
    **kwargs,
) -> str:
    """
    Convenience function to generate a Dockerfile.

    Args:
        model_path: Path to model
        output_dir: Output directory
        backend: Backend to use
        port: Server port
        cuda: Enable CUDA support
        **kwargs: Additional DockerConfig options

    Returns:
        Path to generated Dockerfile
    """
    config = DockerConfig(
        model_path=model_path,
        backend=backend,
        port=port,
        cuda_version="12.1" if cuda else None,
        **kwargs,
    )

    generator = DockerGenerator(config)
    generator.save(output_dir)

    return str(Path(output_dir) / "Dockerfile")
