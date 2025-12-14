"""Docker Compose file generator for LargeForgeAI deployments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from jinja2 import Template

from largeforge.config.base import BaseConfig
from largeforge.deployment.templates import DOCKER_COMPOSE_TEMPLATE
from largeforge.utils import get_logger, ensure_dir

logger = get_logger(__name__)


@dataclass
class ComposeConfig(BaseConfig):
    """Configuration for Docker Compose generation."""

    # Service configuration
    service_name: str = "largeforge-inference"
    container_name: str = "largeforge-inference"

    # Image or build
    image: Optional[str] = None
    build_context: str = "."

    # Model configuration
    model_path: str = ""

    # Networking
    host_port: int = 8000
    container_port: int = 8000

    # GPU configuration
    gpu_enabled: bool = True
    gpu_count: Optional[int] = None
    gpu_ids: Optional[List[int]] = None

    # Resource limits
    memory_limit: Optional[str] = None

    # Environment variables
    environment: Dict[str, str] = field(default_factory=dict)

    # Volumes
    volumes: List[str] = field(default_factory=list)

    # Health check
    healthcheck: bool = True
    healthcheck_start_period: str = "120s"

    # Labels
    labels: Dict[str, str] = field(default_factory=dict)

    # Networks
    networks: List[str] = field(default_factory=list)


class ComposeGenerator:
    """Generates Docker Compose files for LargeForgeAI deployments."""

    def __init__(self, config: ComposeConfig):
        """
        Initialize Docker Compose generator.

        Args:
            config: Compose configuration
        """
        self.config = config

    def generate(self) -> Dict[str, Any]:
        """
        Generate Docker Compose configuration as dictionary.

        Returns:
            Docker Compose configuration dictionary
        """
        compose = {
            "version": "3.8",
            "services": {
                self.config.service_name: self._generate_service()
            }
        }

        if self.config.networks:
            compose["networks"] = self._generate_networks()

        logger.info(f"Generated Docker Compose for service {self.config.service_name}")
        return compose

    def to_yaml(self) -> str:
        """
        Generate Docker Compose configuration as YAML string.

        Returns:
            YAML string
        """
        compose = self.generate()
        return yaml.dump(compose, default_flow_style=False, sort_keys=False)

    def to_template(self) -> str:
        """
        Generate Docker Compose using Jinja2 template.

        Returns:
            Rendered template string
        """
        context = self._build_template_context()
        return Template(DOCKER_COMPOSE_TEMPLATE).render(**context)

    def save(self, output_path: str) -> None:
        """
        Save Docker Compose file.

        Args:
            output_path: Output file path
        """
        output_path = Path(output_path)
        ensure_dir(output_path.parent)

        # Use YAML generation for clean output
        content = self.to_yaml()

        with open(output_path, "w") as f:
            f.write(content)

        logger.info(f"Saved Docker Compose to {output_path}")

    def _generate_service(self) -> Dict[str, Any]:
        """Generate service configuration."""
        service = {}

        # Build or image
        if self.config.image:
            service["image"] = self.config.image
        else:
            service["build"] = {
                "context": self.config.build_context,
                "dockerfile": "Dockerfile",
            }

        service["container_name"] = self.config.container_name
        service["restart"] = "unless-stopped"

        # Ports
        service["ports"] = [
            f"{self.config.host_port}:{self.config.container_port}"
        ]

        # Environment
        if self.config.environment:
            service["environment"] = self.config.environment

        # Volumes
        volumes = list(self.config.volumes)
        if self.config.model_path and not any(
            self.config.model_path in v for v in volumes
        ):
            # Add model volume if using mount mode
            if not self.config.model_path.startswith("/app"):
                volumes.append(
                    f"{self.config.model_path}:/app/model:ro"
                )
        if volumes:
            service["volumes"] = volumes

        # GPU configuration
        if self.config.gpu_enabled:
            service["deploy"] = self._generate_gpu_config()

        # Memory limit
        if self.config.memory_limit:
            service["mem_limit"] = self.config.memory_limit

        # Health check
        if self.config.healthcheck:
            service["healthcheck"] = self._generate_healthcheck()

        # Labels
        if self.config.labels:
            service["labels"] = self.config.labels

        return service

    def _generate_gpu_config(self) -> Dict[str, Any]:
        """Generate GPU deployment configuration."""
        gpu_config = {
            "resources": {
                "reservations": {
                    "devices": [
                        {
                            "driver": "nvidia",
                            "capabilities": ["gpu"],
                        }
                    ]
                }
            }
        }

        device = gpu_config["resources"]["reservations"]["devices"][0]

        if self.config.gpu_ids:
            device["device_ids"] = [str(id) for id in self.config.gpu_ids]
        elif self.config.gpu_count:
            device["count"] = self.config.gpu_count
        else:
            device["count"] = "all"

        return gpu_config

    def _generate_healthcheck(self) -> Dict[str, Any]:
        """Generate health check configuration."""
        return {
            "test": [
                "CMD",
                "curl",
                "-f",
                f"http://localhost:{self.config.container_port}/health",
            ],
            "interval": "30s",
            "timeout": "10s",
            "retries": 3,
            "start_period": self.config.healthcheck_start_period,
        }

    def _generate_networks(self) -> Dict[str, Any]:
        """Generate networks configuration."""
        networks = {}
        for network in self.config.networks:
            networks[network] = {"external": True}
        return networks

    def _build_template_context(self) -> Dict[str, Any]:
        """Build context for Jinja2 template."""
        return {
            "service_name": self.config.service_name,
            "container_name": self.config.container_name,
            "image": self.config.image,
            "build_context": self.config.build_context if not self.config.image else None,
            "host_port": self.config.host_port,
            "container_port": self.config.container_port,
            "environment": self.config.environment,
            "volumes": self.config.volumes,
            "gpu_enabled": self.config.gpu_enabled,
            "gpu_count": self.config.gpu_count,
            "gpu_ids": self.config.gpu_ids,
            "memory_limit": self.config.memory_limit,
            "healthcheck": self.config.healthcheck,
            "healthcheck_start_period": self.config.healthcheck_start_period,
            "labels": self.config.labels,
            "networks": self.config.networks,
        }


def generate_compose(
    model_path: str,
    output_path: str,
    service_name: str = "largeforge-inference",
    port: int = 8000,
    gpu: bool = True,
    **kwargs,
) -> str:
    """
    Convenience function to generate a Docker Compose file.

    Args:
        model_path: Path to model
        output_path: Output file path
        service_name: Service name
        port: Server port
        gpu: Enable GPU support
        **kwargs: Additional ComposeConfig options

    Returns:
        Path to generated file
    """
    config = ComposeConfig(
        model_path=model_path,
        service_name=service_name,
        host_port=port,
        container_port=port,
        gpu_enabled=gpu,
        **kwargs,
    )

    generator = ComposeGenerator(config)
    generator.save(output_path)

    return output_path
