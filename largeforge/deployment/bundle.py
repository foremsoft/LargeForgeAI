"""Complete deployment bundle generator for LargeForgeAI."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional

from jinja2 import Template

from largeforge.config.base import BaseConfig
from largeforge.deployment.dockerfile import DockerConfig, DockerGenerator
from largeforge.deployment.compose import ComposeConfig, ComposeGenerator
from largeforge.deployment.templates import (
    REQUIREMENTS_TEMPLATE,
    SERVICE_MAIN_TEMPLATE,
    README_TEMPLATE,
    CONFIG_YAML_TEMPLATE,
    ENV_EXAMPLE_TEMPLATE,
)
from largeforge.utils import get_logger, ensure_dir

logger = get_logger(__name__)


@dataclass
class BundleConfig(BaseConfig):
    """Configuration for deployment bundle generation."""

    # Required
    model_path: str = ""
    output_dir: str = "./deployment"

    # Bundle naming
    bundle_name: str = "largeforge-deployment"
    container_name: str = "largeforge-inference"

    # Backend
    backend: Literal["auto", "transformers", "vllm"] = "auto"

    # Server configuration
    port: int = 8000
    host: str = "0.0.0.0"

    # GPU configuration
    gpu_enabled: bool = True
    gpu_count: Optional[int] = None
    cuda_version: str = "12.1"

    # Model options
    include_model: bool = False
    quantization: Optional[Literal["awq", "gptq"]] = None

    # Generation options
    generate_readme: bool = True
    generate_config: bool = True
    generate_env_example: bool = True

    # vLLM specific
    tensor_parallel: int = 1
    max_model_len: Optional[int] = None

    # Minimum VRAM requirement
    min_vram_gb: float = 8.0


class DeploymentBundle:
    """Generates complete deployment bundles with all necessary files."""

    def __init__(self, config: BundleConfig):
        """
        Initialize deployment bundle generator.

        Args:
            config: Bundle configuration
        """
        self.config = config
        self.output_path = Path(config.output_dir)
        self._generated_files: List[str] = []

    def generate(self) -> str:
        """
        Generate all deployment files.

        Returns:
            Path to output directory
        """
        logger.info(f"Generating deployment bundle to {self.output_path}")

        # Create output directory
        ensure_dir(self.output_path)
        self._generated_files = []

        # Generate each component
        self._generate_dockerfile()
        self._generate_compose()
        self._generate_requirements()
        self._generate_service_code()

        if self.config.generate_config:
            self._generate_config()

        if self.config.generate_readme:
            self._generate_readme()

        if self.config.generate_env_example:
            self._generate_env_example()

        logger.info(f"Generated {len(self._generated_files)} files")
        return str(self.output_path)

    def get_file_list(self) -> List[str]:
        """Get list of generated files."""
        return self._generated_files

    def _generate_dockerfile(self) -> None:
        """Generate Dockerfile and .dockerignore."""
        docker_config = DockerConfig(
            model_path=self.config.model_path,
            backend=self.config.backend,
            port=self.config.port,
            cuda_version=self.config.cuda_version if self.config.gpu_enabled else None,
            model_copy_mode="copy" if self.config.include_model else "mount",
            tensor_parallel=self.config.tensor_parallel,
            max_model_len=self.config.max_model_len,
            quantization=self.config.quantization,
        )

        generator = DockerGenerator(docker_config)
        generator.save(str(self.output_path))

        self._generated_files.extend(["Dockerfile", ".dockerignore"])

    def _generate_compose(self) -> None:
        """Generate docker-compose.yml."""
        # Build environment variables
        environment = {
            "MODEL_PATH": "/app/model" if self.config.include_model else self.config.model_path,
            "PORT": str(self.config.port),
            "HOST": self.config.host,
        }

        # Build volumes
        volumes = []
        if not self.config.include_model:
            volumes.append(f"{self.config.model_path}:/app/model:ro")

        compose_config = ComposeConfig(
            service_name=self.config.container_name,
            container_name=self.config.container_name,
            model_path=self.config.model_path,
            host_port=self.config.port,
            container_port=self.config.port,
            gpu_enabled=self.config.gpu_enabled,
            gpu_count=self.config.gpu_count,
            environment=environment,
            volumes=volumes,
            healthcheck=True,
            healthcheck_start_period="180s" if self.config.backend == "vllm" else "120s",
        )

        generator = ComposeGenerator(compose_config)
        output_file = self.output_path / "docker-compose.yml"
        generator.save(str(output_file))

        self._generated_files.append("docker-compose.yml")

    def _generate_requirements(self) -> None:
        """Generate requirements.txt."""
        context = {
            "backend": self.config.backend,
            "quantization": self.config.quantization is not None,
        }

        content = Template(REQUIREMENTS_TEMPLATE).render(**context)

        requirements_path = self.output_path / "requirements.txt"
        with open(requirements_path, "w") as f:
            f.write(content)

        self._generated_files.append("requirements.txt")
        logger.debug("Generated requirements.txt")

    def _generate_service_code(self) -> None:
        """Generate main.py service code."""
        context = {
            "model_path": "/app/model" if self.config.include_model else self.config.model_path,
            "port": self.config.port,
        }

        content = Template(SERVICE_MAIN_TEMPLATE).render(**context)

        main_path = self.output_path / "main.py"
        with open(main_path, "w") as f:
            f.write(content)

        self._generated_files.append("main.py")
        logger.debug("Generated main.py")

    def _generate_config(self) -> None:
        """Generate config.yaml."""
        context = {
            "model_path": "/app/model" if self.config.include_model else self.config.model_path,
            "device": "cuda" if self.config.gpu_enabled else "cpu",
            "dtype": "bfloat16" if self.config.gpu_enabled else "float32",
            "port": self.config.port,
            "quantization": self.config.quantization,
            "quant_bits": 4 if self.config.quantization else None,
        }

        content = Template(CONFIG_YAML_TEMPLATE).render(**context)

        config_path = self.output_path / "config.yaml"
        with open(config_path, "w") as f:
            f.write(content)

        self._generated_files.append("config.yaml")
        logger.debug("Generated config.yaml")

    def _generate_readme(self) -> None:
        """Generate README.md."""
        context = {
            "bundle_name": self.config.bundle_name,
            "image_name": self.config.container_name,
            "container_name": self.config.container_name,
            "port": self.config.port,
            "model_path": self.config.model_path,
            "gpu_enabled": self.config.gpu_enabled,
            "cuda_version": self.config.cuda_version,
            "min_vram_gb": self.config.min_vram_gb,
        }

        content = Template(README_TEMPLATE).render(**context)

        readme_path = self.output_path / "README.md"
        with open(readme_path, "w") as f:
            f.write(content)

        self._generated_files.append("README.md")
        logger.debug("Generated README.md")

    def _generate_env_example(self) -> None:
        """Generate .env.example."""
        context = {
            "model_path": "/app/model" if self.config.include_model else self.config.model_path,
            "port": self.config.port,
        }

        content = Template(ENV_EXAMPLE_TEMPLATE).render(**context)

        env_path = self.output_path / ".env.example"
        with open(env_path, "w") as f:
            f.write(content)

        self._generated_files.append(".env.example")
        logger.debug("Generated .env.example")


def generate_deployment(
    model_path: str,
    output_dir: str = "./deployment",
    backend: str = "auto",
    port: int = 8000,
    gpu: bool = True,
    **kwargs,
) -> str:
    """
    Convenience function to generate a complete deployment bundle.

    Args:
        model_path: Path to trained model
        output_dir: Output directory for deployment files
        backend: Inference backend (auto, transformers, vllm)
        port: Server port
        gpu: Enable GPU support
        **kwargs: Additional BundleConfig options

    Returns:
        Path to generated deployment bundle
    """
    config = BundleConfig(
        model_path=model_path,
        output_dir=output_dir,
        backend=backend,
        port=port,
        gpu_enabled=gpu,
        **kwargs,
    )

    bundle = DeploymentBundle(config)
    return bundle.generate()


def verify_bundle(bundle_path: str) -> Dict[str, bool]:
    """
    Verify a deployment bundle has all required files.

    Args:
        bundle_path: Path to deployment bundle

    Returns:
        Dictionary of file names to existence status
    """
    required_files = [
        "Dockerfile",
        "docker-compose.yml",
        "requirements.txt",
        "main.py",
    ]

    optional_files = [
        ".dockerignore",
        "config.yaml",
        "README.md",
        ".env.example",
    ]

    bundle_path = Path(bundle_path)
    results = {}

    for filename in required_files:
        results[filename] = (bundle_path / filename).exists()

    for filename in optional_files:
        if (bundle_path / filename).exists():
            results[filename] = True

    return results
