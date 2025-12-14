"""CLI commands for deployment generation."""

from pathlib import Path
from typing import Optional

import click

from largeforge.utils import get_logger

logger = get_logger(__name__)


@click.group()
def deploy() -> None:
    """Deployment generation commands."""
    pass


@deploy.command("generate")
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default="./deployment", help="Output directory")
@click.option(
    "--backend", "-b",
    type=click.Choice(["auto", "transformers", "vllm"]),
    default="auto",
    help="Inference backend"
)
@click.option("--port", "-p", default=8000, type=int, help="Server port")
@click.option("--gpu/--no-gpu", default=True, help="Enable GPU support")
@click.option("--name", "-n", default="largeforge-inference", help="Service/container name")
@click.option("--include-model/--no-include-model", default=False, help="Copy model into Docker image")
def generate_deployment(
    model_path: str,
    output: str,
    backend: str,
    port: int,
    gpu: bool,
    name: str,
    include_model: bool,
) -> None:
    """Generate a complete deployment bundle for a trained model.

    Creates all necessary files to deploy your model as a Docker service:
    - Dockerfile
    - docker-compose.yml
    - requirements.txt
    - main.py (inference server)
    - README.md
    - config.yaml

    Examples:
        # Basic deployment
        largeforge deploy generate ./output/model

        # Custom output and port
        largeforge deploy generate ./output/model -o ./deploy -p 8080

        # vLLM backend with custom name
        largeforge deploy generate ./output/model --backend vllm --name my-llm-service
    """
    from largeforge.deployment import DeploymentBundle, BundleConfig

    click.echo(f"Generating deployment bundle for: {model_path}")
    click.echo(f"Output directory: {output}")
    click.echo(f"Backend: {backend}")
    click.echo(f"Port: {port}")
    click.echo(f"GPU enabled: {gpu}")
    click.echo("")

    config = BundleConfig(
        model_path=str(Path(model_path).resolve()),
        output_dir=output,
        backend=backend,
        port=port,
        gpu_enabled=gpu,
        container_name=name,
        bundle_name=name,
        include_model=include_model,
    )

    bundle = DeploymentBundle(config)
    bundle.generate()

    files = bundle.get_file_list()
    click.echo(f"\nGenerated {len(files)} files:")
    for f in files:
        click.echo(f"  - {f}")

    click.echo(f"\nDeployment bundle ready at: {output}")
    click.echo("\nTo deploy:")
    click.echo(f"  cd {output}")
    click.echo("  docker compose up -d")


@deploy.command("build")
@click.option("--context", "-c", type=click.Path(exists=True), default=".", help="Build context directory")
@click.option("--tag", "-t", default="largeforge-inference:latest", help="Image tag")
@click.option("--push/--no-push", default=False, help="Push to registry after build")
@click.option("--no-cache", is_flag=True, help="Build without cache")
def build_image(context: str, tag: str, push: bool, no_cache: bool) -> None:
    """Build a Docker image from deployment bundle.

    Examples:
        # Build from current directory
        largeforge deploy build

        # Build with custom tag
        largeforge deploy build -t myregistry/mymodel:v1.0

        # Build and push
        largeforge deploy build -t myregistry/mymodel:v1.0 --push
    """
    import subprocess

    click.echo(f"Building Docker image: {tag}")
    click.echo(f"Context: {context}")

    # Verify Dockerfile exists
    dockerfile_path = Path(context) / "Dockerfile"
    if not dockerfile_path.exists():
        click.echo("Error: Dockerfile not found in context directory", err=True)
        click.echo("Run 'largeforge deploy generate' first", err=True)
        raise click.Abort()

    # Build command
    cmd = ["docker", "build", "-t", tag]
    if no_cache:
        cmd.append("--no-cache")
    cmd.append(context)

    try:
        click.echo("\nBuilding image...")
        result = subprocess.run(cmd, check=True)
        click.echo(f"\nImage built successfully: {tag}")
    except subprocess.CalledProcessError as e:
        click.echo(f"Build failed with exit code {e.returncode}", err=True)
        raise click.Abort()
    except FileNotFoundError:
        click.echo("Error: docker not found. Please install Docker", err=True)
        raise click.Abort()

    # Push if requested
    if push:
        click.echo(f"\nPushing to registry...")
        try:
            subprocess.run(["docker", "push", tag], check=True)
            click.echo(f"Image pushed: {tag}")
        except subprocess.CalledProcessError as e:
            click.echo(f"Push failed with exit code {e.returncode}", err=True)
            raise click.Abort()


@deploy.command("validate")
@click.option("--compose", "-c", type=click.Path(exists=True), default="docker-compose.yml", help="Compose file path")
@click.option("--dockerfile", "-d", type=click.Path(exists=True), help="Dockerfile path")
def validate_deployment(compose: str, dockerfile: Optional[str]) -> None:
    """Validate deployment configuration files.

    Checks that your deployment files are correctly formatted
    and contain all required configurations.

    Examples:
        largeforge deploy validate
        largeforge deploy validate --compose ./deploy/docker-compose.yml
    """
    import yaml

    errors = []
    warnings = []

    click.echo("Validating deployment configuration...")
    click.echo("")

    # Validate docker-compose.yml
    click.echo(f"Checking: {compose}")
    try:
        with open(compose, "r") as f:
            compose_data = yaml.safe_load(f)

        # Check required fields
        if "services" not in compose_data:
            errors.append("docker-compose.yml: missing 'services' section")
        else:
            for service_name, service in compose_data["services"].items():
                if "image" not in service and "build" not in service:
                    errors.append(f"Service '{service_name}': missing 'image' or 'build'")

                if "ports" not in service:
                    warnings.append(f"Service '{service_name}': no ports exposed")

        click.echo(f"  - Format: OK")
        click.echo(f"  - Services: {len(compose_data.get('services', {}))}")

    except yaml.YAMLError as e:
        errors.append(f"docker-compose.yml: invalid YAML - {e}")
    except FileNotFoundError:
        errors.append(f"docker-compose.yml not found at {compose}")

    # Validate Dockerfile if specified or found
    dockerfile_path = dockerfile or "Dockerfile"
    if Path(dockerfile_path).exists():
        click.echo(f"Checking: {dockerfile_path}")

        with open(dockerfile_path, "r") as f:
            dockerfile_content = f.read()

        # Basic checks
        if "FROM" not in dockerfile_content:
            errors.append("Dockerfile: missing FROM instruction")
        if "EXPOSE" not in dockerfile_content:
            warnings.append("Dockerfile: no EXPOSE instruction")
        if "CMD" not in dockerfile_content and "ENTRYPOINT" not in dockerfile_content:
            warnings.append("Dockerfile: no CMD or ENTRYPOINT instruction")

        click.echo("  - Syntax: OK")

    # Display results
    click.echo("")

    if warnings:
        click.echo("Warnings:")
        for warning in warnings:
            click.secho(f"  - {warning}", fg="yellow")

    if errors:
        click.echo("Errors:")
        for error in errors:
            click.secho(f"  - {error}", fg="red")
        click.echo("")
        click.secho("Validation FAILED", fg="red", bold=True)
        raise click.Abort()
    else:
        click.secho("Validation PASSED", fg="green", bold=True)


@deploy.command("run")
@click.option("--compose", "-c", type=click.Path(exists=True), default="docker-compose.yml", help="Compose file")
@click.option("--detach/--no-detach", "-d", default=True, help="Run in background")
@click.option("--build/--no-build", default=False, help="Build images before starting")
def run_deployment(compose: str, detach: bool, build: bool) -> None:
    """Start the deployment using Docker Compose.

    Examples:
        # Start in background
        largeforge deploy run

        # Start with build
        largeforge deploy run --build

        # Start in foreground
        largeforge deploy run --no-detach
    """
    import subprocess

    cmd = ["docker", "compose", "-f", compose, "up"]

    if detach:
        cmd.append("-d")
    if build:
        cmd.append("--build")

    click.echo(f"Starting deployment from {compose}...")

    try:
        subprocess.run(cmd, check=True)
        if detach:
            click.echo("\nDeployment started in background")
            click.echo(f"View logs: docker compose -f {compose} logs -f")
            click.echo(f"Stop: docker compose -f {compose} down")
    except subprocess.CalledProcessError as e:
        click.echo(f"Failed to start: exit code {e.returncode}", err=True)
        raise click.Abort()
    except FileNotFoundError:
        click.echo("Error: docker not found. Please install Docker", err=True)
        raise click.Abort()


@deploy.command("stop")
@click.option("--compose", "-c", type=click.Path(exists=True), default="docker-compose.yml", help="Compose file")
@click.option("--volumes/--no-volumes", "-v", default=False, help="Remove volumes")
def stop_deployment(compose: str, volumes: bool) -> None:
    """Stop the deployment.

    Examples:
        largeforge deploy stop
        largeforge deploy stop --volumes
    """
    import subprocess

    cmd = ["docker", "compose", "-f", compose, "down"]
    if volumes:
        cmd.append("-v")

    click.echo(f"Stopping deployment...")

    try:
        subprocess.run(cmd, check=True)
        click.echo("Deployment stopped")
    except subprocess.CalledProcessError as e:
        click.echo(f"Failed to stop: exit code {e.returncode}", err=True)
        raise click.Abort()
