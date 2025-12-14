"""CLI commands for web server management."""

from pathlib import Path
from typing import Optional

import click

from largeforge.utils import get_logger

logger = get_logger(__name__)


@click.group()
def web() -> None:
    """Web server commands."""
    pass


@web.command()
@click.option("--host", "-h", default="0.0.0.0", help="Host to bind to")
@click.option("--port", "-p", default=7860, type=int, help="Port to bind to")
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to config YAML")
@click.option("--debug/--no-debug", default=False, help="Enable debug mode")
@click.option("--reload/--no-reload", default=False, help="Enable auto-reload for development")
@click.option("--workers", "-w", default=1, type=int, help="Number of worker processes")
def start(
    host: str,
    port: int,
    config: Optional[str],
    debug: bool,
    reload: bool,
    workers: int,
) -> None:
    """Start the LargeForgeAI web server.

    The web server provides:
    - REST API for training job management
    - WebSocket for real-time progress updates
    - React frontend for the web UI

    Examples:
        # Start with defaults
        largeforge web start

        # Start on specific port with debug mode
        largeforge web start --port 8080 --debug

        # Start with custom config
        largeforge web start --config config/web.yaml
    """
    from largeforge.config.web import WebConfig
    from largeforge.web.app import run_server

    click.echo(f"Starting LargeForgeAI Web Server")
    click.echo(f"  Host: {host}")
    click.echo(f"  Port: {port}")
    click.echo(f"  Debug: {debug}")
    click.echo(f"  Workers: {workers}")

    # Load or create config
    if config:
        from largeforge.utils import load_yaml
        config_data = load_yaml(config)
        web_config = WebConfig(**config_data)
    else:
        web_config = WebConfig(
            host=host,
            port=port,
            debug=debug,
        )

    click.echo(f"\nAPI documentation: http://{host}:{port}/docs")
    click.echo(f"Web UI: http://{host}:{port}/")
    click.echo("Press Ctrl+C to stop\n")

    try:
        run_server(
            host=host,
            port=port,
            config=web_config,
            reload=reload,
            workers=workers,
        )
    except KeyboardInterrupt:
        click.echo("\nServer stopped")


@web.command()
@click.option("--output", "-o", type=click.Path(), default="./frontend/dist", help="Output directory")
@click.option("--install/--no-install", default=True, help="Install npm dependencies first")
def build_frontend(output: str, install: bool) -> None:
    """Build the React frontend for production.

    This command builds the React frontend and outputs the static files
    to the specified directory, ready to be served by the web server.

    Examples:
        # Build with defaults
        largeforge web build-frontend

        # Build to custom directory
        largeforge web build-frontend --output ./static
    """
    import subprocess
    import shutil

    frontend_dir = Path(__file__).parent.parent / "web" / "frontend"

    if not frontend_dir.exists():
        click.echo(f"Error: Frontend directory not found at {frontend_dir}", err=True)
        raise click.Abort()

    click.echo(f"Building React frontend from {frontend_dir}")

    # Install dependencies
    if install:
        click.echo("Installing npm dependencies...")
        try:
            subprocess.run(
                ["npm", "install"],
                cwd=str(frontend_dir),
                check=True,
                capture_output=True,
            )
            click.echo("Dependencies installed")
        except subprocess.CalledProcessError as e:
            click.echo(f"Failed to install dependencies: {e.stderr.decode()}", err=True)
            raise click.Abort()
        except FileNotFoundError:
            click.echo("Error: npm not found. Please install Node.js", err=True)
            raise click.Abort()

    # Build
    click.echo("Building production bundle...")
    try:
        subprocess.run(
            ["npm", "run", "build"],
            cwd=str(frontend_dir),
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        click.echo(f"Build failed: {e.stderr.decode()}", err=True)
        raise click.Abort()

    # Copy to output
    build_dir = frontend_dir / "dist"
    output_path = Path(output)

    if build_dir.exists():
        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.copytree(build_dir, output_path)
        click.echo(f"Frontend built successfully to {output_path}")
    else:
        click.echo("Error: Build directory not found", err=True)
        raise click.Abort()


@web.command()
@click.option("--port", "-p", default=7860, type=int, help="Port to check")
def status(port: int) -> None:
    """Check the status of the web server.

    Examples:
        largeforge web status
        largeforge web status --port 8080
    """
    import httpx

    url = f"http://localhost:{port}/health"

    try:
        response = httpx.get(url, timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            click.echo(f"Web server status: {data.get('status', 'unknown')}")
            click.echo(f"  Version: {data.get('version', 'unknown')}")
            click.echo(f"  Uptime: {data.get('uptime_seconds', 0):.0f}s")
            click.echo(f"  Active jobs: {data.get('active_jobs', 0)}")
            click.echo(f"  GPU available: {data.get('gpu_available', False)}")
        else:
            click.echo(f"Server returned status {response.status_code}", err=True)
    except httpx.ConnectError:
        click.echo(f"Server not running on port {port}", err=True)
    except Exception as e:
        click.echo(f"Error checking status: {e}", err=True)


@web.command()
@click.option("--all", "-a", "show_all", is_flag=True, help="Show all jobs including completed")
@click.option("--limit", "-n", default=10, type=int, help="Number of jobs to show")
@click.option("--port", "-p", default=7860, type=int, help="Server port")
def jobs(show_all: bool, limit: int, port: int) -> None:
    """List training jobs from the web server.

    Examples:
        largeforge web jobs
        largeforge web jobs --all --limit 20
    """
    import httpx

    url = f"http://localhost:{port}/api/v1/jobs"
    params = {"limit": limit}

    if not show_all:
        params["status"] = "running"

    try:
        response = httpx.get(url, params=params, timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            jobs = data.get("jobs", [])

            if not jobs:
                click.echo("No jobs found")
                return

            click.echo(f"{'ID':<36} {'Status':<12} {'Type':<6} {'Progress':<10}")
            click.echo("-" * 70)

            for job in jobs:
                job_id = job["job_id"][:36]
                status = job["status"]
                training_type = job["training_type"]

                progress = job.get("progress")
                if progress:
                    pct = (progress["step"] / progress["total_steps"]) * 100 if progress["total_steps"] > 0 else 0
                    progress_str = f"{pct:.1f}%"
                else:
                    progress_str = "-"

                click.echo(f"{job_id:<36} {status:<12} {training_type:<6} {progress_str:<10}")

        else:
            click.echo(f"Error: {response.json().get('detail', 'Unknown error')}", err=True)

    except httpx.ConnectError:
        click.echo(f"Server not running on port {port}", err=True)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
