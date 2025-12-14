"""Main CLI entry point."""

import sys
from typing import Optional

import click

from largeforge.version import __version__


@click.group()
@click.version_option(version=__version__, prog_name="largeforge")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """LargeForgeAI - LLM Training and Deployment Toolkit.

    A comprehensive framework for fine-tuning, deploying, and serving
    large language models with support for SFT, DPO, LoRA, and more.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    if verbose:
        from largeforge.utils import set_log_level
        set_log_level("DEBUG")


@cli.group()
def train() -> None:
    """Training commands for fine-tuning models."""
    pass


@cli.group()
def serve() -> None:
    """Inference server commands."""
    pass


@cli.group()
def data() -> None:
    """Data processing commands."""
    pass


@cli.group()
def model() -> None:
    """Model management commands."""
    pass


# Import and register subcommands
from largeforge.cli.train import sft, dpo
from largeforge.cli.serve import start, generate
from largeforge.cli.data import validate, convert, stats

# Import new command groups
from largeforge.cli.web import web
from largeforge.cli.verify import verify
from largeforge.cli.deploy import deploy
from largeforge.cli.export import export
from largeforge.cli.synthetic import synthetic

# Register train subcommands
train.add_command(sft)
train.add_command(dpo)

# Register serve subcommands
serve.add_command(start)
serve.add_command(generate)

# Register data subcommands
data.add_command(validate)
data.add_command(convert)
data.add_command(stats)

# Register new command groups
cli.add_command(web)
cli.add_command(verify)
cli.add_command(deploy)
cli.add_command(export)
cli.add_command(synthetic)


@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to config file")
def info(config: Optional[str]) -> None:
    """Display system and configuration information."""
    import torch
    from largeforge.utils import get_device, get_device_count, is_bf16_supported

    click.echo("LargeForgeAI System Information")
    click.echo("=" * 40)
    click.echo(f"Version: {__version__}")
    click.echo(f"Python: {sys.version.split()[0]}")
    click.echo(f"PyTorch: {torch.__version__}")
    click.echo(f"Device: {get_device()}")
    click.echo(f"GPU Count: {get_device_count()}")
    click.echo(f"BF16 Support: {is_bf16_supported()}")

    if torch.cuda.is_available():
        click.echo(f"CUDA Version: {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            click.echo(f"GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")

    if config:
        click.echo(f"\nConfig file: {config}")
        from largeforge.utils import load_yaml
        cfg = load_yaml(config)
        for key, value in cfg.items():
            click.echo(f"  {key}: {value}")


def main() -> None:
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
