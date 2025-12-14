"""Synthetic data generation CLI commands."""

from pathlib import Path
from typing import List, Optional

import click

from largeforge.utils import get_logger

logger = get_logger(__name__)


@click.group()
def synthetic() -> None:
    """Synthetic data generation commands."""
    pass


@synthetic.command()
@click.option(
    "--output", "-o",
    type=click.Path(),
    required=True,
    help="Output file path (.jsonl or .json)",
)
@click.option(
    "--num-samples", "-n",
    type=int,
    default=100,
    help="Number of samples to generate",
)
@click.option(
    "--format", "-f",
    type=click.Choice(["sft", "dpo"]),
    default="sft",
    help="Output format (SFT or DPO)",
)
@click.option(
    "--provider", "-p",
    type=click.Choice(["openai", "anthropic"]),
    default="openai",
    help="API provider",
)
@click.option(
    "--model", "-m",
    type=str,
    default=None,
    help="Model to use (defaults: gpt-4 for OpenAI, claude-3-sonnet for Anthropic)",
)
@click.option(
    "--topic", "-t",
    multiple=True,
    help="Topics for generation (can specify multiple)",
)
@click.option(
    "--api-key",
    type=str,
    envvar=["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
    help="API key (or set via environment variable)",
)
@click.option(
    "--temperature",
    type=float,
    default=0.8,
    help="Generation temperature (0.0-2.0)",
)
@click.option(
    "--rate-limit",
    type=float,
    default=0.5,
    help="Delay between API calls in seconds",
)
@click.pass_context
def generate(
    ctx: click.Context,
    output: str,
    num_samples: int,
    format: str,
    provider: str,
    model: Optional[str],
    topic: tuple,
    api_key: Optional[str],
    temperature: float,
    rate_limit: float,
) -> None:
    """Generate synthetic training data using LLM APIs.

    Uses OpenAI or Anthropic APIs to generate high-quality training data
    for SFT (instruction-following) or DPO (preference) training.

    Examples:

        # Generate 100 SFT samples using GPT-4
        largeforge synthetic generate -o data.jsonl -n 100

        # Generate DPO preference data with custom topics
        largeforge synthetic generate -o prefs.jsonl -f dpo -t "coding" -t "math"

        # Use Anthropic Claude
        largeforge synthetic generate -o data.jsonl -p anthropic -m claude-3-opus-20240229
    """
    from largeforge.data.synthetic import SyntheticConfig, SyntheticGenerator

    click.echo(f"Generating {num_samples} {format.upper()} samples")
    click.echo(f"Provider: {provider}")
    click.echo(f"Model: {model or 'default'}")

    # Set default model based on provider
    if model is None:
        model = "gpt-4" if provider == "openai" else "claude-3-sonnet-20240229"

    # Build topics list
    topics = list(topic) if topic else None

    try:
        config = SyntheticConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            num_samples=num_samples,
            format=format,
            topics=topics or config.topics if hasattr(config, 'topics') else None,
            temperature=temperature,
            rate_limit_delay=rate_limit,
        )
    except ValueError as e:
        raise click.ClickException(str(e))

    # Progress callback
    with click.progressbar(length=num_samples, label="Generating") as bar:
        last_progress = 0

        def progress_callback(current: int, total: int) -> None:
            nonlocal last_progress
            bar.update(current - last_progress)
            last_progress = current

        generator = SyntheticGenerator(config)

        try:
            data = generator.generate(progress_callback=progress_callback)
        except Exception as e:
            raise click.ClickException(f"Generation failed: {e}")

    # Save data
    if not data:
        raise click.ClickException("No samples generated successfully")

    output_format = "jsonl" if output.endswith(".jsonl") else "json"
    generator.save(data, output, format=output_format)

    click.echo(f"\nGenerated {len(data)} samples")
    click.echo(f"Saved to: {output}")

    # Show sample
    if data and ctx.obj.get("verbose"):
        click.echo("\nSample output:")
        import json
        click.echo(json.dumps(data[0], indent=2))


@synthetic.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--output", "-o",
    type=click.Path(),
    required=True,
    help="Output file path",
)
@click.option(
    "--factor", "-f",
    type=int,
    default=2,
    help="Augmentation factor (2 = double dataset size)",
)
@click.option(
    "--provider", "-p",
    type=click.Choice(["openai", "anthropic"]),
    default="openai",
    help="API provider",
)
@click.option(
    "--model", "-m",
    type=str,
    default=None,
    help="Model to use",
)
@click.option(
    "--api-key",
    type=str,
    envvar=["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
    help="API key",
)
@click.pass_context
def augment(
    ctx: click.Context,
    input_path: str,
    output: str,
    factor: int,
    provider: str,
    model: Optional[str],
    api_key: Optional[str],
) -> None:
    """Augment existing dataset with synthetic variations.

    Takes an existing SFT dataset and generates variations of each sample
    to increase dataset size while maintaining diversity.

    INPUT_PATH: Path to existing dataset file

    Example:
        largeforge synthetic augment data.jsonl -o augmented.jsonl --factor 3
    """
    from largeforge.data import load_dataset
    from largeforge.data.synthetic import augment_dataset
    from largeforge.utils import save_jsonl, save_json

    click.echo(f"Loading dataset: {input_path}")

    try:
        data = load_dataset(input_path)
    except Exception as e:
        raise click.ClickException(f"Failed to load dataset: {e}")

    click.echo(f"Loaded {len(data)} records")
    click.echo(f"Augmentation factor: {factor}x")

    expected_size = len(data) * factor
    click.echo(f"Expected output size: ~{expected_size} records")

    try:
        augmented = augment_dataset(
            data=data,
            augmentation_factor=factor,
            provider=provider,
            model=model,
            api_key=api_key,
        )
    except Exception as e:
        raise click.ClickException(f"Augmentation failed: {e}")

    # Save output
    output_path = Path(output)
    if output_path.suffix == ".jsonl":
        save_jsonl(augmented, output)
    else:
        save_json(augmented, output)

    click.echo(f"\nAugmented dataset from {len(data)} to {len(augmented)} records")
    click.echo(f"Saved to: {output}")


@synthetic.command()
@click.option(
    "--provider", "-p",
    type=click.Choice(["openai", "anthropic"]),
    default="openai",
    help="API provider",
)
@click.option(
    "--model", "-m",
    type=str,
    default=None,
    help="Model to use",
)
@click.option(
    "--api-key",
    type=str,
    envvar=["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
    help="API key",
)
def test_connection(
    provider: str,
    model: Optional[str],
    api_key: Optional[str],
) -> None:
    """Test API connection and credentials.

    Verifies that the API key is valid and the model is accessible.

    Example:
        largeforge synthetic test-connection --provider openai
    """
    from largeforge.data.synthetic import SyntheticConfig, SyntheticGenerator

    click.echo(f"Testing connection to {provider}...")

    if model is None:
        model = "gpt-4" if provider == "openai" else "claude-3-sonnet-20240229"

    try:
        config = SyntheticConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            num_samples=1,
        )
    except ValueError as e:
        raise click.ClickException(str(e))

    generator = SyntheticGenerator(config)

    try:
        # Simple test call
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Connection successful!' and nothing else."},
        ]
        response = generator._call_api(messages)
        click.echo(f"Response: {response}")
        click.echo(click.style("Connection successful!", fg="green"))

    except Exception as e:
        raise click.ClickException(f"Connection failed: {e}")


@synthetic.command()
def list_topics() -> None:
    """List default topics for data generation.

    Shows the built-in topics that are used when no custom topics are specified.

    Example:
        largeforge synthetic list-topics
    """
    from largeforge.data.synthetic import DEFAULT_TOPICS

    click.echo("Default topics for synthetic data generation:")
    click.echo("-" * 40)
    for i, topic in enumerate(DEFAULT_TOPICS, 1):
        click.echo(f"  {i}. {topic}")

    click.echo("\nUse --topic/-t to specify custom topics:")
    click.echo("  largeforge synthetic generate -o data.jsonl -t 'custom topic 1' -t 'custom topic 2'")


@synthetic.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output file path (defaults to input with _validated suffix)",
)
@click.option(
    "--format", "-f",
    type=click.Choice(["sft", "dpo"]),
    default="sft",
    help="Expected format",
)
@click.option(
    "--fix/--no-fix",
    default=False,
    help="Attempt to fix invalid records",
)
def validate_synthetic(
    input_path: str,
    output: Optional[str],
    format: str,
    fix: bool,
) -> None:
    """Validate synthetic dataset quality.

    Checks that generated data has all required fields and reasonable content.

    INPUT_PATH: Path to synthetic dataset file

    Example:
        largeforge synthetic validate-synthetic data.jsonl --format sft
    """
    from largeforge.data import load_dataset
    from largeforge.utils import save_jsonl

    click.echo(f"Validating synthetic dataset: {input_path}")
    click.echo(f"Expected format: {format.upper()}")

    data = load_dataset(input_path)
    click.echo(f"Loaded {len(data)} records")

    valid_records = []
    invalid_records = []
    fixed_records = []

    # Required fields by format
    required_fields = {
        "sft": ["instruction", "output"],
        "dpo": ["prompt", "chosen", "rejected"],
    }

    required = required_fields.get(format, [])

    for i, record in enumerate(data):
        # Check required fields
        missing = [f for f in required if not record.get(f)]

        if missing:
            invalid_records.append((i, f"Missing fields: {missing}"))
            continue

        # Check for empty content
        empty = [f for f in required if not record.get(f, "").strip()]

        if empty:
            if fix:
                # Can't fix empty content, skip
                invalid_records.append((i, f"Empty fields: {empty}"))
            else:
                invalid_records.append((i, f"Empty fields: {empty}"))
            continue

        # Check for reasonable length
        min_length = 10
        short = [
            f for f in required
            if len(record.get(f, "").strip()) < min_length
        ]

        if short:
            if fix:
                # Mark as potentially low quality but keep
                fixed_records.append(record)
            else:
                invalid_records.append((i, f"Very short content in: {short}"))
                continue

        valid_records.append(record)

    # Report
    click.echo(f"\nValidation Results:")
    click.echo(f"  Valid: {len(valid_records)}")
    click.echo(f"  Invalid: {len(invalid_records)}")
    if fix:
        click.echo(f"  Fixed: {len(fixed_records)}")

    # Show invalid samples
    if invalid_records and len(invalid_records) <= 10:
        click.echo("\nInvalid records:")
        for idx, reason in invalid_records:
            click.echo(f"  Record {idx}: {reason}")
    elif invalid_records:
        click.echo(f"\nShowing first 10 of {len(invalid_records)} invalid records:")
        for idx, reason in invalid_records[:10]:
            click.echo(f"  Record {idx}: {reason}")

    # Save valid records
    if output:
        all_valid = valid_records + fixed_records
        save_jsonl(all_valid, output)
        click.echo(f"\nSaved {len(all_valid)} valid records to: {output}")
