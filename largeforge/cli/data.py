"""Data processing CLI commands."""

from pathlib import Path
from typing import Optional

import click

from largeforge.utils import get_logger

logger = get_logger(__name__)


@click.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--format", "-f", "data_format",
              type=click.Choice(["auto", "alpaca", "sharegpt", "dpo"]),
              default="auto", help="Data format to validate against")
@click.option("--strict/--no-strict", default=False, help="Strict validation mode")
@click.option("--max-errors", type=int, default=10, help="Maximum errors to display")
@click.option("--output", "-o", type=click.Path(), help="Write valid records to file")
@click.pass_context
def validate(
    ctx: click.Context,
    data_path: str,
    data_format: str,
    strict: bool,
    max_errors: int,
    output: Optional[str],
) -> None:
    """Validate a dataset for training.

    DATA_PATH: Path to dataset file (JSON, JSONL)

    Checks data format, required fields, and content quality.

    Example:
        largeforge data validate dataset.jsonl --format alpaca
    """
    click.echo(f"Validating dataset: {data_path}")
    click.echo(f"Format: {data_format}")

    # Load data
    from largeforge.data import load_dataset

    try:
        data = load_dataset(data_path)
    except Exception as e:
        raise click.ClickException(f"Failed to load data: {e}")

    click.echo(f"Loaded {len(data)} records")

    # Auto-detect format if needed
    if data_format == "auto":
        data_format = _detect_format(data[0] if data else {})
        click.echo(f"Detected format: {data_format}")

    # Validate
    from largeforge.data import validate_dataset

    results = validate_dataset(data, format_type=data_format)

    # Display results
    valid_count = sum(1 for r in results if r["valid"])
    invalid_count = len(results) - valid_count

    click.echo(f"\nValidation Results:")
    click.echo(f"  Valid records: {valid_count}")
    click.echo(f"  Invalid records: {invalid_count}")

    if invalid_count > 0:
        click.echo(f"\nFirst {min(max_errors, invalid_count)} errors:")
        error_count = 0
        for i, result in enumerate(results):
            if not result["valid"]:
                click.echo(f"  Record {i}: {result.get('errors', ['Unknown error'])}")
                error_count += 1
                if error_count >= max_errors:
                    remaining = invalid_count - max_errors
                    if remaining > 0:
                        click.echo(f"  ... and {remaining} more errors")
                    break

    # Write valid records if requested
    if output and valid_count > 0:
        valid_data = [data[i] for i, r in enumerate(results) if r["valid"]]

        from largeforge.utils import save_jsonl, save_json

        output_path = Path(output)
        if output_path.suffix == ".jsonl":
            save_jsonl(valid_data, output)
        else:
            save_json(valid_data, output)

        click.echo(f"\nWrote {valid_count} valid records to {output}")

    # Exit with error if validation failed
    if strict and invalid_count > 0:
        raise click.ClickException(f"Validation failed: {invalid_count} invalid records")


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--from", "from_format", required=True,
              type=click.Choice(["alpaca", "sharegpt", "oai"]),
              help="Input format")
@click.option("--to", "to_format", required=True,
              type=click.Choice(["alpaca", "sharegpt", "chat"]),
              help="Output format")
@click.option("--system-prompt", "-s", type=str, help="System prompt to add")
@click.pass_context
def convert(
    ctx: click.Context,
    input_path: str,
    output_path: str,
    from_format: str,
    to_format: str,
    system_prompt: Optional[str],
) -> None:
    """Convert dataset between formats.

    INPUT_PATH: Path to input dataset
    OUTPUT_PATH: Path to output dataset

    Example:
        largeforge data convert data.json out.jsonl --from alpaca --to sharegpt
    """
    click.echo(f"Converting {input_path}")
    click.echo(f"  From: {from_format}")
    click.echo(f"  To: {to_format}")

    # Load data
    from largeforge.data import load_dataset

    data = load_dataset(input_path)
    click.echo(f"Loaded {len(data)} records")

    # Convert
    from largeforge.data.converters import (
        alpaca_to_sharegpt,
        sharegpt_to_alpaca,
        to_chat_format,
    )

    converted = []

    if from_format == "alpaca" and to_format == "sharegpt":
        converted = alpaca_to_sharegpt(data, system_prompt=system_prompt)
    elif from_format == "sharegpt" and to_format == "alpaca":
        converted = sharegpt_to_alpaca(data)
    elif to_format == "chat":
        converted = to_chat_format(data, format=from_format)
    else:
        raise click.ClickException(
            f"Conversion from {from_format} to {to_format} not supported"
        )

    click.echo(f"Converted {len(converted)} records")

    # Save output
    from largeforge.utils import save_jsonl, save_json

    output_path_obj = Path(output_path)
    if output_path_obj.suffix == ".jsonl":
        save_jsonl(converted, output_path)
    else:
        save_json(converted, output_path)

    click.echo(f"Saved to {output_path}")


@click.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--format", "-f", "data_format",
              type=click.Choice(["auto", "alpaca", "sharegpt", "dpo"]),
              default="auto", help="Data format")
@click.option("--detailed/--no-detailed", default=False, help="Show detailed statistics")
@click.pass_context
def stats(
    ctx: click.Context,
    data_path: str,
    data_format: str,
    detailed: bool,
) -> None:
    """Show dataset statistics.

    DATA_PATH: Path to dataset file

    Example:
        largeforge data stats dataset.jsonl --detailed
    """
    click.echo(f"Dataset Statistics: {data_path}")
    click.echo("=" * 50)

    # Load data
    from largeforge.data import load_dataset

    data = load_dataset(data_path)

    # Basic stats
    click.echo(f"Total records: {len(data)}")

    if not data:
        return

    # Auto-detect format
    if data_format == "auto":
        data_format = _detect_format(data[0])
        click.echo(f"Detected format: {data_format}")

    # Format-specific stats
    if data_format == "alpaca":
        _alpaca_stats(data, detailed)
    elif data_format == "sharegpt":
        _sharegpt_stats(data, detailed)
    elif data_format == "dpo":
        _dpo_stats(data, detailed)


def _detect_format(record: dict) -> str:
    """Auto-detect data format from a sample record."""
    if "conversations" in record:
        return "sharegpt"
    elif "chosen" in record and "rejected" in record:
        return "dpo"
    elif "instruction" in record:
        return "alpaca"
    else:
        return "alpaca"  # Default


def _alpaca_stats(data: list, detailed: bool) -> None:
    """Show Alpaca format statistics."""
    has_input = sum(1 for d in data if d.get("input"))
    has_system = sum(1 for d in data if d.get("system"))

    instruction_lens = [len(d.get("instruction", "")) for d in data]
    output_lens = [len(d.get("output", "")) for d in data]

    click.echo(f"\nAlpaca Format Statistics:")
    click.echo(f"  Records with input: {has_input} ({100*has_input/len(data):.1f}%)")
    click.echo(f"  Records with system: {has_system} ({100*has_system/len(data):.1f}%)")
    click.echo(f"\nInstruction length:")
    click.echo(f"  Min: {min(instruction_lens)}, Max: {max(instruction_lens)}")
    click.echo(f"  Avg: {sum(instruction_lens)/len(instruction_lens):.1f}")
    click.echo(f"\nOutput length:")
    click.echo(f"  Min: {min(output_lens)}, Max: {max(output_lens)}")
    click.echo(f"  Avg: {sum(output_lens)/len(output_lens):.1f}")

    if detailed:
        # Word count distribution
        word_counts = [len(d.get("output", "").split()) for d in data]
        click.echo(f"\nOutput word count:")
        click.echo(f"  Min: {min(word_counts)}, Max: {max(word_counts)}")
        click.echo(f"  Avg: {sum(word_counts)/len(word_counts):.1f}")


def _sharegpt_stats(data: list, detailed: bool) -> None:
    """Show ShareGPT format statistics."""
    turn_counts = [len(d.get("conversations", [])) for d in data]
    has_system = sum(
        1 for d in data
        if any(c.get("role") == "system" or c.get("from") == "system"
               for c in d.get("conversations", []))
    )

    click.echo(f"\nShareGPT Format Statistics:")
    click.echo(f"  Records with system: {has_system} ({100*has_system/len(data):.1f}%)")
    click.echo(f"\nConversation turns:")
    click.echo(f"  Min: {min(turn_counts)}, Max: {max(turn_counts)}")
    click.echo(f"  Avg: {sum(turn_counts)/len(turn_counts):.1f}")

    if detailed:
        # Role distribution
        roles = {}
        for d in data:
            for c in d.get("conversations", []):
                role = c.get("role") or c.get("from", "unknown")
                roles[role] = roles.get(role, 0) + 1

        click.echo(f"\nRole distribution:")
        for role, count in sorted(roles.items()):
            click.echo(f"  {role}: {count}")


def _dpo_stats(data: list, detailed: bool) -> None:
    """Show DPO format statistics."""
    prompt_lens = [len(d.get("prompt", "")) for d in data]
    chosen_lens = [len(d.get("chosen", "")) for d in data]
    rejected_lens = [len(d.get("rejected", "")) for d in data]

    click.echo(f"\nDPO Format Statistics:")
    click.echo(f"\nPrompt length:")
    click.echo(f"  Min: {min(prompt_lens)}, Max: {max(prompt_lens)}")
    click.echo(f"  Avg: {sum(prompt_lens)/len(prompt_lens):.1f}")
    click.echo(f"\nChosen length:")
    click.echo(f"  Min: {min(chosen_lens)}, Max: {max(chosen_lens)}")
    click.echo(f"  Avg: {sum(chosen_lens)/len(chosen_lens):.1f}")
    click.echo(f"\nRejected length:")
    click.echo(f"  Min: {min(rejected_lens)}, Max: {max(rejected_lens)}")
    click.echo(f"  Avg: {sum(rejected_lens)/len(rejected_lens):.1f}")

    if detailed:
        # Length difference
        len_diffs = [
            len(d.get("chosen", "")) - len(d.get("rejected", ""))
            for d in data
        ]
        click.echo(f"\nChosen-Rejected length difference:")
        click.echo(f"  Min: {min(len_diffs)}, Max: {max(len_diffs)}")
        click.echo(f"  Avg: {sum(len_diffs)/len(len_diffs):.1f}")
