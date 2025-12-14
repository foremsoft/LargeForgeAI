"""CLI commands for model verification."""

from pathlib import Path
from typing import Optional

import click

from largeforge.utils import get_logger

logger = get_logger(__name__)


@click.group()
def verify() -> None:
    """Model verification commands."""
    pass


@verify.command("run")
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--level", "-l",
    type=click.Choice(["quick", "standard", "thorough"]),
    default="standard",
    help="Validation level"
)
@click.option("--output", "-o", type=click.Path(), help="Save report to file")
@click.option(
    "--format", "-f", "output_format",
    type=click.Choice(["text", "json", "markdown", "html"]),
    default="text",
    help="Report format"
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def run_verification(
    model_path: str,
    level: str,
    output: Optional[str],
    output_format: str,
    verbose: bool,
) -> None:
    """Run comprehensive model verification.

    Performs smoke tests and benchmarks to validate a trained model
    is working correctly and meets performance thresholds.

    Examples:
        # Quick verification
        largeforge verify run ./output/model --level quick

        # Full verification with report
        largeforge verify run ./output/model --output report.md --format markdown

        # Thorough verification with verbose output
        largeforge verify run ./output/model --level thorough -v
    """
    from largeforge.verification import ModelValidator, ValidationLevel
    from largeforge.verification.report import ReportGenerator, ReportFormat

    click.echo(f"Verifying model: {model_path}")
    click.echo(f"Validation level: {level}")
    click.echo("")

    # Map level string to enum
    level_map = {
        "quick": ValidationLevel.QUICK,
        "standard": ValidationLevel.STANDARD,
        "thorough": ValidationLevel.THOROUGH,
    }

    # Run validation
    validator = ModelValidator(model_path)

    with click.progressbar(length=100, label="Validating") as bar:
        if level == "quick":
            result = validator.validate_quick()
            bar.update(100)
        elif level == "thorough":
            result = validator.validate_thorough()
            bar.update(100)
        else:
            result = validator.validate()
            bar.update(100)

    click.echo("")

    # Display results
    if result.passed:
        click.secho("PASSED", fg="green", bold=True)
    else:
        click.secho("FAILED", fg="red", bold=True)

    click.echo(f"\nSummary: {result.summary}")
    click.echo(f"Validation time: {result.validation_time_seconds:.2f}s")

    if verbose:
        if result.smoke_test_result:
            click.echo("\nSmoke Test Results:")
            st = result.smoke_test_result
            click.echo(f"  Model loads: {st.model_loads}")
            click.echo(f"  Generates text: {st.generates_text}")
            click.echo(f"  Text coherent: {st.text_coherent}")
            click.echo(f"  Load time: {st.load_time_seconds:.2f}s")
            click.echo(f"  Tokens/sec: {st.tokens_per_second:.2f}")

        if result.benchmark_results:
            click.echo("\nBenchmark Results:")
            for bench in result.benchmark_results:
                status = click.style("PASS", fg="green") if bench.passed else click.style("FAIL", fg="red")
                click.echo(f"  {bench.name}: {status} (score: {bench.score:.4f})")

    if result.recommendations:
        click.echo("\nRecommendations:")
        for rec in result.recommendations:
            click.echo(f"  - {rec}")

    # Save report if requested
    if output:
        format_map = {
            "text": ReportFormat.TEXT,
            "json": ReportFormat.JSON,
            "markdown": ReportFormat.MARKDOWN,
            "html": ReportFormat.HTML,
        }
        report_format = format_map.get(output_format, ReportFormat.TEXT)

        generator = ReportGenerator(result)
        generator.save(output, report_format)
        click.echo(f"\nReport saved to {output}")


@verify.command("smoke-test")
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--prompts", "-p", multiple=True, help="Custom test prompts")
@click.option("--max-tokens", "-t", default=100, type=int, help="Max tokens to generate")
def smoke_test(model_path: str, prompts: tuple, max_tokens: int) -> None:
    """Run a quick smoke test on a model.

    Verifies the model can:
    - Load successfully
    - Generate text
    - Produce coherent output

    Examples:
        # Basic smoke test
        largeforge verify smoke-test ./output/model

        # With custom prompts
        largeforge verify smoke-test ./output/model -p "Hello, my name is" -p "The weather today is"
    """
    from largeforge.verification import SmokeTest

    click.echo(f"Running smoke test on: {model_path}")
    click.echo("")

    test = SmokeTest(model_path)

    if prompts:
        test.test_prompts = list(prompts)

    result = test.run()

    # Display results
    click.echo("Results:")
    click.echo(f"  Model loads: {click.style('Yes', fg='green') if result.model_loads else click.style('No', fg='red')}")
    click.echo(f"  Generates text: {click.style('Yes', fg='green') if result.generates_text else click.style('No', fg='red')}")
    click.echo(f"  Text coherent: {click.style('Yes', fg='green') if result.text_coherent else click.style('No', fg='red')}")

    click.echo(f"\nLoad time: {result.load_time_seconds:.2f}s")
    click.echo(f"Generation time: {result.generation_time_seconds:.2f}s")
    click.echo(f"Tokens/second: {result.tokens_per_second:.2f}")
    click.echo(f"Memory used: {result.memory_used_gb:.2f}GB")

    if result.sample_outputs:
        click.echo("\nSample outputs:")
        for i, output in enumerate(result.sample_outputs[:3], 1):
            click.echo(f"  {i}. {output[:100]}...")

    if result.errors:
        click.echo("\nErrors:")
        for error in result.errors:
            click.secho(f"  - {error}", fg="red")

    if result.warnings:
        click.echo("\nWarnings:")
        for warning in result.warnings:
            click.secho(f"  - {warning}", fg="yellow")

    click.echo("")
    if result.passed:
        click.secho("Smoke test PASSED", fg="green", bold=True)
    else:
        click.secho("Smoke test FAILED", fg="red", bold=True)


@verify.command("benchmark")
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--benchmarks", "-b",
    multiple=True,
    type=click.Choice(["latency", "throughput", "memory", "consistency"]),
    help="Benchmarks to run (default: all)"
)
@click.option("--runs", "-n", default=10, type=int, help="Number of runs per benchmark")
@click.option("--warmup", "-w", default=3, type=int, help="Number of warmup runs")
@click.option("--output", "-o", type=click.Path(), help="Save results to JSON file")
def benchmark(
    model_path: str,
    benchmarks: tuple,
    runs: int,
    warmup: int,
    output: Optional[str],
) -> None:
    """Run performance benchmarks on a model.

    Available benchmarks:
    - latency: Measures response time (p50, p90, p99)
    - throughput: Measures tokens per second
    - memory: Measures GPU memory usage
    - consistency: Tests output consistency

    Examples:
        # Run all benchmarks
        largeforge verify benchmark ./output/model

        # Run specific benchmarks
        largeforge verify benchmark ./output/model -b latency -b throughput

        # More runs for accuracy
        largeforge verify benchmark ./output/model --runs 50
    """
    from largeforge.verification import BenchmarkSuite
    import json

    click.echo(f"Running benchmarks on: {model_path}")
    click.echo(f"Runs per benchmark: {runs}")
    click.echo(f"Warmup runs: {warmup}")
    click.echo("")

    suite = BenchmarkSuite(
        model_path,
        config={
            "num_runs": runs,
            "num_warmup": warmup,
        }
    )

    # Filter benchmarks if specified
    if benchmarks:
        suite.benchmarks = list(benchmarks)

    click.echo("Running benchmarks...")
    results = suite.run_all()

    # Display results
    click.echo("")
    click.echo(f"{'Benchmark':<15} {'Status':<8} {'Score':<12} {'Details':<30}")
    click.echo("-" * 70)

    for result in results:
        status = click.style("PASS", fg="green") if result.passed else click.style("FAIL", fg="red")

        if result.name == "latency":
            details = f"p50={result.latency_ms.get('p50', 0):.1f}ms, p99={result.latency_ms.get('p99', 0):.1f}ms"
        elif result.name == "throughput":
            details = f"{result.throughput_tokens_per_sec:.1f} tok/s"
        elif result.name == "memory":
            details = f"peak={result.memory_peak_gb:.2f}GB"
        else:
            details = str(result.details)[:30]

        click.echo(f"{result.name:<15} {status:<8} {result.score:<12.4f} {details:<30}")

    # Summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    click.echo("")
    click.echo(f"Passed: {passed}/{total}")

    # Save if requested
    if output:
        data = {
            "model_path": model_path,
            "runs": runs,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "score": r.score,
                    "latency_ms": r.latency_ms,
                    "throughput_tokens_per_sec": r.throughput_tokens_per_sec,
                    "memory_peak_gb": r.memory_peak_gb,
                    "details": r.details,
                }
                for r in results
            ]
        }
        with open(output, "w") as f:
            json.dump(data, f, indent=2)
        click.echo(f"\nResults saved to {output}")
