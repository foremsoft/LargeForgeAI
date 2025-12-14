"""Verification report generation in multiple formats."""

import json
from enum import Enum
from pathlib import Path
from typing import Optional

from largeforge.verification.validator import ValidationResult


class ReportFormat(str, Enum):
    """Supported report formats."""

    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"


class ReportGenerator:
    """Generate verification reports in various formats."""

    def __init__(self, result: ValidationResult):
        """
        Initialize report generator.

        Args:
            result: ValidationResult to generate report from
        """
        self.result = result

    def generate(self, format: ReportFormat = ReportFormat.TEXT) -> str:
        """
        Generate report in specified format.

        Args:
            format: Output format

        Returns:
            Report string
        """
        if format == ReportFormat.TEXT:
            return self._generate_text()
        elif format == ReportFormat.JSON:
            return self._generate_json()
        elif format == ReportFormat.MARKDOWN:
            return self._generate_markdown()
        elif format == ReportFormat.HTML:
            return self._generate_html()
        else:
            raise ValueError(f"Unsupported format: {format}")

    def save(self, path: str, format: Optional[ReportFormat] = None) -> None:
        """
        Save report to file.

        Args:
            path: Output file path
            format: Output format (auto-detected from extension if None)
        """
        if format is None:
            format = self._detect_format(path)

        content = self.generate(format)

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def _detect_format(self, path: str) -> ReportFormat:
        """Detect format from file extension."""
        ext = Path(path).suffix.lower()
        format_map = {
            ".txt": ReportFormat.TEXT,
            ".json": ReportFormat.JSON,
            ".md": ReportFormat.MARKDOWN,
            ".html": ReportFormat.HTML,
            ".htm": ReportFormat.HTML,
        }
        return format_map.get(ext, ReportFormat.TEXT)

    def _generate_text(self) -> str:
        """Generate plain text report with ANSI colors."""
        r = self.result
        lines = []

        # Header
        status = self._colorize("PASSED", "green") if r.passed else self._colorize("FAILED", "red")
        lines.append("=" * 60)
        lines.append(f"Model Validation Report - {status}")
        lines.append("=" * 60)
        lines.append("")

        # Model info
        lines.append(f"Model: {r.model_path}")
        lines.append(f"Level: {r.level.value}")
        lines.append(f"Date: {r.validated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Duration: {r.validation_time_seconds:.1f}s")
        lines.append("")

        # Smoke test results
        if r.smoke_test_result:
            lines.append("-" * 40)
            lines.append("SMOKE TEST")
            lines.append("-" * 40)
            st = r.smoke_test_result
            status = self._colorize("PASS", "green") if st.passed else self._colorize("FAIL", "red")
            lines.append(f"Status: {status}")
            lines.append(f"  Model loads: {'Yes' if st.model_loads else 'No'}")
            lines.append(f"  Generates text: {'Yes' if st.generates_text else 'No'}")
            lines.append(f"  Text coherent: {'Yes' if st.text_coherent else 'No'}")
            lines.append(f"  Load time: {st.load_time_seconds:.2f}s")
            lines.append(f"  Generation time: {st.generation_time_seconds:.2f}s")
            lines.append(f"  Throughput: {st.tokens_per_second:.1f} tokens/s")
            lines.append(f"  Memory used: {st.memory_used_gb:.2f} GB")

            if st.errors:
                lines.append(f"  Errors: {', '.join(st.errors)}")
            if st.warnings:
                lines.append(f"  Warnings: {', '.join(st.warnings)}")
            lines.append("")

        # Benchmark results
        if r.benchmark_results:
            lines.append("-" * 40)
            lines.append("BENCHMARKS")
            lines.append("-" * 40)

            for br in r.benchmark_results:
                status = self._colorize("PASS", "green") if br.passed else self._colorize("FAIL", "red")
                lines.append(f"\n{br.name.upper()}: {status}")
                lines.append(f"  Score: {br.score:.3f}")

                if br.latency_ms:
                    lines.append(f"  Latency P50: {br.latency_ms.get('p50', 0):.0f}ms")
                    lines.append(f"  Latency P99: {br.latency_ms.get('p99', 0):.0f}ms")

                if br.throughput_tokens_per_sec > 0:
                    lines.append(f"  Throughput: {br.throughput_tokens_per_sec:.1f} tokens/s")

                if br.memory_peak_gb > 0:
                    lines.append(f"  Peak memory: {br.memory_peak_gb:.2f} GB")

            lines.append("")

        # Recommendations
        lines.append("-" * 40)
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 40)
        for rec in r.recommendations:
            lines.append(f"  - {rec}")
        lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def _generate_json(self) -> str:
        """Generate JSON report."""
        return json.dumps(self.result.to_dict(), indent=2, default=str)

    def _generate_markdown(self) -> str:
        """Generate Markdown report."""
        r = self.result
        lines = []

        # Header
        status = "PASSED" if r.passed else "FAILED"
        status_emoji = ":white_check_mark:" if r.passed else ":x:"
        lines.append(f"# Model Validation Report {status_emoji}")
        lines.append("")
        lines.append(f"**Status:** {status}")
        lines.append(f"**Model:** `{r.model_path}`")
        lines.append(f"**Level:** {r.level.value}")
        lines.append(f"**Date:** {r.validated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Duration:** {r.validation_time_seconds:.1f}s")
        lines.append("")

        # Smoke test
        if r.smoke_test_result:
            st = r.smoke_test_result
            lines.append("## Smoke Test")
            lines.append("")
            lines.append("| Check | Result |")
            lines.append("|-------|--------|")
            lines.append(f"| Model loads | {'Pass' if st.model_loads else 'Fail'} |")
            lines.append(f"| Generates text | {'Pass' if st.generates_text else 'Fail'} |")
            lines.append(f"| Text coherent | {'Pass' if st.text_coherent else 'Fail'} |")
            lines.append("")
            lines.append("### Metrics")
            lines.append("")
            lines.append(f"- Load time: **{st.load_time_seconds:.2f}s**")
            lines.append(f"- Throughput: **{st.tokens_per_second:.1f} tokens/s**")
            lines.append(f"- Memory: **{st.memory_used_gb:.2f} GB**")
            lines.append("")

        # Benchmarks
        if r.benchmark_results:
            lines.append("## Benchmarks")
            lines.append("")
            lines.append("| Benchmark | Status | Score | Key Metric |")
            lines.append("|-----------|--------|-------|------------|")

            for br in r.benchmark_results:
                status = "Pass" if br.passed else "Fail"
                key_metric = ""
                if br.latency_ms:
                    key_metric = f"P99: {br.latency_ms.get('p99', 0):.0f}ms"
                elif br.throughput_tokens_per_sec > 0:
                    key_metric = f"{br.throughput_tokens_per_sec:.1f} tok/s"
                elif br.memory_peak_gb > 0:
                    key_metric = f"{br.memory_peak_gb:.2f} GB"

                lines.append(f"| {br.name} | {status} | {br.score:.3f} | {key_metric} |")

            lines.append("")

        # Recommendations
        lines.append("## Recommendations")
        lines.append("")
        for rec in r.recommendations:
            lines.append(f"- {rec}")
        lines.append("")

        return "\n".join(lines)

    def _generate_html(self) -> str:
        """Generate HTML report."""
        r = self.result

        status_class = "passed" if r.passed else "failed"
        status_text = "PASSED" if r.passed else "FAILED"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Validation Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .status {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 4px;
            font-weight: bold;
            margin-left: 10px;
        }}
        .status.passed {{ background: #d4edda; color: #155724; }}
        .status.failed {{ background: #f8d7da; color: #721c24; }}
        .info {{ color: #666; margin-bottom: 20px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        .pass {{ color: #28a745; }}
        .fail {{ color: #dc3545; }}
        .metric {{ font-weight: 500; color: #333; }}
        .recommendations {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin-top: 20px;
        }}
        .recommendations.success {{
            background: #d4edda;
            border-color: #28a745;
        }}
        ul {{ margin: 10px 0; padding-left: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Model Validation Report <span class="status {status_class}">{status_text}</span></h1>

        <div class="info">
            <p><strong>Model:</strong> <code>{r.model_path}</code></p>
            <p><strong>Level:</strong> {r.level.value}</p>
            <p><strong>Date:</strong> {r.validated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Duration:</strong> {r.validation_time_seconds:.1f}s</p>
        </div>
"""

        # Smoke test section
        if r.smoke_test_result:
            st = r.smoke_test_result
            html += """
        <h2>Smoke Test</h2>
        <table>
            <tr><th>Check</th><th>Result</th></tr>
"""
            for check, value in [
                ("Model loads", st.model_loads),
                ("Generates text", st.generates_text),
                ("Text coherent", st.text_coherent),
            ]:
                cls = "pass" if value else "fail"
                text = "Pass" if value else "Fail"
                html += f'            <tr><td>{check}</td><td class="{cls}">{text}</td></tr>\n'

            html += f"""        </table>
        <p><span class="metric">Load time:</span> {st.load_time_seconds:.2f}s |
           <span class="metric">Throughput:</span> {st.tokens_per_second:.1f} tokens/s |
           <span class="metric">Memory:</span> {st.memory_used_gb:.2f} GB</p>
"""

        # Benchmarks section
        if r.benchmark_results:
            html += """
        <h2>Benchmarks</h2>
        <table>
            <tr><th>Benchmark</th><th>Status</th><th>Score</th><th>Key Metric</th></tr>
"""
            for br in r.benchmark_results:
                cls = "pass" if br.passed else "fail"
                status = "Pass" if br.passed else "Fail"
                key_metric = ""
                if br.latency_ms:
                    key_metric = f"P99: {br.latency_ms.get('p99', 0):.0f}ms"
                elif br.throughput_tokens_per_sec > 0:
                    key_metric = f"{br.throughput_tokens_per_sec:.1f} tok/s"
                elif br.memory_peak_gb > 0:
                    key_metric = f"{br.memory_peak_gb:.2f} GB"

                html += f'            <tr><td>{br.name}</td><td class="{cls}">{status}</td><td>{br.score:.3f}</td><td>{key_metric}</td></tr>\n'

            html += "        </table>\n"

        # Recommendations
        rec_class = "success" if r.passed else ""
        html += f"""
        <h2>Recommendations</h2>
        <div class="recommendations {rec_class}">
            <ul>
"""
        for rec in r.recommendations:
            html += f"                <li>{rec}</li>\n"

        html += """            </ul>
        </div>
    </div>
</body>
</html>
"""
        return html

    def _colorize(self, text: str, color: str) -> str:
        """Add ANSI color codes to text."""
        colors = {
            "green": "\033[92m",
            "red": "\033[91m",
            "yellow": "\033[93m",
            "reset": "\033[0m",
        }
        return f"{colors.get(color, '')}{text}{colors['reset']}"
