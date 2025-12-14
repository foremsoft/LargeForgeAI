"""Model verification and validation module for LargeForgeAI.

This module provides tools for validating trained models before deployment:
- Smoke tests for quick validation (model loads, generates text)
- Benchmarks for performance evaluation (latency, throughput, memory)
- Unified validation API with configurable thresholds
- Report generation in multiple formats (text, JSON, HTML, Markdown)

Example:
    >>> from largeforge.verification import ModelValidator
    >>> validator = ModelValidator("./my_model")
    >>> result = validator.validate()
    >>> if result.passed:
    ...     print("Model is ready for deployment!")
"""

from largeforge.verification.smoke_test import SmokeTest, SmokeTestResult, run_smoke_test
from largeforge.verification.benchmarks import (
    BenchmarkSuite,
    BenchmarkResult,
    LatencyBenchmark,
    ThroughputBenchmark,
    MemoryBenchmark,
)
from largeforge.verification.validator import (
    ModelValidator,
    ValidationConfig,
    ValidationResult,
    ValidationLevel,
)
from largeforge.verification.report import ReportGenerator, ReportFormat

__all__ = [
    # Smoke test
    "SmokeTest",
    "SmokeTestResult",
    "run_smoke_test",
    # Benchmarks
    "BenchmarkSuite",
    "BenchmarkResult",
    "LatencyBenchmark",
    "ThroughputBenchmark",
    "MemoryBenchmark",
    # Validator
    "ModelValidator",
    "ValidationConfig",
    "ValidationResult",
    "ValidationLevel",
    # Report
    "ReportGenerator",
    "ReportFormat",
]
