"""Unified model validation API."""

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from largeforge.config import BaseConfig
from largeforge.utils import get_logger
from largeforge.verification.smoke_test import SmokeTest, SmokeTestResult
from largeforge.verification.benchmarks import BenchmarkSuite, BenchmarkResult

logger = get_logger(__name__)


class ValidationLevel(str, Enum):
    """Validation thoroughness level."""

    QUICK = "quick"  # Smoke test only
    STANDARD = "standard"  # Smoke test + key benchmarks
    THOROUGH = "thorough"  # All benchmarks with more iterations


@dataclass
class ValidationConfig:
    """Configuration for model validation."""

    level: ValidationLevel = ValidationLevel.STANDARD
    run_smoke_test: bool = True
    run_benchmarks: bool = True
    benchmark_names: Optional[List[str]] = None
    min_tokens_per_sec: float = 10.0
    max_latency_p99_ms: float = 5000.0
    max_memory_gb: Optional[float] = None
    num_benchmark_runs: int = 10
    backend: str = "auto"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "run_smoke_test": self.run_smoke_test,
            "run_benchmarks": self.run_benchmarks,
            "benchmark_names": self.benchmark_names,
            "min_tokens_per_sec": self.min_tokens_per_sec,
            "max_latency_p99_ms": self.max_latency_p99_ms,
            "max_memory_gb": self.max_memory_gb,
            "num_benchmark_runs": self.num_benchmark_runs,
            "backend": self.backend,
        }


@dataclass
class ValidationResult:
    """Results from model validation."""

    passed: bool
    level: ValidationLevel
    model_path: str
    smoke_test_result: Optional[SmokeTestResult] = None
    benchmark_results: List[BenchmarkResult] = field(default_factory=list)
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    validated_at: datetime = field(default_factory=datetime.utcnow)
    validation_time_seconds: float = 0.0
    config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "level": self.level.value,
            "model_path": self.model_path,
            "smoke_test": self.smoke_test_result.to_dict() if self.smoke_test_result else None,
            "benchmarks": [b.to_dict() for b in self.benchmark_results],
            "summary": self.summary,
            "recommendations": self.recommendations,
            "validated_at": self.validated_at.isoformat(),
            "validation_time_seconds": self.validation_time_seconds,
            "config": self.config,
        }


class ModelValidator:
    """Unified model validation interface."""

    def __init__(
        self,
        model_path: str,
        config: Optional[ValidationConfig] = None,
    ):
        """
        Initialize model validator.

        Args:
            model_path: Path to model or HuggingFace model ID
            config: Validation configuration
        """
        self.model_path = model_path
        self.config = config or ValidationConfig()

    def validate(self) -> ValidationResult:
        """
        Run validation based on configured level.

        Returns:
            ValidationResult with all test outcomes
        """
        level = self.config.level

        if level == ValidationLevel.QUICK:
            return self.validate_quick()
        elif level == ValidationLevel.THOROUGH:
            return self.validate_thorough()
        else:
            return self._validate_standard()

    def validate_quick(self) -> ValidationResult:
        """
        Run quick validation (smoke test only).

        Returns:
            ValidationResult
        """
        logger.info(f"Running quick validation on {self.model_path}")
        start_time = time.perf_counter()

        # Run smoke test
        smoke_test = SmokeTest(
            self.model_path,
            backend=self.config.backend,
        )
        smoke_result = smoke_test.run()

        elapsed = time.perf_counter() - start_time

        # Generate summary
        passed = smoke_result.passed
        summary = self._generate_summary(smoke_result, [])
        recommendations = self._generate_recommendations(smoke_result, [])

        return ValidationResult(
            passed=passed,
            level=ValidationLevel.QUICK,
            model_path=self.model_path,
            smoke_test_result=smoke_result,
            benchmark_results=[],
            summary=summary,
            recommendations=recommendations,
            validation_time_seconds=round(elapsed, 2),
            config=self.config.to_dict(),
        )

    def validate_thorough(self) -> ValidationResult:
        """
        Run thorough validation (all benchmarks with more iterations).

        Returns:
            ValidationResult
        """
        # Override config for thorough validation
        thorough_config = ValidationConfig(
            level=ValidationLevel.THOROUGH,
            run_smoke_test=True,
            run_benchmarks=True,
            benchmark_names=None,  # All benchmarks
            min_tokens_per_sec=self.config.min_tokens_per_sec,
            max_latency_p99_ms=self.config.max_latency_p99_ms,
            max_memory_gb=self.config.max_memory_gb,
            num_benchmark_runs=20,  # More runs for thorough
            backend=self.config.backend,
        )

        original_config = self.config
        self.config = thorough_config

        try:
            result = self._validate_standard()
            result.level = ValidationLevel.THOROUGH
            return result
        finally:
            self.config = original_config

    def _validate_standard(self) -> ValidationResult:
        """Run standard validation."""
        logger.info(f"Running standard validation on {self.model_path}")
        start_time = time.perf_counter()

        smoke_result: Optional[SmokeTestResult] = None
        benchmark_results: List[BenchmarkResult] = []
        passed = True

        # Run smoke test if enabled
        if self.config.run_smoke_test:
            smoke_test = SmokeTest(
                self.model_path,
                backend=self.config.backend,
            )
            smoke_result = smoke_test.run()

            if not smoke_result.passed:
                passed = False
                logger.warning("Smoke test failed")

        # Run benchmarks if enabled and smoke test passed (or skipped)
        if self.config.run_benchmarks:
            if smoke_result is None or smoke_result.passed:
                benchmark_results = self._run_benchmarks()

                # Check thresholds
                threshold_passed, threshold_issues = self._check_thresholds(benchmark_results)
                if not threshold_passed:
                    passed = False
                    logger.warning(f"Threshold checks failed: {threshold_issues}")

        elapsed = time.perf_counter() - start_time

        # Generate summary and recommendations
        summary = self._generate_summary(smoke_result, benchmark_results)
        recommendations = self._generate_recommendations(smoke_result, benchmark_results)

        return ValidationResult(
            passed=passed,
            level=self.config.level,
            model_path=self.model_path,
            smoke_test_result=smoke_result,
            benchmark_results=benchmark_results,
            summary=summary,
            recommendations=recommendations,
            validation_time_seconds=round(elapsed, 2),
            config=self.config.to_dict(),
        )

    def _run_benchmarks(self) -> List[BenchmarkResult]:
        """Run configured benchmarks."""
        benchmark_config = {
            "latency": {
                "threshold_p99_ms": self.config.max_latency_p99_ms,
            },
            "throughput": {
                "min_throughput": self.config.min_tokens_per_sec,
            },
            "memory": {
                "max_memory_gb": self.config.max_memory_gb,
            },
        }

        suite = BenchmarkSuite(
            model_path=self.model_path,
            backend=self.config.backend,
            benchmarks=self.config.benchmark_names,
            num_runs=self.config.num_benchmark_runs,
            config=benchmark_config,
        )

        return suite.run_all()

    def _check_thresholds(
        self, results: List[BenchmarkResult]
    ) -> tuple[bool, List[str]]:
        """
        Check if benchmark results meet thresholds.

        Returns:
            Tuple of (all_passed, list_of_issues)
        """
        issues = []

        for result in results:
            if not result.passed:
                issues.append(f"{result.name}: failed threshold check")

            # Additional threshold checks
            if result.name == "throughput":
                if result.throughput_tokens_per_sec < self.config.min_tokens_per_sec:
                    issues.append(
                        f"Throughput {result.throughput_tokens_per_sec:.1f} tokens/s "
                        f"below minimum {self.config.min_tokens_per_sec}"
                    )

            if result.name == "latency":
                p99 = result.latency_ms.get("p99", 0)
                if p99 > self.config.max_latency_p99_ms:
                    issues.append(
                        f"P99 latency {p99:.0f}ms exceeds maximum "
                        f"{self.config.max_latency_p99_ms}ms"
                    )

            if result.name == "memory" and self.config.max_memory_gb:
                if result.memory_peak_gb > self.config.max_memory_gb:
                    issues.append(
                        f"Peak memory {result.memory_peak_gb:.1f}GB exceeds maximum "
                        f"{self.config.max_memory_gb}GB"
                    )

        return len(issues) == 0, issues

    def _generate_summary(
        self,
        smoke_result: Optional[SmokeTestResult],
        benchmark_results: List[BenchmarkResult],
    ) -> str:
        """Generate human-readable summary."""
        lines = []

        if smoke_result:
            status = "PASSED" if smoke_result.passed else "FAILED"
            lines.append(f"Smoke Test: {status}")
            if smoke_result.passed:
                lines.append(f"  - Load time: {smoke_result.load_time_seconds:.2f}s")
                lines.append(f"  - Generation: {smoke_result.tokens_per_second:.1f} tokens/s")

        if benchmark_results:
            passed_count = sum(1 for r in benchmark_results if r.passed)
            total_count = len(benchmark_results)
            lines.append(f"Benchmarks: {passed_count}/{total_count} passed")

            for result in benchmark_results:
                status = "PASS" if result.passed else "FAIL"
                lines.append(f"  - {result.name}: {status} (score: {result.score:.2f})")

        return "\n".join(lines)

    def _generate_recommendations(
        self,
        smoke_result: Optional[SmokeTestResult],
        benchmark_results: List[BenchmarkResult],
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if smoke_result and not smoke_result.passed:
            if not smoke_result.model_loads:
                recommendations.append(
                    "Model failed to load. Check model path and available memory."
                )
            if not smoke_result.generates_text:
                recommendations.append(
                    "Model cannot generate text. Verify model format and tokenizer."
                )
            if not smoke_result.text_coherent:
                recommendations.append(
                    "Generated text lacks coherence. Model may need more training."
                )

        for result in benchmark_results:
            if not result.passed:
                if result.name == "latency":
                    recommendations.append(
                        "High latency detected. Consider quantization or smaller model."
                    )
                elif result.name == "throughput":
                    recommendations.append(
                        "Low throughput. Consider vLLM backend or batch inference."
                    )
                elif result.name == "memory":
                    recommendations.append(
                        "High memory usage. Consider quantization (4-bit/8-bit)."
                    )
                elif result.name == "consistency":
                    recommendations.append(
                        "Inconsistent outputs. Check model training and temperature settings."
                    )

        if not recommendations:
            recommendations.append("Model passed all checks. Ready for deployment.")

        return recommendations
