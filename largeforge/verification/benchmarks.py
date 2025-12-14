"""Benchmark suite for model performance evaluation."""

import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from largeforge.utils import get_logger
from largeforge.utils.device import get_device_memory, empty_cache

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    name: str
    passed: bool
    score: float
    latency_ms: Dict[str, float] = field(default_factory=dict)
    throughput_tokens_per_sec: float = 0.0
    memory_peak_gb: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "name": self.name,
            "passed": self.passed,
            "score": self.score,
            "latency_ms": self.latency_ms,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "memory_peak_gb": self.memory_peak_gb,
            "details": self.details,
        }


class BaseBenchmark(ABC):
    """Abstract base class for benchmarks."""

    name: str = "base"

    def __init__(
        self,
        generator,
        num_warmup: int = 3,
        num_runs: int = 10,
        max_tokens: int = 100,
    ):
        """
        Initialize benchmark.

        Args:
            generator: TextGenerator instance
            num_warmup: Number of warmup runs
            num_runs: Number of benchmark runs
            max_tokens: Maximum tokens to generate
        """
        self.generator = generator
        self.num_warmup = num_warmup
        self.num_runs = num_runs
        self.max_tokens = max_tokens

    @abstractmethod
    def run(self) -> BenchmarkResult:
        """Run the benchmark and return results."""
        pass


class LatencyBenchmark(BaseBenchmark):
    """Benchmark for measuring generation latency."""

    name = "latency"

    def __init__(
        self,
        generator,
        num_warmup: int = 3,
        num_runs: int = 10,
        max_tokens: int = 100,
        threshold_p99_ms: float = 5000.0,
    ):
        """
        Initialize latency benchmark.

        Args:
            generator: TextGenerator instance
            num_warmup: Number of warmup runs
            num_runs: Number of benchmark runs
            max_tokens: Maximum tokens to generate
            threshold_p99_ms: P99 latency threshold in ms for pass
        """
        super().__init__(generator, num_warmup, num_runs, max_tokens)
        self.threshold_p99_ms = threshold_p99_ms
        self.test_prompt = "Write a short paragraph about artificial intelligence:"

    def run(self) -> BenchmarkResult:
        """Run latency benchmark."""
        logger.info(f"Running latency benchmark ({self.num_runs} runs)")

        latencies: List[float] = []

        # Warmup runs
        for _ in range(self.num_warmup):
            try:
                self.generator.generate(
                    self.test_prompt,
                    max_tokens=self.max_tokens,
                    temperature=0.7,
                )
            except Exception:
                pass

        # Benchmark runs
        for i in range(self.num_runs):
            try:
                start = time.perf_counter()
                self.generator.generate(
                    self.test_prompt,
                    max_tokens=self.max_tokens,
                    temperature=0.7,
                )
                elapsed = (time.perf_counter() - start) * 1000  # ms
                latencies.append(elapsed)
            except Exception as e:
                logger.warning(f"Run {i} failed: {e}")

        if not latencies:
            return BenchmarkResult(
                name=self.name,
                passed=False,
                score=0.0,
                details={"error": "All runs failed"},
            )

        # Calculate statistics
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p90 = latencies[int(len(latencies) * 0.9)]
        p99 = latencies[int(len(latencies) * 0.99)] if len(latencies) >= 100 else latencies[-1]
        mean = statistics.mean(latencies)

        passed = p99 <= self.threshold_p99_ms

        # Score: inverse of latency, normalized
        score = min(1.0, 1000.0 / mean) if mean > 0 else 0.0

        return BenchmarkResult(
            name=self.name,
            passed=passed,
            score=score,
            latency_ms={
                "p50": round(p50, 2),
                "p90": round(p90, 2),
                "p99": round(p99, 2),
                "mean": round(mean, 2),
                "min": round(min(latencies), 2),
                "max": round(max(latencies), 2),
            },
            details={
                "num_runs": len(latencies),
                "threshold_p99_ms": self.threshold_p99_ms,
            },
        )


class ThroughputBenchmark(BaseBenchmark):
    """Benchmark for measuring token throughput."""

    name = "throughput"

    def __init__(
        self,
        generator,
        num_warmup: int = 2,
        num_runs: int = 5,
        max_tokens: int = 200,
        min_throughput: float = 10.0,
    ):
        """
        Initialize throughput benchmark.

        Args:
            generator: TextGenerator instance
            num_warmup: Number of warmup runs
            num_runs: Number of benchmark runs
            max_tokens: Maximum tokens to generate
            min_throughput: Minimum tokens/sec for pass
        """
        super().__init__(generator, num_warmup, num_runs, max_tokens)
        self.min_throughput = min_throughput
        self.test_prompt = "Write a detailed explanation of machine learning:"

    def run(self) -> BenchmarkResult:
        """Run throughput benchmark."""
        logger.info(f"Running throughput benchmark ({self.num_runs} runs)")

        throughputs: List[float] = []

        # Warmup
        for _ in range(self.num_warmup):
            try:
                self.generator.generate(
                    self.test_prompt,
                    max_tokens=self.max_tokens,
                    temperature=0.7,
                )
            except Exception:
                pass

        # Benchmark runs
        for i in range(self.num_runs):
            try:
                start = time.perf_counter()
                output = self.generator.generate(
                    self.test_prompt,
                    max_tokens=self.max_tokens,
                    temperature=0.7,
                )
                elapsed = time.perf_counter() - start

                # Estimate token count
                token_count = self.generator.count_tokens(output)
                if token_count > 0 and elapsed > 0:
                    throughput = token_count / elapsed
                    throughputs.append(throughput)

            except Exception as e:
                logger.warning(f"Run {i} failed: {e}")

        if not throughputs:
            return BenchmarkResult(
                name=self.name,
                passed=False,
                score=0.0,
                details={"error": "All runs failed"},
            )

        avg_throughput = statistics.mean(throughputs)
        passed = avg_throughput >= self.min_throughput

        # Score normalized to expected throughput
        score = min(1.0, avg_throughput / 100.0)

        return BenchmarkResult(
            name=self.name,
            passed=passed,
            score=score,
            throughput_tokens_per_sec=round(avg_throughput, 2),
            details={
                "min_throughput": round(min(throughputs), 2),
                "max_throughput": round(max(throughputs), 2),
                "std_dev": round(statistics.stdev(throughputs), 2) if len(throughputs) > 1 else 0,
                "threshold": self.min_throughput,
            },
        )


class MemoryBenchmark(BaseBenchmark):
    """Benchmark for measuring GPU memory usage."""

    name = "memory"

    def __init__(
        self,
        generator,
        num_runs: int = 3,
        max_tokens: int = 256,
        max_memory_gb: Optional[float] = None,
    ):
        """
        Initialize memory benchmark.

        Args:
            generator: TextGenerator instance
            num_runs: Number of runs to measure peak memory
            max_tokens: Maximum tokens to generate
            max_memory_gb: Maximum allowed memory (None for no limit)
        """
        super().__init__(generator, num_warmup=0, num_runs=num_runs, max_tokens=max_tokens)
        self.max_memory_gb = max_memory_gb
        self.test_prompt = "Write a long essay about the history of computing:"

    def run(self) -> BenchmarkResult:
        """Run memory benchmark."""
        logger.info("Running memory benchmark")

        memory_readings: List[float] = []

        # Clear cache first
        empty_cache()
        baseline = get_device_memory().get("allocated_gb", 0) or 0

        for i in range(self.num_runs):
            try:
                # Generate with max tokens to stress memory
                self.generator.generate(
                    self.test_prompt,
                    max_tokens=self.max_tokens,
                    temperature=0.7,
                )

                # Measure peak memory
                current = get_device_memory().get("allocated_gb", 0) or 0
                memory_readings.append(current)

            except Exception as e:
                logger.warning(f"Run {i} failed: {e}")

        if not memory_readings:
            return BenchmarkResult(
                name=self.name,
                passed=False,
                score=0.0,
                details={"error": "All runs failed"},
            )

        peak_memory = max(memory_readings)
        avg_memory = statistics.mean(memory_readings)

        passed = True
        if self.max_memory_gb is not None:
            passed = peak_memory <= self.max_memory_gb

        # Score: lower memory is better
        score = max(0.0, 1.0 - (peak_memory / 80.0))  # Normalize to 80GB max

        return BenchmarkResult(
            name=self.name,
            passed=passed,
            score=score,
            memory_peak_gb=round(peak_memory, 2),
            details={
                "baseline_gb": round(baseline, 2),
                "avg_memory_gb": round(avg_memory, 2),
                "threshold_gb": self.max_memory_gb,
            },
        )


class ConsistencyBenchmark(BaseBenchmark):
    """Benchmark for checking output consistency."""

    name = "consistency"

    def __init__(
        self,
        generator,
        num_runs: int = 5,
        max_tokens: int = 50,
        min_similarity: float = 0.3,
    ):
        """
        Initialize consistency benchmark.

        Args:
            generator: TextGenerator instance
            num_runs: Number of runs to compare
            max_tokens: Maximum tokens to generate
            min_similarity: Minimum similarity score for pass
        """
        super().__init__(generator, num_warmup=0, num_runs=num_runs, max_tokens=max_tokens)
        self.min_similarity = min_similarity
        self.test_prompt = "What is the capital of France?"

    def run(self) -> BenchmarkResult:
        """Run consistency benchmark."""
        logger.info("Running consistency benchmark")

        outputs: List[str] = []

        # Generate multiple outputs with temperature=0 for determinism
        for i in range(self.num_runs):
            try:
                output = self.generator.generate(
                    self.test_prompt,
                    max_tokens=self.max_tokens,
                    temperature=0.0,  # Deterministic
                )
                outputs.append(output.strip().lower())
            except Exception as e:
                logger.warning(f"Run {i} failed: {e}")

        if len(outputs) < 2:
            return BenchmarkResult(
                name=self.name,
                passed=False,
                score=0.0,
                details={"error": "Not enough successful runs"},
            )

        # Calculate similarity using simple word overlap
        similarities = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                sim = self._word_similarity(outputs[i], outputs[j])
                similarities.append(sim)

        avg_similarity = statistics.mean(similarities)
        passed = avg_similarity >= self.min_similarity

        return BenchmarkResult(
            name=self.name,
            passed=passed,
            score=avg_similarity,
            details={
                "avg_similarity": round(avg_similarity, 3),
                "min_similarity": round(min(similarities), 3),
                "max_similarity": round(max(similarities), 3),
                "num_outputs": len(outputs),
                "threshold": self.min_similarity,
            },
        )

    def _word_similarity(self, text1: str, text2: str) -> float:
        """Calculate word-level Jaccard similarity."""
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0


class BenchmarkSuite:
    """Suite of benchmarks for comprehensive model evaluation."""

    AVAILABLE_BENCHMARKS = {
        "latency": LatencyBenchmark,
        "throughput": ThroughputBenchmark,
        "memory": MemoryBenchmark,
        "consistency": ConsistencyBenchmark,
    }

    def __init__(
        self,
        model_path: str,
        backend: str = "auto",
        benchmarks: Optional[List[str]] = None,
        num_warmup: int = 3,
        num_runs: int = 10,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize benchmark suite.

        Args:
            model_path: Path to model or HuggingFace model ID
            backend: Backend to use ("auto", "transformers", "vllm")
            benchmarks: List of benchmark names to run (None for all)
            num_warmup: Number of warmup runs
            num_runs: Number of benchmark runs
            config: Additional configuration for benchmarks
        """
        self.model_path = model_path
        self.backend = backend
        self.benchmark_names = benchmarks or list(self.AVAILABLE_BENCHMARKS.keys())
        self.num_warmup = num_warmup
        self.num_runs = num_runs
        self.config = config or {}
        self._generator = None

    def run_all(self) -> List[BenchmarkResult]:
        """
        Run all configured benchmarks.

        Returns:
            List of BenchmarkResult objects
        """
        results = []

        try:
            self._load_model()

            for name in self.benchmark_names:
                if name in self.AVAILABLE_BENCHMARKS:
                    result = self.run_benchmark(name)
                    results.append(result)
                else:
                    logger.warning(f"Unknown benchmark: {name}")

        finally:
            self._cleanup()

        return results

    def run_benchmark(self, name: str) -> BenchmarkResult:
        """
        Run a single benchmark by name.

        Args:
            name: Benchmark name

        Returns:
            BenchmarkResult
        """
        if name not in self.AVAILABLE_BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {name}")

        if self._generator is None:
            self._load_model()

        benchmark_class = self.AVAILABLE_BENCHMARKS[name]
        benchmark_config = self.config.get(name, {})

        benchmark = benchmark_class(
            generator=self._generator,
            num_warmup=self.num_warmup,
            num_runs=self.num_runs,
            **benchmark_config,
        )

        logger.info(f"Running benchmark: {name}")
        return benchmark.run()

    def get_summary(self) -> Dict[str, Any]:
        """
        Run all benchmarks and return summary.

        Returns:
            Summary dict with overall pass/fail and metrics
        """
        results = self.run_all()

        all_passed = all(r.passed for r in results)
        avg_score = statistics.mean(r.score for r in results) if results else 0.0

        return {
            "passed": all_passed,
            "avg_score": round(avg_score, 3),
            "benchmarks": {r.name: r.to_dict() for r in results},
        }

    def _load_model(self) -> None:
        """Load the model for benchmarking."""
        from largeforge.inference import TextGenerator

        logger.info(f"Loading model: {self.model_path}")
        self._generator = TextGenerator(
            self.model_path,
            backend=self.backend,
        )
        self._generator.load()

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self._generator is not None:
            try:
                self._generator.unload()
            except Exception:
                pass
            self._generator = None
        empty_cache()
