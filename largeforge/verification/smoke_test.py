"""Smoke test for quick model validation."""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from largeforge.utils import get_logger
from largeforge.utils.device import get_device_memory, get_device

logger = get_logger(__name__)


@dataclass
class SmokeTestResult:
    """Results from a smoke test run."""

    passed: bool
    model_loads: bool = False
    generates_text: bool = False
    text_coherent: bool = False
    load_time_seconds: float = 0.0
    generation_time_seconds: float = 0.0
    tokens_per_second: float = 0.0
    memory_used_gb: float = 0.0
    sample_outputs: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "passed": self.passed,
            "model_loads": self.model_loads,
            "generates_text": self.generates_text,
            "text_coherent": self.text_coherent,
            "load_time_seconds": self.load_time_seconds,
            "generation_time_seconds": self.generation_time_seconds,
            "tokens_per_second": self.tokens_per_second,
            "memory_used_gb": self.memory_used_gb,
            "sample_outputs": self.sample_outputs,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class SmokeTest:
    """Quick smoke test to validate a model works correctly."""

    DEFAULT_PROMPTS = [
        "Hello, my name is",
        "The capital of France is",
        "def fibonacci(n):",
        "Explain quantum computing in simple terms:",
    ]

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        backend: str = "auto",
        max_tokens: int = 50,
        temperature: float = 0.7,
        test_prompts: Optional[List[str]] = None,
    ):
        """
        Initialize smoke test.

        Args:
            model_path: Path to model or HuggingFace model ID
            device: Device to use ("auto", "cuda", "cpu")
            backend: Backend to use ("auto", "transformers", "vllm")
            max_tokens: Maximum tokens to generate per test
            temperature: Sampling temperature
            test_prompts: Custom test prompts (uses defaults if None)
        """
        self.model_path = model_path
        self.device = device if device != "auto" else get_device()
        self.backend = backend
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.test_prompts = test_prompts or self.DEFAULT_PROMPTS
        self._generator = None

    def run(self) -> SmokeTestResult:
        """
        Run the smoke test.

        Returns:
            SmokeTestResult with test outcomes
        """
        errors: List[str] = []
        warnings: List[str] = []
        sample_outputs: List[str] = []

        # Test model loading
        model_loads, load_time, load_error = self._test_model_loading()
        if load_error:
            errors.append(f"Model loading failed: {load_error}")

        if not model_loads:
            return SmokeTestResult(
                passed=False,
                model_loads=False,
                errors=errors,
            )

        # Get memory usage after loading
        memory_used = self._get_memory_usage()

        # Test generation
        generates_text = False
        text_coherent = False
        total_gen_time = 0.0
        total_tokens = 0

        for prompt in self.test_prompts:
            success, output, gen_time = self._test_generation(prompt)

            if success:
                generates_text = True
                sample_outputs.append(output)
                total_gen_time += gen_time

                # Estimate tokens (rough approximation)
                estimated_tokens = len(output.split()) * 1.3
                total_tokens += int(estimated_tokens)

                # Check coherence
                if self._check_coherence(output):
                    text_coherent = True
            else:
                warnings.append(f"Generation failed for prompt: {prompt[:30]}...")

        # Calculate metrics
        tokens_per_second = total_tokens / total_gen_time if total_gen_time > 0 else 0

        # Unload model
        self._cleanup()

        # Determine pass/fail
        passed = model_loads and generates_text and text_coherent

        if not generates_text:
            errors.append("Model failed to generate any text")
        if not text_coherent:
            warnings.append("Generated text may lack coherence")

        return SmokeTestResult(
            passed=passed,
            model_loads=model_loads,
            generates_text=generates_text,
            text_coherent=text_coherent,
            load_time_seconds=load_time,
            generation_time_seconds=total_gen_time,
            tokens_per_second=tokens_per_second,
            memory_used_gb=memory_used,
            sample_outputs=sample_outputs,
            errors=errors,
            warnings=warnings,
        )

    def _test_model_loading(self) -> Tuple[bool, float, Optional[str]]:
        """
        Test that the model loads successfully.

        Returns:
            Tuple of (success, load_time_seconds, error_message)
        """
        try:
            from largeforge.inference import TextGenerator

            start_time = time.perf_counter()
            self._generator = TextGenerator(
                self.model_path,
                backend=self.backend,
            )
            self._generator.load()
            load_time = time.perf_counter() - start_time

            logger.info(f"Model loaded in {load_time:.2f}s")
            return True, load_time, None

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False, 0.0, str(e)

    def _test_generation(self, prompt: str) -> Tuple[bool, str, float]:
        """
        Test text generation with a prompt.

        Args:
            prompt: Input prompt

        Returns:
            Tuple of (success, generated_text, generation_time)
        """
        if self._generator is None:
            return False, "", 0.0

        try:
            start_time = time.perf_counter()
            output = self._generator.generate(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            gen_time = time.perf_counter() - start_time

            logger.debug(f"Generated {len(output)} chars in {gen_time:.2f}s")
            return True, output, gen_time

        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            return False, "", 0.0

    def _check_coherence(self, text: str) -> bool:
        """
        Basic coherence check on generated text.

        Args:
            text: Generated text to check

        Returns:
            True if text appears coherent
        """
        if not text or len(text.strip()) == 0:
            return False

        # Check for excessive repetition
        words = text.split()
        if len(words) > 5:
            # Check if more than 70% are the same word
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

            max_count = max(word_counts.values())
            if max_count / len(words) > 0.7:
                return False

        # Check for reasonable character variety
        unique_chars = len(set(text.lower()))
        if len(text) > 20 and unique_chars < 5:
            return False

        return True

    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage in GB."""
        memory_info = get_device_memory()
        return memory_info.get("allocated_gb", 0.0) or 0.0

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self._generator is not None:
            try:
                self._generator.unload()
            except Exception:
                pass
            self._generator = None


def run_smoke_test(
    model_path: str,
    device: str = "auto",
    backend: str = "auto",
    max_tokens: int = 50,
    temperature: float = 0.7,
    test_prompts: Optional[List[str]] = None,
) -> SmokeTestResult:
    """
    Run a smoke test on a model.

    This is a convenience function that creates a SmokeTest instance
    and runs it.

    Args:
        model_path: Path to model or HuggingFace model ID
        device: Device to use ("auto", "cuda", "cpu")
        backend: Backend to use ("auto", "transformers", "vllm")
        max_tokens: Maximum tokens to generate per test
        temperature: Sampling temperature
        test_prompts: Custom test prompts (uses defaults if None)

    Returns:
        SmokeTestResult with test outcomes

    Example:
        >>> result = run_smoke_test("./my_trained_model")
        >>> if result.passed:
        ...     print("Model passed smoke test!")
        >>> else:
        ...     print(f"Errors: {result.errors}")
    """
    test = SmokeTest(
        model_path=model_path,
        device=device,
        backend=backend,
        max_tokens=max_tokens,
        temperature=temperature,
        test_prompts=test_prompts,
    )
    return test.run()
