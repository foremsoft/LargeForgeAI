"""Synthetic data generation using LLM APIs."""

import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional

from largeforge.utils import get_logger, ensure_dir

logger = get_logger(__name__)


# Default topics for data generation
DEFAULT_TOPICS = [
    "science and technology",
    "history and culture",
    "mathematics and logic",
    "programming and software",
    "business and finance",
    "health and medicine",
    "creative writing",
    "education and learning",
    "philosophy and ethics",
    "daily life and practical advice",
]

# Prompt templates for SFT data generation
SFT_SYSTEM_PROMPT = """You are an expert at creating high-quality training data for AI assistants.
Your task is to generate diverse, helpful, and accurate instruction-response pairs.
Each response should be detailed, well-structured, and demonstrate expert knowledge."""

SFT_GENERATION_PROMPT = """Generate a training example for an AI assistant.

Topic: {topic}
Difficulty: {difficulty}

Create a realistic user instruction/question and a comprehensive assistant response.
The instruction should be specific and actionable.
The response should be helpful, accurate, and well-formatted.

Output in JSON format:
{{
    "instruction": "The user's question or instruction",
    "input": "Optional additional context (can be empty string)",
    "output": "The assistant's detailed response"
}}

Only output the JSON, no additional text."""

# Prompt templates for DPO data generation
DPO_SYSTEM_PROMPT = """You are an expert at creating preference data for AI alignment.
Your task is to generate prompts with two responses: one high-quality (chosen) and one lower-quality (rejected).
The difference should be meaningful but both responses should be plausible."""

DPO_GENERATION_PROMPT = """Generate a preference training example.

Topic: {topic}
Difficulty: {difficulty}

Create:
1. A realistic user prompt/question
2. A high-quality "chosen" response (helpful, accurate, well-structured)
3. A lower-quality "rejected" response (less helpful, less detailed, or has minor issues)

The rejected response should NOT be completely wrong - it should be a plausible but inferior response.

Output in JSON format:
{{
    "prompt": "The user's question or instruction",
    "chosen": "The high-quality preferred response",
    "rejected": "The lower-quality response"
}}

Only output the JSON, no additional text."""


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""

    provider: Literal["openai", "anthropic"] = "openai"
    model: str = "gpt-4"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    num_samples: int = 100
    format: Literal["sft", "dpo"] = "sft"
    topics: List[str] = field(default_factory=lambda: DEFAULT_TOPICS.copy())
    difficulties: List[str] = field(
        default_factory=lambda: ["easy", "medium", "hard"]
    )
    system_prompt: Optional[str] = None
    generation_prompt: Optional[str] = None
    temperature: float = 0.8
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_delay: float = 0.5
    batch_size: int = 10

    def __post_init__(self):
        """Load API key from environment if not provided."""
        if self.api_key is None:
            if self.provider == "openai":
                self.api_key = os.environ.get("OPENAI_API_KEY")
            elif self.provider == "anthropic":
                self.api_key = os.environ.get("ANTHROPIC_API_KEY")

        if self.api_key is None:
            raise ValueError(
                f"API key not provided and {self.provider.upper()}_API_KEY "
                "environment variable not set"
            )


class SyntheticGenerator:
    """Generator for synthetic training data using LLM APIs."""

    def __init__(self, config: SyntheticConfig):
        """
        Initialize the synthetic data generator.

        Args:
            config: Configuration for generation
        """
        self.config = config
        self._client = None

    @property
    def client(self):
        """Lazy-load the API client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self):
        """Create the appropriate API client."""
        if self.config.provider == "openai":
            try:
                from openai import OpenAI

                return OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                )
            except ImportError:
                raise ImportError(
                    "openai package required for OpenAI provider. "
                    "Install with: pip install openai"
                )

        elif self.config.provider == "anthropic":
            try:
                from anthropic import Anthropic

                return Anthropic(api_key=self.config.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package required for Anthropic provider. "
                    "Install with: pip install anthropic"
                )

        raise ValueError(f"Unknown provider: {self.config.provider}")

    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        """
        Call the LLM API with retry logic.

        Args:
            messages: List of message dictionaries

        Returns:
            Generated text response
        """
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                if self.config.provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.config.model,
                        messages=messages,
                        temperature=self.config.temperature,
                    )
                    return response.choices[0].message.content

                elif self.config.provider == "anthropic":
                    # Convert messages format for Anthropic
                    system = None
                    user_messages = []
                    for msg in messages:
                        if msg["role"] == "system":
                            system = msg["content"]
                        else:
                            user_messages.append(msg)

                    response = self.client.messages.create(
                        model=self.config.model,
                        max_tokens=4096,
                        system=system or "",
                        messages=user_messages,
                        temperature=self.config.temperature,
                    )
                    return response.content[0].text

            except Exception as e:
                last_error = e
                logger.warning(
                    f"API call failed (attempt {attempt + 1}/{self.config.max_retries}): {e}"
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))

        raise RuntimeError(f"API call failed after {self.config.max_retries} retries: {last_error}")

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON from API response.

        Args:
            response: Raw API response

        Returns:
            Parsed dictionary or None if parsing fails
        """
        # Try to extract JSON from response
        response = response.strip()

        # Handle markdown code blocks
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]

        response = response.strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return None

    def generate_sft_sample(
        self,
        topic: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> Optional[Dict[str, str]]:
        """
        Generate a single SFT training sample.

        Args:
            topic: Topic for generation (random if None)
            difficulty: Difficulty level (random if None)

        Returns:
            Dictionary with instruction, input, output or None if failed
        """
        topic = topic or random.choice(self.config.topics)
        difficulty = difficulty or random.choice(self.config.difficulties)

        system_prompt = self.config.system_prompt or SFT_SYSTEM_PROMPT
        generation_prompt = self.config.generation_prompt or SFT_GENERATION_PROMPT

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": generation_prompt.format(
                    topic=topic, difficulty=difficulty
                ),
            },
        ]

        response = self._call_api(messages)
        data = self._parse_json_response(response)

        if data and all(k in data for k in ["instruction", "output"]):
            # Ensure input field exists
            data.setdefault("input", "")
            return {
                "instruction": data["instruction"],
                "input": data["input"],
                "output": data["output"],
            }

        return None

    def generate_dpo_sample(
        self,
        topic: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> Optional[Dict[str, str]]:
        """
        Generate a single DPO preference sample.

        Args:
            topic: Topic for generation (random if None)
            difficulty: Difficulty level (random if None)

        Returns:
            Dictionary with prompt, chosen, rejected or None if failed
        """
        topic = topic or random.choice(self.config.topics)
        difficulty = difficulty or random.choice(self.config.difficulties)

        system_prompt = self.config.system_prompt or DPO_SYSTEM_PROMPT
        generation_prompt = self.config.generation_prompt or DPO_GENERATION_PROMPT

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": generation_prompt.format(
                    topic=topic, difficulty=difficulty
                ),
            },
        ]

        response = self._call_api(messages)
        data = self._parse_json_response(response)

        if data and all(k in data for k in ["prompt", "chosen", "rejected"]):
            return {
                "prompt": data["prompt"],
                "chosen": data["chosen"],
                "rejected": data["rejected"],
            }

        return None

    def generate(
        self,
        num_samples: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict[str, str]]:
        """
        Generate synthetic training data.

        Args:
            num_samples: Number of samples to generate (uses config if None)
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of generated samples
        """
        num_samples = num_samples or self.config.num_samples
        samples = []
        failed = 0

        logger.info(
            f"Generating {num_samples} {self.config.format.upper()} samples "
            f"using {self.config.provider}/{self.config.model}"
        )

        generate_fn = (
            self.generate_sft_sample
            if self.config.format == "sft"
            else self.generate_dpo_sample
        )

        for i in range(num_samples):
            try:
                sample = generate_fn()
                if sample:
                    samples.append(sample)
                else:
                    failed += 1
                    logger.warning(f"Sample {i + 1} generation returned None")

            except Exception as e:
                failed += 1
                logger.warning(f"Failed to generate sample {i + 1}: {e}")

            if progress_callback:
                progress_callback(i + 1, num_samples)

            # Rate limiting
            if i < num_samples - 1:
                time.sleep(self.config.rate_limit_delay)

        logger.info(
            f"Generated {len(samples)} samples successfully, {failed} failed"
        )
        return samples

    def generate_iter(
        self,
        num_samples: Optional[int] = None,
    ) -> Iterator[Dict[str, str]]:
        """
        Generate synthetic data as an iterator (streaming).

        Args:
            num_samples: Number of samples to generate

        Yields:
            Individual generated samples
        """
        num_samples = num_samples or self.config.num_samples

        generate_fn = (
            self.generate_sft_sample
            if self.config.format == "sft"
            else self.generate_dpo_sample
        )

        for i in range(num_samples):
            try:
                sample = generate_fn()
                if sample:
                    yield sample

            except Exception as e:
                logger.warning(f"Failed to generate sample {i + 1}: {e}")

            # Rate limiting
            if i < num_samples - 1:
                time.sleep(self.config.rate_limit_delay)

    def save(
        self,
        data: List[Dict[str, str]],
        path: str,
        format: Literal["jsonl", "json"] = "jsonl",
    ) -> None:
        """
        Save generated data to file.

        Args:
            data: List of generated samples
            path: Output file path
            format: Output format (jsonl or json)
        """
        output_path = Path(path)
        ensure_dir(output_path.parent)

        if format == "jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for sample in data:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(data)} samples to {output_path}")


def generate_synthetic_data(
    num_samples: int = 100,
    format: Literal["sft", "dpo"] = "sft",
    provider: Literal["openai", "anthropic"] = "openai",
    model: Optional[str] = None,
    topics: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> List[Dict[str, str]]:
    """
    Generate synthetic training data.

    Args:
        num_samples: Number of samples to generate
        format: Output format (sft or dpo)
        provider: API provider (openai or anthropic)
        model: Model to use (defaults based on provider)
        topics: List of topics to generate for
        output_path: Optional path to save generated data
        api_key: API key (uses environment variable if not provided)
        **kwargs: Additional config options

    Returns:
        List of generated samples

    Example:
        >>> data = generate_synthetic_data(
        ...     num_samples=50,
        ...     format="sft",
        ...     provider="openai",
        ...     model="gpt-4",
        ...     topics=["programming", "mathematics"],
        ... )
        >>> print(len(data))
        50
    """
    # Set default model based on provider
    if model is None:
        model = "gpt-4" if provider == "openai" else "claude-3-sonnet-20240229"

    config = SyntheticConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        num_samples=num_samples,
        format=format,
        topics=topics or DEFAULT_TOPICS.copy(),
        **kwargs,
    )

    generator = SyntheticGenerator(config)
    data = generator.generate()

    if output_path:
        file_format = "jsonl" if output_path.endswith(".jsonl") else "json"
        generator.save(data, output_path, format=file_format)

    return data


def augment_dataset(
    data: List[Dict[str, Any]],
    augmentation_factor: int = 2,
    provider: Literal["openai", "anthropic"] = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Augment existing dataset by generating variations.

    Args:
        data: Existing training data
        augmentation_factor: How many variations per sample
        provider: API provider
        model: Model to use
        api_key: API key

    Returns:
        Augmented dataset (original + variations)
    """
    if model is None:
        model = "gpt-4" if provider == "openai" else "claude-3-sonnet-20240229"

    config = SyntheticConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        num_samples=1,  # We'll generate one at a time
        format="sft",
    )

    generator = SyntheticGenerator(config)
    augmented = list(data)  # Start with original data

    augmentation_prompt = """Given this training example, create a variation that teaches the same concept but with different wording, context, or scenario.

Original:
Instruction: {instruction}
Input: {input}
Output: {output}

Create a new variation in JSON format:
{{
    "instruction": "A different but related instruction",
    "input": "Different or empty context",
    "output": "Appropriate response teaching same concept"
}}

Only output the JSON, no additional text."""

    for sample in data:
        for _ in range(augmentation_factor - 1):
            try:
                messages = [
                    {"role": "system", "content": SFT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": augmentation_prompt.format(
                            instruction=sample.get("instruction", ""),
                            input=sample.get("input", ""),
                            output=sample.get("output", ""),
                        ),
                    },
                ]

                response = generator._call_api(messages)
                variation = generator._parse_json_response(response)

                if variation and "instruction" in variation and "output" in variation:
                    variation.setdefault("input", "")
                    augmented.append(variation)

                time.sleep(config.rate_limit_delay)

            except Exception as e:
                logger.warning(f"Failed to augment sample: {e}")

    logger.info(
        f"Augmented dataset from {len(data)} to {len(augmented)} samples"
    )
    return augmented
