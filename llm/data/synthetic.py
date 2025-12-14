"""Synthetic data generation utilities."""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    temperature: float = 0.7
    max_new_tokens: int = 512
    num_samples: int = 1000
    batch_size: int = 8


class SyntheticDataGenerator:
    """Generate synthetic training data using a teacher model."""

    def __init__(self, config: SyntheticConfig | None = None):
        self.config = config or SyntheticConfig()
        self.model = None
        self.tokenizer = None

    def load_model(self, device: str = "auto"):
        """Load the teacher model for generation."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map=device,
            torch_dtype="auto",
        )

    def generate_from_template(
        self,
        template: str,
        variables: dict[str, list[str]],
        num_samples: int | None = None,
    ) -> list[dict]:
        """
        Generate synthetic data by filling template with random variables.

        Args:
            template: Prompt template with {variable} placeholders
            variables: Dict mapping variable names to possible values
            num_samples: Number of samples to generate

        Returns:
            List of {"prompt": ..., "response": ...} dicts
        """
        if self.model is None:
            self.load_model()

        num_samples = num_samples or self.config.num_samples
        results = []

        for _ in range(num_samples):
            # Fill template with random variable choices
            filled = template
            for var_name, var_options in variables.items():
                filled = filled.replace(f"{{{var_name}}}", random.choice(var_options))

            # Generate response
            inputs = self.tokenizer(filled, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )

            results.append({"prompt": filled, "response": response.strip()})

        return results

    def generate_with_seed_prompts(
        self,
        seed_prompts: list[str],
        system_prompt: str = "You are a helpful AI assistant.",
    ) -> list[dict]:
        """Generate responses for a list of seed prompts."""
        if self.model is None:
            self.load_model()

        results = []
        for prompt in seed_prompts:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )

            results.append({
                "system": system_prompt,
                "prompt": prompt,
                "response": response.strip(),
            })

        return results


def save_jsonl(data: list[dict], path: str | Path):
    """Save data to JSONL format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    """Load data from JSONL format."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
