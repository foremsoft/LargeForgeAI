"""Expert model management and registry."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExpertModel:
    """Configuration for an expert model."""

    name: str
    base_model: str
    adapter_path: str | None = None
    description: str = ""
    specialization: str = ""
    keywords: list[str] = field(default_factory=list)

    # Training info
    trained_on: str | None = None
    training_steps: int | None = None

    # Serving config
    quantization: str | None = None  # awq, gptq, or None
    port: int | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ExpertModel":
        """Create from dictionary."""
        return cls(**data)


class ExpertRegistry:
    """Registry for managing expert models."""

    def __init__(self, config_path: str | Path | None = None):
        self.experts: dict[str, ExpertModel] = {}
        self.config_path = Path(config_path) if config_path else None

        if self.config_path and self.config_path.exists():
            self.load()

    def register(self, expert: ExpertModel):
        """Register a new expert model."""
        self.experts[expert.name] = expert

    def unregister(self, name: str):
        """Remove an expert from the registry."""
        if name in self.experts:
            del self.experts[name]

    def get(self, name: str) -> ExpertModel | None:
        """Get an expert by name."""
        return self.experts.get(name)

    def list(self) -> list[ExpertModel]:
        """List all registered experts."""
        return list(self.experts.values())

    def find_by_keyword(self, keyword: str) -> list[ExpertModel]:
        """Find experts by keyword."""
        keyword_lower = keyword.lower()
        return [
            expert
            for expert in self.experts.values()
            if keyword_lower in [k.lower() for k in expert.keywords]
        ]

    def find_by_specialization(self, spec: str) -> list[ExpertModel]:
        """Find experts by specialization."""
        spec_lower = spec.lower()
        return [
            expert
            for expert in self.experts.values()
            if spec_lower in expert.specialization.lower()
        ]

    def save(self, path: str | Path | None = None):
        """Save registry to JSON file."""
        path = Path(path) if path else self.config_path
        if not path:
            raise ValueError("No config path specified")

        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"experts": [expert.to_dict() for expert in self.experts.values()]}

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str | Path | None = None):
        """Load registry from JSON file."""
        path = Path(path) if path else self.config_path
        if not path:
            raise ValueError("No config path specified")

        with open(path, "r") as f:
            data = json.load(f)

        self.experts = {}
        for expert_data in data.get("experts", []):
            expert = ExpertModel.from_dict(expert_data)
            self.experts[expert.name] = expert


# Default expert configurations
DEFAULT_EXPERTS = [
    ExpertModel(
        name="code",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        description="Code generation and programming assistance",
        specialization="code",
        keywords=["code", "python", "javascript", "programming", "debug", "function"],
        port=8001,
    ),
    ExpertModel(
        name="math",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        description="Mathematical reasoning and calculations",
        specialization="math",
        keywords=["math", "calculate", "equation", "solve", "formula"],
        port=8002,
    ),
    ExpertModel(
        name="writing",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        description="Creative writing and text generation",
        specialization="writing",
        keywords=["write", "essay", "story", "summarize", "creative"],
        port=8003,
    ),
    ExpertModel(
        name="general",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        description="General knowledge and conversation",
        specialization="general",
        keywords=[],
        port=8000,
    ),
]


def create_default_registry(config_path: str | Path | None = None) -> ExpertRegistry:
    """Create a registry with default experts."""
    registry = ExpertRegistry(config_path)
    for expert in DEFAULT_EXPERTS:
        registry.register(expert)
    return registry
