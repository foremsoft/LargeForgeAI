"""Base configuration classes for LargeForgeAI."""

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, field_validator


class BaseConfig(BaseModel):
    """Base configuration class with serialization support."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
    )

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "BaseConfig":
        """Create config from dictionary."""
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str) -> "BaseConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


class ModelConfig(BaseConfig):
    """Configuration for model loading."""

    name: str
    revision: str = "main"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = False
    use_cache: bool = True
    attn_implementation: Optional[str] = None

    @field_validator("torch_dtype")
    @classmethod
    def validate_torch_dtype(cls, v: str) -> str:
        """Validate torch dtype."""
        valid_dtypes = {"float16", "bfloat16", "float32"}
        if v not in valid_dtypes:
            raise ValueError(f"torch_dtype must be one of {valid_dtypes}")
        return v

    @field_validator("device_map")
    @classmethod
    def validate_device_map(cls, v: str) -> str:
        """Validate device map."""
        valid_maps = {"auto", "cpu", "cuda", "balanced", "sequential"}
        if v not in valid_maps and not v.startswith("cuda:"):
            raise ValueError(f"device_map must be one of {valid_maps} or cuda:N")
        return v
