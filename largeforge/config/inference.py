"""Inference configuration classes for LargeForgeAI."""

from typing import List, Optional

from pydantic import Field, field_validator, model_validator

from largeforge.config.base import BaseConfig


class GenerationConfig(BaseConfig):
    """Configuration for text generation."""

    max_tokens: int = Field(default=256, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.9, gt=0, le=1)
    top_k: int = Field(default=50, ge=0)
    repetition_penalty: float = Field(default=1.0, ge=1)
    stop: Optional[List[str]] = None
    stream: bool = False
    do_sample: bool = True

    @model_validator(mode="after")
    def validate_sampling(self) -> "GenerationConfig":
        """Set do_sample based on temperature."""
        if self.temperature == 0:
            object.__setattr__(self, "do_sample", False)
        return self


class InferenceConfig(BaseConfig):
    """Configuration for inference server."""

    model_path: str
    backend: str = "auto"
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    max_model_len: int = Field(default=4096, ge=128)
    gpu_memory_utilization: float = Field(default=0.9, gt=0, le=1)
    tensor_parallel_size: int = Field(default=1, ge=1)
    quantization: Optional[str] = None
    dtype: str = "auto"
    trust_remote_code: bool = False
    enable_prefix_caching: bool = False

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        """Validate inference backend."""
        valid_backends = {"auto", "vllm", "transformers"}
        if v not in valid_backends:
            raise ValueError(f"backend must be one of {valid_backends}")
        return v

    @field_validator("quantization")
    @classmethod
    def validate_quantization(cls, v: Optional[str]) -> Optional[str]:
        """Validate quantization method."""
        if v is not None:
            valid_quant = {"awq", "gptq", "squeezellm"}
            if v.lower() not in valid_quant:
                raise ValueError(f"quantization must be one of {valid_quant}")
            return v.lower()
        return v


class RouterConfig(BaseConfig):
    """Configuration for expert router."""

    classifier_type: str = "hybrid"
    confidence_threshold: float = Field(default=0.6, ge=0, le=1)
    fallback_expert: str = "general"
    keyword_weight: float = Field(default=0.3, ge=0, le=1)
    neural_weight: float = Field(default=0.7, ge=0, le=1)
    neural_model_path: Optional[str] = None
    cache_size: int = Field(default=1000, ge=0)
    timeout_seconds: float = Field(default=30.0, gt=0)

    @field_validator("classifier_type")
    @classmethod
    def validate_classifier_type(cls, v: str) -> str:
        """Validate classifier type."""
        valid_types = {"keyword", "neural", "hybrid"}
        if v not in valid_types:
            raise ValueError(f"classifier_type must be one of {valid_types}")
        return v

    @model_validator(mode="after")
    def validate_weights(self) -> "RouterConfig":
        """Ensure weights sum to 1.0 for hybrid classifier."""
        if self.classifier_type == "hybrid":
            total = self.keyword_weight + self.neural_weight
            if abs(total - 1.0) > 0.001:
                raise ValueError(
                    f"keyword_weight + neural_weight must equal 1.0, got {total}"
                )
        return self
