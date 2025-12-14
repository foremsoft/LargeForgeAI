"""Training configuration classes for LargeForgeAI."""

from typing import List, Optional

from pydantic import Field, field_validator, model_validator

from largeforge.config.base import BaseConfig


class TrainingConfig(BaseConfig):
    """Base training configuration."""

    output_dir: str
    num_train_epochs: int = Field(default=3, ge=1, le=100)
    per_device_train_batch_size: int = Field(default=4, ge=1)
    per_device_eval_batch_size: int = Field(default=4, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=2e-5, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_ratio: float = Field(default=0.1, ge=0, le=1)
    max_grad_norm: float = Field(default=1.0, gt=0)
    logging_steps: int = Field(default=10, ge=1)
    save_steps: int = Field(default=500, ge=1)
    eval_steps: int = Field(default=500, ge=1)
    save_total_limit: int = Field(default=3, ge=1)
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = False
    seed: int = 42
    report_to: str = "none"
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"

    @model_validator(mode="after")
    def validate_precision(self) -> "TrainingConfig":
        """Ensure fp16 and bf16 are not both enabled."""
        if self.fp16 and self.bf16:
            raise ValueError("Cannot enable both fp16 and bf16")
        return self


class LoRAConfig(BaseConfig):
    """LoRA adapter configuration."""

    r: int = Field(default=8, ge=1, le=256)
    lora_alpha: int = Field(default=16, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0, le=1)
    target_modules: List[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    modules_to_save: Optional[List[str]] = None

    @field_validator("bias")
    @classmethod
    def validate_bias(cls, v: str) -> str:
        """Validate bias setting."""
        valid_bias = {"none", "all", "lora_only"}
        if v not in valid_bias:
            raise ValueError(f"bias must be one of {valid_bias}")
        return v


class SFTConfig(TrainingConfig):
    """Supervised Fine-Tuning configuration."""

    max_seq_length: int = Field(default=2048, ge=128)
    packing: bool = False
    dataset_text_field: str = "text"
    neftune_noise_alpha: Optional[float] = None


class DPOConfig(TrainingConfig):
    """Direct Preference Optimization configuration."""

    beta: float = Field(default=0.1, gt=0, le=1)
    max_length: int = Field(default=1024, ge=128)
    max_prompt_length: int = Field(default=512, ge=64)
    loss_type: str = "sigmoid"
    label_smoothing: float = Field(default=0.0, ge=0, le=0.5)
    generate_during_eval: bool = False

    @field_validator("loss_type")
    @classmethod
    def validate_loss_type(cls, v: str) -> str:
        """Validate DPO loss type."""
        valid_types = {"sigmoid", "hinge", "ipo", "kto_pair"}
        if v not in valid_types:
            raise ValueError(f"loss_type must be one of {valid_types}")
        return v

    @model_validator(mode="after")
    def validate_lengths(self) -> "DPOConfig":
        """Ensure prompt length is less than total length."""
        if self.max_prompt_length >= self.max_length:
            raise ValueError("max_prompt_length must be less than max_length")
        return self
