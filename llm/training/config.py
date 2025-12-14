"""Training configuration classes."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class BaseTrainingConfig:
    """Base configuration for all training types."""

    model_name: str = "Qwen/Qwen2.5-7B"
    output_dir: str = "./output"

    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Precision
    fp16: bool = False
    bf16: bool = True

    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3

    # Data
    max_seq_length: int = 2048

    # Device
    device_map: str = "auto"


@dataclass
class LoraConfig:
    """Configuration for LoRA adapters."""

    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class SFTConfig(BaseTrainingConfig):
    """Configuration for supervised fine-tuning."""

    # LoRA settings
    use_lora: bool = True
    lora: LoraConfig = field(default_factory=LoraConfig)

    # Quantization
    load_in_4bit: bool = True
    load_in_8bit: bool = False

    # Data
    dataset_path: str = ""
    dataset_format: Literal["jsonl", "parquet", "huggingface"] = "jsonl"


@dataclass
class PretrainConfig(BaseTrainingConfig):
    """Configuration for continued pretraining."""

    # Usually full fine-tuning for pretraining
    use_lora: bool = False
    lora: LoraConfig = field(default_factory=LoraConfig)

    # Pretraining specific
    dataset_path: str = ""
    dataset_format: Literal["jsonl", "parquet", "huggingface", "text"] = "jsonl"

    # Typically use larger batch sizes for pretraining
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 16


@dataclass
class DPOConfig(BaseTrainingConfig):
    """Configuration for DPO training."""

    # DPO specific
    beta: float = 0.1
    loss_type: Literal["sigmoid", "hinge", "ipo"] = "sigmoid"

    # LoRA settings
    use_lora: bool = True
    lora: LoraConfig = field(default_factory=LoraConfig)

    # Quantization
    load_in_4bit: bool = True

    # Data
    dataset_path: str = ""
    dataset_format: Literal["jsonl", "parquet", "huggingface"] = "jsonl"

    # Reference model
    ref_model: str | None = None  # If None, uses copy of base model
