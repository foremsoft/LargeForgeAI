"""LoRA (Low-Rank Adaptation) utilities for LargeForgeAI."""

from typing import Any, Dict, List, Optional, Union

from largeforge.config import LoRAConfig
from largeforge.utils import get_logger

logger = get_logger(__name__)

# Common target modules for different model architectures
MODEL_TARGET_MODULES = {
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
    "gpt2": ["c_attn", "c_proj", "c_fc"],
    "opt": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    "falcon": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "bloom": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "qwen": ["c_attn", "c_proj", "w1", "w2"],
}


def get_lora_target_modules(
    model_name_or_config: str,
    include_mlp: bool = True,
) -> List[str]:
    """
    Get the recommended LoRA target modules for a model.

    Args:
        model_name_or_config: Model name or architecture type
        include_mlp: Whether to include MLP layers

    Returns:
        List of target module names
    """
    model_lower = model_name_or_config.lower()

    # Check for known architectures
    for arch_name, modules in MODEL_TARGET_MODULES.items():
        if arch_name in model_lower:
            if include_mlp:
                return modules
            else:
                # Filter to attention-only modules
                attn_keywords = ["q_", "k_", "v_", "o_", "query", "key", "value", "attn"]
                return [m for m in modules if any(k in m for k in attn_keywords)]

    # Default to common attention module names
    logger.warning(f"Unknown model architecture: {model_name_or_config}, using default modules")
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def create_lora_config(
    config: Optional[LoRAConfig] = None,
    r: int = 8,
    lora_alpha: int = 16,
    target_modules: Optional[List[str]] = None,
    lora_dropout: float = 0.05,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
) -> Any:
    """
    Create a PEFT LoraConfig object.

    Args:
        config: Optional LoRAConfig from largeforge
        r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        target_modules: Modules to apply LoRA to
        lora_dropout: Dropout probability
        bias: Bias training mode ("none", "all", "lora_only")
        task_type: Task type for PEFT

    Returns:
        PEFT LoraConfig object

    Example:
        >>> lora_config = create_lora_config(r=16, lora_alpha=32)
        >>> model = get_peft_model(model, lora_config)
    """
    try:
        from peft import LoraConfig as PeftLoraConfig, TaskType
    except ImportError:
        raise ImportError("peft is required for LoRA. Install with: pip install peft")

    # Use provided config if available
    if config is not None:
        r = config.r
        lora_alpha = config.lora_alpha
        target_modules = config.target_modules
        lora_dropout = config.lora_dropout
        bias = config.bias

    # Map task type string to enum
    task_type_map = {
        "CAUSAL_LM": TaskType.CAUSAL_LM,
        "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
        "SEQ_CLS": TaskType.SEQ_CLS,
        "TOKEN_CLS": TaskType.TOKEN_CLS,
        "QUESTION_ANS": TaskType.QUESTION_ANS,
        "FEATURE_EXTRACTION": TaskType.FEATURE_EXTRACTION,
    }

    peft_task_type = task_type_map.get(task_type, TaskType.CAUSAL_LM)

    lora_config = PeftLoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=peft_task_type,
    )

    logger.info(f"Created LoRA config: r={r}, alpha={lora_alpha}, targets={target_modules}")
    return lora_config


def prepare_model_for_lora(
    model,
    lora_config: Optional[Any] = None,
    config: Optional[LoRAConfig] = None,
    **kwargs,
):
    """
    Prepare a model for LoRA training.

    Args:
        model: The base model
        lora_config: PEFT LoraConfig object
        config: LargeForge LoRAConfig
        **kwargs: Additional arguments for create_lora_config

    Returns:
        Model with LoRA adapters

    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> model = prepare_model_for_lora(model, r=16)
    """
    try:
        from peft import get_peft_model, prepare_model_for_kbit_training
    except ImportError:
        raise ImportError("peft is required for LoRA. Install with: pip install peft")

    # Check if model needs kbit preparation (quantized models)
    if hasattr(model, "is_loaded_in_8bit") and model.is_loaded_in_8bit:
        logger.info("Preparing 8-bit model for training")
        model = prepare_model_for_kbit_training(model)
    elif hasattr(model, "is_loaded_in_4bit") and model.is_loaded_in_4bit:
        logger.info("Preparing 4-bit model for training")
        model = prepare_model_for_kbit_training(model)

    # Create LoRA config if not provided
    if lora_config is None:
        lora_config = create_lora_config(config=config, **kwargs)

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / total_params

    logger.info(
        f"LoRA applied: {trainable_params:,} trainable params "
        f"({trainable_pct:.2f}% of {total_params:,} total)"
    )

    return model


def merge_lora_weights(
    model,
    output_dir: str,
    safe_serialization: bool = True,
) -> str:
    """
    Merge LoRA weights into the base model and save.

    Args:
        model: PEFT model with LoRA adapters
        output_dir: Directory to save merged model
        safe_serialization: Use safetensors format

    Returns:
        Path to saved model

    Example:
        >>> merged_path = merge_lora_weights(peft_model, "./merged_model")
    """
    from pathlib import Path

    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError("peft is required. Install with: pip install peft")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Merging LoRA weights to {output_dir}")

    # Check if model is PEFT model
    if not isinstance(model, PeftModel):
        raise ValueError("Model must be a PEFT model with LoRA adapters")

    # Merge and unload
    merged_model = model.merge_and_unload()

    # Save
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=safe_serialization,
    )

    logger.info(f"Merged model saved to {output_dir}")
    return str(output_dir)


def save_lora_weights(
    model,
    output_dir: str,
    adapter_name: str = "default",
) -> str:
    """
    Save only the LoRA adapter weights.

    Args:
        model: PEFT model with LoRA adapters
        output_dir: Directory to save adapter weights
        adapter_name: Name of the adapter to save

    Returns:
        Path to saved adapter
    """
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving LoRA adapter to {output_dir}")

    model.save_pretrained(output_dir)

    logger.info(f"LoRA adapter saved to {output_dir}")
    return str(output_dir)


def load_lora_weights(
    model,
    adapter_path: str,
    adapter_name: str = "default",
):
    """
    Load LoRA weights into a model.

    Args:
        model: Base model
        adapter_path: Path to LoRA adapter
        adapter_name: Name for the loaded adapter

    Returns:
        Model with LoRA adapters loaded
    """
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError("peft is required. Install with: pip install peft")

    logger.info(f"Loading LoRA adapter from {adapter_path}")

    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        adapter_name=adapter_name,
    )

    logger.info("LoRA adapter loaded successfully")
    return model


def get_trainable_parameters(model) -> Dict[str, int]:
    """
    Get statistics about trainable parameters.

    Args:
        model: The model (can be PEFT or regular)

    Returns:
        Dictionary with parameter counts
    """
    trainable = 0
    total = 0
    trainable_names = []

    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
            trainable_names.append(name)

    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_percent": 100 * trainable / total if total > 0 else 0,
        "trainable_layers": len(trainable_names),
    }
