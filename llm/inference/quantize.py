"""Model quantization utilities."""

from pathlib import Path


def quantize_awq(
    model_name: str,
    output_dir: str,
    bits: int = 4,
    group_size: int = 128,
    zero_point: bool = True,
    calib_data: str = "pileval",
    calib_samples: int = 512,
):
    """
    Quantize a model using AWQ (Activation-aware Weight Quantization).

    AWQ preserves important weights for better accuracy.

    Args:
        model_name: HuggingFace model name or local path
        output_dir: Directory to save quantized model
        bits: Quantization bits (4 recommended)
        group_size: Group size for quantization
        zero_point: Use zero-point quantization
        calib_data: Calibration dataset name
        calib_samples: Number of calibration samples
    """
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("AutoAWQ not installed. Install with: pip install autoawq")

    print(f"Loading model: {model_name}")
    model = AutoAWQForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    quant_config = {
        "w_bit": bits,
        "q_group_size": group_size,
        "zero_point": zero_point,
    }

    print(f"Quantizing with config: {quant_config}")
    model.quantize(tokenizer, quant_config=quant_config)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving to: {output_dir}")
    model.save_quantized(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    print("Quantization complete!")
    return str(output_path)


def quantize_gptq(
    model_name: str,
    output_dir: str,
    bits: int = 4,
    group_size: int = 128,
    dataset: str = "c4",
    desc_act: bool = False,
):
    """
    Quantize a model using GPTQ.

    GPTQ is a post-training quantization method.

    Args:
        model_name: HuggingFace model name or local path
        output_dir: Directory to save quantized model
        bits: Quantization bits (4 recommended)
        group_size: Group size for quantization
        dataset: Calibration dataset
        desc_act: Use descending activation order
    """
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("AutoGPTQ not installed. Install with: pip install auto-gptq")

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
    )

    model = AutoGPTQForCausalLM.from_pretrained(
        model_name,
        quantize_config=quantize_config,
        trust_remote_code=True,
    )

    print(f"Quantizing with {bits}-bit GPTQ")
    model.quantize(
        examples=get_gptq_calib_data(tokenizer, dataset),
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving to: {output_dir}")
    model.save_quantized(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    print("Quantization complete!")
    return str(output_path)


def get_gptq_calib_data(tokenizer, dataset_name: str = "c4", n_samples: int = 128):
    """Get calibration data for GPTQ quantization."""
    from datasets import load_dataset

    if dataset_name == "c4":
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
        samples = []
        for i, example in enumerate(dataset):
            if i >= n_samples:
                break
            samples.append(tokenizer(example["text"], truncation=True, max_length=2048))
        return samples
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def merge_lora_weights(
    base_model: str,
    lora_path: str,
    output_dir: str,
):
    """
    Merge LoRA adapter weights into the base model.

    Args:
        base_model: Base model name or path
        lora_path: Path to LoRA adapter
        output_dir: Directory to save merged model
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print(f"Loading LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)

    print("Merging weights...")
    model = model.merge_and_unload()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving to: {output_dir}")
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    print("Merge complete!")
    return str(output_path)
