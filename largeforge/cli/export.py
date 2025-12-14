"""CLI commands for model export and conversion."""

from pathlib import Path
from typing import Optional

import click

from largeforge.utils import get_logger

logger = get_logger(__name__)


@click.group()
def export() -> None:
    """Model export and conversion commands."""
    pass


@export.command("model")
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), required=True, help="Output directory")
@click.option(
    "--format", "-f", "output_format",
    type=click.Choice(["safetensors", "pytorch"]),
    default="safetensors",
    help="Output format"
)
@click.option("--merge-lora/--no-merge-lora", default=True, help="Merge LoRA adapters if present")
@click.option("--shard-size", "-s", default="5GB", help="Maximum shard size for large models")
def export_model(
    model_path: str,
    output: str,
    output_format: str,
    merge_lora: bool,
    shard_size: str,
) -> None:
    """Export a model to a specific format.

    Exports trained models with optional LoRA merging and format conversion.
    Useful for preparing models for deployment or sharing.

    Examples:
        # Export with LoRA merged
        largeforge export model ./output/model -o ./exported

        # Export to PyTorch format
        largeforge export model ./output/model -o ./exported --format pytorch

        # Export without merging LoRA
        largeforge export model ./output/model -o ./exported --no-merge-lora
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    click.echo(f"Exporting model from: {model_path}")
    click.echo(f"Output directory: {output}")
    click.echo(f"Format: {output_format}")
    click.echo(f"Merge LoRA: {merge_lora}")
    click.echo("")

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if this is a LoRA adapter
    adapter_config = Path(model_path) / "adapter_config.json"
    is_lora = adapter_config.exists()

    if is_lora:
        click.echo("Detected LoRA adapter")

        if merge_lora:
            click.echo("Loading base model and merging LoRA...")

            from peft import PeftModel, PeftConfig

            # Load adapter config to get base model
            import json
            with open(adapter_config, "r") as f:
                config = json.load(f)
            base_model_name = config.get("base_model_name_or_path")

            if not base_model_name:
                click.echo("Error: Could not determine base model from adapter config", err=True)
                raise click.Abort()

            # Load base model
            click.echo(f"Loading base model: {base_model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="cpu",
                trust_remote_code=True,
            )

            # Load and merge LoRA
            click.echo("Loading LoRA adapter...")
            model = PeftModel.from_pretrained(model, model_path)

            click.echo("Merging weights...")
            model = model.merge_and_unload()

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        else:
            click.echo("Exporting LoRA adapter only (not merged)")
            import shutil
            shutil.copytree(model_path, output)
            click.echo(f"\nAdapter exported to: {output}")
            return
    else:
        click.echo("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Save in requested format
    click.echo(f"Saving model in {output_format} format...")

    if output_format == "safetensors":
        model.save_pretrained(
            output_path,
            safe_serialization=True,
            max_shard_size=shard_size,
        )
    else:
        model.save_pretrained(
            output_path,
            safe_serialization=False,
            max_shard_size=shard_size,
        )

    tokenizer.save_pretrained(output_path)

    # List output files
    files = list(output_path.glob("*"))
    total_size = sum(f.stat().st_size for f in files if f.is_file())

    click.echo(f"\nExported {len(files)} files ({total_size / 1e9:.2f}GB)")
    click.echo(f"Output: {output_path}")


@export.command("merge-lora")
@click.argument("base_model", type=str)
@click.argument("adapter_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), required=True, help="Output directory")
@click.option("--device", "-d", default="cpu", help="Device to use for merging")
def merge_lora(base_model: str, adapter_path: str, output: str, device: str) -> None:
    """Merge LoRA adapters into the base model.

    Creates a standalone model by merging LoRA weights into the base model.
    The resulting model can be used without PEFT.

    Examples:
        # Merge LoRA adapter
        largeforge export merge-lora Qwen/Qwen2.5-7B ./lora-adapter -o ./merged-model

        # Use GPU for faster merging
        largeforge export merge-lora Qwen/Qwen2.5-7B ./lora-adapter -o ./merged --device cuda
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    click.echo(f"Base model: {base_model}")
    click.echo(f"Adapter: {adapter_path}")
    click.echo(f"Output: {output}")
    click.echo(f"Device: {device}")
    click.echo("")

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load base model
    click.echo("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map=device if device == "auto" else None,
        trust_remote_code=True,
    )

    if device not in ("auto", "cpu"):
        model = model.to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Load adapter
    click.echo("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)

    # Merge
    click.echo("Merging weights...")
    model = model.merge_and_unload()

    # Save
    click.echo("Saving merged model...")
    model.save_pretrained(
        output_path,
        safe_serialization=True,
    )
    tokenizer.save_pretrained(output_path)

    click.echo(f"\nMerged model saved to: {output_path}")


@export.command("onnx")
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), required=True, help="Output directory")
@click.option("--opset", default=14, type=int, help="ONNX opset version")
def export_onnx(model_path: str, output: str, opset: int) -> None:
    """Export model to ONNX format.

    Exports the model to ONNX format for use with ONNX Runtime
    or other ONNX-compatible inference engines.

    Examples:
        largeforge export onnx ./output/model -o ./onnx-model
    """
    click.echo(f"Exporting to ONNX from: {model_path}")
    click.echo(f"Output: {output}")
    click.echo(f"Opset: {opset}")
    click.echo("")

    try:
        from optimum.exporters.onnx import main_export
    except ImportError:
        click.echo("Error: optimum not installed", err=True)
        click.echo("Install with: pip install optimum[exporters]", err=True)
        raise click.Abort()

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    click.echo("Converting to ONNX...")

    try:
        main_export(
            model_name_or_path=model_path,
            output=output_path,
            task="text-generation",
            opset=opset,
        )
        click.echo(f"\nONNX model saved to: {output_path}")
    except Exception as e:
        click.echo(f"Export failed: {e}", err=True)
        raise click.Abort()


@export.command("quantize")
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), required=True, help="Output directory")
@click.option(
    "--method", "-m",
    type=click.Choice(["awq", "gptq", "gguf"]),
    default="awq",
    help="Quantization method"
)
@click.option("--bits", "-b", default=4, type=click.Choice([4, 8]), help="Quantization bits")
def quantize_model(model_path: str, output: str, method: str, bits: int) -> None:
    """Quantize a model for efficient inference.

    Reduces model size and improves inference speed through quantization.

    Methods:
    - awq: Activation-aware Weight Quantization (recommended)
    - gptq: GPTQ quantization
    - gguf: GGML/GGUF format for llama.cpp

    Examples:
        # AWQ quantization (recommended)
        largeforge export quantize ./output/model -o ./quantized --method awq

        # 8-bit GPTQ
        largeforge export quantize ./output/model -o ./quantized --method gptq --bits 8
    """
    click.echo(f"Quantizing model: {model_path}")
    click.echo(f"Method: {method}")
    click.echo(f"Bits: {bits}")
    click.echo(f"Output: {output}")
    click.echo("")

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    if method == "awq":
        try:
            from awq import AutoAWQForCausalLM
            from transformers import AutoTokenizer
        except ImportError:
            click.echo("Error: autoawq not installed", err=True)
            click.echo("Install with: pip install autoawq", err=True)
            raise click.Abort()

        click.echo("Loading model for AWQ quantization...")
        model = AutoAWQForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        quant_config = {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": bits,
            "version": "GEMM",
        }

        click.echo("Quantizing (this may take a while)...")
        model.quantize(tokenizer, quant_config=quant_config)

        click.echo("Saving quantized model...")
        model.save_quantized(str(output_path))
        tokenizer.save_pretrained(output_path)

    elif method == "gptq":
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
            from transformers import AutoTokenizer
        except ImportError:
            click.echo("Error: auto-gptq not installed", err=True)
            click.echo("Install with: pip install auto-gptq", err=True)
            raise click.Abort()

        click.echo("Loading model for GPTQ quantization...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        quantize_config = BaseQuantizeConfig(
            bits=bits,
            group_size=128,
            desc_act=False,
        )

        model = AutoGPTQForCausalLM.from_pretrained(
            model_path,
            quantize_config=quantize_config,
            trust_remote_code=True,
        )

        click.echo("Quantizing (this may take a while)...")
        # Note: GPTQ needs calibration data
        click.echo("Warning: Using default calibration. For best results, provide calibration data.", fg="yellow")

        model.quantize([])  # Empty calibration for demo

        click.echo("Saving quantized model...")
        model.save_quantized(str(output_path))
        tokenizer.save_pretrained(output_path)

    elif method == "gguf":
        click.echo("GGUF export requires llama.cpp convert script")
        click.echo("Please use llama.cpp's convert.py for GGUF conversion")
        click.echo("See: https://github.com/ggerganov/llama.cpp")
        raise click.Abort()

    click.echo(f"\nQuantized model saved to: {output_path}")
