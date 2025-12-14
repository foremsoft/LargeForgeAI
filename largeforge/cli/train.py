"""Training CLI commands."""

from pathlib import Path
from typing import Optional

import click

from largeforge.utils import get_logger

logger = get_logger(__name__)


@click.command()
@click.argument("model_path", type=str)
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), required=True, help="Output directory")
@click.option("--epochs", "-e", type=int, default=3, help="Number of training epochs")
@click.option("--batch-size", "-b", type=int, default=4, help="Training batch size")
@click.option("--learning-rate", "-lr", type=float, default=2e-5, help="Learning rate")
@click.option("--max-length", type=int, default=2048, help="Maximum sequence length")
@click.option("--lora/--no-lora", default=True, help="Use LoRA for training")
@click.option("--lora-r", type=int, default=16, help="LoRA rank")
@click.option("--lora-alpha", type=int, default=32, help="LoRA alpha")
@click.option("--gradient-accumulation", type=int, default=4, help="Gradient accumulation steps")
@click.option("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
@click.option("--fp16/--no-fp16", default=False, help="Use FP16 training")
@click.option("--bf16/--no-bf16", default=True, help="Use BF16 training")
@click.option("--config", "-c", type=click.Path(exists=True), help="Config file path")
@click.option("--resume", type=click.Path(exists=True), help="Resume from checkpoint")
@click.pass_context
def sft(
    ctx: click.Context,
    model_path: str,
    data_path: str,
    output: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    lora: bool,
    lora_r: int,
    lora_alpha: int,
    gradient_accumulation: int,
    warmup_ratio: float,
    fp16: bool,
    bf16: bool,
    config: Optional[str],
    resume: Optional[str],
) -> None:
    """Run Supervised Fine-Tuning (SFT) on a model.

    MODEL_PATH: HuggingFace model ID or local path
    DATA_PATH: Path to training data (JSON, JSONL, or directory)

    Example:
        largeforge train sft meta-llama/Llama-2-7b-hf data.jsonl -o ./output
    """
    verbose = ctx.obj.get("verbose", False)

    click.echo(f"Starting SFT training...")
    click.echo(f"Model: {model_path}")
    click.echo(f"Data: {data_path}")
    click.echo(f"Output: {output}")

    # Load config if provided
    training_config = None
    if config:
        from largeforge.utils import load_yaml
        from largeforge.config import SFTConfig
        cfg_data = load_yaml(config)
        training_config = SFTConfig(**cfg_data)
        click.echo(f"Loaded config from {config}")
    else:
        from largeforge.config import SFTConfig, LoRAConfig

        lora_config = None
        if lora:
            lora_config = LoRAConfig(r=lora_r, lora_alpha=lora_alpha)

        training_config = SFTConfig(
            output_dir=output,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            max_seq_length=max_length,
            gradient_accumulation_steps=gradient_accumulation,
            warmup_ratio=warmup_ratio,
            fp16=fp16,
            bf16=bf16 and not fp16,
            lora=lora_config,
        )

    # Create output directory
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_path / "training_config.yaml"
    training_config.to_yaml(config_path)
    click.echo(f"Saved config to {config_path}")

    # Load data
    click.echo("Loading training data...")
    from largeforge.data import load_dataset

    data = load_dataset(data_path)
    click.echo(f"Loaded {len(data)} training examples")

    # Initialize trainer
    click.echo("Initializing trainer...")
    from largeforge.training import SFTTrainer

    trainer = SFTTrainer(
        model_path=model_path,
        config=training_config,
    )

    # Add callbacks
    from largeforge.training.callbacks import LoggingCallback, CheckpointCallback

    trainer.add_callback(LoggingCallback(
        log_dir=str(output_path / "logs"),
        log_to_file=True,
    ))
    trainer.add_callback(CheckpointCallback(
        save_dir=str(output_path / "checkpoints"),
        save_every_n_steps=500,
    ))

    # Train
    click.echo("Starting training...")
    try:
        trainer.train(
            train_data=data,
            resume_from=resume,
        )
        click.echo(f"Training complete! Model saved to {output}")
    except KeyboardInterrupt:
        click.echo("\nTraining interrupted by user")
        trainer.save_checkpoint(output_path / "interrupted_checkpoint")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise click.ClickException(str(e))


@click.command()
@click.argument("model_path", type=str)
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), required=True, help="Output directory")
@click.option("--epochs", "-e", type=int, default=1, help="Number of training epochs")
@click.option("--batch-size", "-b", type=int, default=2, help="Training batch size")
@click.option("--learning-rate", "-lr", type=float, default=5e-6, help="Learning rate")
@click.option("--beta", type=float, default=0.1, help="DPO beta parameter")
@click.option("--loss-type", type=click.Choice(["sigmoid", "hinge", "ipo", "kto_pair"]),
              default="sigmoid", help="DPO loss type")
@click.option("--max-length", type=int, default=1024, help="Maximum sequence length")
@click.option("--max-prompt-length", type=int, default=512, help="Maximum prompt length")
@click.option("--lora/--no-lora", default=True, help="Use LoRA for training")
@click.option("--lora-r", type=int, default=16, help="LoRA rank")
@click.option("--ref-model", type=str, help="Reference model path (optional)")
@click.option("--config", "-c", type=click.Path(exists=True), help="Config file path")
@click.pass_context
def dpo(
    ctx: click.Context,
    model_path: str,
    data_path: str,
    output: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    beta: float,
    loss_type: str,
    max_length: int,
    max_prompt_length: int,
    lora: bool,
    lora_r: int,
    ref_model: Optional[str],
    config: Optional[str],
) -> None:
    """Run Direct Preference Optimization (DPO) training.

    MODEL_PATH: HuggingFace model ID or local path (policy model)
    DATA_PATH: Path to preference data with chosen/rejected pairs

    Example:
        largeforge train dpo ./sft_model preferences.jsonl -o ./dpo_output
    """
    verbose = ctx.obj.get("verbose", False)

    click.echo(f"Starting DPO training...")
    click.echo(f"Policy Model: {model_path}")
    click.echo(f"Data: {data_path}")
    click.echo(f"Output: {output}")
    click.echo(f"Loss Type: {loss_type}")
    click.echo(f"Beta: {beta}")

    # Load config if provided
    if config:
        from largeforge.utils import load_yaml
        from largeforge.config import DPOConfig
        cfg_data = load_yaml(config)
        training_config = DPOConfig(**cfg_data)
        click.echo(f"Loaded config from {config}")
    else:
        from largeforge.config import DPOConfig, LoRAConfig

        lora_config = None
        if lora:
            lora_config = LoRAConfig(r=lora_r)

        training_config = DPOConfig(
            output_dir=output,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            beta=beta,
            loss_type=loss_type,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            lora=lora_config,
        )

    # Create output directory
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_path / "dpo_config.yaml"
    training_config.to_yaml(config_path)

    # Load data
    click.echo("Loading preference data...")
    from largeforge.data import load_dataset

    data = load_dataset(data_path)
    click.echo(f"Loaded {len(data)} preference pairs")

    # Initialize trainer
    click.echo("Initializing DPO trainer...")
    from largeforge.training import DPOTrainer

    trainer = DPOTrainer(
        model_path=model_path,
        ref_model_path=ref_model,
        config=training_config,
    )

    # Add callbacks
    from largeforge.training.callbacks import LoggingCallback

    trainer.add_callback(LoggingCallback(
        log_dir=str(output_path / "logs"),
        log_to_file=True,
    ))

    # Train
    click.echo("Starting DPO training...")
    try:
        trainer.train(train_data=data)
        click.echo(f"DPO training complete! Model saved to {output}")
    except KeyboardInterrupt:
        click.echo("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"DPO training failed: {e}")
        raise click.ClickException(str(e))
