"""Supervised Fine-Tuning (SFT) trainer."""

import torch
from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from llm.data import load_sft_dataset
from llm.training.config import SFTConfig


def get_quantization_config(config: SFTConfig) -> BitsAndBytesConfig | None:
    """Get BitsAndBytes quantization config."""
    if config.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif config.load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def get_peft_config(config: SFTConfig) -> PeftLoraConfig | None:
    """Get PEFT LoRA config."""
    if not config.use_lora:
        return None

    return PeftLoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
        task_type=config.lora.task_type,
    )


def train_sft(config: SFTConfig):
    """
    Run supervised fine-tuning.

    Args:
        config: SFT configuration
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization
    quant_config = get_quantization_config(config)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quant_config,
        device_map=config.device_map,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        trust_remote_code=True,
    )

    # Prepare model for training
    if config.use_lora and quant_config:
        model = prepare_model_for_kbit_training(model)

    # Load dataset
    dataset = load_sft_dataset(
        config.dataset_path,
        format=config.dataset_format,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        fp16=config.fp16,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        optim="paged_adamw_8bit" if quant_config else "adamw_torch",
        report_to="none",
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=get_peft_config(config),
        max_seq_length=config.max_seq_length,
    )

    # Train
    trainer.train()

    # Save
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)

    return trainer
