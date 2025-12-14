"""Continued pretraining trainer."""

import torch
from peft import LoraConfig as PeftLoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from llm.data import load_pretrain_dataset
from llm.training.config import PretrainConfig


def train_pretrain(config: PretrainConfig):
    """
    Run continued pretraining on domain-specific data.

    Args:
        config: Pretraining configuration
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map=config.device_map,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        trust_remote_code=True,
    )

    # Optional LoRA for memory efficiency
    if config.use_lora:
        lora_config = PeftLoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
            target_modules=config.lora.target_modules,
            bias=config.lora.bias,
            task_type=config.lora.task_type,
        )
        model = get_peft_model(model, lora_config)

    # Load dataset
    dataset = load_pretrain_dataset(
        config.dataset_path,
        format=config.dataset_format,
    )

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.max_seq_length,
            padding=False,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
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
        gradient_checkpointing=True,  # Memory efficiency for pretraining
        report_to="none",
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Train
    trainer.train()

    # Save
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)

    return trainer
