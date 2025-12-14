"""Direct Preference Optimization (DPO) and ORPO training."""

import torch
from peft import LoraConfig as PeftLoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import DPOConfig as TRLDPOConfig, DPOTrainer, ORPOConfig, ORPOTrainer

from llm.data import load_dpo_dataset
from llm.training.config import DPOConfig


def get_quantization_config(load_in_4bit: bool) -> BitsAndBytesConfig | None:
    """Get BitsAndBytes quantization config."""
    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    return None


def train_dpo(config: DPOConfig):
    """
    Run DPO (Direct Preference Optimization) training.

    DPO directly optimizes for human preferences without a separate reward model.

    Args:
        config: DPO configuration
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    quant_config = get_quantization_config(config.load_in_4bit)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quant_config,
        device_map=config.device_map,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        trust_remote_code=True,
    )

    # Prepare for LoRA training
    peft_config = None
    if config.use_lora:
        if quant_config:
            model = prepare_model_for_kbit_training(model)

        peft_config = PeftLoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
            target_modules=config.lora.target_modules,
            bias=config.lora.bias,
            task_type=config.lora.task_type,
        )

    # Load reference model (for DPO loss calculation)
    ref_model = None
    if config.ref_model:
        ref_model = AutoModelForCausalLM.from_pretrained(
            config.ref_model,
            quantization_config=quant_config,
            device_map=config.device_map,
            torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            trust_remote_code=True,
        )

    # Load dataset
    dataset = load_dpo_dataset(
        config.dataset_path,
        format=config.dataset_format,
    )

    # DPO training config
    training_args = TRLDPOConfig(
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
        beta=config.beta,
        loss_type=config.loss_type,
        max_length=config.max_seq_length,
        max_prompt_length=config.max_seq_length // 2,
        report_to="none",
    )

    # Create DPO trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Train
    trainer.train()

    # Save
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)

    return trainer


def train_orpo(
    model_name: str,
    dataset_path: str,
    output_dir: str = "./output/orpo",
    num_epochs: int = 1,
    batch_size: int = 1,
    learning_rate: float = 8e-6,
    beta: float = 0.1,
    max_length: int = 2048,
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    load_in_4bit: bool = True,
):
    """
    Run ORPO (Odds Ratio Preference Optimization) training.

    ORPO combines SFT and preference optimization in a single stage,
    eliminating the need for a reference model.

    Args:
        model_name: Base model name or path
        dataset_path: Path to preference dataset
        output_dir: Output directory for trained model
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        learning_rate: Learning rate (ORPO typically uses lower LR)
        beta: ORPO beta parameter
        max_length: Maximum sequence length
        use_lora: Whether to use LoRA
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        load_in_4bit: Whether to use 4-bit quantization
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    quant_config = get_quantization_config(load_in_4bit)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # LoRA config
    peft_config = None
    if use_lora:
        if quant_config:
            model = prepare_model_for_kbit_training(model)

        peft_config = PeftLoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

    # Load dataset
    dataset = load_dpo_dataset(dataset_path)

    # ORPO config
    orpo_config = ORPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_length // 2,
        report_to="none",
    )

    # Create ORPO trainer
    trainer = ORPOTrainer(
        model=model,
        args=orpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Train
    trainer.train()

    # Save
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    return trainer
