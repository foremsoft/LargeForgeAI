"""Knowledge distillation from teacher to student model."""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)

from llm.training.config import LoraConfig


@dataclass
class DistillConfig:
    """Configuration for knowledge distillation."""

    teacher_model: str = "Qwen/Qwen2.5-72B-Instruct"
    student_model: str = "Qwen/Qwen2.5-7B"
    output_dir: str = "./output/distill"

    # Distillation params
    temperature: float = 2.0
    alpha: float = 0.5  # Weight for distillation loss vs hard label loss

    # Training params
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_seq_length: int = 2048

    # LoRA for student
    use_lora: bool = True
    lora: LoraConfig = field(default_factory=LoraConfig)
    load_in_4bit: bool = True

    # Teacher settings
    teacher_device: str = "auto"
    student_device: str = "auto"


class Distiller:
    """Knowledge distillation trainer."""

    def __init__(self, config: DistillConfig):
        self.config = config
        self.teacher = None
        self.student = None
        self.tokenizer = None

    def load_models(self):
        """Load teacher and student models."""
        # Load tokenizer (use student tokenizer, assuming same family)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.student_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load teacher (inference mode, no gradients)
        print(f"Loading teacher: {self.config.teacher_model}")
        self.teacher = AutoModelForCausalLM.from_pretrained(
            self.config.teacher_model,
            device_map=self.config.teacher_device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Load student
        print(f"Loading student: {self.config.student_model}")
        quant_config = None
        if self.config.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        self.student = AutoModelForCausalLM.from_pretrained(
            self.config.student_model,
            quantization_config=quant_config,
            device_map=self.config.student_device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        if self.config.use_lora:
            if quant_config:
                self.student = prepare_model_for_kbit_training(self.student)

            lora_config = PeftLoraConfig(
                r=self.config.lora.r,
                lora_alpha=self.config.lora.lora_alpha,
                lora_dropout=self.config.lora.lora_dropout,
                target_modules=self.config.lora.target_modules,
                bias=self.config.lora.bias,
                task_type=self.config.lora.task_type,
            )
            self.student = get_peft_model(self.student, lora_config)

    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
        alpha: float,
    ) -> torch.Tensor:
        """
        Compute distillation loss combining soft and hard targets.

        Args:
            student_logits: Student model output logits
            teacher_logits: Teacher model output logits
            labels: Ground truth labels
            temperature: Softmax temperature for soft targets
            alpha: Weight for distillation loss (1-alpha for hard loss)
        """
        # Soft target loss (KL divergence)
        soft_student = F.log_softmax(student_logits / temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
        distill_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (
            temperature**2
        )

        # Hard target loss (cross entropy)
        hard_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        # Combined loss
        return alpha * distill_loss + (1 - alpha) * hard_loss

    def train(self, dataset: Dataset):
        """
        Run distillation training.

        Args:
            dataset: Dataset with 'input_ids' and 'labels' columns
        """
        if self.teacher is None:
            self.load_models()

        # Prepare dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.config.learning_rate,
        )

        num_training_steps = (
            len(dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        )
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Training loop
        self.student.train()
        global_step = 0

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

            for step, batch in enumerate(progress):
                input_ids = batch["input_ids"].to(self.student.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.student.device)
                labels = batch.get("labels", input_ids).to(self.student.device)

                # Get teacher logits
                with torch.no_grad():
                    teacher_outputs = self.teacher(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

                # Get student logits
                student_outputs = self.student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                # Compute loss
                loss = self.distillation_loss(
                    student_outputs.logits,
                    teacher_outputs.logits,
                    labels,
                    self.config.temperature,
                    self.config.alpha,
                )

                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                epoch_loss += loss.item()

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                progress.set_postfix({"loss": loss.item() * self.config.gradient_accumulation_steps})

            print(f"Epoch {epoch + 1} - Avg Loss: {epoch_loss / len(dataloader):.4f}")

        # Save
        self.student.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)

        return self.student


def train_distill(config: DistillConfig, dataset: Dataset):
    """Convenience function to run distillation."""
    distiller = Distiller(config)
    return distiller.train(dataset)
