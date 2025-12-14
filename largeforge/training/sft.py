"""Supervised Fine-Tuning trainer for LargeForgeAI."""

from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset

from largeforge.config import SFTConfig
from largeforge.training.base import BaseTrainer, TrainingCallback
from largeforge.utils import get_logger

logger = get_logger(__name__)


class SFTTrainer(BaseTrainer):
    """Trainer for Supervised Fine-Tuning."""

    def __init__(
        self,
        model,
        config: SFTConfig,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer=None,
        callbacks: Optional[List[TrainingCallback]] = None,
        data_collator=None,
    ):
        """
        Initialize the SFT trainer.

        Args:
            model: The model to train
            config: SFT training configuration
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            tokenizer: Tokenizer for the model
            callbacks: List of training callbacks
            data_collator: Optional data collator
        """
        super().__init__(
            model=model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks,
        )
        self.data_collator = data_collator or self._default_data_collator

    def _default_data_collator(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Default data collator for SFT."""
        batch = {}

        # Get all keys from first feature
        keys = features[0].keys()

        for key in keys:
            values = [f[key] for f in features]

            # Convert to tensors and pad
            if isinstance(values[0], list):
                # Pad sequences
                max_len = max(len(v) for v in values)
                padded = []
                for v in values:
                    if len(v) < max_len:
                        # Pad with tokenizer's pad_token_id or 0
                        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
                        v = v + [pad_id] * (max_len - len(v))
                    padded.append(v)
                batch[key] = torch.tensor(padded)
            elif isinstance(values[0], torch.Tensor):
                batch[key] = torch.stack(values)
            else:
                batch[key] = values

        return batch

    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create the optimizer for SFT training."""
        # Separate parameters that should and shouldn't have weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if "bias" in name or "LayerNorm" in name or "layer_norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )

        return optimizer

    def create_scheduler(
        self, optimizer: torch.optim.Optimizer, num_training_steps: int
    ):
        """Create the learning rate scheduler."""
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        if self.config.lr_scheduler_type == "linear":
            from torch.optim.lr_scheduler import LambdaLR

            def lr_lambda(current_step: int):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                return max(
                    0.0,
                    float(num_training_steps - current_step)
                    / float(max(1, num_training_steps - num_warmup_steps)),
                )

            return LambdaLR(optimizer, lr_lambda)

        elif self.config.lr_scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR

            return CosineAnnealingLR(
                optimizer, T_max=num_training_steps - num_warmup_steps
            )

        elif self.config.lr_scheduler_type == "constant":
            from torch.optim.lr_scheduler import LambdaLR

            return LambdaLR(optimizer, lambda _: 1.0)

        else:
            logger.warning(f"Unknown scheduler type: {self.config.lr_scheduler_type}")
            return None

    def compute_loss(self, model, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Compute the SFT loss (causal language modeling loss).

        Args:
            model: The model
            inputs: Batch inputs

        Returns:
            Loss tensor
        """
        # Forward pass
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            labels=inputs.get("labels", inputs["input_ids"]),
        )

        return outputs.loss

    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation on the eval dataset.

        Returns:
            Dictionary of evaluation metrics
        """
        if self.eval_dataset is None:
            return {}

        logger.info("Running evaluation...")
        self.model.eval()
        eval_dataloader = self.get_eval_dataloader()

        total_loss = 0.0
        total_steps = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                batch = self._prepare_inputs(batch)
                loss = self.compute_loss(self.model, batch)
                total_loss += loss.item()
                total_steps += 1

        avg_loss = total_loss / max(total_steps, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        metrics = {
            "eval_loss": avg_loss,
            "eval_perplexity": perplexity,
        }

        logger.info(f"Evaluation results: {metrics}")
        self.model.train()

        return metrics

    def get_train_dataloader(self):
        """Create the training dataloader with data collator."""
        if self.train_dataset is None:
            raise ValueError("No training dataset provided")

        from torch.utils.data import DataLoader

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.dataloader_pin_memory,
            drop_last=self.config.dataloader_drop_last,
            collate_fn=self.data_collator,
        )

    def get_eval_dataloader(self):
        """Create the evaluation dataloader with data collator."""
        if self.eval_dataset is None:
            return None

        from torch.utils.data import DataLoader

        return DataLoader(
            self.eval_dataset,
            batch_size=self.config.per_device_eval_batch_size,
            shuffle=False,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.dataloader_pin_memory,
            collate_fn=self.data_collator,
        )


def train_sft(
    model,
    tokenizer,
    train_data: List[Dict[str, Any]],
    config: Optional[SFTConfig] = None,
    eval_data: Optional[List[Dict[str, Any]]] = None,
    callbacks: Optional[List[TrainingCallback]] = None,
):
    """
    Convenience function to train a model with SFT.

    Args:
        model: The model to train
        tokenizer: Tokenizer for the model
        train_data: Training data (list of dicts with input_ids, attention_mask, labels)
        config: Optional SFT configuration
        eval_data: Optional evaluation data
        callbacks: Optional list of callbacks

    Returns:
        Tuple of (trained_model, training_state)

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model, state = train_sft(model, tokenizer, train_data)
    """
    if config is None:
        config = SFTConfig(output_dir="./sft_output")

    # Create dataset from data
    from torch.utils.data import Dataset

    class ListDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    train_dataset = ListDataset(train_data)
    eval_dataset = ListDataset(eval_data) if eval_data else None

    trainer = SFTTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    state = trainer.train()
    return model, state
