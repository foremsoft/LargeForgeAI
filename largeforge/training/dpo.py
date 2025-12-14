"""Direct Preference Optimization trainer for LargeForgeAI."""

import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from largeforge.config import DPOConfig
from largeforge.training.base import BaseTrainer, TrainingCallback
from largeforge.utils import get_logger

logger = get_logger(__name__)


class DPOTrainer(BaseTrainer):
    """Trainer for Direct Preference Optimization (DPO)."""

    def __init__(
        self,
        model,
        config: DPOConfig,
        ref_model=None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer=None,
        callbacks: Optional[List[TrainingCallback]] = None,
    ):
        """
        Initialize the DPO trainer.

        Args:
            model: The policy model to train
            config: DPO training configuration
            ref_model: Reference model (frozen). If None, a copy of model is used.
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            tokenizer: Tokenizer for the model
            callbacks: List of training callbacks
        """
        super().__init__(
            model=model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks,
        )

        # Create or use reference model
        if ref_model is None:
            logger.info("Creating reference model from policy model")
            self.ref_model = copy.deepcopy(model)
        else:
            self.ref_model = ref_model

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.beta = config.beta
        self.loss_type = config.loss_type
        self.label_smoothing = config.label_smoothing

    def _get_log_probs(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probabilities of the labels under the model.

        Args:
            model: The model
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Labels for computing log probs

        Returns:
            Log probabilities tensor
        """
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log probs for the labels
        per_token_log_probs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask padding tokens
        label_mask = (shift_labels != -100).float()
        per_token_log_probs = per_token_log_probs * label_mask

        # Sum log probs for each sequence
        return per_token_log_probs.sum(dim=-1)

    def compute_loss(self, model, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Compute the DPO loss.

        Args:
            model: The policy model
            inputs: Batch inputs with chosen and rejected sequences

        Returns:
            DPO loss tensor
        """
        # Get chosen and rejected inputs
        chosen_input_ids = inputs["chosen_input_ids"]
        chosen_attention_mask = inputs["chosen_attention_mask"]
        rejected_input_ids = inputs["rejected_input_ids"]
        rejected_attention_mask = inputs["rejected_attention_mask"]

        # Create labels (same as input_ids for causal LM)
        chosen_labels = chosen_input_ids.clone()
        rejected_labels = rejected_input_ids.clone()

        # Mask prompt tokens in labels if prompt_input_ids provided
        if "prompt_input_ids" in inputs:
            prompt_len = inputs["prompt_input_ids"].shape[1]
            chosen_labels[:, :prompt_len] = -100
            rejected_labels[:, :prompt_len] = -100

        # Compute log probs for policy model
        policy_chosen_log_probs = self._get_log_probs(
            model, chosen_input_ids, chosen_attention_mask, chosen_labels
        )
        policy_rejected_log_probs = self._get_log_probs(
            model, rejected_input_ids, rejected_attention_mask, rejected_labels
        )

        # Compute log probs for reference model
        with torch.no_grad():
            ref_chosen_log_probs = self._get_log_probs(
                self.ref_model, chosen_input_ids, chosen_attention_mask, chosen_labels
            )
            ref_rejected_log_probs = self._get_log_probs(
                self.ref_model, rejected_input_ids, rejected_attention_mask, rejected_labels
            )

        # Compute log ratios
        policy_log_ratio = policy_chosen_log_probs - policy_rejected_log_probs
        ref_log_ratio = ref_chosen_log_probs - ref_rejected_log_probs

        # Compute DPO loss based on loss type
        logits = self.beta * (policy_log_ratio - ref_log_ratio)

        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(logits)
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - logits)
        elif self.loss_type == "ipo":
            # IPO loss: (logits - 1/(2*beta))^2
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "kto_pair":
            # KTO pair loss
            chosen_kl = (policy_chosen_log_probs - ref_chosen_log_probs).detach()
            rejected_kl = (policy_rejected_log_probs - ref_rejected_log_probs).detach()
            kl = 0.5 * (chosen_kl + rejected_kl)

            losses = -F.logsigmoid(logits) + 0.1 * kl
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Apply label smoothing
        if self.label_smoothing > 0:
            losses = (1 - self.label_smoothing) * losses + self.label_smoothing * 0.5

        return losses.mean()

    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create the optimizer for DPO training."""
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

        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )

    def create_scheduler(
        self, optimizer: torch.optim.Optimizer, num_training_steps: int
    ):
        """Create the learning rate scheduler."""
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)

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

    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation on the eval dataset.

        Returns:
            Dictionary of evaluation metrics including:
            - eval_loss: Average DPO loss
            - eval_accuracy: Preference accuracy (chosen > rejected)
            - eval_chosen_reward: Average reward for chosen responses
            - eval_rejected_reward: Average reward for rejected responses
        """
        if self.eval_dataset is None:
            return {}

        logger.info("Running DPO evaluation...")
        self.model.eval()
        eval_dataloader = self.get_eval_dataloader()

        total_loss = 0.0
        total_chosen_rewards = 0.0
        total_rejected_rewards = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                batch = self._prepare_inputs(batch)

                # Compute loss
                loss = self.compute_loss(self.model, batch)
                total_loss += loss.item()

                # Compute rewards for metrics
                chosen_input_ids = batch["chosen_input_ids"]
                rejected_input_ids = batch["rejected_input_ids"]

                chosen_labels = chosen_input_ids.clone()
                rejected_labels = rejected_input_ids.clone()

                policy_chosen = self._get_log_probs(
                    self.model,
                    chosen_input_ids,
                    batch["chosen_attention_mask"],
                    chosen_labels,
                )
                policy_rejected = self._get_log_probs(
                    self.model,
                    rejected_input_ids,
                    batch["rejected_attention_mask"],
                    rejected_labels,
                )

                ref_chosen = self._get_log_probs(
                    self.ref_model,
                    chosen_input_ids,
                    batch["chosen_attention_mask"],
                    chosen_labels,
                )
                ref_rejected = self._get_log_probs(
                    self.ref_model,
                    rejected_input_ids,
                    batch["rejected_attention_mask"],
                    rejected_labels,
                )

                # Implicit rewards
                chosen_rewards = self.beta * (policy_chosen - ref_chosen)
                rejected_rewards = self.beta * (policy_rejected - ref_rejected)

                total_chosen_rewards += chosen_rewards.sum().item()
                total_rejected_rewards += rejected_rewards.sum().item()

                # Accuracy: chosen reward > rejected reward
                correct += (chosen_rewards > rejected_rewards).sum().item()
                total += chosen_rewards.shape[0]

        num_batches = len(eval_dataloader)
        metrics = {
            "eval_loss": total_loss / num_batches,
            "eval_accuracy": correct / total if total > 0 else 0.0,
            "eval_chosen_reward": total_chosen_rewards / total if total > 0 else 0.0,
            "eval_rejected_reward": total_rejected_rewards / total if total > 0 else 0.0,
            "eval_reward_margin": (total_chosen_rewards - total_rejected_rewards) / total if total > 0 else 0.0,
        }

        logger.info(f"DPO Evaluation results: {metrics}")
        self.model.train()

        return metrics

    def _default_data_collator(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Data collator for DPO data."""
        batch = {}
        keys = features[0].keys()

        for key in keys:
            values = [f[key] for f in features]

            if isinstance(values[0], list):
                max_len = max(len(v) for v in values)
                padded = []
                for v in values:
                    pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
                    v = v + [pad_id] * (max_len - len(v))
                    padded.append(v)
                batch[key] = torch.tensor(padded)
            elif isinstance(values[0], torch.Tensor):
                batch[key] = torch.stack(values)
            elif isinstance(values[0], str):
                batch[key] = values
            else:
                batch[key] = torch.tensor(values)

        return batch

    def get_train_dataloader(self):
        """Create the training dataloader."""
        if self.train_dataset is None:
            raise ValueError("No training dataset provided")

        from torch.utils.data import DataLoader

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.dataloader_pin_memory,
            drop_last=True,  # Important for DPO to maintain batch consistency
            collate_fn=self._default_data_collator,
        )

    def get_eval_dataloader(self):
        """Create the evaluation dataloader."""
        if self.eval_dataset is None:
            return None

        from torch.utils.data import DataLoader

        return DataLoader(
            self.eval_dataset,
            batch_size=self.config.per_device_eval_batch_size,
            shuffle=False,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.dataloader_pin_memory,
            collate_fn=self._default_data_collator,
        )


def train_dpo(
    model,
    tokenizer,
    train_data: List[Dict[str, Any]],
    config: Optional[DPOConfig] = None,
    ref_model=None,
    eval_data: Optional[List[Dict[str, Any]]] = None,
    callbacks: Optional[List[TrainingCallback]] = None,
):
    """
    Convenience function to train a model with DPO.

    Args:
        model: The policy model to train
        tokenizer: Tokenizer for the model
        train_data: Training preference data
        config: Optional DPO configuration
        ref_model: Optional reference model
        eval_data: Optional evaluation data
        callbacks: Optional list of callbacks

    Returns:
        Tuple of (trained_model, training_state)
    """
    if config is None:
        config = DPOConfig(output_dir="./dpo_output")

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

    trainer = DPOTrainer(
        model=model,
        config=config,
        ref_model=ref_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    state = trainer.train()
    return model, state
