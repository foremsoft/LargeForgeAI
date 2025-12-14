"""Data loading utilities for various formats."""

import json
from pathlib import Path
from typing import Literal

from datasets import Dataset, load_dataset


def load_sft_dataset(
    path: str | Path,
    format: Literal["jsonl", "parquet", "huggingface"] = "jsonl",
    prompt_field: str = "prompt",
    response_field: str = "response",
    system_field: str | None = "system",
) -> Dataset:
    """
    Load a dataset for supervised fine-tuning.

    Args:
        path: Path to local file or HuggingFace dataset name
        format: Data format
        prompt_field: Field name for user prompts
        response_field: Field name for assistant responses
        system_field: Optional field name for system prompts

    Returns:
        HuggingFace Dataset with standardized columns
    """
    if format == "huggingface":
        dataset = load_dataset(str(path), split="train")
    elif format == "parquet":
        dataset = Dataset.from_parquet(str(path))
    else:
        dataset = Dataset.from_json(str(path))

    # Standardize column names
    def format_example(example):
        messages = []
        if system_field and system_field in example and example[system_field]:
            messages.append({"role": "system", "content": example[system_field]})
        messages.append({"role": "user", "content": example[prompt_field]})
        messages.append({"role": "assistant", "content": example[response_field]})
        return {"messages": messages}

    return dataset.map(format_example, remove_columns=dataset.column_names)


def load_dpo_dataset(
    path: str | Path,
    format: Literal["jsonl", "parquet", "huggingface"] = "jsonl",
    prompt_field: str = "prompt",
    chosen_field: str = "chosen",
    rejected_field: str = "rejected",
) -> Dataset:
    """
    Load a dataset for DPO/preference training.

    Args:
        path: Path to local file or HuggingFace dataset name
        format: Data format
        prompt_field: Field name for prompts
        chosen_field: Field name for preferred responses
        rejected_field: Field name for rejected responses

    Returns:
        HuggingFace Dataset with prompt, chosen, rejected columns
    """
    if format == "huggingface":
        dataset = load_dataset(str(path), split="train")
    elif format == "parquet":
        dataset = Dataset.from_parquet(str(path))
    else:
        dataset = Dataset.from_json(str(path))

    # Standardize column names if needed
    rename_map = {}
    if prompt_field != "prompt":
        rename_map[prompt_field] = "prompt"
    if chosen_field != "chosen":
        rename_map[chosen_field] = "chosen"
    if rejected_field != "rejected":
        rename_map[rejected_field] = "rejected"

    if rename_map:
        dataset = dataset.rename_columns(rename_map)

    return dataset


def load_pretrain_dataset(
    path: str | Path,
    format: Literal["jsonl", "parquet", "huggingface", "text"] = "jsonl",
    text_field: str = "text",
) -> Dataset:
    """
    Load a dataset for continued pretraining.

    Args:
        path: Path to local file or HuggingFace dataset name
        format: Data format
        text_field: Field name for text content

    Returns:
        HuggingFace Dataset with text column
    """
    if format == "huggingface":
        dataset = load_dataset(str(path), split="train")
    elif format == "text":
        with open(path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        dataset = Dataset.from_dict({"text": texts})
    elif format == "parquet":
        dataset = Dataset.from_parquet(str(path))
    else:
        dataset = Dataset.from_json(str(path))

    if text_field != "text" and text_field in dataset.column_names:
        dataset = dataset.rename_column(text_field, "text")

    return dataset


def prepare_chat_format(
    dataset: Dataset,
    tokenizer,
    max_length: int = 2048,
) -> Dataset:
    """
    Prepare dataset for chat-style training with proper tokenization.

    Args:
        dataset: Dataset with 'messages' column
        tokenizer: HuggingFace tokenizer with chat template
        max_length: Maximum sequence length

    Returns:
        Tokenized dataset ready for training
    """

    def tokenize(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    return dataset.map(tokenize, remove_columns=["messages"])
