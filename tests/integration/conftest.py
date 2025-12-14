"""Pytest fixtures for integration tests."""

import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch


@pytest.fixture
def integration_data_dir(tmp_path):
    """Create a temporary directory with test datasets."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create alpaca format file
    alpaca_data = [
        {"instruction": "Summarize this text", "input": "AI is transforming industries.", "output": "AI revolutionizes industries."},
        {"instruction": "Translate to Spanish", "input": "Hello world", "output": "Hola mundo"},
        {"instruction": "What is Python?", "input": "", "output": "Python is a programming language."},
    ]
    alpaca_file = data_dir / "alpaca.jsonl"
    with open(alpaca_file, "w") as f:
        for item in alpaca_data:
            f.write(json.dumps(item) + "\n")

    # Create sharegpt format file
    sharegpt_data = [
        {
            "conversations": [
                {"from": "human", "value": "What is machine learning?"},
                {"from": "gpt", "value": "Machine learning is a subset of AI."},
            ]
        },
        {
            "conversations": [
                {"from": "system", "value": "You are helpful."},
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi! How can I help?"},
            ]
        },
    ]
    sharegpt_file = data_dir / "sharegpt.json"
    with open(sharegpt_file, "w") as f:
        json.dump(sharegpt_data, f)

    # Create DPO format file
    dpo_data = [
        {"prompt": "Write a greeting", "chosen": "Hello! Welcome!", "rejected": "Hi"},
        {"prompt": "Explain AI", "chosen": "AI is artificial intelligence.", "rejected": "Computers."},
    ]
    dpo_file = data_dir / "dpo.jsonl"
    with open(dpo_file, "w") as f:
        for item in dpo_data:
            f.write(json.dumps(item) + "\n")

    return data_dir


@pytest.fixture
def integration_model_dir(tmp_path):
    """Create a temporary directory for model outputs."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def integration_config_dir(tmp_path):
    """Create a temporary directory with config files."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    # Create training config
    train_config = {
        "model_name": "gpt2",
        "max_length": 512,
        "learning_rate": 2e-5,
        "epochs": 1,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "warmup_ratio": 0.1,
        "save_steps": 100,
    }
    train_file = config_dir / "train.json"
    with open(train_file, "w") as f:
        json.dump(train_config, f)

    # Create inference config
    inference_config = {
        "backend": "transformers",
        "dtype": "float32",
        "max_length": 256,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    inference_file = config_dir / "inference.json"
    with open(inference_file, "w") as f:
        json.dump(inference_config, f)

    return config_dir


@pytest.fixture
def mock_model():
    """Create a mock model for integration testing."""
    model = MagicMock()
    model.config = MagicMock()
    model.config.vocab_size = 50257
    model.config.hidden_size = 768
    model.config.num_attention_heads = 12
    model.config.num_hidden_layers = 12
    model.parameters.return_value = iter([MagicMock()])
    model.named_parameters.return_value = iter([("weight", MagicMock())])
    model.to.return_value = model
    model.eval.return_value = model
    model.train.return_value = model
    return model


@pytest.fixture
def mock_pipeline():
    """Create a mock text generation pipeline."""
    pipeline = MagicMock()
    pipeline.return_value = [{"generated_text": "This is a test response."}]
    return pipeline
