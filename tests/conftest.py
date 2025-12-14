"""Pytest fixtures and configuration for LargeForgeAI tests."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock


@pytest.fixture
def sample_alpaca_data():
    """Sample Alpaca format training data."""
    return [
        {
            "instruction": "Summarize the following text",
            "input": "Machine learning is a subset of artificial intelligence.",
            "output": "ML is a branch of AI focused on learning from data.",
        },
        {
            "instruction": "Translate to French",
            "input": "Hello, how are you?",
            "output": "Bonjour, comment allez-vous?",
        },
        {
            "instruction": "What is the capital of France?",
            "input": "",
            "output": "The capital of France is Paris.",
        },
    ]


@pytest.fixture
def sample_sharegpt_data():
    """Sample ShareGPT format conversation data."""
    return [
        {
            "conversations": [
                {"from": "system", "value": "You are a helpful assistant."},
                {"from": "human", "value": "What is Python?"},
                {"from": "gpt", "value": "Python is a programming language."},
            ]
        },
        {
            "conversations": [
                {"from": "human", "value": "Hello!"},
                {"from": "gpt", "value": "Hi there! How can I help you?"},
                {"from": "human", "value": "What's the weather like?"},
                {"from": "gpt", "value": "I don't have weather data access."},
            ]
        },
    ]


@pytest.fixture
def sample_dpo_data():
    """Sample DPO format preference data."""
    return [
        {
            "prompt": "Write a greeting message",
            "chosen": "Hello! Welcome to our service. How may I assist you today?",
            "rejected": "Hi",
        },
        {
            "prompt": "Explain machine learning",
            "chosen": "Machine learning is a branch of AI that enables systems to learn from data.",
            "rejected": "It's computers learning stuff.",
        },
        {
            "prompt": "What is 2+2?",
            "chosen": "2 + 2 equals 4.",
            "rejected": "The answer is four, obviously.",
        },
    ]


@pytest.fixture
def tmp_model_dir(tmp_path):
    """Temporary directory for model outputs."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.bos_token_id = 1
    tokenizer.model_max_length = 2048

    def mock_encode(text, **kwargs):
        return list(range(len(text.split())))

    def mock_decode(ids, **kwargs):
        return " ".join(["token"] * len(ids))

    tokenizer.encode = mock_encode
    tokenizer.decode = mock_decode
    tokenizer.__call__ = lambda text, **kwargs: {
        "input_ids": mock_encode(text),
        "attention_mask": [1] * len(mock_encode(text)),
    }

    return tokenizer


@pytest.fixture
def small_model_name():
    """Name of a small model for testing."""
    return "gpt2"


@pytest.fixture
def config_dir(tmp_path):
    """Temporary directory for config files."""
    config_path = tmp_path / "configs"
    config_path.mkdir()
    return config_path


# Markers for special test types
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU")
    config.addinivalue_line("markers", "slow: marks tests as slow running")
    config.addinivalue_line("markers", "integration: marks integration tests")
