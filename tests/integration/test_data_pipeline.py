"""Integration tests for the data processing pipeline."""

import pytest
import json
from pathlib import Path

from largeforge.data import (
    load_dataset,
    validate_dataset,
    detect_format,
)
from largeforge.data.converters import (
    alpaca_to_sharegpt,
    sharegpt_to_alpaca,
    to_chat_format,
)
from largeforge.utils import (
    load_json,
    save_json,
    load_jsonl,
    save_jsonl,
)


@pytest.mark.integration
class TestDataLoadingPipeline:
    """Test end-to-end data loading."""

    def test_load_jsonl_and_validate_alpaca(self, integration_data_dir):
        """Test loading JSONL and validating Alpaca format."""
        # Load data
        data_path = integration_data_dir / "alpaca.jsonl"
        data = load_dataset(str(data_path))

        assert len(data) == 3
        assert all("instruction" in d for d in data)
        assert all("output" in d for d in data)

        # Validate
        results = validate_dataset(data, format_type="alpaca")
        assert len(results) == 3
        assert all(r["valid"] for r in results)

    def test_load_json_and_validate_sharegpt(self, integration_data_dir):
        """Test loading JSON and validating ShareGPT format."""
        # Load data
        data_path = integration_data_dir / "sharegpt.json"
        data = load_dataset(str(data_path))

        assert len(data) == 2
        assert all("conversations" in d for d in data)

        # Validate
        results = validate_dataset(data, format_type="sharegpt")
        assert len(results) == 2
        assert all(r["valid"] for r in results)

    def test_load_and_validate_dpo(self, integration_data_dir):
        """Test loading and validating DPO format."""
        # Load data
        data_path = integration_data_dir / "dpo.jsonl"
        data = load_dataset(str(data_path))

        assert len(data) == 2
        assert all("prompt" in d for d in data)
        assert all("chosen" in d for d in data)
        assert all("rejected" in d for d in data)

        # Validate
        results = validate_dataset(data, format_type="dpo")
        assert len(results) == 2
        assert all(r["valid"] for r in results)

    def test_auto_detect_format(self, integration_data_dir):
        """Test automatic format detection."""
        # Alpaca format
        alpaca_data = load_dataset(str(integration_data_dir / "alpaca.jsonl"))
        assert detect_format(alpaca_data) == "alpaca"

        # ShareGPT format
        sharegpt_data = load_dataset(str(integration_data_dir / "sharegpt.json"))
        assert detect_format(sharegpt_data) == "sharegpt"

        # DPO format
        dpo_data = load_dataset(str(integration_data_dir / "dpo.jsonl"))
        assert detect_format(dpo_data) == "dpo"


@pytest.mark.integration
class TestDataConversionPipeline:
    """Test end-to-end data conversion."""

    def test_alpaca_to_sharegpt_pipeline(self, integration_data_dir, tmp_path):
        """Test converting Alpaca to ShareGPT format."""
        # Load original data
        alpaca_data = load_dataset(str(integration_data_dir / "alpaca.jsonl"))

        # Convert
        sharegpt_data = alpaca_to_sharegpt(alpaca_data)

        # Verify conversion
        assert len(sharegpt_data) == len(alpaca_data)
        for item in sharegpt_data:
            assert "conversations" in item
            assert len(item["conversations"]) >= 2

        # Save converted data
        output_path = tmp_path / "converted.json"
        save_json(sharegpt_data, str(output_path))

        # Reload and verify
        reloaded = load_json(str(output_path))
        assert len(reloaded) == len(sharegpt_data)

    def test_sharegpt_to_alpaca_pipeline(self, integration_data_dir, tmp_path):
        """Test converting ShareGPT to Alpaca format."""
        # Load original data
        sharegpt_data = load_dataset(str(integration_data_dir / "sharegpt.json"))

        # Convert
        alpaca_data = sharegpt_to_alpaca(sharegpt_data)

        # Verify conversion
        assert len(alpaca_data) > 0
        for item in alpaca_data:
            assert "instruction" in item
            assert "output" in item

        # Save converted data
        output_path = tmp_path / "converted.jsonl"
        save_jsonl(alpaca_data, str(output_path))

        # Reload and verify
        reloaded = load_jsonl(str(output_path))
        assert len(reloaded) == len(alpaca_data)

    def test_to_chat_format_pipeline(self, integration_data_dir):
        """Test converting to chat format."""
        # Load Alpaca data
        alpaca_data = load_dataset(str(integration_data_dir / "alpaca.jsonl"))

        # Convert to chat format
        chat_data = to_chat_format(alpaca_data, format="alpaca")

        # Verify conversion - returns list of message lists
        assert len(chat_data) == len(alpaca_data)
        for messages in chat_data:
            # Should have user and assistant messages
            roles = [m["role"] for m in messages]
            assert "user" in roles
            assert "assistant" in roles


@pytest.mark.integration
class TestDataSaveLoadCycle:
    """Test complete save/load cycles."""

    def test_json_save_load_cycle(self, tmp_path):
        """Test JSON save and load cycle."""
        test_data = [
            {"key": "value1", "nested": {"a": 1}},
            {"key": "value2", "nested": {"b": 2}},
        ]

        file_path = tmp_path / "test.json"
        save_json(test_data, str(file_path))

        loaded = load_json(str(file_path))
        assert loaded == test_data

    def test_jsonl_save_load_cycle(self, tmp_path):
        """Test JSONL save and load cycle."""
        test_data = [
            {"id": 1, "text": "First entry"},
            {"id": 2, "text": "Second entry"},
            {"id": 3, "text": "Third entry"},
        ]

        file_path = tmp_path / "test.jsonl"
        save_jsonl(test_data, str(file_path))

        loaded = load_jsonl(str(file_path))
        assert loaded == test_data

    def test_unicode_data_handling(self, tmp_path):
        """Test handling of unicode in data."""
        test_data = [
            {"text": "Hello 世界 🌍"},
            {"text": "Привет мир"},
            {"text": "مرحبا بالعالم"},
        ]

        file_path = tmp_path / "unicode.jsonl"
        save_jsonl(test_data, str(file_path))

        loaded = load_jsonl(str(file_path))
        assert loaded == test_data


@pytest.mark.integration
class TestValidationWithErrors:
    """Test validation error handling."""

    def test_validate_invalid_alpaca_records(self, tmp_path):
        """Test validation catches invalid Alpaca records."""
        invalid_data = [
            {"instruction": "Valid instruction", "output": "Valid output"},
            {"output": "Missing instruction"},  # Invalid
            {"instruction": "Missing output"},  # Invalid
            {},  # Empty - Invalid
        ]

        # Save and load
        file_path = tmp_path / "invalid.jsonl"
        save_jsonl(invalid_data, str(file_path))
        data = load_dataset(str(file_path))

        # Validate
        results = validate_dataset(data, format_type="alpaca")
        valid_count = sum(1 for r in results if r["valid"])
        invalid_count = len(results) - valid_count

        assert valid_count == 1
        assert invalid_count == 3

    def test_validate_invalid_sharegpt_records(self, tmp_path):
        """Test validation catches invalid ShareGPT records."""
        invalid_data = [
            {"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"}]},  # Valid
            {"conversations": []},  # Empty conversations
            {"no_conversations": "field"},  # Missing conversations
        ]

        file_path = tmp_path / "invalid.json"
        save_json(invalid_data, str(file_path))
        data = load_dataset(str(file_path))

        results = validate_dataset(data, format_type="sharegpt")
        valid_count = sum(1 for r in results if r["valid"])

        assert valid_count == 1  # Only first record is valid

    def test_validate_invalid_dpo_records(self, tmp_path):
        """Test validation catches invalid DPO records."""
        invalid_data = [
            {"prompt": "Q", "chosen": "Good", "rejected": "Bad"},  # Valid
            {"prompt": "Q", "chosen": "Good"},  # Missing rejected
            {"prompt": "Q", "rejected": "Bad"},  # Missing chosen
            {"chosen": "Good", "rejected": "Bad"},  # Missing prompt
        ]

        file_path = tmp_path / "invalid.jsonl"
        save_jsonl(invalid_data, str(file_path))
        data = load_dataset(str(file_path))

        results = validate_dataset(data, format_type="dpo")
        valid_count = sum(1 for r in results if r["valid"])

        assert valid_count == 1
