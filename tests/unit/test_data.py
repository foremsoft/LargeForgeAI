"""Unit tests for data pipeline modules."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from largeforge.data.loaders import JSONLoader, JSONLLoader, load_dataset
from largeforge.data.validators import (
    AlpacaValidator, ShareGPTValidator, DPOValidator,
    validate_dataset, ValidationError,
)
from largeforge.data.converters import (
    FormatConverter, alpaca_to_sharegpt, sharegpt_to_alpaca, to_chat_format,
)
from largeforge.data.generators import SFTDatasetGenerator, DPODatasetGenerator


class TestJSONLoader:
    """Tests for JSONLoader."""

    def test_load_json_file(self, tmp_path):
        """Test loading a JSON file."""
        data = [{"id": 1}, {"id": 2}]
        json_path = tmp_path / "test.json"
        json_path.write_text(json.dumps(data))

        loader = JSONLoader(json_path)
        loaded = loader.load()

        assert loaded == data

    def test_load_json_file_not_found(self, tmp_path):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            JSONLoader(tmp_path / "nonexistent.json")

    def test_load_invalid_json_structure(self, tmp_path):
        """Test loading JSON that isn't a list."""
        json_path = tmp_path / "invalid.json"
        json_path.write_text('{"not": "a list"}')

        loader = JSONLoader(json_path)
        with pytest.raises(ValueError, match="must contain a list"):
            loader.load()

    def test_iter_load_batches(self, tmp_path):
        """Test batch loading."""
        data = [{"id": i} for i in range(10)]
        json_path = tmp_path / "test.json"
        json_path.write_text(json.dumps(data))

        loader = JSONLoader(json_path)
        batches = list(loader.iter_load(batch_size=3))

        assert len(batches) == 4  # 3, 3, 3, 1
        assert batches[0] == [{"id": 0}, {"id": 1}, {"id": 2}]


class TestJSONLLoader:
    """Tests for JSONLLoader."""

    def test_load_jsonl_file(self, tmp_path):
        """Test loading a JSONL file."""
        jsonl_path = tmp_path / "test.jsonl"
        lines = ['{"id": 1}\n', '{"id": 2}\n', '{"id": 3}\n']
        jsonl_path.write_text("".join(lines))

        loader = JSONLLoader(jsonl_path)
        loaded = loader.load()

        assert loaded == [{"id": 1}, {"id": 2}, {"id": 3}]

    def test_load_jsonl_skips_blank_lines(self, tmp_path):
        """Test that blank lines are skipped."""
        jsonl_path = tmp_path / "test.jsonl"
        content = '{"id": 1}\n\n{"id": 2}\n   \n{"id": 3}\n'
        jsonl_path.write_text(content)

        loader = JSONLLoader(jsonl_path)
        loaded = loader.load()

        assert len(loaded) == 3

    def test_count_lines(self, tmp_path):
        """Test counting lines."""
        jsonl_path = tmp_path / "test.jsonl"
        content = '{"id": 1}\n{"id": 2}\n\n{"id": 3}\n'
        jsonl_path.write_text(content)

        loader = JSONLLoader(jsonl_path)
        count = loader.count_lines()

        assert count == 3


class TestLoadDataset:
    """Tests for load_dataset function."""

    def test_auto_detect_json(self, tmp_path):
        """Test auto-detection of JSON format."""
        data = [{"id": 1}]
        json_path = tmp_path / "data.json"
        json_path.write_text(json.dumps(data))

        loaded = load_dataset(json_path)
        assert loaded == data

    def test_auto_detect_jsonl(self, tmp_path):
        """Test auto-detection of JSONL format."""
        jsonl_path = tmp_path / "data.jsonl"
        jsonl_path.write_text('{"id": 1}\n{"id": 2}\n')

        loaded = load_dataset(jsonl_path)
        assert loaded == [{"id": 1}, {"id": 2}]

    def test_explicit_format(self, tmp_path):
        """Test explicit format specification."""
        data = [{"id": 1}]
        json_path = tmp_path / "data.txt"  # Wrong extension
        json_path.write_text(json.dumps(data))

        loaded = load_dataset(json_path, format="json")
        assert loaded == data


class TestAlpacaValidator:
    """Tests for AlpacaValidator."""

    def test_valid_alpaca_record(self, sample_alpaca_data):
        """Test validating valid Alpaca data."""
        validator = AlpacaValidator()
        valid, errors = validator.validate(sample_alpaca_data, raise_on_error=False)

        assert len(errors) == 0
        assert len(valid) == len(sample_alpaca_data)

    def test_missing_instruction(self):
        """Test missing instruction field."""
        data = [{"output": "response"}]
        validator = AlpacaValidator()

        with pytest.raises(ValidationError):
            validator.validate(data)

    def test_missing_output(self):
        """Test missing output field."""
        data = [{"instruction": "hello"}]
        validator = AlpacaValidator()

        with pytest.raises(ValidationError):
            validator.validate(data)

    def test_require_input(self):
        """Test require_input option."""
        data = [{"instruction": "hello", "output": "hi"}]
        validator = AlpacaValidator(require_input=True)

        _, errors = validator.validate(data, raise_on_error=False)
        assert len(errors) > 0

    def test_max_length(self):
        """Test max_length constraint."""
        data = [{"instruction": "x" * 1000, "output": "y"}]
        validator = AlpacaValidator(max_length=100)

        _, errors = validator.validate(data, raise_on_error=False)
        assert any("too long" in e for e in errors)


class TestShareGPTValidator:
    """Tests for ShareGPTValidator."""

    def test_valid_sharegpt_record(self, sample_sharegpt_data):
        """Test validating valid ShareGPT data."""
        validator = ShareGPTValidator()
        valid, errors = validator.validate(sample_sharegpt_data, raise_on_error=False)

        assert len(errors) == 0
        assert len(valid) == len(sample_sharegpt_data)

    def test_missing_conversations(self):
        """Test missing conversations field."""
        data = [{"other": "field"}]
        validator = ShareGPTValidator()

        with pytest.raises(ValidationError):
            validator.validate(data)

    def test_invalid_role(self):
        """Test invalid role in conversation."""
        data = [{"conversations": [{"from": "invalid_role", "value": "hello"}]}]
        validator = ShareGPTValidator(min_turns=1)

        _, errors = validator.validate(data, raise_on_error=False)
        assert any("Invalid role" in e for e in errors)

    def test_require_system(self):
        """Test require_system option."""
        data = [{"conversations": [
            {"from": "human", "value": "Hi"},
            {"from": "gpt", "value": "Hello"},
        ]}]
        validator = ShareGPTValidator(require_system=True)

        _, errors = validator.validate(data, raise_on_error=False)
        assert any("system message" in e for e in errors)

    def test_min_turns(self):
        """Test min_turns constraint."""
        data = [{"conversations": [{"from": "human", "value": "Hi"}]}]
        validator = ShareGPTValidator(min_turns=2)

        _, errors = validator.validate(data, raise_on_error=False)
        assert any("Too few turns" in e for e in errors)


class TestDPOValidator:
    """Tests for DPOValidator."""

    def test_valid_dpo_record(self, sample_dpo_data):
        """Test validating valid DPO data."""
        validator = DPOValidator()
        valid, errors = validator.validate(sample_dpo_data, raise_on_error=False)

        assert len(errors) == 0
        assert len(valid) == len(sample_dpo_data)

    def test_missing_fields(self):
        """Test missing required fields."""
        data = [{"prompt": "hello"}]  # Missing chosen and rejected
        validator = DPOValidator()

        with pytest.raises(ValidationError):
            validator.validate(data)

    def test_identical_chosen_rejected(self):
        """Test when chosen equals rejected."""
        data = [{
            "prompt": "hello",
            "chosen": "same response",
            "rejected": "same response",
        }]
        validator = DPOValidator(check_different=True)

        _, errors = validator.validate(data, raise_on_error=False)
        assert any("identical" in e for e in errors)


class TestValidateDataset:
    """Tests for validate_dataset function."""

    def test_validate_alpaca(self, sample_alpaca_data):
        """Test validating with alpaca format."""
        results = validate_dataset(sample_alpaca_data, format="alpaca")
        valid_count = sum(1 for r in results if r["valid"])
        assert valid_count == len(sample_alpaca_data)

    def test_validate_sharegpt(self, sample_sharegpt_data):
        """Test validating with sharegpt format."""
        results = validate_dataset(sample_sharegpt_data, format="sharegpt")
        valid_count = sum(1 for r in results if r["valid"])
        assert valid_count == len(sample_sharegpt_data)

    def test_validate_dpo(self, sample_dpo_data):
        """Test validating with dpo format."""
        results = validate_dataset(sample_dpo_data, format="dpo")
        valid_count = sum(1 for r in results if r["valid"])
        assert valid_count == len(sample_dpo_data)

    def test_invalid_format(self, sample_alpaca_data):
        """Test invalid format raises error."""
        with pytest.raises(ValueError, match="Unknown format"):
            validate_dataset(sample_alpaca_data, format="invalid_format")


class TestFormatConverter:
    """Tests for FormatConverter."""

    def test_alpaca_to_sharegpt_basic(self):
        """Test basic Alpaca to ShareGPT conversion."""
        record = {
            "instruction": "Hello",
            "input": "",
            "output": "Hi there!",
        }

        result = FormatConverter.alpaca_to_sharegpt(record)

        assert "conversations" in result
        assert len(result["conversations"]) == 2
        assert result["conversations"][0]["from"] == "human"
        assert result["conversations"][1]["from"] == "gpt"

    def test_alpaca_to_sharegpt_with_input(self):
        """Test conversion with input field."""
        record = {
            "instruction": "Summarize:",
            "input": "Some text to summarize",
            "output": "Summary",
        }

        result = FormatConverter.alpaca_to_sharegpt(record)

        human_msg = result["conversations"][0]["value"]
        assert "Summarize:" in human_msg
        assert "Some text to summarize" in human_msg

    def test_alpaca_to_sharegpt_with_system(self):
        """Test conversion with system prompt."""
        record = {"instruction": "Hello", "output": "Hi"}

        result = FormatConverter.alpaca_to_sharegpt(
            record, system_prompt="You are helpful."
        )

        assert len(result["conversations"]) == 3
        assert result["conversations"][0]["from"] == "system"

    def test_sharegpt_to_alpaca(self):
        """Test ShareGPT to Alpaca conversion."""
        record = {
            "conversations": [
                {"from": "human", "value": "What is 2+2?"},
                {"from": "gpt", "value": "4"},
            ]
        }

        result = FormatConverter.sharegpt_to_alpaca(record)

        assert result["instruction"] == "What is 2+2?"
        assert result["output"] == "4"

    def test_to_chat_messages_alpaca(self):
        """Test conversion to chat format from Alpaca."""
        record = {"instruction": "Hello", "output": "Hi", "input": ""}

        messages = FormatConverter.to_chat_messages(record, "alpaca")

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_to_chat_messages_sharegpt(self):
        """Test conversion to chat format from ShareGPT."""
        record = {
            "conversations": [
                {"from": "system", "value": "You are helpful."},
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello!"},
            ]
        }

        messages = FormatConverter.to_chat_messages(record, "sharegpt")

        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"


class TestConversionFunctions:
    """Tests for conversion helper functions."""

    def test_alpaca_to_sharegpt_batch(self, sample_alpaca_data):
        """Test batch Alpaca to ShareGPT conversion."""
        result = alpaca_to_sharegpt(sample_alpaca_data)

        assert len(result) == len(sample_alpaca_data)
        assert all("conversations" in r for r in result)

    def test_sharegpt_to_alpaca_batch(self, sample_sharegpt_data):
        """Test batch ShareGPT to Alpaca conversion."""
        result = sharegpt_to_alpaca(sample_sharegpt_data)

        assert len(result) == len(sample_sharegpt_data)
        assert all("instruction" in r for r in result)

    def test_to_chat_format_auto(self, sample_alpaca_data):
        """Test auto-detection in to_chat_format."""
        result = to_chat_format(sample_alpaca_data)

        assert len(result) == len(sample_alpaca_data)
        assert all(isinstance(msgs, list) for msgs in result)


class TestSFTDatasetGenerator:
    """Tests for SFTDatasetGenerator."""

    def test_process_record(self, mock_tokenizer):
        """Test processing a single record."""
        generator = SFTDatasetGenerator(
            tokenizer=mock_tokenizer,
            max_length=512,
        )

        record = {"instruction": "Hello", "output": "Hi"}
        result = generator.process_record(record, format="alpaca")

        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result

    def test_generate_batch(self, mock_tokenizer, sample_alpaca_data):
        """Test generating a batch."""
        generator = SFTDatasetGenerator(
            tokenizer=mock_tokenizer,
            max_length=512,
        )

        result = generator.generate(sample_alpaca_data, format="alpaca")

        assert len(result) == len(sample_alpaca_data)


class TestDPODatasetGenerator:
    """Tests for DPODatasetGenerator."""

    def test_process_record(self, mock_tokenizer):
        """Test processing a DPO record."""
        generator = DPODatasetGenerator(
            tokenizer=mock_tokenizer,
            max_length=512,
        )

        record = {
            "prompt": "What is 2+2?",
            "chosen": "4",
            "rejected": "5",
        }
        result = generator.process_record(record)

        assert "chosen_input_ids" in result
        assert "rejected_input_ids" in result
        assert "prompt_input_ids" in result

    def test_generate_batch(self, mock_tokenizer, sample_dpo_data):
        """Test generating a batch."""
        generator = DPODatasetGenerator(
            tokenizer=mock_tokenizer,
            max_length=512,
        )

        result = generator.generate(sample_dpo_data)

        assert len(result) == len(sample_dpo_data)
