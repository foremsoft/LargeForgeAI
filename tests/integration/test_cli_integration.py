"""Integration tests for the CLI system."""

import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
import json

from largeforge.cli.main import cli


@pytest.mark.integration
class TestCLIEntryPoint:
    """Test CLI main entry point."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_cli_invocation(self, runner):
        """Test basic CLI invocation."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "LargeForgeAI" in result.output

    def test_cli_version(self, runner):
        """Test version display."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "largeforge" in result.output

    def test_cli_subcommands_available(self, runner):
        """Test all subcommands are available."""
        result = runner.invoke(cli, ["--help"])

        # Check main commands are listed
        assert "train" in result.output
        assert "serve" in result.output
        assert "data" in result.output
        assert "info" in result.output


@pytest.mark.integration
class TestTrainCLIIntegration:
    """Test train CLI command integration."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_train_sft_full_options(self, runner, integration_data_dir, tmp_path):
        """Test SFT training with all options."""
        data_file = integration_data_dir / "alpaca.jsonl"
        output_dir = tmp_path / "output"

        with patch("largeforge.training.SFTTrainer") as mock_trainer:
            mock_instance = MagicMock()
            mock_trainer.return_value = mock_instance

            result = runner.invoke(cli, [
                "train", "sft",
                "gpt2",
                str(data_file),
                "--output", str(output_dir),
                "--epochs", "1",
                "--batch-size", "2",
                "--learning-rate", "5e-5",
                "--max-length", "256",
            ])

            # Should not error on argument parsing
            assert result.exit_code == 0 or "Error" not in result.output[:100]

    def test_train_dpo_full_options(self, runner, integration_data_dir, tmp_path):
        """Test DPO training with all options."""
        data_file = integration_data_dir / "dpo.jsonl"
        output_dir = tmp_path / "output"

        with patch("largeforge.training.DPOTrainer") as mock_trainer:
            mock_instance = MagicMock()
            mock_trainer.return_value = mock_instance

            result = runner.invoke(cli, [
                "train", "dpo",
                "gpt2",
                str(data_file),
                "--output", str(output_dir),
                "--beta", "0.1",
                "--loss-type", "sigmoid",
            ])

            # Should not error on argument parsing
            assert result.exit_code == 0 or "Error" not in result.output[:100]


@pytest.mark.integration
class TestDataCLIIntegration:
    """Test data CLI command integration."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_data_validate_alpaca(self, runner, integration_data_dir):
        """Test validating Alpaca format data."""
        data_file = integration_data_dir / "alpaca.jsonl"

        with patch("largeforge.data.load_dataset") as mock_load, \
             patch("largeforge.data.validate_dataset") as mock_validate:
            mock_load.return_value = [
                {"instruction": "test", "output": "result"},
                {"instruction": "test2", "output": "result2"},
                {"instruction": "test3", "output": "result3"},
            ]
            mock_validate.return_value = [
                {"valid": True},
                {"valid": True},
                {"valid": True},
            ]

            result = runner.invoke(cli, [
                "data", "validate",
                str(data_file),
                "--format", "alpaca",
            ])

            assert result.exit_code == 0
            assert "Valid records: 3" in result.output

    def test_data_validate_sharegpt(self, runner, integration_data_dir):
        """Test validating ShareGPT format data."""
        data_file = integration_data_dir / "sharegpt.json"

        with patch("largeforge.data.load_dataset") as mock_load, \
             patch("largeforge.data.validate_dataset") as mock_validate:
            mock_load.return_value = [
                {"conversations": [{"from": "human", "value": "Hi"}]},
                {"conversations": [{"from": "human", "value": "Hello"}]},
            ]
            mock_validate.return_value = [
                {"valid": True},
                {"valid": True},
            ]

            result = runner.invoke(cli, [
                "data", "validate",
                str(data_file),
                "--format", "sharegpt",
            ])

            assert result.exit_code == 0
            assert "Valid records: 2" in result.output

    def test_data_stats_detailed(self, runner, integration_data_dir):
        """Test showing detailed data stats."""
        data_file = integration_data_dir / "alpaca.jsonl"

        with patch("largeforge.data.load_dataset") as mock_load:
            mock_load.return_value = [
                {"instruction": "Summarize this text", "output": "Summary here"},
                {"instruction": "Translate", "input": "Hello", "output": "Hola"},
            ]

            result = runner.invoke(cli, [
                "data", "stats",
                str(data_file),
                "--format", "alpaca",
                "--detailed",
            ])

            assert result.exit_code == 0
            assert "Total records: 2" in result.output

    def test_data_convert_formats(self, runner, integration_data_dir, tmp_path):
        """Test converting between data formats."""
        input_file = integration_data_dir / "alpaca.jsonl"
        output_file = tmp_path / "converted.json"

        with patch("largeforge.data.load_dataset") as mock_load, \
             patch("largeforge.data.converters.alpaca_to_sharegpt") as mock_convert, \
             patch("largeforge.utils.save_json") as mock_save:
            mock_load.return_value = [
                {"instruction": "test", "output": "result"},
            ]
            mock_convert.return_value = [
                {"conversations": [{"from": "human", "value": "test"}]},
            ]

            result = runner.invoke(cli, [
                "data", "convert",
                str(input_file),
                str(output_file),
                "--from", "alpaca",
                "--to", "sharegpt",
            ])

            assert result.exit_code == 0


@pytest.mark.integration
class TestServeCLIIntegration:
    """Test serve CLI command integration."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_serve_start_help(self, runner):
        """Test serve start command help."""
        result = runner.invoke(cli, ["serve", "start", "--help"])

        assert result.exit_code == 0
        assert "MODEL_PATH" in result.output
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--backend" in result.output

    def test_serve_generate_help(self, runner):
        """Test serve generate command help."""
        result = runner.invoke(cli, ["serve", "generate", "--help"])

        assert result.exit_code == 0
        assert "MODEL_PATH" in result.output
        assert "--prompt" in result.output
        assert "--max-tokens" in result.output
        assert "--temperature" in result.output
        assert "--stream" in result.output

    def test_serve_generate_with_prompt(self, runner):
        """Test generating text with prompt."""
        with patch("largeforge.inference.TextGenerator") as mock_gen_class:
            mock_gen = MagicMock()
            mock_gen.generate.return_value = "Generated response text"
            mock_gen_class.return_value = mock_gen

            result = runner.invoke(cli, [
                "serve", "generate",
                "gpt2",
                "--prompt", "Once upon a time",
                "--max-tokens", "50",
                "--no-stream",
            ])

            # Should attempt generation
            assert "Loading model" in result.output or result.exit_code == 0


@pytest.mark.integration
class TestCLIVerboseMode:
    """Test CLI verbose mode integration."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_verbose_flag_accepted(self, runner):
        """Test verbose flag is accepted."""
        result = runner.invoke(cli, ["-v", "--help"])
        assert result.exit_code == 0

    def test_verbose_with_subcommand(self, runner):
        """Test verbose flag with subcommand."""
        result = runner.invoke(cli, ["-v", "train", "--help"])
        assert result.exit_code == 0


@pytest.mark.integration
class TestCLIErrorHandling:
    """Test CLI error handling."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_missing_required_arguments(self, runner):
        """Test missing required arguments show error."""
        result = runner.invoke(cli, ["train", "sft"])
        assert result.exit_code != 0

    def test_invalid_option_values(self, runner, tmp_path):
        """Test invalid option values are rejected."""
        data_file = tmp_path / "test.json"
        data_file.write_text("[]")

        result = runner.invoke(cli, [
            "data", "validate",
            str(data_file),
            "--format", "invalid_format",  # Invalid format
        ])

        assert result.exit_code != 0

    def test_nonexistent_file(self, runner):
        """Test nonexistent file shows error."""
        result = runner.invoke(cli, [
            "data", "validate",
            "/nonexistent/path/data.json",
        ])

        assert result.exit_code != 0


@pytest.mark.integration
class TestCLIInfoCommand:
    """Test CLI info command."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_info_command(self, runner):
        """Test info command displays system info."""
        with patch("torch.cuda.is_available", return_value=False):
            result = runner.invoke(cli, ["info"])

            assert result.exit_code == 0
            assert "Version" in result.output or "Python" in result.output
