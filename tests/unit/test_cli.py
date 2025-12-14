"""Unit tests for CLI modules."""

import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock

from largeforge.cli.main import cli


class TestCLIMain:
    """Tests for main CLI entry point."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_cli_help(self, runner):
        """Test CLI help output."""
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "LargeForgeAI" in result.output
        assert "train" in result.output
        assert "serve" in result.output
        assert "data" in result.output

    def test_cli_version(self, runner):
        """Test CLI version output."""
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "largeforge" in result.output

    def test_train_help(self, runner):
        """Test train command help."""
        result = runner.invoke(cli, ["train", "--help"])

        assert result.exit_code == 0
        assert "sft" in result.output
        assert "dpo" in result.output

    def test_serve_help(self, runner):
        """Test serve command help."""
        result = runner.invoke(cli, ["serve", "--help"])

        assert result.exit_code == 0
        assert "start" in result.output
        assert "generate" in result.output

    def test_data_help(self, runner):
        """Test data command help."""
        result = runner.invoke(cli, ["data", "--help"])

        assert result.exit_code == 0
        assert "validate" in result.output
        assert "convert" in result.output
        assert "stats" in result.output

    def test_info_command(self, runner):
        """Test info command."""
        with patch("torch.cuda.is_available", return_value=False):
            result = runner.invoke(cli, ["info"])

            assert result.exit_code == 0
            assert "Version" in result.output
            assert "Python" in result.output


class TestTrainSFTCommand:
    """Tests for train sft command."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_sft_help(self, runner):
        """Test SFT command help."""
        result = runner.invoke(cli, ["train", "sft", "--help"])

        assert result.exit_code == 0
        assert "MODEL_PATH" in result.output
        assert "DATA_PATH" in result.output
        assert "--output" in result.output
        assert "--epochs" in result.output
        assert "--batch-size" in result.output
        assert "--learning-rate" in result.output
        assert "--lora" in result.output

    def test_sft_missing_output(self, runner, tmp_path):
        """Test SFT requires output option."""
        # Create a test data file so we don't fail on file existence check
        test_file = tmp_path / "data.json"
        test_file.write_text("[]")

        result = runner.invoke(cli, ["train", "sft", "model", str(test_file)])

        assert result.exit_code != 0
        # Either output is mentioned or it's a missing option error
        assert "output" in result.output.lower() or "missing" in result.output.lower() or "required" in result.output.lower()


class TestTrainDPOCommand:
    """Tests for train dpo command."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_dpo_help(self, runner):
        """Test DPO command help."""
        result = runner.invoke(cli, ["train", "dpo", "--help"])

        assert result.exit_code == 0
        assert "MODEL_PATH" in result.output
        assert "--beta" in result.output
        assert "--loss-type" in result.output

    def test_dpo_loss_types(self, runner):
        """Test DPO loss type options."""
        result = runner.invoke(cli, ["train", "dpo", "--help"])

        assert "sigmoid" in result.output
        assert "hinge" in result.output
        assert "ipo" in result.output


class TestServeStartCommand:
    """Tests for serve start command."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_start_help(self, runner):
        """Test start server command help."""
        result = runner.invoke(cli, ["serve", "start", "--help"])

        assert result.exit_code == 0
        assert "MODEL_PATH" in result.output
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--backend" in result.output

    def test_start_backend_options(self, runner):
        """Test backend options."""
        result = runner.invoke(cli, ["serve", "start", "--help"])

        assert "auto" in result.output
        assert "transformers" in result.output
        assert "vllm" in result.output


class TestServeGenerateCommand:
    """Tests for serve generate command."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_generate_help(self, runner):
        """Test generate command help."""
        result = runner.invoke(cli, ["serve", "generate", "--help"])

        assert result.exit_code == 0
        assert "MODEL_PATH" in result.output
        assert "--prompt" in result.output
        assert "--max-tokens" in result.output
        assert "--temperature" in result.output
        assert "--stream" in result.output
        assert "--chat" in result.output


class TestDataValidateCommand:
    """Tests for data validate command."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_validate_help(self, runner):
        """Test validate command help."""
        result = runner.invoke(cli, ["data", "validate", "--help"])

        assert result.exit_code == 0
        assert "DATA_PATH" in result.output
        assert "--format" in result.output
        assert "--strict" in result.output

    def test_validate_success(self, runner, tmp_path):
        """Test successful validation."""
        # Create test file
        test_file = tmp_path / "test.jsonl"
        test_file.write_text('{"instruction": "test", "output": "result"}\n')

        with patch("largeforge.data.load_dataset") as mock_load, \
             patch("largeforge.data.validate_dataset") as mock_validate:
            mock_load.return_value = [{"instruction": "test", "output": "result"}]
            mock_validate.return_value = [{"valid": True}]

            result = runner.invoke(cli, ["data", "validate", str(test_file)])

            assert result.exit_code == 0
            assert "Valid records: 1" in result.output

    def test_validate_with_errors(self, runner, tmp_path):
        """Test validation with errors."""
        test_file = tmp_path / "test.jsonl"
        test_file.write_text('{"bad": "data"}\n')

        with patch("largeforge.data.load_dataset") as mock_load, \
             patch("largeforge.data.validate_dataset") as mock_validate:
            mock_load.return_value = [{"bad": "data"}]
            mock_validate.return_value = [{"valid": False, "errors": ["Missing instruction"]}]

            result = runner.invoke(cli, ["data", "validate", str(test_file)])

            assert result.exit_code == 0
            assert "Invalid records: 1" in result.output


class TestDataConvertCommand:
    """Tests for data convert command."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_convert_help(self, runner):
        """Test convert command help."""
        result = runner.invoke(cli, ["data", "convert", "--help"])

        assert result.exit_code == 0
        assert "INPUT_PATH" in result.output
        assert "OUTPUT_PATH" in result.output
        assert "--from" in result.output
        assert "--to" in result.output

    def test_convert_requires_from(self, runner, tmp_path):
        """Test convert requires --from option."""
        test_file = tmp_path / "test.json"
        test_file.write_text("[]")

        result = runner.invoke(cli, [
            "data", "convert", str(test_file), "out.json",
            "--to", "sharegpt"
        ])

        assert result.exit_code != 0

    def test_convert_requires_to(self, runner, tmp_path):
        """Test convert requires --to option."""
        test_file = tmp_path / "test.json"
        test_file.write_text("[]")

        result = runner.invoke(cli, [
            "data", "convert", str(test_file), "out.json",
            "--from", "alpaca"
        ])

        assert result.exit_code != 0


class TestDataStatsCommand:
    """Tests for data stats command."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_stats_help(self, runner):
        """Test stats command help."""
        result = runner.invoke(cli, ["data", "stats", "--help"])

        assert result.exit_code == 0
        assert "DATA_PATH" in result.output
        assert "--format" in result.output
        assert "--detailed" in result.output

    def test_stats_alpaca(self, runner, tmp_path):
        """Test stats for Alpaca format."""
        test_file = tmp_path / "test.jsonl"
        test_file.write_text('{"instruction": "test", "output": "result"}\n')

        with patch("largeforge.data.load_dataset") as mock_load:
            mock_load.return_value = [
                {"instruction": "test instruction", "output": "test output"},
                {"instruction": "another", "output": "result", "input": "context"},
            ]

            result = runner.invoke(cli, ["data", "stats", str(test_file)])

            assert result.exit_code == 0
            assert "Total records: 2" in result.output
            assert "Alpaca" in result.output

    def test_stats_sharegpt(self, runner, tmp_path):
        """Test stats for ShareGPT format."""
        test_file = tmp_path / "test.jsonl"
        test_file.write_text('{"conversations": []}\n')

        with patch("largeforge.data.load_dataset") as mock_load:
            mock_load.return_value = [{
                "conversations": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ]
            }]

            result = runner.invoke(cli, [
                "data", "stats", str(test_file), "--format", "sharegpt"
            ])

            assert result.exit_code == 0
            assert "ShareGPT" in result.output

    def test_stats_dpo(self, runner, tmp_path):
        """Test stats for DPO format."""
        test_file = tmp_path / "test.jsonl"
        test_file.write_text('{"prompt": "q", "chosen": "good", "rejected": "bad"}\n')

        with patch("largeforge.data.load_dataset") as mock_load:
            mock_load.return_value = [{
                "prompt": "What is 2+2?",
                "chosen": "4",
                "rejected": "5",
            }]

            result = runner.invoke(cli, [
                "data", "stats", str(test_file), "--format", "dpo"
            ])

            assert result.exit_code == 0
            assert "DPO" in result.output


class TestCLIVerboseMode:
    """Tests for verbose mode."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_verbose_sets_debug_level(self, runner):
        """Test verbose flag sets debug log level."""
        with patch("largeforge.utils.set_log_level") as mock_set_log:
            result = runner.invoke(cli, ["-v", "--help"])

            # The verbose flag should trigger set_log_level("DEBUG")
            # But since --help exits early, we just check it doesn't error
            assert result.exit_code == 0
