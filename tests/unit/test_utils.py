"""Unit tests for utility modules."""

import json
import logging
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from largeforge.utils.logging import get_logger, timed_operation, set_log_level
from largeforge.utils.io import (
    load_json, save_json, load_jsonl, save_jsonl,
    ensure_dir, atomic_write, read_text, write_text,
    get_file_size, format_size,
)


class TestLogging:
    """Tests for logging utilities."""

    def test_get_logger_creates_logger(self):
        """Test logger creation."""
        logger = get_logger("test_logger")
        assert logger is not None
        assert logger.name == "test_logger"

    def test_get_logger_caches_loggers(self):
        """Test that loggers are cached."""
        logger1 = get_logger("cached_test")
        logger2 = get_logger("cached_test")
        assert logger1 is logger2

    def test_get_logger_with_level(self):
        """Test logger with custom level."""
        logger = get_logger("level_test", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_timed_operation(self, capsys):
        """Test timed operation context manager."""
        logger = get_logger("timed_test")
        with timed_operation(logger, "Test operation"):
            pass  # Do nothing

        captured = capsys.readouterr()
        assert "Starting: Test operation" in captured.out
        assert "Completed: Test operation" in captured.out

    def test_set_log_level(self):
        """Test setting log level."""
        logger = get_logger("set_level_test")
        set_log_level("debug")
        # Level should be changed to DEBUG
        assert logger.level == logging.DEBUG


class TestIOJson:
    """Tests for JSON I/O utilities."""

    def test_save_and_load_json(self, tmp_path):
        """Test JSON save and load roundtrip."""
        data = {"name": "test", "value": 42, "nested": {"a": 1}}
        json_path = tmp_path / "test.json"

        save_json(data, json_path)
        loaded = load_json(json_path)

        assert loaded == data

    def test_load_json_file_not_found(self, tmp_path):
        """Test loading non-existent JSON file."""
        with pytest.raises(FileNotFoundError):
            load_json(tmp_path / "nonexistent.json")

    def test_save_json_creates_dirs(self, tmp_path):
        """Test that save_json creates parent directories."""
        data = {"test": True}
        nested_path = tmp_path / "nested" / "dir" / "test.json"

        save_json(data, nested_path)

        assert nested_path.exists()
        assert load_json(nested_path) == data


class TestIOJsonl:
    """Tests for JSONL I/O utilities."""

    def test_save_and_load_jsonl(self, tmp_path):
        """Test JSONL save and load roundtrip."""
        data = [
            {"id": 1, "text": "First"},
            {"id": 2, "text": "Second"},
            {"id": 3, "text": "Third"},
        ]
        jsonl_path = tmp_path / "test.jsonl"

        save_jsonl(data, jsonl_path)
        loaded = load_jsonl(jsonl_path)

        assert loaded == data

    def test_load_jsonl_handles_blank_lines(self, tmp_path):
        """Test that blank lines in JSONL are skipped."""
        jsonl_path = tmp_path / "blank_lines.jsonl"
        content = '{"id": 1}\n\n{"id": 2}\n   \n{"id": 3}\n'
        jsonl_path.write_text(content)

        loaded = load_jsonl(jsonl_path)

        assert len(loaded) == 3


class TestIOText:
    """Tests for text I/O utilities."""

    def test_read_and_write_text(self, tmp_path):
        """Test text read and write."""
        content = "Hello, World!\nLine 2"
        text_path = tmp_path / "test.txt"

        write_text(text_path, content)
        loaded = read_text(text_path)

        assert loaded == content

    def test_atomic_write(self, tmp_path):
        """Test atomic write operation."""
        content = "Atomic content"
        path = tmp_path / "atomic.txt"

        atomic_write(path, content)

        assert path.read_text() == content

    def test_atomic_write_creates_dirs(self, tmp_path):
        """Test that atomic_write creates parent directories."""
        content = "Nested atomic"
        nested_path = tmp_path / "nested" / "atomic.txt"

        atomic_write(nested_path, content)

        assert nested_path.exists()


class TestIODir:
    """Tests for directory utilities."""

    def test_ensure_dir_creates_directory(self, tmp_path):
        """Test directory creation."""
        new_dir = tmp_path / "new_directory"

        result = ensure_dir(new_dir)

        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir

    def test_ensure_dir_nested(self, tmp_path):
        """Test nested directory creation."""
        nested = tmp_path / "a" / "b" / "c"

        ensure_dir(nested)

        assert nested.exists()

    def test_ensure_dir_existing(self, tmp_path):
        """Test ensure_dir with existing directory."""
        # Should not raise
        result = ensure_dir(tmp_path)
        assert result == tmp_path


class TestIOFileSize:
    """Tests for file size utilities."""

    def test_get_file_size(self, tmp_path):
        """Test getting file size."""
        file_path = tmp_path / "sized.txt"
        content = "Hello" * 100  # 500 bytes
        file_path.write_text(content)

        size = get_file_size(file_path)

        assert size == 500

    def test_format_size_bytes(self):
        """Test formatting bytes."""
        assert format_size(500) == "500.0 B"

    def test_format_size_kb(self):
        """Test formatting kilobytes."""
        assert format_size(1024) == "1.0 KB"

    def test_format_size_mb(self):
        """Test formatting megabytes."""
        assert format_size(1024 * 1024) == "1.0 MB"

    def test_format_size_gb(self):
        """Test formatting gigabytes."""
        assert format_size(1024 * 1024 * 1024) == "1.0 GB"


class TestDevice:
    """Tests for device utilities."""

    def test_get_device_returns_string(self):
        """Test that get_device returns a string."""
        from largeforge.utils.device import get_device

        device = get_device()
        assert device in ["cuda", "cpu"]

    def test_get_device_count_returns_int(self):
        """Test that get_device_count returns an integer."""
        from largeforge.utils.device import get_device_count

        count = get_device_count()
        assert isinstance(count, int)
        assert count >= 0

    @patch('largeforge.utils.device.torch.cuda.is_available')
    def test_get_device_cpu_fallback(self, mock_cuda):
        """Test CPU fallback when CUDA unavailable."""
        mock_cuda.return_value = False
        from largeforge.utils.device import get_device

        # Need to reimport to get fresh function
        import importlib
        import largeforge.utils.device as device_module
        importlib.reload(device_module)

        assert device_module.get_device() == "cpu"

    def test_is_bf16_supported_returns_bool(self):
        """Test that is_bf16_supported returns a boolean."""
        from largeforge.utils.device import is_bf16_supported

        result = is_bf16_supported()
        assert isinstance(result, bool)

    def test_get_optimal_dtype_returns_dtype(self):
        """Test that get_optimal_dtype returns a torch dtype."""
        import torch
        from largeforge.utils.device import get_optimal_dtype

        dtype = get_optimal_dtype()
        assert dtype in [torch.float16, torch.bfloat16, torch.float32]

    def test_select_device_auto(self):
        """Test select_device with auto preference."""
        import torch
        from largeforge.utils.device import select_device

        device = select_device("auto")
        assert isinstance(device, torch.device)

    def test_select_device_cpu(self):
        """Test select_device with cpu preference."""
        import torch
        from largeforge.utils.device import select_device

        device = select_device("cpu")
        assert device == torch.device("cpu")
