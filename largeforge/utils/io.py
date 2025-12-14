"""File I/O utilities for LargeForgeAI."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, List, Union

PathLike = Union[str, Path]


def load_json(path: PathLike) -> dict:
    """
    Load a JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Parsed JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: PathLike, indent: int = 2) -> None:
    """
    Save data to a JSON file.

    Args:
        data: Data to save
        path: Output path
        indent: JSON indentation level
    """
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_jsonl(path: PathLike) -> List[dict]:
    """
    Load a JSON Lines file.

    Args:
        path: Path to JSONL file

    Returns:
        List of parsed JSON objects
    """
    path = Path(path)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[dict], path: PathLike) -> None:
    """
    Save data to a JSON Lines file.

    Args:
        data: List of dicts to save
        path: Output path
    """
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def ensure_dir(path: PathLike) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size(path: PathLike) -> int:
    """
    Get the size of a file in bytes.

    Args:
        path: Path to file

    Returns:
        File size in bytes
    """
    return Path(path).stat().st_size


def format_size(size_bytes: int) -> str:
    """
    Format byte size as human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def atomic_write(path: PathLike, content: str) -> None:
    """
    Write content to a file atomically.

    Uses a temporary file and rename to ensure the file is either
    completely written or not modified at all.

    Args:
        path: Target file path
        content: Content to write
    """
    path = Path(path)
    ensure_dir(path.parent)

    # Write to temp file in same directory (same filesystem)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except Exception:
        # Clean up temp file on failure
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def read_text(path: PathLike) -> str:
    """
    Read text content from a file.

    Args:
        path: Path to file

    Returns:
        File content as string
    """
    return Path(path).read_text(encoding="utf-8")


def write_text(path: PathLike, content: str) -> None:
    """
    Write text content to a file.

    Args:
        path: Path to file
        content: Content to write
    """
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")
