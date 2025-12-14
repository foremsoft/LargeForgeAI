"""Data loading utilities for LargeForgeAI."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from largeforge.utils import get_logger

logger = get_logger(__name__)

PathLike = Union[str, Path]


class DataLoader(ABC):
    """Abstract base class for data loaders."""

    def __init__(self, path: PathLike):
        """
        Initialize the data loader.

        Args:
            path: Path to the data file or directory
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Path not found: {self.path}")

    @abstractmethod
    def load(self) -> List[Dict[str, Any]]:
        """Load all data at once."""
        pass

    @abstractmethod
    def iter_load(self, batch_size: int = 1000) -> Iterator[List[Dict[str, Any]]]:
        """Load data in batches."""
        pass

    def __len__(self) -> int:
        """Return the number of records."""
        return len(self.load())


class JSONLoader(DataLoader):
    """Loader for JSON files containing a list of records."""

    def load(self) -> List[Dict[str, Any]]:
        """Load all data from JSON file."""
        logger.info(f"Loading JSON from {self.path}")
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of records")

        logger.info(f"Loaded {len(data)} records from {self.path}")
        return data

    def iter_load(self, batch_size: int = 1000) -> Iterator[List[Dict[str, Any]]]:
        """Load JSON data in batches."""
        data = self.load()
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]


class JSONLLoader(DataLoader):
    """Loader for JSON Lines files."""

    def load(self) -> List[Dict[str, Any]]:
        """Load all data from JSONL file."""
        logger.info(f"Loading JSONL from {self.path}")
        data = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")

        logger.info(f"Loaded {len(data)} records from {self.path}")
        return data

    def iter_load(self, batch_size: int = 1000) -> Iterator[List[Dict[str, Any]]]:
        """Load JSONL data in batches (streaming)."""
        batch = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        batch.append(json.loads(line))
                        if len(batch) >= batch_size:
                            yield batch
                            batch = []
                    except json.JSONDecodeError:
                        continue

        if batch:
            yield batch

    def count_lines(self) -> int:
        """Count the number of lines without loading all data."""
        count = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count


class ParquetLoader(DataLoader):
    """Loader for Parquet files."""

    def __init__(self, path: PathLike, columns: Optional[List[str]] = None):
        """
        Initialize the Parquet loader.

        Args:
            path: Path to the Parquet file
            columns: Optional list of columns to load
        """
        super().__init__(path)
        self.columns = columns

    def load(self) -> List[Dict[str, Any]]:
        """Load all data from Parquet file."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "pyarrow is required for Parquet support. "
                "Install with: pip install pyarrow"
            )

        logger.info(f"Loading Parquet from {self.path}")
        table = pq.read_table(self.path, columns=self.columns)
        data = table.to_pylist()
        logger.info(f"Loaded {len(data)} records from {self.path}")
        return data

    def iter_load(self, batch_size: int = 1000) -> Iterator[List[Dict[str, Any]]]:
        """Load Parquet data in batches."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "pyarrow is required for Parquet support. "
                "Install with: pip install pyarrow"
            )

        parquet_file = pq.ParquetFile(self.path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=self.columns):
            yield batch.to_pylist()


class HuggingFaceLoader(DataLoader):
    """Loader for Hugging Face datasets."""

    def __init__(
        self,
        path: str,
        name: Optional[str] = None,
        split: str = "train",
        streaming: bool = False,
    ):
        """
        Initialize the Hugging Face dataset loader.

        Args:
            path: Dataset name on Hugging Face Hub
            name: Dataset configuration name
            split: Dataset split to load
            streaming: Whether to use streaming mode
        """
        self._path = path
        self.name = name
        self.split = split
        self.streaming = streaming
        self._dataset = None

    @property
    def path(self) -> str:
        return self._path

    def _load_dataset(self):
        """Load the Hugging Face dataset."""
        try:
            from datasets import load_dataset as hf_load_dataset
        except ImportError:
            raise ImportError(
                "datasets library is required. Install with: pip install datasets"
            )

        if self._dataset is None:
            logger.info(f"Loading dataset {self._path} (split={self.split})")
            self._dataset = hf_load_dataset(
                self._path,
                name=self.name,
                split=self.split,
                streaming=self.streaming,
            )

        return self._dataset

    def load(self) -> List[Dict[str, Any]]:
        """Load all data from Hugging Face dataset."""
        dataset = self._load_dataset()

        if self.streaming:
            raise ValueError("Cannot use load() with streaming=True. Use iter_load() instead.")

        data = [dict(item) for item in dataset]
        logger.info(f"Loaded {len(data)} records from {self._path}")
        return data

    def iter_load(self, batch_size: int = 1000) -> Iterator[List[Dict[str, Any]]]:
        """Load Hugging Face dataset in batches."""
        dataset = self._load_dataset()

        if self.streaming:
            batch = []
            for item in dataset:
                batch.append(dict(item))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
        else:
            for i in range(0, len(dataset), batch_size):
                yield [dict(item) for item in dataset[i:i + batch_size]]


def load_dataset(
    path: PathLike,
    format: Optional[str] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Load a dataset from a file.

    Args:
        path: Path to the data file
        format: File format ("json", "jsonl", "parquet", "hf")
                Auto-detected if not specified
        **kwargs: Additional arguments passed to the loader

    Returns:
        List of data records

    Example:
        >>> data = load_dataset("data/train.jsonl")
        >>> data = load_dataset("meta-llama/alpaca", format="hf")
    """
    path = Path(path) if format != "hf" else path

    # Auto-detect format
    if format is None:
        if isinstance(path, Path):
            suffix = path.suffix.lower()
            format_map = {
                ".json": "json",
                ".jsonl": "jsonl",
                ".parquet": "parquet",
                ".pq": "parquet",
            }
            format = format_map.get(suffix)
            if format is None:
                raise ValueError(f"Unknown file format: {suffix}")
        else:
            format = "hf"  # Assume Hugging Face dataset name

    # Create appropriate loader
    loaders = {
        "json": JSONLoader,
        "jsonl": JSONLLoader,
        "parquet": ParquetLoader,
        "hf": HuggingFaceLoader,
    }

    loader_class = loaders.get(format)
    if loader_class is None:
        raise ValueError(f"Unknown format: {format}")

    loader = loader_class(path, **kwargs)
    return loader.load()
