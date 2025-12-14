"""Dataset generators for training."""

from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from largeforge.utils import get_logger
from largeforge.data.converters import FormatConverter

logger = get_logger(__name__)


class SFTDatasetGenerator:
    """Generator for Supervised Fine-Tuning datasets."""

    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
        truncation: bool = True,
        padding: bool = False,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the SFT dataset generator.

        Args:
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            truncation: Whether to truncate long sequences
            padding: Whether to pad sequences
            system_prompt: Optional system prompt to prepend
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding
        self.system_prompt = system_prompt

    def _format_prompt(self, record: Dict[str, Any], format: str) -> str:
        """Format a record into a prompt string."""
        messages = FormatConverter.to_chat_messages(record, format)

        # Add system prompt if specified and not already present
        if self.system_prompt and (not messages or messages[0]["role"] != "system"):
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        # Apply chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

        # Fallback: simple concatenation
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|{role}|>\n{content}")
        return "\n".join(parts) + self.tokenizer.eos_token

    def process_record(
        self,
        record: Dict[str, Any],
        format: str = "auto",
    ) -> Dict[str, Any]:
        """
        Process a single record into model inputs.

        Args:
            record: Data record
            format: Input format ("alpaca", "sharegpt", "auto")

        Returns:
            Dictionary with input_ids, attention_mask, labels
        """
        prompt = self._format_prompt(record, format)

        # Tokenize
        encodings = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=self.truncation,
            padding="max_length" if self.padding else False,
            return_tensors=None,
        )

        # For SFT, labels are the same as input_ids
        encodings["labels"] = encodings["input_ids"].copy()

        return encodings

    def generate(
        self,
        data: List[Dict[str, Any]],
        format: str = "auto",
    ) -> List[Dict[str, Any]]:
        """
        Generate SFT dataset from raw data.

        Args:
            data: List of records
            format: Input format

        Returns:
            List of processed records
        """
        logger.info(f"Generating SFT dataset from {len(data)} records")
        processed = []
        for record in data:
            try:
                processed.append(self.process_record(record, format))
            except Exception as e:
                logger.warning(f"Failed to process record: {e}")

        logger.info(f"Generated {len(processed)} training examples")
        return processed

    def iter_generate(
        self,
        data: Iterator[Dict[str, Any]],
        format: str = "auto",
        batch_size: int = 100,
    ) -> Iterator[List[Dict[str, Any]]]:
        """
        Generate SFT dataset in batches (streaming).

        Args:
            data: Iterator of records
            format: Input format
            batch_size: Batch size

        Yields:
            Batches of processed records
        """
        batch = []
        for record in data:
            try:
                batch.append(self.process_record(record, format))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            except Exception as e:
                logger.warning(f"Failed to process record: {e}")

        if batch:
            yield batch


class DPODatasetGenerator:
    """Generator for Direct Preference Optimization datasets."""

    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
        max_prompt_length: int = 1024,
        truncation: bool = True,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the DPO dataset generator.

        Args:
            tokenizer: Hugging Face tokenizer
            max_length: Maximum total sequence length
            max_prompt_length: Maximum prompt length
            truncation: Whether to truncate long sequences
            system_prompt: Optional system prompt
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.system_prompt = system_prompt

    def _format_prompt(self, prompt: str) -> str:
        """Format the prompt with optional system message."""
        messages = [{"role": "user", "content": prompt}]

        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Fallback
        parts = [f"<|{msg['role']}|>\n{msg['content']}" for msg in messages]
        return "\n".join(parts) + "\n<|assistant|>\n"

    def _format_response(self, response: str) -> str:
        """Format a response."""
        return response + self.tokenizer.eos_token

    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single DPO record.

        Args:
            record: DPO record with prompt, chosen, rejected

        Returns:
            Processed record ready for DPO training
        """
        prompt = self._format_prompt(record["prompt"])
        chosen = self._format_response(record["chosen"])
        rejected = self._format_response(record["rejected"])

        # Tokenize prompt
        prompt_encodings = self.tokenizer(
            prompt,
            max_length=self.max_prompt_length,
            truncation=self.truncation,
            return_tensors=None,
        )

        # Tokenize chosen response
        chosen_encodings = self.tokenizer(
            prompt + chosen,
            max_length=self.max_length,
            truncation=self.truncation,
            return_tensors=None,
        )

        # Tokenize rejected response
        rejected_encodings = self.tokenizer(
            prompt + rejected,
            max_length=self.max_length,
            truncation=self.truncation,
            return_tensors=None,
        )

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "prompt_input_ids": prompt_encodings["input_ids"],
            "chosen_input_ids": chosen_encodings["input_ids"],
            "rejected_input_ids": rejected_encodings["input_ids"],
            "chosen_attention_mask": chosen_encodings["attention_mask"],
            "rejected_attention_mask": rejected_encodings["attention_mask"],
        }

    def generate(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate DPO dataset from raw preference data.

        Args:
            data: List of DPO records

        Returns:
            List of processed records
        """
        logger.info(f"Generating DPO dataset from {len(data)} records")
        processed = []
        for record in data:
            try:
                processed.append(self.process_record(record))
            except Exception as e:
                logger.warning(f"Failed to process record: {e}")

        logger.info(f"Generated {len(processed)} preference pairs")
        return processed

    def iter_generate(
        self,
        data: Iterator[Dict[str, Any]],
        batch_size: int = 100,
    ) -> Iterator[List[Dict[str, Any]]]:
        """
        Generate DPO dataset in batches.

        Args:
            data: Iterator of records
            batch_size: Batch size

        Yields:
            Batches of processed records
        """
        batch = []
        for record in data:
            try:
                batch.append(self.process_record(record))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            except Exception as e:
                logger.warning(f"Failed to process record: {e}")

        if batch:
            yield batch


def create_sft_dataset(
    data: List[Dict[str, Any]],
    tokenizer,
    max_length: int = 2048,
    format: str = "auto",
    system_prompt: Optional[str] = None,
    return_hf_dataset: bool = False,
):
    """
    Create an SFT dataset from raw data.

    Args:
        data: List of training records
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
        format: Input format ("alpaca", "sharegpt", "auto")
        system_prompt: Optional system prompt
        return_hf_dataset: Return as HuggingFace Dataset

    Returns:
        List of processed records or HF Dataset

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> dataset = create_sft_dataset(data, tokenizer)
    """
    generator = SFTDatasetGenerator(
        tokenizer=tokenizer,
        max_length=max_length,
        system_prompt=system_prompt,
    )

    processed = generator.generate(data, format=format)

    if return_hf_dataset:
        try:
            from datasets import Dataset
            return Dataset.from_list(processed)
        except ImportError:
            logger.warning("datasets not installed, returning list")
            return processed

    return processed


def create_dpo_dataset(
    data: List[Dict[str, Any]],
    tokenizer,
    max_length: int = 2048,
    max_prompt_length: int = 1024,
    system_prompt: Optional[str] = None,
    return_hf_dataset: bool = False,
):
    """
    Create a DPO dataset from preference data.

    Args:
        data: List of preference records
        tokenizer: Hugging Face tokenizer
        max_length: Maximum total sequence length
        max_prompt_length: Maximum prompt length
        system_prompt: Optional system prompt
        return_hf_dataset: Return as HuggingFace Dataset

    Returns:
        List of processed records or HF Dataset

    Example:
        >>> dataset = create_dpo_dataset(preference_data, tokenizer)
    """
    generator = DPODatasetGenerator(
        tokenizer=tokenizer,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        system_prompt=system_prompt,
    )

    processed = generator.generate(data)

    if return_hf_dataset:
        try:
            from datasets import Dataset
            return Dataset.from_list(processed)
        except ImportError:
            logger.warning("datasets not installed, returning list")
            return processed

    return processed
