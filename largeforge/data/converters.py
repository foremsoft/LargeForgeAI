"""Data format conversion utilities for LargeForgeAI."""

from typing import Any, Dict, List, Optional

from largeforge.utils import get_logger

logger = get_logger(__name__)


class FormatConverter:
    """Converts between different data formats."""

    @staticmethod
    def alpaca_to_sharegpt(
        record: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convert an Alpaca format record to ShareGPT format.

        Args:
            record: Alpaca format record with instruction, input, output
            system_prompt: Optional system prompt to include

        Returns:
            ShareGPT format record with conversations
        """
        conversations = []

        # Add system prompt if provided
        if system_prompt:
            conversations.append({
                "from": "system",
                "value": system_prompt,
            })

        # Build human message
        instruction = record["instruction"]
        input_text = record.get("input", "")

        if input_text:
            human_message = f"{instruction}\n\n{input_text}"
        else:
            human_message = instruction

        conversations.append({
            "from": "human",
            "value": human_message,
        })

        # Add assistant response
        conversations.append({
            "from": "gpt",
            "value": record["output"],
        })

        return {"conversations": conversations}

    @staticmethod
    def sharegpt_to_alpaca(record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a ShareGPT format record to Alpaca format.

        Note: This conversion may lose information as Alpaca format
        only supports single-turn conversations.

        Args:
            record: ShareGPT format record with conversations

        Returns:
            Alpaca format record
        """
        conversations = record["conversations"]

        # Skip system messages and find first human/gpt pair
        instruction = ""
        output = ""

        for turn in conversations:
            role = turn["from"]
            if role in ("human", "user") and not instruction:
                instruction = turn["value"]
            elif role in ("gpt", "assistant") and instruction and not output:
                output = turn["value"]
                break

        if not instruction or not output:
            raise ValueError("Could not extract instruction/output pair from conversation")

        return {
            "instruction": instruction,
            "input": "",
            "output": output,
        }

    @staticmethod
    def to_chat_messages(
        record: Dict[str, Any],
        format: str = "auto",
    ) -> List[Dict[str, str]]:
        """
        Convert a record to OpenAI chat message format.

        Args:
            record: Data record in alpaca or sharegpt format
            format: Input format ("alpaca", "sharegpt", "auto")

        Returns:
            List of messages in chat format
        """
        # Auto-detect format
        if format == "auto":
            if "conversations" in record:
                format = "sharegpt"
            elif "instruction" in record:
                format = "alpaca"
            else:
                raise ValueError("Could not auto-detect format")

        if format == "sharegpt":
            return FormatConverter._sharegpt_to_chat(record)
        elif format == "alpaca":
            return FormatConverter._alpaca_to_chat(record)
        else:
            raise ValueError(f"Unknown format: {format}")

    @staticmethod
    def _sharegpt_to_chat(record: Dict[str, Any]) -> List[Dict[str, str]]:
        """Convert ShareGPT to chat messages."""
        role_map = {
            "system": "system",
            "human": "user",
            "user": "user",
            "gpt": "assistant",
            "assistant": "assistant",
        }

        messages = []
        for turn in record["conversations"]:
            role = role_map.get(turn["from"], turn["from"])
            messages.append({
                "role": role,
                "content": turn["value"],
            })

        return messages

    @staticmethod
    def _alpaca_to_chat(record: Dict[str, Any]) -> List[Dict[str, str]]:
        """Convert Alpaca to chat messages."""
        instruction = record["instruction"]
        input_text = record.get("input", "")

        if input_text:
            user_content = f"{instruction}\n\n{input_text}"
        else:
            user_content = instruction

        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": record["output"]},
        ]


def alpaca_to_sharegpt(
    data: List[Dict[str, Any]],
    system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Convert a list of Alpaca format records to ShareGPT format.

    Args:
        data: List of Alpaca format records
        system_prompt: Optional system prompt to include

    Returns:
        List of ShareGPT format records

    Example:
        >>> alpaca_data = [{"instruction": "Hello", "output": "Hi!"}]
        >>> sharegpt_data = alpaca_to_sharegpt(alpaca_data)
    """
    logger.info(f"Converting {len(data)} records from Alpaca to ShareGPT format")
    converted = []
    for record in data:
        converted.append(
            FormatConverter.alpaca_to_sharegpt(record, system_prompt)
        )
    return converted


def sharegpt_to_alpaca(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert a list of ShareGPT format records to Alpaca format.

    Note: Multi-turn conversations will be truncated to first turn.

    Args:
        data: List of ShareGPT format records

    Returns:
        List of Alpaca format records
    """
    logger.info(f"Converting {len(data)} records from ShareGPT to Alpaca format")
    converted = []
    for record in data:
        try:
            converted.append(FormatConverter.sharegpt_to_alpaca(record))
        except ValueError as e:
            logger.warning(f"Skipping record: {e}")
    return converted


def to_chat_format(
    data: List[Dict[str, Any]],
    format: str = "auto",
) -> List[List[Dict[str, str]]]:
    """
    Convert dataset to OpenAI chat format.

    Args:
        data: List of records in alpaca or sharegpt format
        format: Input format ("alpaca", "sharegpt", "auto")

    Returns:
        List of chat message lists

    Example:
        >>> chat_data = to_chat_format(data)
        >>> # [
        >>> #   [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
        >>> #   ...
        >>> # ]
    """
    logger.info(f"Converting {len(data)} records to chat format")
    return [FormatConverter.to_chat_messages(record, format) for record in data]


def apply_chat_template(
    messages: List[Dict[str, str]],
    tokenizer,
    add_generation_prompt: bool = False,
) -> str:
    """
    Apply tokenizer's chat template to messages.

    Args:
        messages: List of chat messages
        tokenizer: Hugging Face tokenizer with chat template
        add_generation_prompt: Whether to add generation prompt

    Returns:
        Formatted string ready for tokenization
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Tokenizer does not support chat templates")

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def merge_conversations(
    records: List[Dict[str, Any]],
    max_turns: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Merge multiple ShareGPT records into a single multi-turn conversation.

    Args:
        records: List of ShareGPT format records
        max_turns: Maximum number of turns to include

    Returns:
        Single merged ShareGPT record
    """
    merged_conversations = []

    for record in records:
        conversations = record.get("conversations", [])
        for turn in conversations:
            # Skip duplicate system messages
            if turn["from"] == "system" and any(
                t["from"] == "system" for t in merged_conversations
            ):
                continue
            merged_conversations.append(turn)

            if max_turns and len(merged_conversations) >= max_turns:
                return {"conversations": merged_conversations}

    return {"conversations": merged_conversations}
