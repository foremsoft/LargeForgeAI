"""Data validation utilities for LargeForgeAI."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

from largeforge.utils import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Raised when data validation fails."""

    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or []


class DataValidator(ABC):
    """Abstract base class for data validators."""

    @abstractmethod
    def validate_record(self, record: Dict[str, Any], index: int) -> List[str]:
        """
        Validate a single record.

        Args:
            record: Data record to validate
            index: Record index for error reporting

        Returns:
            List of error messages (empty if valid)
        """
        pass

    def validate(
        self,
        data: List[Dict[str, Any]],
        raise_on_error: bool = True,
        max_errors: int = 100,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Validate a dataset.

        Args:
            data: List of records to validate
            raise_on_error: Whether to raise ValidationError on first error
            max_errors: Maximum number of errors to collect

        Returns:
            Tuple of (valid_records, all_errors)

        Raises:
            ValidationError: If raise_on_error=True and validation fails
        """
        valid_records = []
        all_errors = []

        for i, record in enumerate(data):
            errors = self.validate_record(record, i)

            if errors:
                all_errors.extend(errors)
                if raise_on_error:
                    raise ValidationError(
                        f"Validation failed at record {i}",
                        errors=errors
                    )
                if len(all_errors) >= max_errors:
                    logger.warning(f"Stopped validation after {max_errors} errors")
                    break
            else:
                valid_records.append(record)

        if all_errors:
            logger.warning(f"Found {len(all_errors)} validation errors")
        else:
            logger.info(f"Validated {len(valid_records)} records successfully")

        return valid_records, all_errors


class AlpacaValidator(DataValidator):
    """Validator for Alpaca format data."""

    REQUIRED_FIELDS = {"instruction", "output"}
    OPTIONAL_FIELDS = {"input", "text"}

    def __init__(
        self,
        require_input: bool = False,
        max_length: Optional[int] = None,
        min_length: int = 1,
    ):
        """
        Initialize the Alpaca validator.

        Args:
            require_input: Whether the 'input' field is required
            max_length: Maximum text length (characters)
            min_length: Minimum text length (characters)
        """
        self.require_input = require_input
        self.max_length = max_length
        self.min_length = min_length

    def validate_record(self, record: Dict[str, Any], index: int) -> List[str]:
        """Validate an Alpaca format record."""
        errors = []

        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in record:
                errors.append(f"Record {index}: Missing required field '{field}'")
            elif not isinstance(record[field], str):
                errors.append(f"Record {index}: Field '{field}' must be a string")

        if self.require_input and "input" not in record:
            errors.append(f"Record {index}: Missing required field 'input'")

        # Check text lengths
        for field in ["instruction", "output", "input"]:
            if field in record and isinstance(record[field], str):
                text = record[field]
                if len(text) < self.min_length and field != "input":
                    errors.append(
                        f"Record {index}: Field '{field}' is too short "
                        f"({len(text)} < {self.min_length})"
                    )
                if self.max_length and len(text) > self.max_length:
                    errors.append(
                        f"Record {index}: Field '{field}' is too long "
                        f"({len(text)} > {self.max_length})"
                    )

        return errors


class ShareGPTValidator(DataValidator):
    """Validator for ShareGPT conversation format."""

    VALID_ROLES = {"system", "human", "gpt", "user", "assistant"}

    def __init__(
        self,
        require_system: bool = False,
        min_turns: int = 2,
        max_turns: Optional[int] = None,
    ):
        """
        Initialize the ShareGPT validator.

        Args:
            require_system: Whether system message is required
            min_turns: Minimum number of conversation turns
            max_turns: Maximum number of conversation turns
        """
        self.require_system = require_system
        self.min_turns = min_turns
        self.max_turns = max_turns

    def validate_record(self, record: Dict[str, Any], index: int) -> List[str]:
        """Validate a ShareGPT format record."""
        errors = []

        # Check conversations field
        if "conversations" not in record:
            errors.append(f"Record {index}: Missing 'conversations' field")
            return errors

        conversations = record["conversations"]
        if not isinstance(conversations, list):
            errors.append(f"Record {index}: 'conversations' must be a list")
            return errors

        # Check turn count
        if len(conversations) < self.min_turns:
            errors.append(
                f"Record {index}: Too few turns "
                f"({len(conversations)} < {self.min_turns})"
            )

        if self.max_turns and len(conversations) > self.max_turns:
            errors.append(
                f"Record {index}: Too many turns "
                f"({len(conversations)} > {self.max_turns})"
            )

        # Validate each turn
        has_system = False
        for turn_idx, turn in enumerate(conversations):
            if not isinstance(turn, dict):
                errors.append(
                    f"Record {index}, turn {turn_idx}: Must be a dictionary"
                )
                continue

            # Check required fields
            if "from" not in turn:
                errors.append(
                    f"Record {index}, turn {turn_idx}: Missing 'from' field"
                )
            else:
                role = turn["from"]
                if role not in self.VALID_ROLES:
                    errors.append(
                        f"Record {index}, turn {turn_idx}: Invalid role '{role}'"
                    )
                if role == "system":
                    has_system = True

            if "value" not in turn:
                errors.append(
                    f"Record {index}, turn {turn_idx}: Missing 'value' field"
                )
            elif not isinstance(turn["value"], str):
                errors.append(
                    f"Record {index}, turn {turn_idx}: 'value' must be a string"
                )

        if self.require_system and not has_system:
            errors.append(f"Record {index}: Missing required system message")

        return errors


class DPOValidator(DataValidator):
    """Validator for DPO preference data."""

    REQUIRED_FIELDS = {"prompt", "chosen", "rejected"}

    def __init__(
        self,
        min_length: int = 1,
        max_length: Optional[int] = None,
        check_different: bool = True,
    ):
        """
        Initialize the DPO validator.

        Args:
            min_length: Minimum text length (characters)
            max_length: Maximum text length (characters)
            check_different: Ensure chosen != rejected
        """
        self.min_length = min_length
        self.max_length = max_length
        self.check_different = check_different

    def validate_record(self, record: Dict[str, Any], index: int) -> List[str]:
        """Validate a DPO format record."""
        errors = []

        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in record:
                errors.append(f"Record {index}: Missing required field '{field}'")
            elif not isinstance(record[field], str):
                errors.append(f"Record {index}: Field '{field}' must be a string")

        # If basic fields missing, skip further validation
        if errors:
            return errors

        # Check text lengths
        for field in self.REQUIRED_FIELDS:
            text = record[field]
            if len(text) < self.min_length:
                errors.append(
                    f"Record {index}: Field '{field}' is too short "
                    f"({len(text)} < {self.min_length})"
                )
            if self.max_length and len(text) > self.max_length:
                errors.append(
                    f"Record {index}: Field '{field}' is too long "
                    f"({len(text)} > {self.max_length})"
                )

        # Check chosen != rejected
        if self.check_different:
            if record["chosen"] == record["rejected"]:
                errors.append(
                    f"Record {index}: 'chosen' and 'rejected' are identical"
                )

        return errors


def validate_dataset(
    data: List[Dict[str, Any]],
    format: str,
    raise_on_error: bool = True,
    **kwargs,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Validate a dataset with the appropriate validator.

    Args:
        data: List of records to validate
        format: Data format ("alpaca", "sharegpt", "dpo")
        raise_on_error: Whether to raise on validation failure
        **kwargs: Additional arguments for the validator

    Returns:
        Tuple of (valid_records, errors)

    Example:
        >>> valid_data, errors = validate_dataset(data, "alpaca")
        >>> valid_data, errors = validate_dataset(
        ...     data, "sharegpt", require_system=True
        ... )
    """
    validators = {
        "alpaca": AlpacaValidator,
        "sharegpt": ShareGPTValidator,
        "dpo": DPOValidator,
    }

    validator_class = validators.get(format.lower())
    if validator_class is None:
        raise ValueError(f"Unknown format: {format}. Valid: {list(validators.keys())}")

    validator = validator_class(**kwargs)
    return validator.validate(data, raise_on_error=raise_on_error)
