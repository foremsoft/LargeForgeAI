"""Base classes for inference backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from largeforge.config import GenerationConfig
from largeforge.utils import get_logger

logger = get_logger(__name__)


@dataclass
class GenerationRequest:
    """Request for text generation."""

    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    stop: Optional[List[str]] = None
    stream: bool = False
    n: int = 1  # Number of completions

    # Optional parameters
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logprobs: Optional[int] = None
    echo: bool = False

    @classmethod
    def from_config(cls, prompt: str, config: GenerationConfig) -> "GenerationRequest":
        """Create request from GenerationConfig."""
        return cls(
            prompt=prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
            stop=config.stop,
            stream=config.stream,
        )


@dataclass
class GenerationResponse:
    """Response from text generation."""

    text: str
    prompt: str
    finish_reason: str = "stop"  # "stop", "length", "error"
    usage: Dict[str, int] = field(default_factory=dict)
    model: str = ""
    logprobs: Optional[List[float]] = None

    @property
    def prompt_tokens(self) -> int:
        """Get number of prompt tokens."""
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        """Get number of completion tokens."""
        return self.usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        """Get total number of tokens."""
        return self.usage.get("total_tokens", 0)


@dataclass
class StreamingResponse:
    """Streaming response chunk."""

    delta: str
    finish_reason: Optional[str] = None
    is_final: bool = False

    # Accumulated state
    accumulated_text: str = ""
    token_count: int = 0


class InferenceBackend(ABC):
    """Abstract base class for inference backends."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
    ):
        """
        Initialize the inference backend.

        Args:
            model_path: Path to model or HuggingFace model ID
            device: Device to run on ("auto", "cuda", "cpu", "cuda:N")
            dtype: Data type ("auto", "float16", "bfloat16", "float32")
            trust_remote_code: Whether to trust remote code
        """
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code

        self.model = None
        self.tokenizer = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    @abstractmethod
    def load(self) -> None:
        """Load the model and tokenizer."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload the model to free memory."""
        pass

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate text from a prompt.

        Args:
            request: Generation request

        Returns:
            Generation response
        """
        pass

    @abstractmethod
    def generate_stream(
        self, request: GenerationRequest
    ) -> Iterator[StreamingResponse]:
        """
        Generate text with streaming output.

        Args:
            request: Generation request

        Yields:
            Streaming response chunks
        """
        pass

    def generate_batch(
        self, requests: List[GenerationRequest]
    ) -> List[GenerationResponse]:
        """
        Generate text for multiple requests.

        Default implementation processes sequentially.
        Subclasses may override for batch optimization.

        Args:
            requests: List of generation requests

        Returns:
            List of generation responses
        """
        return [self.generate(req) for req in requests]

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        return len(self.tokenizer.encode(text))

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model info
        """
        return {
            "model_path": self.model_path,
            "device": self.device,
            "dtype": self.dtype,
            "is_loaded": self._is_loaded,
        }


class AsyncInferenceBackend(InferenceBackend):
    """Async-capable inference backend."""

    @abstractmethod
    async def generate_async(self, request: GenerationRequest) -> GenerationResponse:
        """
        Asynchronously generate text.

        Args:
            request: Generation request

        Returns:
            Generation response
        """
        pass

    @abstractmethod
    async def generate_stream_async(
        self, request: GenerationRequest
    ) -> AsyncIterator[StreamingResponse]:
        """
        Asynchronously generate text with streaming.

        Args:
            request: Generation request

        Yields:
            Streaming response chunks
        """
        pass

    async def generate_batch_async(
        self, requests: List[GenerationRequest]
    ) -> List[GenerationResponse]:
        """
        Asynchronously generate text for multiple requests.

        Args:
            requests: List of generation requests

        Returns:
            List of generation responses
        """
        import asyncio
        tasks = [self.generate_async(req) for req in requests]
        return await asyncio.gather(*tasks)
