"""High-level text generation interface."""

from typing import Any, Dict, Iterator, List, Optional, Union

from largeforge.config import GenerationConfig, InferenceConfig
from largeforge.inference.base import (
    InferenceBackend,
    GenerationRequest,
    GenerationResponse,
    StreamingResponse,
)
from largeforge.utils import get_logger

logger = get_logger(__name__)


class TextGenerator:
    """High-level text generation interface."""

    def __init__(
        self,
        model_path: str,
        backend: str = "auto",
        config: Optional[InferenceConfig] = None,
        **kwargs,
    ):
        """
        Initialize the text generator.

        Args:
            model_path: Path to model or HuggingFace model ID
            backend: Backend to use ("auto", "transformers", "vllm")
            config: Optional inference configuration
            **kwargs: Additional backend-specific arguments
        """
        self.model_path = model_path
        self.backend_name = backend
        self.config = config

        # Initialize backend
        self._backend: Optional[InferenceBackend] = None
        self._backend_kwargs = kwargs

    @property
    def backend(self) -> InferenceBackend:
        """Get or create the inference backend."""
        if self._backend is None:
            self._backend = self._create_backend()
        return self._backend

    def _create_backend(self) -> InferenceBackend:
        """Create the appropriate inference backend."""
        backend_name = self.backend_name

        # Auto-select backend
        if backend_name == "auto":
            try:
                import vllm
                backend_name = "vllm"
                logger.info("Auto-selected vLLM backend")
            except ImportError:
                backend_name = "transformers"
                logger.info("Auto-selected Transformers backend")

        # Create backend
        if backend_name == "vllm":
            from largeforge.inference.vllm_backend import VLLMBackend
            return VLLMBackend(self.model_path, **self._backend_kwargs)
        elif backend_name == "transformers":
            from largeforge.inference.transformers_backend import TransformersBackend
            return TransformersBackend(self.model_path, **self._backend_kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend_name}")

    def load(self) -> "TextGenerator":
        """Load the model."""
        self.backend.load()
        return self

    def unload(self) -> None:
        """Unload the model."""
        if self._backend is not None:
            self._backend.unload()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            **kwargs: Additional generation parameters

        Returns:
            Generated text

        Example:
            >>> generator = TextGenerator("gpt2").load()
            >>> text = generator.generate("Once upon a time")
        """
        request = GenerationRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            **kwargs,
        )

        response = self.backend.generate(request)
        return response.text

    def generate_with_info(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> GenerationResponse:
        """
        Generate text and return full response with metadata.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            Full generation response
        """
        request = GenerationRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        return self.backend.generate(request)

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> Iterator[str]:
        """
        Generate text with streaming output.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Yields:
            Generated text chunks

        Example:
            >>> for chunk in generator.generate_stream("Tell me a story"):
            ...     print(chunk, end="", flush=True)
        """
        request = GenerationRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            **kwargs,
        )

        for response in self.backend.generate_stream(request):
            if response.delta:
                yield response.delta

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> List[str]:
        """
        Generate text for multiple prompts.

        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            List of generated texts
        """
        requests = [
            GenerationRequest(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            for prompt in prompts
        ]

        responses = self.backend.generate_batch(requests)
        return [r.text for r in responses]

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """
        Generate response for chat messages.

        Args:
            messages: List of chat messages [{"role": "...", "content": "..."}]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            Assistant response

        Example:
            >>> messages = [
            ...     {"role": "user", "content": "Hello!"}
            ... ]
            >>> response = generator.chat(messages)
        """
        # Format messages into prompt
        if hasattr(self.backend, "tokenizer") and self.backend.tokenizer is not None:
            tokenizer = self.backend.tokenizer
            if hasattr(tokenizer, "apply_chat_template"):
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = self._format_messages_simple(messages)
        else:
            prompt = self._format_messages_simple(messages)

        return self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    def _format_messages_simple(self, messages: List[Dict[str, str]]) -> str:
        """Simple message formatting fallback."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|{role}|>\n{content}")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return self.backend.count_tokens(text)

    def get_info(self) -> Dict[str, Any]:
        """Get generator and model information."""
        return {
            "model_path": self.model_path,
            "backend": self.backend_name,
            **self.backend.get_model_info(),
        }

    def __enter__(self) -> "TextGenerator":
        """Context manager entry."""
        self.load()
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.unload()


def generate(
    prompt: str,
    model_path: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    backend: str = "auto",
    **kwargs,
) -> str:
    """
    One-shot text generation.

    Args:
        prompt: Input prompt
        model_path: Path to model
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        backend: Backend to use
        **kwargs: Additional parameters

    Returns:
        Generated text

    Example:
        >>> text = generate("Hello, world!", model_path="gpt2")
    """
    with TextGenerator(model_path, backend=backend, **kwargs) as generator:
        return generator.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )


def generate_batch(
    prompts: List[str],
    model_path: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    backend: str = "auto",
    **kwargs,
) -> List[str]:
    """
    One-shot batch text generation.

    Args:
        prompts: List of input prompts
        model_path: Path to model
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        backend: Backend to use
        **kwargs: Additional parameters

    Returns:
        List of generated texts
    """
    with TextGenerator(model_path, backend=backend, **kwargs) as generator:
        return generator.generate_batch(
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
        )
