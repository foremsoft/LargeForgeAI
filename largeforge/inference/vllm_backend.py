"""vLLM-based inference backend for high-throughput serving."""

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from largeforge.inference.base import (
    AsyncInferenceBackend,
    GenerationRequest,
    GenerationResponse,
    StreamingResponse,
)
from largeforge.utils import get_logger

logger = get_logger(__name__)


class VLLMBackend(AsyncInferenceBackend):
    """Inference backend using vLLM for high-throughput serving."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        quantization: Optional[str] = None,
        enforce_eager: bool = False,
    ):
        """
        Initialize the vLLM backend.

        Args:
            model_path: Path to model or HuggingFace model ID
            device: Device to run on (vLLM manages this)
            dtype: Data type
            trust_remote_code: Whether to trust remote code
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum context length
            quantization: Quantization method (awq, gptq, squeezellm)
            enforce_eager: Disable CUDA graphs for debugging
        """
        super().__init__(model_path, device, dtype, trust_remote_code)
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.quantization = quantization
        self.enforce_eager = enforce_eager

        self.engine = None
        self.sampling_params_class = None

    def load(self) -> None:
        """Load the model using vLLM."""
        if self._is_loaded:
            logger.info("Model already loaded")
            return

        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vllm is required for VLLMBackend. "
                "Install with: pip install vllm"
            )

        logger.info(f"Loading model with vLLM: {self.model_path}")

        # Prepare engine args
        engine_kwargs = {
            "model": self.model_path,
            "trust_remote_code": self.trust_remote_code,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enforce_eager": self.enforce_eager,
        }

        if self.dtype != "auto":
            engine_kwargs["dtype"] = self.dtype

        if self.max_model_len is not None:
            engine_kwargs["max_model_len"] = self.max_model_len

        if self.quantization is not None:
            engine_kwargs["quantization"] = self.quantization

        # Create engine
        self.engine = LLM(**engine_kwargs)
        self.sampling_params_class = SamplingParams

        # Get tokenizer from engine
        self.tokenizer = self.engine.get_tokenizer()

        self._is_loaded = True
        logger.info("Model loaded with vLLM")

    def unload(self) -> None:
        """Unload the model."""
        if not self._is_loaded:
            return

        del self.engine
        self.engine = None
        self.tokenizer = None

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._is_loaded = False
        logger.info("Model unloaded")

    def _create_sampling_params(self, request: GenerationRequest):
        """Create vLLM SamplingParams from request."""
        params = {
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k if request.top_k > 0 else -1,
            "repetition_penalty": request.repetition_penalty,
            "n": request.n,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
        }

        if request.stop:
            params["stop"] = request.stop

        if request.logprobs is not None:
            params["logprobs"] = request.logprobs

        return self.sampling_params_class(**params)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate text from a prompt.

        Args:
            request: Generation request

        Returns:
            Generation response
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        sampling_params = self._create_sampling_params(request)

        # Generate
        outputs = self.engine.generate([request.prompt], sampling_params)

        # Process output
        output = outputs[0]
        completion = output.outputs[0]

        # Get token counts
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = len(completion.token_ids)

        return GenerationResponse(
            text=completion.text,
            prompt=request.prompt,
            finish_reason=completion.finish_reason,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            model=self.model_path,
            logprobs=list(completion.logprobs) if completion.logprobs else None,
        )

    def generate_stream(
        self, request: GenerationRequest
    ) -> Iterator[StreamingResponse]:
        """
        Generate text with streaming output.

        Note: This implementation collects tokens and yields them.
        For true streaming, use async methods.

        Args:
            request: Generation request

        Yields:
            Streaming response chunks
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # vLLM's synchronous API doesn't support true streaming
        # We generate and yield token by token
        response = self.generate(request)

        # Simulate streaming by yielding characters
        accumulated = ""
        for char in response.text:
            accumulated += char
            yield StreamingResponse(
                delta=char,
                accumulated_text=accumulated,
                is_final=False,
            )

        yield StreamingResponse(
            delta="",
            finish_reason=response.finish_reason,
            is_final=True,
            accumulated_text=accumulated,
            token_count=response.completion_tokens,
        )

    def generate_batch(
        self, requests: List[GenerationRequest]
    ) -> List[GenerationResponse]:
        """
        Generate text for multiple requests efficiently.

        vLLM excels at batch processing.

        Args:
            requests: List of generation requests

        Returns:
            List of generation responses
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not requests:
            return []

        prompts = [req.prompt for req in requests]

        # Use first request's params (in production, group by params)
        sampling_params = self._create_sampling_params(requests[0])

        # Generate batch
        outputs = self.engine.generate(prompts, sampling_params)

        # Process outputs
        responses = []
        for output, request in zip(outputs, requests):
            completion = output.outputs[0]
            prompt_tokens = len(output.prompt_token_ids)
            completion_tokens = len(completion.token_ids)

            responses.append(GenerationResponse(
                text=completion.text,
                prompt=request.prompt,
                finish_reason=completion.finish_reason,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                model=self.model_path,
            ))

        return responses

    async def generate_async(self, request: GenerationRequest) -> GenerationResponse:
        """
        Asynchronously generate text.

        Args:
            request: Generation request

        Returns:
            Generation response
        """
        # vLLM's LLM class is synchronous
        # For async, use the AsyncLLMEngine
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None, self.generate, request
        )

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
        # Wrap synchronous streaming in async
        import asyncio

        for chunk in self.generate_stream(request):
            yield chunk
            await asyncio.sleep(0)  # Allow other tasks to run

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = super().get_model_info()

        info.update({
            "backend": "vllm",
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "quantization": self.quantization,
        })

        return info


class AsyncVLLMBackend(VLLMBackend):
    """Fully async vLLM backend using AsyncLLMEngine."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.async_engine = None

    def load(self) -> None:
        """Load the model with async engine."""
        try:
            from vllm import AsyncLLMEngine, AsyncEngineArgs
        except ImportError:
            raise ImportError("vllm required for AsyncVLLMBackend")

        logger.info(f"Loading async vLLM engine: {self.model_path}")

        engine_args = AsyncEngineArgs(
            model=self.model_path,
            trust_remote_code=self.trust_remote_code,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enforce_eager=self.enforce_eager,
        )

        if self.dtype != "auto":
            engine_args.dtype = self.dtype

        if self.max_model_len is not None:
            engine_args.max_model_len = self.max_model_len

        if self.quantization is not None:
            engine_args.quantization = self.quantization

        self.async_engine = AsyncLLMEngine.from_engine_args(engine_args)

        from vllm import SamplingParams
        self.sampling_params_class = SamplingParams

        self._is_loaded = True
        logger.info("Async vLLM engine loaded")

    async def generate_async(self, request: GenerationRequest) -> GenerationResponse:
        """Truly async generation."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")

        sampling_params = self._create_sampling_params(request)

        import uuid
        request_id = str(uuid.uuid4())

        final_output = None
        async for output in self.async_engine.generate(
            request.prompt, sampling_params, request_id
        ):
            final_output = output

        completion = final_output.outputs[0]
        prompt_tokens = len(final_output.prompt_token_ids)
        completion_tokens = len(completion.token_ids)

        return GenerationResponse(
            text=completion.text,
            prompt=request.prompt,
            finish_reason=completion.finish_reason,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            model=self.model_path,
        )

    async def generate_stream_async(
        self, request: GenerationRequest
    ) -> AsyncIterator[StreamingResponse]:
        """Truly async streaming generation."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")

        sampling_params = self._create_sampling_params(request)

        import uuid
        request_id = str(uuid.uuid4())

        accumulated = ""
        prev_text = ""

        async for output in self.async_engine.generate(
            request.prompt, sampling_params, request_id
        ):
            completion = output.outputs[0]
            new_text = completion.text

            if new_text != prev_text:
                delta = new_text[len(prev_text):]
                accumulated = new_text
                prev_text = new_text

                yield StreamingResponse(
                    delta=delta,
                    accumulated_text=accumulated,
                    is_final=False,
                )

        yield StreamingResponse(
            delta="",
            finish_reason="stop",
            is_final=True,
            accumulated_text=accumulated,
        )
