"""Transformers-based inference backend."""

from typing import Any, Dict, Iterator, List, Optional

import torch

from largeforge.inference.base import (
    InferenceBackend,
    GenerationRequest,
    GenerationResponse,
    StreamingResponse,
)
from largeforge.utils import get_logger, get_device, get_optimal_dtype

logger = get_logger(__name__)


class TransformersBackend(InferenceBackend):
    """Inference backend using HuggingFace Transformers."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_flash_attention: bool = True,
    ):
        """
        Initialize the Transformers backend.

        Args:
            model_path: Path to model or HuggingFace model ID
            device: Device to run on
            dtype: Data type
            trust_remote_code: Whether to trust remote code
            load_in_8bit: Load model in 8-bit quantization
            load_in_4bit: Load model in 4-bit quantization
            use_flash_attention: Use Flash Attention 2 if available
        """
        super().__init__(model_path, device, dtype, trust_remote_code)
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.use_flash_attention = use_flash_attention
        self.streamer = None

    def load(self) -> None:
        """Load the model and tokenizer."""
        if self._is_loaded:
            logger.info("Model already loaded")
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required. Install with: pip install transformers"
            )

        logger.info(f"Loading model from {self.model_path}")

        # Determine device
        if self.device == "auto":
            device_str = get_device()
        else:
            device_str = self.device

        # Determine dtype
        if self.dtype == "auto":
            torch_dtype = get_optimal_dtype()
        else:
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(self.dtype, torch.float16)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )

        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Prepare model kwargs
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": self.trust_remote_code,
        }

        # Quantization config
        if self.load_in_4bit or self.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=self.load_in_4bit,
                    load_in_8bit=self.load_in_8bit,
                    bnb_4bit_compute_dtype=torch_dtype,
                )
                model_kwargs["quantization_config"] = quantization_config
            except ImportError:
                logger.warning("bitsandbytes not installed, skipping quantization")

        # Flash attention
        if self.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # Device map
        if device_str == "cuda" or device_str.startswith("cuda:"):
            model_kwargs["device_map"] = device_str if device_str.startswith("cuda:") else "auto"
        else:
            model_kwargs["device_map"] = None

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **model_kwargs,
        )

        # Move to device if not using device_map
        if model_kwargs.get("device_map") is None:
            self.model = self.model.to(device_str)

        self.model.eval()
        self._is_loaded = True

        logger.info(f"Model loaded on {device_str} with dtype {torch_dtype}")

    def unload(self) -> None:
        """Unload the model to free memory."""
        if not self._is_loaded:
            return

        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._is_loaded = False
        logger.info("Model unloaded")

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

        # Tokenize
        inputs = self.tokenizer(
            request.prompt,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        prompt_tokens = inputs.input_ids.shape[1]

        # Prepare generation config
        gen_kwargs = {
            "max_new_tokens": request.max_tokens,
            "temperature": max(request.temperature, 1e-7),  # Avoid division by zero
            "top_p": request.top_p,
            "top_k": request.top_k,
            "repetition_penalty": request.repetition_penalty,
            "do_sample": request.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # Add stop sequences
        if request.stop:
            stop_token_ids = []
            for stop_seq in request.stop:
                tokens = self.tokenizer.encode(stop_seq, add_special_tokens=False)
                if tokens:
                    stop_token_ids.append(tokens[0])
            if stop_token_ids:
                gen_kwargs["eos_token_id"] = stop_token_ids

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
            )

        # Decode
        generated_ids = outputs[0][prompt_tokens:]
        generated_text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )

        # Determine finish reason
        finish_reason = "stop"
        if len(generated_ids) >= request.max_tokens:
            finish_reason = "length"

        return GenerationResponse(
            text=generated_text,
            prompt=request.prompt,
            finish_reason=finish_reason,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(generated_ids),
                "total_tokens": prompt_tokens + len(generated_ids),
            },
            model=self.model_path,
        )

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
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        try:
            from transformers import TextIteratorStreamer
            from threading import Thread
        except ImportError:
            raise ImportError("transformers required for streaming")

        # Tokenize
        inputs = self.tokenizer(
            request.prompt,
            return_tensors="pt",
        ).to(self.model.device)

        prompt_tokens = inputs.input_ids.shape[1]

        # Create streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # Prepare generation config
        gen_kwargs = {
            "max_new_tokens": request.max_tokens,
            "temperature": max(request.temperature, 1e-7),
            "top_p": request.top_p,
            "top_k": request.top_k,
            "repetition_penalty": request.repetition_penalty,
            "do_sample": request.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "streamer": streamer,
        }

        # Start generation in thread
        generation_kwargs = dict(inputs, **gen_kwargs)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield chunks
        accumulated = ""
        token_count = 0

        for text_chunk in streamer:
            accumulated += text_chunk
            token_count += 1

            yield StreamingResponse(
                delta=text_chunk,
                accumulated_text=accumulated,
                token_count=token_count,
                is_final=False,
            )

        # Final response
        yield StreamingResponse(
            delta="",
            finish_reason="stop",
            is_final=True,
            accumulated_text=accumulated,
            token_count=token_count,
        )

        thread.join()

    def generate_batch(
        self, requests: List[GenerationRequest]
    ) -> List[GenerationResponse]:
        """
        Generate text for multiple requests with batching.

        Args:
            requests: List of generation requests

        Returns:
            List of generation responses
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not requests:
            return []

        # For simplicity, use same generation params for all
        # In production, you'd want to group by params
        prompts = [req.prompt for req in requests]

        # Tokenize batch
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        # Use first request's params
        req = requests[0]
        gen_kwargs = {
            "max_new_tokens": req.max_tokens,
            "temperature": max(req.temperature, 1e-7),
            "top_p": req.top_p,
            "do_sample": req.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode and create responses
        responses = []
        for i, (output, request) in enumerate(zip(outputs, requests)):
            prompt_len = inputs.input_ids[i].shape[0]
            generated_ids = output[prompt_len:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            responses.append(GenerationResponse(
                text=text,
                prompt=request.prompt,
                finish_reason="stop",
                usage={
                    "prompt_tokens": prompt_len,
                    "completion_tokens": len(generated_ids),
                    "total_tokens": prompt_len + len(generated_ids),
                },
                model=self.model_path,
            ))

        return responses

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = super().get_model_info()

        if self._is_loaded and self.model is not None:
            info.update({
                "num_parameters": sum(p.numel() for p in self.model.parameters()),
                "vocab_size": self.model.config.vocab_size,
                "hidden_size": getattr(self.model.config, "hidden_size", None),
                "num_layers": getattr(self.model.config, "num_hidden_layers", None),
            })

        return info
