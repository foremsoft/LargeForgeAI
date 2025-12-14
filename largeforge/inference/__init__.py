"""Inference modules for LargeForgeAI."""

from largeforge.inference.base import (
    InferenceBackend,
    GenerationRequest,
    GenerationResponse,
    StreamingResponse,
)
from largeforge.inference.transformers_backend import TransformersBackend
from largeforge.inference.vllm_backend import VLLMBackend
from largeforge.inference.generator import (
    TextGenerator,
    generate,
    generate_batch,
)
from largeforge.inference.server import (
    create_app,
    run_server,
)

__all__ = [
    # Base
    "InferenceBackend",
    "GenerationRequest",
    "GenerationResponse",
    "StreamingResponse",
    # Backends
    "TransformersBackend",
    "VLLMBackend",
    # Generator
    "TextGenerator",
    "generate",
    "generate_batch",
    # Server
    "create_app",
    "run_server",
]
