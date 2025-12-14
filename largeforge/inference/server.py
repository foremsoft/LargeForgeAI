"""FastAPI-based inference server."""

from typing import Any, Dict, List, Optional

from largeforge.config import InferenceConfig, GenerationConfig
from largeforge.inference.base import GenerationRequest, GenerationResponse
from largeforge.inference.generator import TextGenerator
from largeforge.utils import get_logger

logger = get_logger(__name__)


def create_app(
    model_path: str,
    backend: str = "auto",
    config: Optional[InferenceConfig] = None,
):
    """
    Create a FastAPI application for inference.

    Args:
        model_path: Path to model or HuggingFace model ID
        backend: Backend to use ("auto", "transformers", "vllm")
        config: Optional inference configuration

    Returns:
        FastAPI application

    Example:
        >>> app = create_app("meta-llama/Llama-2-7b-hf")
        >>> import uvicorn
        >>> uvicorn.run(app, host="0.0.0.0", port=8000)
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel, Field
    except ImportError:
        raise ImportError(
            "fastapi is required for the inference server. "
            "Install with: pip install fastapi uvicorn"
        )

    # Define request/response models
    class CompletionRequest(BaseModel):
        """OpenAI-compatible completion request."""

        model: str = Field(default="default")
        prompt: str
        max_tokens: int = Field(default=256, ge=1, le=4096)
        temperature: float = Field(default=0.7, ge=0, le=2)
        top_p: float = Field(default=0.9, gt=0, le=1)
        n: int = Field(default=1, ge=1, le=10)
        stream: bool = False
        stop: Optional[List[str]] = None
        presence_penalty: float = Field(default=0, ge=-2, le=2)
        frequency_penalty: float = Field(default=0, ge=-2, le=2)

    class ChatMessage(BaseModel):
        """Chat message."""

        role: str
        content: str

    class ChatCompletionRequest(BaseModel):
        """OpenAI-compatible chat completion request."""

        model: str = Field(default="default")
        messages: List[ChatMessage]
        max_tokens: int = Field(default=256, ge=1, le=4096)
        temperature: float = Field(default=0.7, ge=0, le=2)
        top_p: float = Field(default=0.9, gt=0, le=1)
        stream: bool = False
        stop: Optional[List[str]] = None

    class Choice(BaseModel):
        """Completion choice."""

        index: int
        text: str
        finish_reason: str

    class ChatChoice(BaseModel):
        """Chat completion choice."""

        index: int
        message: ChatMessage
        finish_reason: str

    class Usage(BaseModel):
        """Token usage."""

        prompt_tokens: int
        completion_tokens: int
        total_tokens: int

    class CompletionResponse(BaseModel):
        """OpenAI-compatible completion response."""

        id: str
        object: str = "text_completion"
        created: int
        model: str
        choices: List[Choice]
        usage: Usage

    class ChatCompletionResponse(BaseModel):
        """OpenAI-compatible chat completion response."""

        id: str
        object: str = "chat.completion"
        created: int
        model: str
        choices: List[ChatChoice]
        usage: Usage

    # Create app
    app = FastAPI(
        title="LargeForgeAI Inference Server",
        description="OpenAI-compatible API for LLM inference",
        version="1.0.0",
    )

    # Initialize generator
    generator_kwargs = {}
    if config is not None:
        generator_kwargs.update({
            "dtype": config.dtype,
            "trust_remote_code": config.trust_remote_code,
        })

    generator = TextGenerator(model_path, backend=backend, **generator_kwargs)

    @app.on_event("startup")
    async def startup():
        """Load model on startup."""
        logger.info("Loading model...")
        generator.load()
        logger.info("Model loaded, server ready")

    @app.on_event("shutdown")
    async def shutdown():
        """Unload model on shutdown."""
        logger.info("Shutting down, unloading model...")
        generator.unload()

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "model_loaded": generator.backend.is_loaded}

    @app.get("/v1/models")
    async def list_models():
        """List available models."""
        return {
            "object": "list",
            "data": [
                {
                    "id": model_path,
                    "object": "model",
                    "owned_by": "largeforge",
                }
            ],
        }

    @app.post("/v1/completions", response_model=CompletionResponse)
    async def create_completion(request: CompletionRequest):
        """Create a completion."""
        import time
        import uuid

        gen_request = GenerationRequest(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            n=request.n,
        )

        try:
            response = generator.backend.generate(gen_request)
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=model_path,
            choices=[
                Choice(
                    index=0,
                    text=response.text,
                    finish_reason=response.finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
            ),
        )

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def create_chat_completion(request: ChatCompletionRequest):
        """Create a chat completion."""
        import time
        import uuid

        # Convert messages to list of dicts
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        try:
            response_text = generator.chat(
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
            )
        except Exception as e:
            logger.error(f"Chat generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        # Estimate token counts (simplified)
        prompt_tokens = sum(len(m.content.split()) for m in request.messages)
        completion_tokens = len(response_text.split())

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=model_path,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    @app.post("/generate")
    async def generate_simple(
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ):
        """Simple generation endpoint."""
        try:
            text = generator.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return {"text": text, "prompt": prompt}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


def run_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    backend: str = "auto",
    config: Optional[InferenceConfig] = None,
    workers: int = 1,
    reload: bool = False,
):
    """
    Run the inference server.

    Args:
        model_path: Path to model
        host: Host to bind to
        port: Port to bind to
        backend: Backend to use
        config: Optional inference configuration
        workers: Number of workers
        reload: Enable auto-reload for development

    Example:
        >>> run_server("gpt2", port=8000)
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn is required. Install with: pip install uvicorn")

    app = create_app(model_path, backend=backend, config=config)

    logger.info(f"Starting server at {host}:{port}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Backend: {backend}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers,
        reload=reload,
    )
