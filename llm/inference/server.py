"""Inference server for serving LLM models."""

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Optional vLLM import
try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# Fallback to transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


class ChatMessage(BaseModel):
    """Chat message format."""

    role: Literal["system", "user", "assistant"]
    content: str


class GenerateRequest(BaseModel):
    """Request format for generation."""

    prompt: str | None = None
    messages: list[ChatMessage] | None = None
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stop: list[str] | None = None


class GenerateResponse(BaseModel):
    """Response format for generation."""

    text: str
    tokens_used: int | None = None
    finish_reason: str | None = None


class ServerConfig(BaseModel):
    """Server configuration."""

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    use_vllm: bool = True
    quantization: str | None = "awq"  # awq, gptq, or None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096
    device: str = "auto"


# Global state
model = None
tokenizer = None
config: ServerConfig = ServerConfig()


def load_vllm_model(cfg: ServerConfig):
    """Load model using vLLM for high-performance inference."""
    if not VLLM_AVAILABLE:
        raise RuntimeError("vLLM not installed. Install with: pip install vllm")

    return LLM(
        model=cfg.model_name,
        quantization=cfg.quantization,
        tensor_parallel_size=cfg.tensor_parallel_size,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_model_len=cfg.max_model_len,
        trust_remote_code=True,
    )


def load_transformers_model(cfg: ServerConfig):
    """Load model using transformers (fallback)."""
    global tokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine dtype and device
    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    if device != "cuda":
        model = model.to(device)

    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage server lifecycle."""
    global model

    if config.use_vllm and VLLM_AVAILABLE:
        print(f"Loading model with vLLM: {config.model_name}")
        model = load_vllm_model(config)
    else:
        print(f"Loading model with transformers: {config.model_name}")
        model = load_transformers_model(config)

    yield

    # Cleanup
    del model


app = FastAPI(
    title="LargeForgeAI Inference Server",
    description="High-performance LLM inference server",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model": config.model_name}


@app.get("/model")
async def model_info():
    """Get model information."""
    return {
        "model": config.model_name,
        "quantization": config.quantization,
        "use_vllm": config.use_vllm and VLLM_AVAILABLE,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """Generate text from prompt or messages."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build prompt
    if request.messages:
        if tokenizer is not None:
            prompt = tokenizer.apply_chat_template(
                [{"role": m.role, "content": m.content} for m in request.messages],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # For vLLM without separate tokenizer
            prompt = "\n".join([f"{m.role}: {m.content}" for m in request.messages])
            prompt += "\nassistant:"
    elif request.prompt:
        prompt = request.prompt
    else:
        raise HTTPException(status_code=400, detail="Either prompt or messages required")

    # Generate
    if config.use_vllm and VLLM_AVAILABLE:
        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
        )
        outputs = model.generate([prompt], sampling_params)
        output = outputs[0]
        generated_text = output.outputs[0].text
        tokens_used = len(output.outputs[0].token_ids)
        finish_reason = output.outputs[0].finish_reason

    else:
        # Transformers generation
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs.input_ids.shape[1] :]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        tokens_used = len(generated_tokens)
        finish_reason = "stop"

    return GenerateResponse(
        text=generated_text.strip(),
        tokens_used=tokens_used,
        finish_reason=finish_reason,
    )


def run(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    host: str = "0.0.0.0",
    port: int = 8000,
    use_vllm: bool = True,
    quantization: str | None = "awq",
):
    """Run the inference server."""
    import uvicorn

    global config
    config = ServerConfig(
        model_name=model_name,
        use_vllm=use_vllm,
        quantization=quantization,
    )

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
