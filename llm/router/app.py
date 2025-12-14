"""FastAPI router service for expert model routing."""

import os
from contextlib import asynccontextmanager
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llm.router.classifier import DEFAULT_EXPERTS, ExpertConfig, HybridClassifier


class ChatMessage(BaseModel):
    """Chat message format."""

    role: Literal["system", "user", "assistant"]
    content: str


class GenerateRequest(BaseModel):
    """Request format for text generation."""

    prompt: str | None = None
    messages: list[ChatMessage] | None = None
    max_tokens: int = 512
    temperature: float = 0.7
    expert: str | None = None  # Override automatic routing


class GenerateResponse(BaseModel):
    """Response format for text generation."""

    text: str
    expert: str
    tokens_used: int | None = None


class RouterConfig(BaseModel):
    """Router configuration."""

    experts: list[ExpertConfig] = DEFAULT_EXPERTS
    neural_model: str | None = None
    timeout: float = 60.0


# Global state
classifier: HybridClassifier | None = None
http_client: httpx.AsyncClient | None = None
config: RouterConfig = RouterConfig()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage router lifecycle."""
    global classifier, http_client

    # Initialize
    classifier = HybridClassifier(
        experts=config.experts,
        neural_model=config.neural_model,
    )
    http_client = httpx.AsyncClient(timeout=config.timeout)

    yield

    # Cleanup
    await http_client.aclose()


app = FastAPI(
    title="LargeForgeAI Router",
    description="Routes queries to specialized expert models",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/experts")
async def list_experts():
    """List available expert models."""
    return {
        "experts": [
            {"name": e.name, "description": e.description, "endpoint": e.endpoint}
            for e in config.experts
        ]
    }


@app.post("/route")
async def route_query(query: str) -> dict:
    """
    Route a query to the appropriate expert.

    Returns the expert name without generating a response.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not initialized")

    expert = classifier.classify(query)
    endpoint = classifier.get_endpoint(expert)

    return {"expert": expert, "endpoint": endpoint}


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """
    Generate a response by routing to the appropriate expert.
    """
    if classifier is None or http_client is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Determine query text for classification
    if request.prompt:
        query_text = request.prompt
    elif request.messages:
        # Use the last user message for classification
        user_messages = [m for m in request.messages if m.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")
        query_text = user_messages[-1].content
    else:
        raise HTTPException(status_code=400, detail="Either prompt or messages required")

    # Route to expert
    if request.expert:
        expert_name = request.expert
    else:
        expert_name = classifier.classify(query_text)

    endpoint = classifier.get_endpoint(expert_name)
    if not endpoint:
        raise HTTPException(status_code=404, detail=f"Expert '{expert_name}' not found")

    # Forward request to expert
    try:
        response = await http_client.post(
            endpoint,
            json={
                "prompt": request.prompt,
                "messages": [m.model_dump() for m in request.messages] if request.messages else None,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            },
        )
        response.raise_for_status()
        result = response.json()

        return GenerateResponse(
            text=result.get("text", result.get("generated_text", "")),
            expert=expert_name,
            tokens_used=result.get("tokens_used"),
        )

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail=f"Expert '{expert_name}' timed out")
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Expert '{expert_name}' error: {e.response.text}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling expert: {str(e)}")


def run(host: str = "0.0.0.0", port: int = 8080):
    """Run the router server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
