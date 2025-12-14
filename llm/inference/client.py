"""Client for interacting with inference servers."""

from typing import Literal

import httpx


class InferenceClient:
    """Client for the LargeForgeAI inference server."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._client.close()

    def close(self):
        """Close the client."""
        self._client.close()

    def health(self) -> dict:
        """Check server health."""
        response = self._client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def generate(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
    ) -> dict:
        """
        Generate text from prompt or messages.

        Args:
            prompt: Text prompt for generation
            messages: List of chat messages (role, content dicts)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences

        Returns:
            Dict with 'text', 'tokens_used', 'finish_reason'
        """
        payload = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if prompt:
            payload["prompt"] = prompt
        if messages:
            payload["messages"] = messages
        if stop:
            payload["stop"] = stop

        response = self._client.post(f"{self.base_url}/generate", json=payload)
        response.raise_for_status()
        return response.json()

    def chat(
        self,
        message: str,
        system: str | None = None,
        history: list[dict] | None = None,
        **kwargs,
    ) -> str:
        """
        Simple chat interface.

        Args:
            message: User message
            system: Optional system prompt
            history: Previous conversation history
            **kwargs: Additional generation parameters

        Returns:
            Assistant response text
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": message})

        result = self.generate(messages=messages, **kwargs)
        return result["text"]


class AsyncInferenceClient:
    """Async client for the LargeForgeAI inference server."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self._client.aclose()

    async def close(self):
        """Close the client."""
        await self._client.aclose()

    async def health(self) -> dict:
        """Check server health."""
        response = await self._client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    async def generate(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
    ) -> dict:
        """Generate text from prompt or messages."""
        payload = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if prompt:
            payload["prompt"] = prompt
        if messages:
            payload["messages"] = messages
        if stop:
            payload["stop"] = stop

        response = await self._client.post(f"{self.base_url}/generate", json=payload)
        response.raise_for_status()
        return response.json()

    async def chat(
        self,
        message: str,
        system: str | None = None,
        history: list[dict] | None = None,
        **kwargs,
    ) -> str:
        """Simple chat interface."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": message})

        result = await self.generate(messages=messages, **kwargs)
        return result["text"]


class RouterClient:
    """Client for the LargeForgeAI router service."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._client.close()

    def close(self):
        """Close the client."""
        self._client.close()

    def list_experts(self) -> list[dict]:
        """List available experts."""
        response = self._client.get(f"{self.base_url}/experts")
        response.raise_for_status()
        return response.json()["experts"]

    def route(self, query: str) -> dict:
        """Get routing decision without generating."""
        response = self._client.post(f"{self.base_url}/route", params={"query": query})
        response.raise_for_status()
        return response.json()

    def generate(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        expert: str | None = None,
        **kwargs,
    ) -> dict:
        """Generate using automatic expert routing."""
        payload = {**kwargs}
        if prompt:
            payload["prompt"] = prompt
        if messages:
            payload["messages"] = messages
        if expert:
            payload["expert"] = expert

        response = self._client.post(f"{self.base_url}/generate", json=payload)
        response.raise_for_status()
        return response.json()
