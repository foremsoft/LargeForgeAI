"""LargeForgeAI - Low-cost LLM training and deployment stack."""

__version__ = "0.1.0"

from llm import data, experts, inference, router, training

__all__ = [
    "__version__",
    "data",
    "training",
    "experts",
    "router",
    "inference",
]
