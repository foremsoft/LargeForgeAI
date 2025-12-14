"""Router module for intelligent request routing.

This module provides expert routing capabilities for directing
requests to the most appropriate model or expert based on
semantic similarity, keywords, or hybrid approaches.
"""

from largeforge.router.base import (
    Router,
    RoutingResult,
    Expert,
)
from largeforge.router.semantic import SemanticRouter
from largeforge.router.keyword import KeywordRouter
from largeforge.router.hybrid import HybridRouter

__all__ = [
    # Base classes
    "Router",
    "RoutingResult",
    "Expert",
    # Router implementations
    "SemanticRouter",
    "KeywordRouter",
    "HybridRouter",
]
