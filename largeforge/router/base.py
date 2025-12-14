"""Base classes for routing system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class Expert:
    """Represents an expert/model that can handle specific types of requests.

    Attributes:
        name: Unique identifier for the expert
        description: Human-readable description of expert capabilities
        keywords: Keywords associated with this expert
        examples: Example queries this expert handles well
        metadata: Additional metadata about the expert
    """

    name: str
    description: str = ""
    keywords: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "keywords": self.keywords,
            "examples": self.examples,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Expert":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            keywords=data.get("keywords", []),
            examples=data.get("examples", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RoutingResult:
    """Result of routing a query to experts.

    Attributes:
        query: The original query
        expert: Selected expert
        score: Confidence score (0-1)
        all_scores: Scores for all experts
        method: Routing method used
        metadata: Additional routing metadata
    """

    query: str
    expert: Expert
    score: float
    all_scores: Dict[str, float] = field(default_factory=dict)
    method: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def expert_name(self) -> str:
        """Get the selected expert's name."""
        return self.expert.name

    @property
    def confidence(self) -> str:
        """Get confidence level as string."""
        if self.score >= 0.8:
            return "high"
        elif self.score >= 0.5:
            return "medium"
        else:
            return "low"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "expert": self.expert.to_dict(),
            "score": self.score,
            "all_scores": self.all_scores,
            "method": self.method,
            "metadata": self.metadata,
        }


class Router(ABC):
    """Abstract base class for routers.

    Routers determine which expert is best suited to handle
    a given query based on various criteria.
    """

    def __init__(self, experts: Optional[List[Expert]] = None):
        """Initialize the router.

        Args:
            experts: List of available experts
        """
        self._experts: Dict[str, Expert] = {}
        if experts:
            for expert in experts:
                self.add_expert(expert)

    @property
    def experts(self) -> List[Expert]:
        """Get list of all experts."""
        return list(self._experts.values())

    @property
    def expert_names(self) -> List[str]:
        """Get list of expert names."""
        return list(self._experts.keys())

    def add_expert(self, expert: Expert) -> None:
        """Add an expert to the router.

        Args:
            expert: Expert to add
        """
        self._experts[expert.name] = expert
        self._on_expert_added(expert)

    def remove_expert(self, name: str) -> Optional[Expert]:
        """Remove an expert from the router.

        Args:
            name: Name of expert to remove

        Returns:
            Removed expert or None if not found
        """
        expert = self._experts.pop(name, None)
        if expert:
            self._on_expert_removed(expert)
        return expert

    def get_expert(self, name: str) -> Optional[Expert]:
        """Get expert by name.

        Args:
            name: Expert name

        Returns:
            Expert or None if not found
        """
        return self._experts.get(name)

    def _on_expert_added(self, expert: Expert) -> None:
        """Hook called when an expert is added.

        Override to perform initialization when expert is added.

        Args:
            expert: The added expert
        """
        pass

    def _on_expert_removed(self, expert: Expert) -> None:
        """Hook called when an expert is removed.

        Override to perform cleanup when expert is removed.

        Args:
            expert: The removed expert
        """
        pass

    @abstractmethod
    def route(self, query: str) -> RoutingResult:
        """Route a query to the best expert.

        Args:
            query: The query to route

        Returns:
            Routing result with selected expert and scores
        """
        pass

    def route_batch(self, queries: List[str]) -> List[RoutingResult]:
        """Route multiple queries.

        Args:
            queries: List of queries to route

        Returns:
            List of routing results
        """
        return [self.route(query) for query in queries]

    def get_top_k(self, query: str, k: int = 3) -> List[RoutingResult]:
        """Get top-k experts for a query.

        Args:
            query: The query to route
            k: Number of top experts to return

        Returns:
            List of routing results sorted by score
        """
        # Default implementation routes and returns top-k
        result = self.route(query)
        sorted_scores = sorted(
            result.all_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        results = []
        for name, score in sorted_scores:
            expert = self.get_expert(name)
            if expert:
                results.append(
                    RoutingResult(
                        query=query,
                        expert=expert,
                        score=score,
                        all_scores=result.all_scores,
                        method=result.method,
                        metadata=result.metadata,
                    )
                )

        return results

    def get_info(self) -> Dict[str, Any]:
        """Get router information.

        Returns:
            Dictionary with router details
        """
        return {
            "type": self.__class__.__name__,
            "num_experts": len(self._experts),
            "expert_names": self.expert_names,
        }
