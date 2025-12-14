"""Hybrid router combining semantic and keyword-based routing."""

from typing import Any, Callable, Dict, List, Optional

from largeforge.router.base import Router, Expert, RoutingResult
from largeforge.router.semantic import SemanticRouter
from largeforge.router.keyword import KeywordRouter
from largeforge.config import RouterConfig
from largeforge.utils import get_logger

logger = get_logger(__name__)


class HybridRouter(Router):
    """Router that combines semantic and keyword-based routing.

    Weights the scores from both methods to determine the best expert.
    """

    def __init__(
        self,
        experts: Optional[List[Expert]] = None,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        config: Optional[RouterConfig] = None,
        case_sensitive: bool = False,
        similarity_threshold: float = 0.3,
        default_expert: Optional[str] = None,
    ):
        """Initialize the hybrid router.

        Args:
            experts: List of available experts
            semantic_weight: Weight for semantic similarity scores (0-1)
            keyword_weight: Weight for keyword match scores (0-1)
            embedding_fn: Function to generate embeddings
            config: Optional router configuration
            case_sensitive: Whether keyword matching is case-sensitive
            similarity_threshold: Minimum combined score threshold
            default_expert: Default expert when no good match found
        """
        # Apply config if provided
        if config is not None:
            semantic_weight = config.neural_weight
            keyword_weight = config.keyword_weight
            similarity_threshold = config.confidence_threshold

        # Normalize weights
        total_weight = semantic_weight + keyword_weight
        if total_weight > 0:
            self.semantic_weight = semantic_weight / total_weight
            self.keyword_weight = keyword_weight / total_weight
        else:
            self.semantic_weight = 0.5
            self.keyword_weight = 0.5

        self.similarity_threshold = similarity_threshold
        self.default_expert_name = default_expert

        # Initialize component routers (without experts - we'll add them later)
        self._semantic_router = SemanticRouter(
            embedding_fn=embedding_fn,
            similarity_threshold=0.0,  # We handle threshold ourselves
        )

        self._keyword_router = KeywordRouter(
            case_sensitive=case_sensitive,
            default_expert=None,  # We handle default ourselves
        )

        # Initialize base class (this adds experts)
        super().__init__(experts)

    def _on_expert_added(self, expert: Expert) -> None:
        """Add expert to component routers."""
        self._semantic_router.add_expert(expert)
        self._keyword_router.add_expert(expert)

    def _on_expert_removed(self, expert: Expert) -> None:
        """Remove expert from component routers."""
        self._semantic_router.remove_expert(expert.name)
        self._keyword_router.remove_expert(expert.name)

    def route(self, query: str) -> RoutingResult:
        """Route a query using combined semantic and keyword scores.

        Args:
            query: The query to route

        Returns:
            Routing result with selected expert and combined scores
        """
        if not self.experts:
            raise ValueError("No experts registered")

        # Get results from both routers
        semantic_result = self._semantic_router.route(query)
        keyword_result = self._keyword_router.route(query)

        # Combine scores
        all_scores: Dict[str, float] = {}
        semantic_scores: Dict[str, float] = {}
        keyword_scores: Dict[str, float] = {}

        for expert in self.experts:
            name = expert.name
            sem_score = semantic_result.all_scores.get(name, 0.0)
            kw_score = keyword_result.all_scores.get(name, 0.0)

            combined = (
                sem_score * self.semantic_weight +
                kw_score * self.keyword_weight
            )

            all_scores[name] = combined
            semantic_scores[name] = sem_score
            keyword_scores[name] = kw_score

        # Find best expert
        best_name = max(all_scores, key=all_scores.get)  # type: ignore
        best_score = all_scores[best_name]

        # Check threshold
        if best_score < self.similarity_threshold:
            logger.warning(
                f"Best combined score {best_score:.3f} below threshold "
                f"{self.similarity_threshold:.3f}"
            )

            if self.default_expert_name:
                default_expert = self.get_expert(self.default_expert_name)
                if default_expert:
                    return RoutingResult(
                        query=query,
                        expert=default_expert,
                        score=best_score,
                        all_scores=all_scores,
                        method="hybrid",
                        metadata={
                            "semantic_scores": semantic_scores,
                            "keyword_scores": keyword_scores,
                            "semantic_weight": self.semantic_weight,
                            "keyword_weight": self.keyword_weight,
                            "used_default": True,
                            "below_threshold": True,
                        },
                    )

        best_expert = self.get_expert(best_name)

        return RoutingResult(
            query=query,
            expert=best_expert,  # type: ignore
            score=best_score,
            all_scores=all_scores,
            method="hybrid",
            metadata={
                "semantic_scores": semantic_scores,
                "keyword_scores": keyword_scores,
                "semantic_weight": self.semantic_weight,
                "keyword_weight": self.keyword_weight,
                "used_default": False,
                "below_threshold": best_score < self.similarity_threshold,
            },
        )

    def route_semantic_only(self, query: str) -> RoutingResult:
        """Route using only semantic similarity.

        Args:
            query: The query to route

        Returns:
            Routing result from semantic router
        """
        return self._semantic_router.route(query)

    def route_keyword_only(self, query: str) -> RoutingResult:
        """Route using only keyword matching.

        Args:
            query: The query to route

        Returns:
            Routing result from keyword router
        """
        return self._keyword_router.route(query)

    def set_weights(self, semantic_weight: float, keyword_weight: float) -> None:
        """Update the routing weights.

        Args:
            semantic_weight: New semantic weight
            keyword_weight: New keyword weight
        """
        total = semantic_weight + keyword_weight
        if total > 0:
            self.semantic_weight = semantic_weight / total
            self.keyword_weight = keyword_weight / total
        else:
            raise ValueError("Weights cannot both be zero")

        logger.info(
            f"Updated weights: semantic={self.semantic_weight:.2f}, "
            f"keyword={self.keyword_weight:.2f}"
        )

    def add_keywords(self, expert_name: str, keywords: List[str]) -> None:
        """Add keywords to an expert.

        Args:
            expert_name: Name of the expert
            keywords: Keywords to add
        """
        self._keyword_router.add_keywords(expert_name, keywords)

        # Also update the base expert
        expert = self.get_expert(expert_name)
        if expert:
            expert.keywords.extend(keywords)

    def update_embeddings(self) -> None:
        """Recompute all semantic embeddings."""
        self._semantic_router.update_embeddings()

    def get_component_results(
        self, query: str
    ) -> Dict[str, RoutingResult]:
        """Get routing results from each component router.

        Args:
            query: The query to route

        Returns:
            Dictionary with 'semantic', 'keyword', and 'hybrid' results
        """
        return {
            "semantic": self._semantic_router.route(query),
            "keyword": self._keyword_router.route(query),
            "hybrid": self.route(query),
        }

    def get_info(self) -> Dict[str, Any]:
        """Get router information."""
        info = super().get_info()
        info.update({
            "method": "hybrid",
            "semantic_weight": self.semantic_weight,
            "keyword_weight": self.keyword_weight,
            "similarity_threshold": self.similarity_threshold,
            "default_expert": self.default_expert_name,
            "semantic_router": self._semantic_router.get_info(),
            "keyword_router": self._keyword_router.get_info(),
        })
        return info

    @classmethod
    def from_config(
        cls,
        config: RouterConfig,
        experts: Optional[List[Expert]] = None,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
    ) -> "HybridRouter":
        """Create a hybrid router from configuration.

        Args:
            config: Router configuration
            experts: List of experts
            embedding_fn: Embedding function

        Returns:
            Configured HybridRouter instance
        """
        return cls(
            experts=experts,
            config=config,
            embedding_fn=embedding_fn,
        )
