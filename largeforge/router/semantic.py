"""Semantic router using embeddings for similarity-based routing."""

from typing import Any, Dict, List, Optional, Callable
import numpy as np

from largeforge.router.base import Router, Expert, RoutingResult
from largeforge.utils import get_logger

logger = get_logger(__name__)


class SemanticRouter(Router):
    """Router that uses semantic similarity for expert selection.

    Uses embeddings to compute similarity between queries and
    expert descriptions/examples.
    """

    def __init__(
        self,
        experts: Optional[List[Expert]] = None,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        similarity_threshold: float = 0.5,
        use_examples: bool = True,
        use_descriptions: bool = True,
    ):
        """Initialize the semantic router.

        Args:
            experts: List of available experts
            embedding_fn: Function to generate embeddings. If None,
                uses a simple TF-IDF based approach.
            similarity_threshold: Minimum similarity score
            use_examples: Whether to use expert examples for routing
            use_descriptions: Whether to use expert descriptions for routing
        """
        self.embedding_fn = embedding_fn
        self.similarity_threshold = similarity_threshold
        self.use_examples = use_examples
        self.use_descriptions = use_descriptions

        # Cache for expert embeddings
        self._expert_embeddings: Dict[str, np.ndarray] = {}
        self._example_embeddings: Dict[str, List[np.ndarray]] = {}

        # Simple TF-IDF vocabulary for fallback
        self._vocabulary: Dict[str, int] = {}
        self._idf: Optional[np.ndarray] = None

        super().__init__(experts)

    def _on_expert_added(self, expert: Expert) -> None:
        """Compute embeddings when expert is added."""
        self._update_expert_embeddings(expert)

    def _on_expert_removed(self, expert: Expert) -> None:
        """Remove embeddings when expert is removed."""
        self._expert_embeddings.pop(expert.name, None)
        self._example_embeddings.pop(expert.name, None)

    def _update_expert_embeddings(self, expert: Expert) -> None:
        """Update embeddings for an expert."""
        if self.use_descriptions and expert.description:
            self._expert_embeddings[expert.name] = self._get_embedding(
                expert.description
            )

        if self.use_examples and expert.examples:
            self._example_embeddings[expert.name] = [
                self._get_embedding(ex) for ex in expert.examples
            ]

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if self.embedding_fn is not None:
            embedding = self.embedding_fn(text)
            return np.array(embedding)

        # Fallback: simple TF-IDF-like approach
        return self._simple_embedding(text)

    def _simple_embedding(self, text: str) -> np.ndarray:
        """Simple TF-IDF-like embedding for fallback.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Tokenize
        tokens = text.lower().split()

        # Build vocabulary if needed
        if not self._vocabulary:
            self._build_vocabulary()

        # Create sparse vector
        vector = np.zeros(len(self._vocabulary))
        for token in tokens:
            if token in self._vocabulary:
                idx = self._vocabulary[token]
                vector[idx] += 1

        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    def _build_vocabulary(self) -> None:
        """Build vocabulary from all expert descriptions and examples."""
        all_tokens = set()

        for expert in self.experts:
            if expert.description:
                all_tokens.update(expert.description.lower().split())
            for example in expert.examples:
                all_tokens.update(example.lower().split())

        self._vocabulary = {token: idx for idx, token in enumerate(sorted(all_tokens))}

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity (0-1)
        """
        # Handle dimension mismatch
        if len(a) != len(b):
            min_len = min(len(a), len(b))
            a = a[:min_len]
            b = b[:min_len]

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def _compute_similarity(
        self, query_embedding: np.ndarray, expert: Expert
    ) -> float:
        """Compute similarity between query and expert.

        Args:
            query_embedding: Query embedding
            expert: Expert to compare against

        Returns:
            Similarity score (0-1)
        """
        scores = []

        # Compare with description
        if self.use_descriptions and expert.name in self._expert_embeddings:
            desc_sim = self._cosine_similarity(
                query_embedding, self._expert_embeddings[expert.name]
            )
            scores.append(desc_sim)

        # Compare with examples
        if self.use_examples and expert.name in self._example_embeddings:
            for example_emb in self._example_embeddings[expert.name]:
                ex_sim = self._cosine_similarity(query_embedding, example_emb)
                scores.append(ex_sim)

        if not scores:
            return 0.0

        # Return max similarity
        return max(scores)

    def route(self, query: str) -> RoutingResult:
        """Route a query to the best expert based on semantic similarity.

        Args:
            query: The query to route

        Returns:
            Routing result with selected expert and scores
        """
        if not self.experts:
            raise ValueError("No experts registered")

        # Get query embedding
        query_embedding = self._get_embedding(query)

        # Compute similarities for all experts
        all_scores: Dict[str, float] = {}
        for expert in self.experts:
            score = self._compute_similarity(query_embedding, expert)
            all_scores[expert.name] = score

        # Find best expert
        best_name = max(all_scores, key=all_scores.get)  # type: ignore
        best_expert = self.get_expert(best_name)
        best_score = all_scores[best_name]

        # Apply threshold
        if best_score < self.similarity_threshold:
            logger.warning(
                f"Best score {best_score:.3f} below threshold "
                f"{self.similarity_threshold:.3f}"
            )

        return RoutingResult(
            query=query,
            expert=best_expert,  # type: ignore
            score=best_score,
            all_scores=all_scores,
            method="semantic",
            metadata={
                "embedding_size": len(query_embedding),
                "threshold": self.similarity_threshold,
            },
        )

    def route_batch(self, queries: List[str]) -> List[RoutingResult]:
        """Route multiple queries efficiently.

        Args:
            queries: List of queries to route

        Returns:
            List of routing results
        """
        # For batch routing, we could optimize by batching embedding calls
        # For now, use the default implementation
        return [self.route(query) for query in queries]

    def update_embeddings(self) -> None:
        """Recompute all expert embeddings.

        Call this after modifying expert descriptions or examples
        without using add_expert/remove_expert.
        """
        # Rebuild vocabulary for simple embeddings
        self._vocabulary = {}
        self._build_vocabulary()

        # Update all expert embeddings
        self._expert_embeddings = {}
        self._example_embeddings = {}

        for expert in self.experts:
            self._update_expert_embeddings(expert)

    def get_info(self) -> Dict[str, Any]:
        """Get router information."""
        info = super().get_info()
        info.update({
            "method": "semantic",
            "similarity_threshold": self.similarity_threshold,
            "use_examples": self.use_examples,
            "use_descriptions": self.use_descriptions,
            "has_custom_embedding_fn": self.embedding_fn is not None,
            "vocabulary_size": len(self._vocabulary),
        })
        return info
