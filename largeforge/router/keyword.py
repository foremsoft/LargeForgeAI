"""Keyword-based router for rule-based routing."""

import re
from typing import Any, Dict, List, Optional, Pattern, Set

from largeforge.router.base import Router, Expert, RoutingResult
from largeforge.utils import get_logger

logger = get_logger(__name__)


class KeywordRouter(Router):
    """Router that uses keyword matching for expert selection.

    Supports exact matching, pattern matching, and weighted keywords.
    """

    def __init__(
        self,
        experts: Optional[List[Expert]] = None,
        case_sensitive: bool = False,
        use_word_boundaries: bool = True,
        default_expert: Optional[str] = None,
    ):
        """Initialize the keyword router.

        Args:
            experts: List of available experts
            case_sensitive: Whether keyword matching is case-sensitive
            default_expert: Name of default expert when no match found
        """
        self.case_sensitive = case_sensitive
        self.use_word_boundaries = use_word_boundaries
        self.default_expert_name = default_expert

        # Keyword patterns for each expert
        self._keyword_patterns: Dict[str, List[Pattern[str]]] = {}

        # Keyword weights (default = 1.0)
        self._keyword_weights: Dict[str, Dict[str, float]] = {}

        super().__init__(experts)

    def _on_expert_added(self, expert: Expert) -> None:
        """Compile keyword patterns when expert is added."""
        self._compile_patterns(expert)

    def _on_expert_removed(self, expert: Expert) -> None:
        """Remove patterns when expert is removed."""
        self._keyword_patterns.pop(expert.name, None)
        self._keyword_weights.pop(expert.name, None)

    def _compile_patterns(self, expert: Expert) -> None:
        """Compile keyword patterns for an expert.

        Args:
            expert: Expert to compile patterns for
        """
        patterns = []
        weights = {}

        for keyword in expert.keywords:
            # Handle weighted keywords (format: "keyword:weight")
            weight = 1.0
            if ":" in keyword:
                parts = keyword.rsplit(":", 1)
                try:
                    weight = float(parts[1])
                    keyword = parts[0]
                except ValueError:
                    pass  # Not a weight, keep full keyword

            # Create pattern
            flags = 0 if self.case_sensitive else re.IGNORECASE

            if self.use_word_boundaries:
                pattern_str = r"\b" + re.escape(keyword) + r"\b"
            else:
                pattern_str = re.escape(keyword)

            try:
                pattern = re.compile(pattern_str, flags)
                patterns.append(pattern)
                weights[keyword.lower() if not self.case_sensitive else keyword] = weight
            except re.error as e:
                logger.warning(f"Invalid keyword pattern '{keyword}': {e}")

        self._keyword_patterns[expert.name] = patterns
        self._keyword_weights[expert.name] = weights

    def _count_matches(self, text: str, expert: Expert) -> float:
        """Count weighted keyword matches for an expert.

        Args:
            text: Text to search
            expert: Expert to check

        Returns:
            Weighted match score
        """
        if expert.name not in self._keyword_patterns:
            return 0.0

        total_score = 0.0
        matched_keywords: Set[str] = set()

        for pattern in self._keyword_patterns[expert.name]:
            matches = pattern.findall(text)
            if matches:
                # Get base keyword for weight lookup
                keyword = pattern.pattern.replace(r"\b", "").replace("\\", "")
                if not self.case_sensitive:
                    keyword = keyword.lower()

                weight = self._keyword_weights[expert.name].get(keyword, 1.0)
                total_score += len(matches) * weight
                matched_keywords.add(keyword)

        return total_score

    def route(self, query: str) -> RoutingResult:
        """Route a query to the best expert based on keyword matching.

        Args:
            query: The query to route

        Returns:
            Routing result with selected expert and scores
        """
        if not self.experts:
            raise ValueError("No experts registered")

        # Count matches for all experts
        all_scores: Dict[str, float] = {}
        for expert in self.experts:
            score = self._count_matches(query, expert)
            all_scores[expert.name] = score

        # Find best expert
        best_name = max(all_scores, key=all_scores.get)  # type: ignore
        best_score = all_scores[best_name]

        # Normalize scores to 0-1 range
        max_score = max(all_scores.values()) if all_scores else 0
        if max_score > 0:
            normalized_scores = {
                name: score / max_score for name, score in all_scores.items()
            }
        else:
            normalized_scores = {name: 0.0 for name in all_scores}

        # If no matches found, use default expert
        if best_score == 0 and self.default_expert_name:
            default_expert = self.get_expert(self.default_expert_name)
            if default_expert:
                return RoutingResult(
                    query=query,
                    expert=default_expert,
                    score=0.0,
                    all_scores=normalized_scores,
                    method="keyword",
                    metadata={
                        "raw_scores": all_scores,
                        "used_default": True,
                    },
                )

        best_expert = self.get_expert(best_name)

        return RoutingResult(
            query=query,
            expert=best_expert,  # type: ignore
            score=normalized_scores[best_name],
            all_scores=normalized_scores,
            method="keyword",
            metadata={
                "raw_scores": all_scores,
                "used_default": False,
            },
        )

    def add_keywords(self, expert_name: str, keywords: List[str]) -> None:
        """Add keywords to an existing expert.

        Args:
            expert_name: Name of the expert
            keywords: Keywords to add
        """
        expert = self.get_expert(expert_name)
        if expert is None:
            raise ValueError(f"Expert '{expert_name}' not found")

        expert.keywords.extend(keywords)
        self._compile_patterns(expert)

    def remove_keywords(self, expert_name: str, keywords: List[str]) -> None:
        """Remove keywords from an expert.

        Args:
            expert_name: Name of the expert
            keywords: Keywords to remove
        """
        expert = self.get_expert(expert_name)
        if expert is None:
            raise ValueError(f"Expert '{expert_name}' not found")

        for keyword in keywords:
            if keyword in expert.keywords:
                expert.keywords.remove(keyword)

        self._compile_patterns(expert)

    def set_keyword_weight(
        self, expert_name: str, keyword: str, weight: float
    ) -> None:
        """Set weight for a specific keyword.

        Args:
            expert_name: Name of the expert
            keyword: Keyword to set weight for
            weight: Weight value (higher = more important)
        """
        if expert_name not in self._keyword_weights:
            self._keyword_weights[expert_name] = {}

        key = keyword.lower() if not self.case_sensitive else keyword
        self._keyword_weights[expert_name][key] = weight

    def get_matching_keywords(self, query: str, expert_name: str) -> List[str]:
        """Get all keywords that match a query for an expert.

        Args:
            query: Query to check
            expert_name: Expert to check

        Returns:
            List of matching keywords
        """
        expert = self.get_expert(expert_name)
        if expert is None or expert_name not in self._keyword_patterns:
            return []

        matching = []
        for pattern in self._keyword_patterns[expert_name]:
            if pattern.search(query):
                # Extract original keyword from pattern
                keyword = pattern.pattern.replace(r"\b", "").replace("\\", "")
                matching.append(keyword)

        return matching

    def get_info(self) -> Dict[str, Any]:
        """Get router information."""
        info = super().get_info()

        keyword_counts = {
            name: len(patterns)
            for name, patterns in self._keyword_patterns.items()
        }

        info.update({
            "method": "keyword",
            "case_sensitive": self.case_sensitive,
            "use_word_boundaries": self.use_word_boundaries,
            "default_expert": self.default_expert_name,
            "keywords_per_expert": keyword_counts,
            "total_keywords": sum(keyword_counts.values()),
        })
        return info
