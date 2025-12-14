"""Integration tests for the router system."""

import pytest
from unittest.mock import MagicMock, patch

from largeforge.router import (
    Router,
    SemanticRouter,
    KeywordRouter,
    HybridRouter,
    Expert,
    RoutingResult,
)
from largeforge.config import RouterConfig


@pytest.mark.integration
class TestRouterExpertManagement:
    """Test expert management across router types."""

    def test_add_experts_to_all_routers(self):
        """Test adding experts to all router types."""
        experts = [
            Expert(
                name="code",
                description="Programming and coding assistant",
                keywords=["python", "javascript", "code", "programming"],
                examples=["Write a function", "Debug this code"],
            ),
            Expert(
                name="writing",
                description="Writing and content assistant",
                keywords=["write", "essay", "article", "story"],
                examples=["Write an essay", "Help me write a story"],
            ),
            Expert(
                name="math",
                description="Mathematics assistant",
                keywords=["math", "calculate", "equation", "algebra"],
                examples=["Solve this equation", "Calculate the integral"],
            ),
        ]

        # Test with each router type
        routers = [
            SemanticRouter(),
            KeywordRouter(),
            HybridRouter(),
        ]

        for router in routers:
            for expert in experts:
                router.add_expert(expert)

            assert len(router.experts) == 3
            assert router.get_expert("code") is not None
            assert router.get_expert("writing") is not None
            assert router.get_expert("math") is not None

    def test_remove_experts(self):
        """Test removing experts from routers."""
        router = HybridRouter()

        router.add_expert(Expert(name="test1", description="Test 1"))
        router.add_expert(Expert(name="test2", description="Test 2"))

        assert len(router.experts) == 2

        router.remove_expert("test1")
        assert len(router.experts) == 1
        assert router.get_expert("test1") is None
        assert router.get_expert("test2") is not None


@pytest.mark.integration
class TestKeywordRouterIntegration:
    """Test keyword router integration scenarios."""

    def test_keyword_routing_accuracy(self):
        """Test keyword routing produces accurate results."""
        router = KeywordRouter()

        router.add_expert(Expert(
            name="weather",
            description="Weather information",
            keywords=["weather", "temperature", "forecast", "rain", "sunny"],
        ))
        router.add_expert(Expert(
            name="news",
            description="News information",
            keywords=["news", "headline", "breaking", "report"],
        ))
        router.add_expert(Expert(
            name="sports",
            description="Sports information",
            keywords=["sports", "game", "score", "team", "player"],
        ))

        # Test weather query
        result = router.route("What's the weather forecast for today?")
        assert result.expert.name == "weather"

        # Test news query
        result = router.route("Show me the breaking news headlines")
        assert result.expert.name == "news"

        # Test sports query
        result = router.route("What was the score of the game?")
        assert result.expert.name == "sports"

    def test_keyword_weighted_routing(self):
        """Test weighted keyword routing."""
        router = KeywordRouter()

        # Add expert with weighted keywords
        router.add_expert(Expert(
            name="urgent",
            description="Urgent issues",
            keywords=["urgent:5", "emergency:5", "help:2", "issue:1"],
        ))
        router.add_expert(Expert(
            name="general",
            description="General questions",
            keywords=["question:1", "help:1", "ask:1"],
        ))

        # Emergency should route to urgent
        result = router.route("This is an urgent emergency!")
        assert result.expert.name == "urgent"
        assert result.score > 0


@pytest.mark.integration
class TestSemanticRouterIntegration:
    """Test semantic router integration scenarios."""

    def test_semantic_routing_with_examples(self):
        """Test semantic routing using examples."""
        router = SemanticRouter()

        router.add_expert(Expert(
            name="translation",
            description="Language translation",
            examples=[
                "Translate this to French",
                "How do you say hello in Spanish?",
                "Convert this text to German",
            ],
        ))
        router.add_expert(Expert(
            name="summarization",
            description="Text summarization",
            examples=[
                "Summarize this article",
                "Give me the key points",
                "What's the main idea?",
            ],
        ))

        # Should route translation query
        result = router.route("Can you translate this to Italian?")
        assert result is not None
        assert result.score > 0

    def test_semantic_routing_with_custom_embeddings(self):
        """Test semantic routing with custom embedding function."""
        # Simple mock embedding function
        def simple_embed(text):
            # Return normalized vector based on text length
            import math
            length = len(text)
            return [length / 100, math.sin(length), math.cos(length)]

        router = SemanticRouter(embedding_fn=simple_embed)

        router.add_expert(Expert(
            name="short",
            description="Short queries",
            examples=["Hi", "Hello", "Hey"],
        ))
        router.add_expert(Expert(
            name="long",
            description="Long queries",
            examples=[
                "This is a very long query that should match long expert",
                "Another lengthy query with many words in it",
            ],
        ))

        result = router.route("Test query")
        assert result is not None


@pytest.mark.integration
class TestHybridRouterIntegration:
    """Test hybrid router integration scenarios."""

    def test_hybrid_routing_combines_methods(self):
        """Test hybrid routing combines semantic and keyword methods."""
        config = RouterConfig(
            neural_weight=0.5,
            keyword_weight=0.5,
            confidence_threshold=0.1,
        )

        router = HybridRouter(config=config)

        router.add_expert(Expert(
            name="python",
            description="Python programming",
            keywords=["python", "pip", "django", "flask"],
            examples=["Write Python code", "How to use pip?"],
        ))
        router.add_expert(Expert(
            name="javascript",
            description="JavaScript programming",
            keywords=["javascript", "nodejs", "react", "npm"],
            examples=["Write JavaScript code", "How to use npm?"],
        ))

        # Test with keyword match
        result = router.route("How do I install packages with pip?")
        assert result.expert.name == "python"

        # Test with JavaScript keywords
        result = router.route("How to create a React component with npm?")
        assert result.expert.name == "javascript"

    def test_hybrid_router_with_different_weights(self):
        """Test hybrid router with different weight configurations."""
        experts = [
            Expert(
                name="expert_a",
                description="Expert A",
                keywords=["alpha", "first"],
                examples=["Alpha query"],
            ),
            Expert(
                name="expert_b",
                description="Expert B",
                keywords=["beta", "second"],
                examples=["Beta query"],
            ),
        ]

        # Test with keyword-heavy weights
        config_keyword = RouterConfig(
            neural_weight=0.2,
            keyword_weight=0.8,
        )
        router_keyword = HybridRouter(config=config_keyword)
        for e in experts:
            router_keyword.add_expert(e)

        # Test with semantic-heavy weights
        config_semantic = RouterConfig(
            neural_weight=0.8,
            keyword_weight=0.2,
        )
        router_semantic = HybridRouter(config=config_semantic)
        for e in experts:
            router_semantic.add_expert(e)

        # Both should route, possibly with different scores
        result_keyword = router_keyword.route("Tell me about alpha")
        result_semantic = router_semantic.route("Tell me about alpha")

        assert result_keyword is not None
        assert result_semantic is not None


@pytest.mark.integration
class TestRouterWithInference:
    """Test router integration with inference system."""

    def test_router_expert_metadata_for_inference(self):
        """Test router can store inference-relevant metadata."""
        router = HybridRouter()

        router.add_expert(Expert(
            name="code_gen",
            description="Code generation expert",
            keywords=["code", "generate", "write"],
            metadata={
                "model_path": "codellama/CodeLlama-7b-hf",
                "temperature": 0.2,
                "max_tokens": 1024,
            },
        ))
        router.add_expert(Expert(
            name="chat",
            description="General chat expert",
            keywords=["chat", "talk", "conversation"],
            metadata={
                "model_path": "meta-llama/Llama-2-7b-chat-hf",
                "temperature": 0.7,
                "max_tokens": 512,
            },
        ))

        result = router.route("Write a Python function")
        assert result.expert.metadata["model_path"] == "codellama/CodeLlama-7b-hf"
        assert result.expert.metadata["temperature"] == 0.2

    def test_routing_result_contains_all_scores(self):
        """Test routing result contains all expert scores."""
        router = HybridRouter()

        experts = [
            Expert(name="a", keywords=["alpha"]),
            Expert(name="b", keywords=["beta"]),
            Expert(name="c", keywords=["gamma"]),
        ]
        for e in experts:
            router.add_expert(e)

        result = router.route("Test alpha query")

        # Should have scores for all experts
        assert result.all_scores is not None
        assert len(result.all_scores) == 3


@pytest.mark.integration
class TestRouterBatchRouting:
    """Test batch routing functionality."""

    def test_route_multiple_queries(self):
        """Test routing multiple queries at once."""
        router = KeywordRouter()

        router.add_expert(Expert(name="math", keywords=["math", "calculate"]))
        router.add_expert(Expert(name="science", keywords=["science", "physics"]))
        router.add_expert(Expert(name="history", keywords=["history", "war"]))

        queries = [
            "Calculate the sum",
            "Explain physics",
            "Tell me about the war",
        ]

        results = router.route_batch(queries)

        assert len(results) == 3
        assert results[0].expert.name == "math"
        assert results[1].expert.name == "science"
        assert results[2].expert.name == "history"
