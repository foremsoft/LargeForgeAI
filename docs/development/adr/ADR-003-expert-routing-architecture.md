# ADR-003: Expert Routing Architecture

## Status

Accepted

## Date

2024-07-01

## Context

LargeForgeAI supports multiple expert models specialized for different domains. We need a routing system that:

1. Accurately classifies incoming queries
2. Routes to the most appropriate expert
3. Handles failure gracefully
4. Scales with increasing experts
5. Provides observable routing decisions
6. Supports A/B testing and gradual rollout

Key questions:
- How should classification work?
- What happens when an expert is unavailable?
- How do we handle ambiguous queries?
- How do we balance routing accuracy vs latency?

## Decision

We will implement a **hybrid routing architecture** that combines:

1. **Keyword matching** for fast, rule-based routing
2. **Neural classification** for accurate semantic understanding
3. **Confidence-based fallback** for uncertain queries

### Architecture Overview

```
Query → Hybrid Classifier → Expert Selection → Load Balancer → Expert
              │                    │
              ├─ Keyword Match     ├─ Circuit Breaker
              ├─ Neural Score      ├─ Health Check
              └─ Confidence        └─ Fallback Handler
```

### Classification Strategy

| Scenario | Router Behavior |
|----------|-----------------|
| High keyword match (>0.8) | Route directly, skip neural |
| High neural confidence (>0.9) | Route to top expert |
| Medium confidence (0.6-0.9) | Use hybrid score |
| Low confidence (<0.6) | Route to general expert |

## Consequences

### Positive

- **Accuracy**: Neural classifier handles semantic nuance
- **Speed**: Keyword matching provides fast path for obvious cases
- **Robustness**: Multiple routing methods reduce single points of failure
- **Flexibility**: Easy to add new experts with keywords + training data
- **Observability**: Clear routing decisions for debugging

### Negative

- **Complexity**: Two routing methods to maintain
- **Latency**: Neural classification adds ~20ms
- **Training requirement**: Neural classifier needs labeled routing data
- **Calibration**: Confidence thresholds need tuning

### Neutral

- Need to collect routing data for classifier training
- Hybrid weight tuning requires experimentation

## Alternatives Considered

### Alternative 1: Keyword-Only Routing

**Description**: Route solely based on keyword matching

**Pros**:
- Very fast (<1ms)
- No model required
- Easy to understand and debug

**Cons**:
- Misses semantic understanding
- Keyword list maintenance burden
- Poor with paraphrased queries

**Why not chosen**: Accuracy insufficient for production use

### Alternative 2: Neural-Only Routing

**Description**: Route using only neural classifier

**Pros**:
- Best semantic understanding
- Handles paraphrasing well
- Less manual configuration

**Cons**:
- Adds latency to every request
- Requires training data
- Single point of failure

**Why not chosen**: Latency and lack of fallback mechanism

### Alternative 3: LLM-Based Routing

**Description**: Use an LLM to decide routing

**Pros**:
- Most flexible
- Can handle complex reasoning
- Zero-shot capability

**Cons**:
- Highest latency (100ms+)
- Most expensive
- Overkill for most queries

**Why not chosen**: Cost/latency too high for routing decision

### Alternative 4: Embedding Similarity

**Description**: Route based on query embedding similarity to expert descriptions

**Pros**:
- Semantic understanding
- No classifier training
- Easy to add experts

**Cons**:
- Sensitive to embedding quality
- Requires expert description tuning
- May not capture domain boundaries well

**Why not chosen**: Boundary definition less precise than trained classifier

## Implementation Notes

### Hybrid Classifier Implementation

```python
class HybridClassifier:
    def __init__(self, config):
        self.keyword_matcher = KeywordMatcher(config.keyword_config)
        self.neural_classifier = NeuralClassifier(config.neural_config)
        self.keyword_weight = config.keyword_weight  # e.g., 0.3
        self.neural_weight = config.neural_weight    # e.g., 0.7
        self.confidence_threshold = config.confidence_threshold

    def classify(self, query: str) -> RoutingDecision:
        # Fast path: strong keyword match
        keyword_result = self.keyword_matcher.match(query)
        if keyword_result.confidence > 0.8:
            return RoutingDecision(
                expert=keyword_result.expert,
                confidence=keyword_result.confidence,
                method="keyword"
            )

        # Neural classification
        neural_result = self.neural_classifier.classify(query)

        # Combine scores
        if keyword_result.expert == neural_result.expert:
            # Agreement boosts confidence
            combined_confidence = min(1.0,
                keyword_result.confidence + neural_result.confidence * 0.5
            )
        else:
            # Disagreement: use weighted combination
            combined_confidence = (
                self.keyword_weight * keyword_result.confidence +
                self.neural_weight * neural_result.confidence
            )

        # Determine winner
        if neural_result.confidence > keyword_result.confidence:
            winner = neural_result.expert
        else:
            winner = keyword_result.expert

        # Fallback for low confidence
        if combined_confidence < self.confidence_threshold:
            winner = "general"

        return RoutingDecision(
            expert=winner,
            confidence=combined_confidence,
            method="hybrid",
            alternatives=[keyword_result, neural_result]
        )
```

### Circuit Breaker for Expert Failures

```python
class ExpertCircuitBreaker:
    def __init__(self, failure_threshold=5, reset_timeout=60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = defaultdict(int)
        self.last_failure = {}
        self.state = defaultdict(lambda: "closed")  # closed, open, half-open

    def can_call(self, expert: str) -> bool:
        if self.state[expert] == "closed":
            return True
        elif self.state[expert] == "open":
            if time.time() - self.last_failure[expert] > self.reset_timeout:
                self.state[expert] = "half-open"
                return True
            return False
        else:  # half-open
            return True

    def record_success(self, expert: str):
        self.failures[expert] = 0
        self.state[expert] = "closed"

    def record_failure(self, expert: str):
        self.failures[expert] += 1
        self.last_failure[expert] = time.time()
        if self.failures[expert] >= self.failure_threshold:
            self.state[expert] = "open"
```

### Metrics to Track

```python
# Prometheus metrics
routing_decisions = Counter(
    'routing_decisions_total',
    'Total routing decisions',
    ['expert', 'method', 'confidence_bucket']
)

routing_latency = Histogram(
    'routing_latency_seconds',
    'Routing classification latency',
    ['method']
)

routing_fallback = Counter(
    'routing_fallback_total',
    'Times fallback expert was used',
    ['reason']
)
```

## References

- [Mixture of Experts Paper](https://arxiv.org/abs/1701.06538)
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)
- [Router Design Patterns](./design/router_patterns.md)

---

*ADR created by: Core Team*
*Last reviewed: 2024-12-01*
