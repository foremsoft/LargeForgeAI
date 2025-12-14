# LargeForgeAI REST API Reference

## Overview

The LargeForgeAI API provides endpoints for inference, routing, and model management. The API is designed to be compatible with the OpenAI API format while providing additional features for expert routing.

**Base URL:** `http://localhost:8000` (local) or `https://api.largeforge.ai/v1` (production)

---

## Authentication

All API requests require authentication using a Bearer token.

```http
Authorization: Bearer YOUR_API_KEY
```

### Obtaining an API Key

API keys can be generated through:
- Environment variable: `LARGEFORGE_API_KEY`
- Configuration file: `~/.largeforge/credentials`
- Admin API (enterprise only)

---

## Rate Limiting

Rate limits are applied per API key:

| Tier | Requests/Minute | Tokens/Day |
|------|-----------------|------------|
| Free | 60 | 100,000 |
| Pro | 600 | 1,000,000 |
| Enterprise | Custom | Custom |

Rate limit headers are included in all responses:

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1701432000
```

When rate limited, the API returns `429 Too Many Requests` with a `Retry-After` header.

---

## Endpoints

### Completions

#### Create Completion

Creates a text completion for the provided prompt.

```http
POST /v1/completions
```

**Request Body:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| model | string | Yes | - | Model ID to use |
| prompt | string | Yes | - | The prompt to complete |
| max_tokens | integer | No | 256 | Maximum tokens to generate (1-4096) |
| temperature | number | No | 0.7 | Sampling temperature (0-2) |
| top_p | number | No | 0.9 | Nucleus sampling probability (0-1) |
| top_k | integer | No | 50 | Top-k sampling (1-100) |
| stop | array | No | null | Stop sequences (max 4) |
| stream | boolean | No | false | Enable streaming |
| presence_penalty | number | No | 0 | Presence penalty (-2 to 2) |
| frequency_penalty | number | No | 0 | Frequency penalty (-2 to 2) |
| user | string | No | null | User identifier for tracking |

**Example Request:**

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "largeforge-7b",
    "prompt": "Explain quantum computing in simple terms:",
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

**Example Response:**

```json
{
  "id": "cmpl-abc123def456",
  "object": "text_completion",
  "created": 1701432000,
  "model": "largeforge-7b",
  "choices": [
    {
      "index": 0,
      "text": "\n\nQuantum computing is a type of computation that harnesses quantum mechanical phenomena like superposition and entanglement...",
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 156,
    "total_tokens": 164
  }
}
```

**Streaming Response:**

When `stream: true`, the response is delivered as Server-Sent Events:

```text
data: {"id":"cmpl-abc123","object":"text_completion.chunk","created":1701432000,"model":"largeforge-7b","choices":[{"index":0,"text":"Quantum","finish_reason":null}]}

data: {"id":"cmpl-abc123","object":"text_completion.chunk","created":1701432000,"model":"largeforge-7b","choices":[{"index":0,"text":" computing","finish_reason":null}]}

data: {"id":"cmpl-abc123","object":"text_completion.chunk","created":1701432000,"model":"largeforge-7b","choices":[{"index":0,"text":"","finish_reason":"stop"}]}

data: [DONE]
```

---

### Chat Completions

#### Create Chat Completion

Creates a completion for a chat conversation.

```http
POST /v1/chat/completions
```

**Request Body:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| model | string | Yes | - | Model ID to use |
| messages | array | Yes | - | Array of chat messages |
| max_tokens | integer | No | 256 | Maximum tokens to generate |
| temperature | number | No | 0.7 | Sampling temperature |
| top_p | number | No | 0.9 | Nucleus sampling |
| stream | boolean | No | false | Enable streaming |
| stop | array | No | null | Stop sequences |

**Message Object:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| role | string | Yes | "system", "user", or "assistant" |
| content | string | Yes | Message content |

**Example Request:**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "largeforge-7b",
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "Write a Python function to calculate factorial."}
    ],
    "max_tokens": 512,
    "temperature": 0.3
  }'
```

**Example Response:**

```json
{
  "id": "chatcmpl-abc123def456",
  "object": "chat.completion",
  "created": 1701432000,
  "model": "largeforge-7b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Here's a Python function to calculate factorial:\n\n```python\ndef factorial(n):\n    if n < 0:\n        raise ValueError(\"Factorial not defined for negative numbers\")\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n```\n\nYou can also use an iterative approach..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 128,
    "total_tokens": 153
  }
}
```

---

### Models

#### List Models

Returns a list of available models.

```http
GET /v1/models
```

**Example Request:**

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer $API_KEY"
```

**Example Response:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "largeforge-7b",
      "object": "model",
      "created": 1701000000,
      "owned_by": "largeforge",
      "root": "mistralai/Mistral-7B-v0.1",
      "parent": null
    },
    {
      "id": "code-expert-7b",
      "object": "model",
      "created": 1701100000,
      "owned_by": "largeforge",
      "root": "largeforge-7b",
      "parent": "largeforge-7b"
    }
  ]
}
```

#### Get Model

Returns information about a specific model.

```http
GET /v1/models/{model_id}
```

**Example Request:**

```bash
curl http://localhost:8000/v1/models/largeforge-7b \
  -H "Authorization: Bearer $API_KEY"
```

---

### Router

#### Route Query

Classifies a query and returns the recommended expert.

```http
POST /route
```

**Request Body:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| query | string | Yes | The query to route |
| context | object | No | Additional routing context |
| prefer_expert | string | No | Preferred expert hint |

**Example Request:**

```bash
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "query": "How do I implement a binary search tree?"
  }'
```

**Example Response:**

```json
{
  "expert": "code-expert",
  "confidence": 0.92,
  "alternatives": [
    {"expert": "general-assistant", "confidence": 0.45},
    {"expert": "math-expert", "confidence": 0.32}
  ]
}
```

#### Route and Generate

Routes a query to the appropriate expert and generates a response.

```http
POST /generate
```

**Request Body:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| query | string | Yes | - | The query |
| max_tokens | integer | No | 256 | Maximum tokens |
| temperature | number | No | 0.7 | Sampling temperature |
| stream | boolean | No | false | Enable streaming |
| prefer_expert | string | No | null | Preferred expert |

**Example Request:**

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "query": "Explain how to implement quicksort in Python",
    "max_tokens": 512
  }'
```

**Example Response:**

```json
{
  "expert": "code-expert",
  "confidence": 0.95,
  "response": "Quicksort is a divide-and-conquer sorting algorithm...\n\n```python\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n```",
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 245,
    "total_tokens": 257
  }
}
```

---

### Experts

#### List Experts

Returns a list of registered experts.

```http
GET /experts
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| domain | string | null | Filter by domain |
| status | string | "active" | Filter by status (active/inactive/all) |

**Example Request:**

```bash
curl http://localhost:8000/experts \
  -H "Authorization: Bearer $API_KEY"
```

**Example Response:**

```json
{
  "experts": [
    {
      "name": "code-expert",
      "model_path": "./models/code-expert",
      "description": "Specialized in code generation and review",
      "domains": ["programming", "debugging"],
      "keywords": ["code", "function", "debug"],
      "endpoint": "http://localhost:8001",
      "status": "active",
      "load": 0.35,
      "request_count": 12453,
      "avg_latency_ms": 245.5
    },
    {
      "name": "writing-expert",
      "model_path": "./models/writing-expert",
      "description": "Specialized in creative and technical writing",
      "domains": ["creative-writing", "documentation"],
      "keywords": ["write", "essay", "article"],
      "endpoint": "http://localhost:8002",
      "status": "active",
      "load": 0.22,
      "request_count": 8921,
      "avg_latency_ms": 312.3
    }
  ]
}
```

#### Register Expert

Registers a new expert model.

```http
POST /experts
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| name | string | Yes | Unique expert name (lowercase, alphanumeric) |
| model_path | string | Yes | Path to model |
| description | string | Yes | Expert description |
| domains | array | No | Domain categories |
| keywords | array | No | Routing keywords |
| endpoint | string | No | Inference endpoint URL |
| priority | integer | No | Routing priority (default: 0) |

**Example Request:**

```bash
curl -X POST http://localhost:8000/experts \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "name": "math-expert",
    "model_path": "./models/math-expert",
    "description": "Specialized in mathematics and calculations",
    "domains": ["mathematics", "statistics"],
    "keywords": ["calculate", "equation", "math", "formula"],
    "endpoint": "http://localhost:8003"
  }'
```

#### Remove Expert

Removes an expert from the registry.

```http
DELETE /experts/{expert_name}
```

**Example Request:**

```bash
curl -X DELETE http://localhost:8000/experts/math-expert \
  -H "Authorization: Bearer $API_KEY"
```

---

### Health & Monitoring

#### Health Check

Returns the health status of the service.

```http
GET /health
```

**Example Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 86400,
  "components": [
    {"name": "inference", "status": "healthy", "latency_ms": 2.5},
    {"name": "router", "status": "healthy", "latency_ms": 0.8},
    {"name": "database", "status": "healthy", "latency_ms": 1.2}
  ]
}
```

#### Readiness Check

Returns whether the service is ready to accept requests.

```http
GET /health/ready
```

Returns `200 OK` when ready, `503 Service Unavailable` when not ready.

#### Liveness Check

Returns whether the service is alive.

```http
GET /health/live
```

Returns `200 OK` when alive.

#### Prometheus Metrics

Returns Prometheus-compatible metrics.

```http
GET /metrics
```

**Example Response:**

```text
# HELP inference_requests_total Total inference requests
# TYPE inference_requests_total counter
inference_requests_total{model="largeforge-7b",status="success"} 12453
inference_requests_total{model="largeforge-7b",status="error"} 23

# HELP inference_latency_seconds Inference request latency
# TYPE inference_latency_seconds histogram
inference_latency_seconds_bucket{model="largeforge-7b",le="0.1"} 8234
inference_latency_seconds_bucket{model="largeforge-7b",le="0.5"} 11892
inference_latency_seconds_bucket{model="largeforge-7b",le="1.0"} 12341
inference_latency_seconds_bucket{model="largeforge-7b",le="+Inf"} 12453

# HELP tokens_generated_total Total tokens generated
# TYPE tokens_generated_total counter
tokens_generated_total{model="largeforge-7b"} 2847562

# HELP gpu_memory_used_bytes GPU memory utilization
# TYPE gpu_memory_used_bytes gauge
gpu_memory_used_bytes{device="0"} 21474836480

# HELP active_requests Currently processing requests
# TYPE active_requests gauge
active_requests{model="largeforge-7b"} 5
```

---

## Error Handling

### Error Response Format

All errors follow a consistent format:

```json
{
  "error": {
    "code": "error_code",
    "message": "Human-readable error message",
    "type": "error_type",
    "param": "optional_parameter_name"
  }
}
```

### Error Codes

| HTTP Status | Error Code | Description |
|-------------|------------|-------------|
| 400 | invalid_request | Malformed request body |
| 400 | invalid_parameter | Invalid parameter value |
| 401 | authentication_failed | Invalid or missing API key |
| 403 | forbidden | Insufficient permissions |
| 404 | not_found | Resource not found |
| 404 | model_not_found | Requested model not available |
| 429 | rate_limited | Rate limit exceeded |
| 500 | internal_error | Internal server error |
| 503 | model_overloaded | Model at capacity |
| 504 | timeout | Request processing timeout |

### Example Error Responses

**Invalid Request (400):**

```json
{
  "error": {
    "code": "invalid_request",
    "message": "Invalid JSON in request body",
    "type": "invalid_request_error"
  }
}
```

**Authentication Failed (401):**

```json
{
  "error": {
    "code": "authentication_failed",
    "message": "Invalid API key provided",
    "type": "authentication_error"
  }
}
```

**Rate Limited (429):**

```json
{
  "error": {
    "code": "rate_limited",
    "message": "Rate limit exceeded. Please retry after 60 seconds.",
    "type": "rate_limit_error"
  }
}
```

---

## Best Practices

### Request Optimization

1. **Use streaming for long outputs**: Reduces time-to-first-token perception
2. **Batch requests when possible**: Use parallel requests for multiple prompts
3. **Set appropriate max_tokens**: Avoid generating more tokens than needed
4. **Use stop sequences**: Terminate generation at logical points

### Error Handling

1. **Implement exponential backoff**: For rate limiting and transient errors
2. **Handle streaming errors**: Check for `[DONE]` or error events
3. **Validate inputs client-side**: Reduce unnecessary API calls
4. **Log request IDs**: Include `id` from responses for debugging

### Security

1. **Never expose API keys**: Use environment variables or secret managers
2. **Rotate keys regularly**: Especially after team member changes
3. **Use HTTPS**: Always use encrypted connections in production
4. **Implement request signing**: For high-security applications

---

## SDKs and Libraries

- **Python**: `pip install largeforge` - [SDK Reference](./SDK_REFERENCE.md)
- **JavaScript/TypeScript**: `npm install largeforge` (coming soon)
- **Go**: `go get github.com/largeforgeai/largeforge-go` (coming soon)

---

## Changelog

### v1.0.0 (December 2024)
- Initial API release
- OpenAI-compatible endpoints
- Expert routing system
- Streaming support

---

*For more examples, see the [examples](./examples/) directory.*
