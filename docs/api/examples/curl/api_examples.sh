#!/bin/bash
# LargeForgeAI API - cURL Examples
# Set your API key: export API_KEY="your-api-key"

BASE_URL="${LARGEFORGE_URL:-http://localhost:8000}"

echo "=== LargeForgeAI cURL Examples ==="
echo "Base URL: $BASE_URL"
echo ""

# =============================================================================
# Health Check
# =============================================================================

echo "--- Health Check ---"
curl -s "$BASE_URL/health" | jq .
echo ""

# =============================================================================
# List Models
# =============================================================================

echo "--- List Models ---"
curl -s "$BASE_URL/v1/models" \
  -H "Authorization: Bearer $API_KEY" | jq .
echo ""

# =============================================================================
# Text Completion
# =============================================================================

echo "--- Text Completion ---"
curl -s "$BASE_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "largeforge-7b",
    "prompt": "Explain quantum computing in simple terms:",
    "max_tokens": 256,
    "temperature": 0.7
  }' | jq .
echo ""

# =============================================================================
# Streaming Completion
# =============================================================================

echo "--- Streaming Completion ---"
curl -s "$BASE_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "largeforge-7b",
    "prompt": "Write a haiku about programming:",
    "max_tokens": 64,
    "stream": true
  }'
echo ""
echo ""

# =============================================================================
# Chat Completion
# =============================================================================

echo "--- Chat Completion ---"
curl -s "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "largeforge-7b",
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "What is a Python decorator?"}
    ],
    "max_tokens": 512,
    "temperature": 0.5
  }' | jq .
echo ""

# =============================================================================
# Route Query
# =============================================================================

echo "--- Route Query ---"
curl -s "$BASE_URL/route" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "query": "How do I implement a binary search tree?"
  }' | jq .
echo ""

# =============================================================================
# Route and Generate
# =============================================================================

echo "--- Route and Generate ---"
curl -s "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "query": "Write a Python function to sort a list",
    "max_tokens": 300
  }' | jq .
echo ""

# =============================================================================
# List Experts
# =============================================================================

echo "--- List Experts ---"
curl -s "$BASE_URL/experts" \
  -H "Authorization: Bearer $API_KEY" | jq .
echo ""

# =============================================================================
# Register Expert
# =============================================================================

echo "--- Register Expert ---"
curl -s -X POST "$BASE_URL/experts" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "name": "math-expert",
    "model_path": "./models/math-expert",
    "description": "Specialized in mathematics",
    "domains": ["mathematics", "statistics"],
    "keywords": ["calculate", "equation", "math"]
  }' | jq .
echo ""

# =============================================================================
# Get Metrics (Prometheus format)
# =============================================================================

echo "--- Prometheus Metrics ---"
curl -s "$BASE_URL/metrics" | head -30
echo ""
echo "..."
echo ""

echo "=== Examples Complete ==="
