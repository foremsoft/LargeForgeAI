# LargeForgeAI Data Dictionary

## Overview

This document defines all data structures, formats, and schemas used in LargeForgeAI.

---

## 1. Training Data Formats

### 1.1 Alpaca Format

Standard instruction-following format.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| instruction | string | Yes | The task instruction |
| input | string | No | Additional input context |
| output | string | Yes | Expected response |

**Schema:**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["instruction", "output"],
  "properties": {
    "instruction": {
      "type": "string",
      "minLength": 1,
      "maxLength": 10000
    },
    "input": {
      "type": "string",
      "maxLength": 10000
    },
    "output": {
      "type": "string",
      "minLength": 1,
      "maxLength": 50000
    }
  }
}
```

**Example:**
```json
{
  "instruction": "Summarize the following text",
  "input": "Large language models are neural networks trained on vast amounts of text...",
  "output": "LLMs are neural networks trained on text data for natural language tasks."
}
```

### 1.2 ShareGPT Format

Multi-turn conversation format.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| conversations | array | Yes | List of conversation turns |
| conversations[].from | string | Yes | Speaker: "human", "gpt", "system" |
| conversations[].value | string | Yes | Message content |

**Schema:**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["conversations"],
  "properties": {
    "conversations": {
      "type": "array",
      "minItems": 2,
      "items": {
        "type": "object",
        "required": ["from", "value"],
        "properties": {
          "from": {
            "type": "string",
            "enum": ["human", "gpt", "system"]
          },
          "value": {
            "type": "string",
            "minLength": 1
          }
        }
      }
    }
  }
}
```

**Example:**
```json
{
  "conversations": [
    {"from": "system", "value": "You are a helpful assistant."},
    {"from": "human", "value": "What is Python?"},
    {"from": "gpt", "value": "Python is a high-level programming language..."},
    {"from": "human", "value": "What are its main uses?"},
    {"from": "gpt", "value": "Python is commonly used for web development..."}
  ]
}
```

### 1.3 DPO Format

Preference data format for Direct Preference Optimization.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| prompt | string | Yes | Input prompt |
| chosen | string | Yes | Preferred response |
| rejected | string | Yes | Non-preferred response |

**Schema:**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["prompt", "chosen", "rejected"],
  "properties": {
    "prompt": {
      "type": "string",
      "minLength": 1
    },
    "chosen": {
      "type": "string",
      "minLength": 1
    },
    "rejected": {
      "type": "string",
      "minLength": 1
    }
  }
}
```

**Example:**
```json
{
  "prompt": "Write a greeting message",
  "chosen": "Hello! How can I assist you today? I'm here to help with any questions you might have.",
  "rejected": "Hi"
}
```

---

## 2. Configuration Data

### 2.1 TrainingConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| model_name | string | - | Model identifier or path |
| output_dir | string | - | Output directory for checkpoints |
| num_epochs | int | 3 | Number of training epochs |
| per_device_batch_size | int | 4 | Batch size per device |
| gradient_accumulation_steps | int | 4 | Gradient accumulation steps |
| learning_rate | float | 2e-5 | Learning rate |
| weight_decay | float | 0.01 | Weight decay coefficient |
| warmup_ratio | float | 0.1 | Warmup ratio |
| max_grad_norm | float | 1.0 | Max gradient norm for clipping |
| fp16 | bool | false | Use FP16 precision |
| bf16 | bool | true | Use BF16 precision |
| optim | string | "adamw_torch" | Optimizer type |
| lr_scheduler_type | string | "cosine" | LR scheduler type |
| save_steps | int | 500 | Save checkpoint every N steps |
| save_total_limit | int | 3 | Keep only N checkpoints |
| logging_steps | int | 10 | Log every N steps |
| eval_steps | int | 500 | Evaluate every N steps |

### 2.2 LoraConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| r | int | 8 | LoRA rank |
| lora_alpha | int | 16 | LoRA alpha (scaling) |
| lora_dropout | float | 0.05 | Dropout probability |
| target_modules | list[str] | ["q_proj", "k_proj", "v_proj", "o_proj"] | Modules to adapt |
| bias | string | "none" | Bias training mode |
| task_type | string | "CAUSAL_LM" | Task type |

### 2.3 ExpertConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| name | string | - | Unique expert name |
| model_path | string | - | Path to model |
| description | string | - | Human-readable description |
| domains | list[str] | [] | Domain categories |
| keywords | list[str] | [] | Routing keywords |
| endpoint | string | null | Inference endpoint URL |
| priority | int | 0 | Routing priority |
| max_tokens | int | 2048 | Max context length |

---

## 3. API Data Structures

### 3.1 CompletionRequest

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| model | string | Yes | - | Model identifier |
| prompt | string | Yes | - | Input prompt |
| max_tokens | int | No | 256 | Max tokens to generate |
| temperature | float | No | 0.7 | Sampling temperature |
| top_p | float | No | 0.9 | Nucleus sampling |
| top_k | int | No | 50 | Top-k sampling |
| stop | list[str] | No | null | Stop sequences |
| stream | bool | No | false | Enable streaming |

### 3.2 CompletionResponse

| Field | Type | Description |
|-------|------|-------------|
| id | string | Unique completion ID |
| object | string | "text_completion" |
| created | int | Unix timestamp |
| model | string | Model used |
| choices | list[Choice] | Generated completions |
| usage | Usage | Token usage statistics |

### 3.3 ChatMessage

| Field | Type | Description |
|-------|------|-------------|
| role | string | "system", "user", or "assistant" |
| content | string | Message content |

### 3.4 Usage

| Field | Type | Description |
|-------|------|-------------|
| prompt_tokens | int | Tokens in prompt |
| completion_tokens | int | Tokens generated |
| total_tokens | int | Total tokens |

### 3.5 RouteResponse

| Field | Type | Description |
|-------|------|-------------|
| expert | string | Selected expert name |
| confidence | float | Routing confidence (0-1) |
| alternatives | list[Alternative] | Other possible experts |

### 3.6 HealthResponse

| Field | Type | Description |
|-------|------|-------------|
| status | string | "healthy", "degraded", "unhealthy" |
| version | string | Service version |
| uptime_seconds | int | Uptime in seconds |
| components | list[Component] | Component health |

---

## 4. Model Artifacts

### 4.1 Checkpoint Structure

```
checkpoint/
├── config.json           # Model configuration
├── generation_config.json # Generation defaults
├── model.safetensors     # Model weights
├── tokenizer.json        # Tokenizer
├── tokenizer_config.json # Tokenizer config
├── special_tokens_map.json
├── trainer_state.json    # Training state
└── training_args.bin     # Training arguments
```

### 4.2 LoRA Adapter Structure

```
lora-adapter/
├── adapter_config.json   # LoRA configuration
├── adapter_model.safetensors  # Adapter weights
└── README.md             # Model card
```

### 4.3 Quantized Model Structure

```
quantized-model/
├── config.json           # Model configuration
├── generation_config.json
├── model.safetensors     # Quantized weights
├── quant_config.json     # Quantization config
├── tokenizer.json
└── tokenizer_config.json
```

---

## 5. Metrics Data

### 5.1 Training Metrics

| Metric | Type | Description |
|--------|------|-------------|
| loss | float | Training loss |
| learning_rate | float | Current learning rate |
| epoch | float | Current epoch |
| global_step | int | Global training step |
| grad_norm | float | Gradient norm |
| train_samples_per_second | float | Training throughput |

### 5.2 Evaluation Metrics

| Metric | Type | Description |
|--------|------|-------------|
| eval_loss | float | Validation loss |
| perplexity | float | Model perplexity |
| accuracy | float | Token prediction accuracy |

### 5.3 Inference Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| inference_requests_total | counter | model, status | Total requests |
| inference_latency_seconds | histogram | model | Latency distribution |
| tokens_generated_total | counter | model | Total tokens generated |
| tokens_per_second | gauge | model | Generation speed |
| gpu_memory_used_bytes | gauge | device | GPU memory usage |
| active_requests | gauge | model | Concurrent requests |

### 5.4 Router Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| routing_decisions_total | counter | expert | Routing decisions |
| routing_latency_seconds | histogram | classifier | Classification time |
| expert_availability | gauge | expert | Expert health status |

---

## 6. Log Data

### 6.1 Structured Log Format

```json
{
  "timestamp": "2024-12-01T10:30:00.000Z",
  "level": "INFO",
  "logger": "llm.inference.server",
  "message": "Request completed",
  "request_id": "req-abc123",
  "trace_id": "trace-xyz789",
  "context": {
    "model": "largeforge-7b",
    "tokens_generated": 156,
    "latency_ms": 245.5
  }
}
```

### 6.2 Log Fields

| Field | Type | Description |
|-------|------|-------------|
| timestamp | string | ISO 8601 timestamp |
| level | string | DEBUG, INFO, WARNING, ERROR, CRITICAL |
| logger | string | Logger name |
| message | string | Log message |
| request_id | string | Request identifier |
| trace_id | string | Distributed trace ID |
| context | object | Additional context |

---

## 7. Data Validation

### 7.1 Validation Rules

| Data Type | Rule | Error |
|-----------|------|-------|
| model_name | Non-empty, valid path | "Model name required" |
| prompt | Max 100K chars | "Prompt too long" |
| max_tokens | 1-4096 | "Invalid max_tokens" |
| temperature | 0-2 | "Temperature out of range" |
| learning_rate | > 0 | "Learning rate must be positive" |
| batch_size | >= 1 | "Batch size must be at least 1" |

### 7.2 Validation Example

```python
from pydantic import BaseModel, Field, field_validator

class TrainingConfig(BaseModel):
    model_name: str = Field(..., min_length=1)
    learning_rate: float = Field(2e-5, gt=0)
    num_epochs: int = Field(3, ge=1, le=100)

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if ".." in v:
            raise ValueError("Invalid model path")
        return v
```

---

*Last Updated: December 2024*
