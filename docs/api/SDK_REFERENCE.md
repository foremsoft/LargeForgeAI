# LargeForgeAI Python SDK Reference

## Installation

```bash
pip install largeforge
```

Or install from source:

```bash
git clone https://github.com/largeforgeai/largeforgeai.git
cd largeforgeai
pip install -e .
```

---

## Quick Start

```python
from largeforge import LargeForgeClient

# Initialize client
client = LargeForgeClient(api_key="your-api-key")

# Generate completion
response = client.completions.create(
    model="largeforge-7b",
    prompt="Explain quantum computing:",
    max_tokens=256
)

print(response.choices[0].text)
```

---

## Client Configuration

### LargeForgeClient

The main client class for interacting with the LargeForgeAI API.

```python
class LargeForgeClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "http://localhost:8000",
        timeout: float = 60.0,
        max_retries: int = 3,
        **kwargs
    )
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| api_key | str | None | API key (reads from `LARGEFORGE_API_KEY` if not provided) |
| base_url | str | "http://localhost:8000" | Base URL of the API |
| timeout | float | 60.0 | Request timeout in seconds |
| max_retries | int | 3 | Maximum retry attempts |

**Example:**

```python
from largeforge import LargeForgeClient

# From environment variable
client = LargeForgeClient()

# Explicit configuration
client = LargeForgeClient(
    api_key="sk-...",
    base_url="https://api.largeforge.ai/v1",
    timeout=120.0
)
```

### AsyncLargeForgeClient

Async version of the client for use with asyncio.

```python
import asyncio
from largeforge import AsyncLargeForgeClient

async def main():
    client = AsyncLargeForgeClient(api_key="your-api-key")

    response = await client.completions.create(
        model="largeforge-7b",
        prompt="Hello, world!",
        max_tokens=100
    )

    print(response.choices[0].text)

asyncio.run(main())
```

---

## Completions

### Create Completion

```python
client.completions.create(
    model: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    stop: list[str] | None = None,
    stream: bool = False,
    **kwargs
) -> CompletionResponse | Iterator[CompletionChunk]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model | str | required | Model ID |
| prompt | str | required | Input prompt |
| max_tokens | int | 256 | Maximum tokens to generate |
| temperature | float | 0.7 | Sampling temperature (0-2) |
| top_p | float | 0.9 | Nucleus sampling |
| top_k | int | 50 | Top-k sampling |
| stop | list[str] | None | Stop sequences |
| stream | bool | False | Enable streaming |

**Example:**

```python
# Basic completion
response = client.completions.create(
    model="largeforge-7b",
    prompt="Write a haiku about programming:",
    max_tokens=64,
    temperature=0.9
)

print(response.choices[0].text)
print(f"Tokens used: {response.usage.total_tokens}")
```

### Streaming Completions

```python
# Streaming response
stream = client.completions.create(
    model="largeforge-7b",
    prompt="Tell me a story:",
    max_tokens=500,
    stream=True
)

for chunk in stream:
    print(chunk.choices[0].text, end="", flush=True)
print()  # Newline at end
```

### Async Streaming

```python
async def stream_completion():
    async for chunk in await client.completions.create(
        model="largeforge-7b",
        prompt="Explain relativity:",
        max_tokens=300,
        stream=True
    ):
        print(chunk.choices[0].text, end="", flush=True)
```

---

## Chat Completions

### Create Chat Completion

```python
client.chat.completions.create(
    model: str,
    messages: list[dict],
    max_tokens: int = 256,
    temperature: float = 0.7,
    stream: bool = False,
    **kwargs
) -> ChatCompletionResponse | Iterator[ChatCompletionChunk]
```

**Message Format:**

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "What is Python?"}
]
```

**Example:**

```python
response = client.chat.completions.create(
    model="largeforge-7b",
    messages=[
        {"role": "system", "content": "You are a coding expert."},
        {"role": "user", "content": "Explain decorators in Python."}
    ],
    max_tokens=512,
    temperature=0.5
)

print(response.choices[0].message.content)
```

### Chat with History

```python
class Conversation:
    def __init__(self, client, model, system_prompt=None):
        self.client = client
        self.model = model
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def chat(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            max_tokens=512
        )

        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})

        return assistant_message

# Usage
conv = Conversation(client, "largeforge-7b", "You are a helpful tutor.")
print(conv.chat("What is machine learning?"))
print(conv.chat("Can you give me an example?"))
```

---

## Models

### List Models

```python
client.models.list() -> ModelList
```

**Example:**

```python
models = client.models.list()

for model in models.data:
    print(f"{model.id}: {model.owned_by}")
```

### Get Model

```python
client.models.retrieve(model_id: str) -> Model
```

**Example:**

```python
model = client.models.retrieve("largeforge-7b")
print(f"Model: {model.id}")
print(f"Created: {model.created}")
print(f"Base: {model.root}")
```

---

## Router

### Route Query

```python
client.router.route(
    query: str,
    context: dict | None = None,
    prefer_expert: str | None = None
) -> RouteResponse
```

**Example:**

```python
result = client.router.route(
    query="How do I implement a REST API in FastAPI?"
)

print(f"Expert: {result.expert}")
print(f"Confidence: {result.confidence:.2%}")

for alt in result.alternatives:
    print(f"  Alternative: {alt.expert} ({alt.confidence:.2%})")
```

### Route and Generate

```python
client.router.generate(
    query: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    stream: bool = False,
    prefer_expert: str | None = None
) -> GenerateResponse | Iterator[GenerateChunk]
```

**Example:**

```python
# Non-streaming
response = client.router.generate(
    query="Write a Python function to sort a list",
    max_tokens=300
)

print(f"Routed to: {response.expert}")
print(f"Response: {response.response}")

# Streaming
for chunk in client.router.generate(
    query="Explain neural networks",
    max_tokens=500,
    stream=True
):
    if chunk.chunk:
        print(chunk.chunk, end="", flush=True)
```

---

## Experts

### List Experts

```python
client.experts.list(
    domain: str | None = None,
    status: str = "active"
) -> ExpertList
```

**Example:**

```python
experts = client.experts.list()

for expert in experts.experts:
    print(f"{expert.name}: {expert.description}")
    print(f"  Status: {expert.status}, Load: {expert.load:.2%}")
    print(f"  Domains: {', '.join(expert.domains)}")
```

### Register Expert

```python
client.experts.create(
    name: str,
    model_path: str,
    description: str,
    domains: list[str] | None = None,
    keywords: list[str] | None = None,
    endpoint: str | None = None,
    priority: int = 0
) -> Expert
```

**Example:**

```python
expert = client.experts.create(
    name="math-expert",
    model_path="./models/math-expert",
    description="Specialized in mathematics",
    domains=["mathematics", "statistics", "calculus"],
    keywords=["calculate", "equation", "integral", "derivative"],
    endpoint="http://localhost:8003"
)

print(f"Created expert: {expert.name}")
```

### Get Expert

```python
client.experts.retrieve(name: str) -> Expert
```

### Remove Expert

```python
client.experts.delete(name: str) -> None
```

---

## Training

### Training Configuration

```python
from largeforge.training import (
    TrainingConfig,
    LoraConfig,
    DPOConfig,
    SFTTrainer,
    DPOTrainer
)

# SFT Training configuration
training_config = TrainingConfig(
    model_name="mistralai/Mistral-7B-v0.1",
    output_dir="./output/my-model",
    num_epochs=3,
    per_device_batch_size=4,
    learning_rate=2e-5,
    bf16=True
)

# LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
```

### SFT Training

```python
from largeforge.training import SFTTrainer
from largeforge.data import DatasetLoader

# Load dataset
dataset = DatasetLoader.load("./data/training.json", format="alpaca")

# Initialize trainer
trainer = SFTTrainer(
    model_name="mistralai/Mistral-7B-v0.1",
    training_config=training_config,
    lora_config=lora_config,
    quantization="4bit"
)

# Train
result = trainer.train(
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"]
)

print(f"Training complete. Final loss: {result.final_loss:.4f}")

# Save model
trainer.save_model(merge_lora=True)
```

### DPO Training

```python
from largeforge.training import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    beta=0.1,
    loss_type="sigmoid"
)

trainer = DPOTrainer(
    model_path="./output/sft-model",
    dpo_config=dpo_config,
    training_config=training_config,
    lora_config=lora_config
)

result = trainer.train(
    train_dataset=preference_dataset["train"],
    eval_dataset=preference_dataset["validation"]
)
```

---

## Data Generation

### Synthetic Data Generator

```python
from largeforge.data import SyntheticDataGenerator

# Initialize generator
generator = SyntheticDataGenerator(
    teacher_model="meta-llama/Llama-2-70b-chat-hf",
    quantization="4bit"
)

# Generate instruction data
prompts = [
    "Explain how to sort a list in Python",
    "What are the benefits of using TypeScript?",
    "How does garbage collection work in Java?"
]

data = generator.generate_instruction_data(
    prompts=prompts,
    system_prompt="You are a helpful coding assistant.",
    temperature=0.7,
    max_tokens=1024
)

# Save dataset
generator.save_dataset(data, "./data/synthetic.jsonl", format="jsonl")
```

### Preference Data Generation

```python
preference_data = generator.generate_preference_data(
    prompts=prompts,
    num_responses=2,
    temperature_range=(0.5, 1.0)
)

generator.save_dataset(preference_data, "./data/preferences.jsonl")
```

---

## Quantization

### AWQ Quantization

```python
from largeforge.inference import quantize_awq

result = quantize_awq(
    model_path="./models/my-model",
    output_path="./models/my-model-awq",
    calibration_dataset="c4",
    num_calibration_samples=128,
    bits=4
)

print(f"Compression ratio: {result.compression_ratio:.2f}x")
print(f"Original size: {result.original_size_gb:.2f} GB")
print(f"Quantized size: {result.quantized_size_gb:.2f} GB")
```

### GPTQ Quantization

```python
from largeforge.inference import quantize_gptq

result = quantize_gptq(
    model_path="./models/my-model",
    output_path="./models/my-model-gptq",
    bits=4,
    group_size=128
)
```

### Merge LoRA Weights

```python
from largeforge.inference import merge_lora_weights

merge_lora_weights(
    base_model_path="mistralai/Mistral-7B-v0.1",
    adapter_path="./output/lora-adapter",
    output_path="./models/merged-model"
)
```

---

## Data Types

### CompletionResponse

```python
@dataclass
class CompletionResponse:
    id: str
    object: str  # "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage

@dataclass
class CompletionChoice:
    index: int
    text: str
    finish_reason: str  # "stop", "length"

@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
```

### ChatCompletionResponse

```python
@dataclass
class ChatCompletionResponse:
    id: str
    object: str  # "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage

@dataclass
class ChatChoice:
    index: int
    message: ChatMessage
    finish_reason: str

@dataclass
class ChatMessage:
    role: str  # "system", "user", "assistant"
    content: str
```

### Expert

```python
@dataclass
class Expert:
    name: str
    model_path: str
    description: str
    domains: list[str]
    keywords: list[str]
    endpoint: str | None
    status: str  # "active", "inactive", "loading", "error"
    load: float
    request_count: int
    avg_latency_ms: float
```

---

## Error Handling

### Exception Types

```python
from largeforge.exceptions import (
    LargeForgeError,        # Base exception
    AuthenticationError,     # Invalid API key
    RateLimitError,         # Rate limit exceeded
    InvalidRequestError,     # Bad request
    ModelNotFoundError,      # Model doesn't exist
    APIConnectionError,      # Network issues
    APITimeoutError,         # Request timeout
    InternalServerError      # Server error
)
```

### Handling Errors

```python
from largeforge.exceptions import RateLimitError, APIConnectionError

try:
    response = client.completions.create(
        model="largeforge-7b",
        prompt="Hello!",
        max_tokens=100
    )
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after} seconds")
except APIConnectionError as e:
    print(f"Connection error: {e}")
except LargeForgeError as e:
    print(f"API error: {e.code} - {e.message}")
```

### Retry Logic

```python
import time
from largeforge.exceptions import RateLimitError

def generate_with_retry(client, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.completions.create(
                model="largeforge-7b",
                prompt=prompt,
                max_tokens=256
            )
        except RateLimitError as e:
            if attempt < max_retries - 1:
                time.sleep(e.retry_after or 60)
            else:
                raise
```

---

## Advanced Usage

### Custom HTTP Client

```python
import httpx

# Custom HTTP client with proxy
http_client = httpx.Client(
    proxies="http://proxy.example.com:8080",
    verify=False  # For self-signed certs
)

client = LargeForgeClient(
    api_key="your-key",
    http_client=http_client
)
```

### Request Hooks

```python
def log_request(request):
    print(f"Request: {request.method} {request.url}")

def log_response(response):
    print(f"Response: {response.status_code}")

client = LargeForgeClient(
    api_key="your-key",
    request_hooks=[log_request],
    response_hooks=[log_response]
)
```

### Context Manager

```python
with LargeForgeClient(api_key="your-key") as client:
    response = client.completions.create(
        model="largeforge-7b",
        prompt="Hello!",
        max_tokens=100
    )
# Client is properly closed
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| LARGEFORGE_API_KEY | API key for authentication |
| LARGEFORGE_BASE_URL | Custom API base URL |
| LARGEFORGE_TIMEOUT | Request timeout in seconds |
| LARGEFORGE_MAX_RETRIES | Maximum retry attempts |
| LARGEFORGE_LOG_LEVEL | Logging level (DEBUG, INFO, etc.) |

---

## Changelog

### v1.0.0
- Initial SDK release
- Completions and Chat Completions API
- Expert routing
- Training utilities
- Quantization tools

---

*For more examples, see the [examples](./examples/) directory.*
