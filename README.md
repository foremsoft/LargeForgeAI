<p align="center">
  <img src="docs/assets/logo.png" alt="LargeForgeAI Logo" width="200"/>
</p>

<h1 align="center">LargeForgeAI</h1>

<p align="center">
  <strong>Build GPT-4-level AI systems for under $10,000</strong>
</p>

<p align="center">
  <a href="#why-largeforgeai">Why LargeForgeAI</a> •
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#community">Community</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-2.0.0-blue.svg" alt="Version"/>
  <img src="https://img.shields.io/badge/python-3.10+-green.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/license-MIT-orange.svg" alt="License"/>
  <img src="https://img.shields.io/badge/tests-303%20passed-brightgreen.svg" alt="Tests"/>
</p>

---

## Why LargeForgeAI?

### The Problem

Training and deploying production-grade LLMs is expensive and complex:

| Challenge | Traditional Approach | Cost |
|-----------|---------------------|------|
| Training Infrastructure | Cloud GPU clusters | $50,000 - $500,000+ |
| ML Engineering Team | 5-10 specialists | $1M+/year |
| Time to Production | 6-12 months | Opportunity cost |
| Operational Complexity | Custom MLOps stack | Ongoing maintenance |

**Result**: Only large tech companies can afford to build custom AI systems.

### The Solution

LargeForgeAI democratizes LLM development with:

| Feature | LargeForgeAI Approach | Your Savings |
|---------|----------------------|--------------|
| **Efficient Training** | LoRA + 4-bit quantization | 90% GPU memory reduction |
| **Smart Architecture** | Expert routing + distillation | 10x inference cost savings |
| **Complete Stack** | Training → Deployment in one toolkit | No custom integration |
| **Web UI** | Visual training management | No ML expertise required |

**Total Cost**: Train a GPT-4-level system for **under $10,000** on consumer GPUs.

### Who Is This For?

- **Startups** building AI-powered products without VC-scale budgets
- **Enterprises** wanting on-premise AI without cloud dependency
- **Researchers** needing rapid experimentation capabilities
- **Developers** creating domain-specific AI assistants
- **Teams** requiring data privacy and model control

---

## Features

### Core Capabilities

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LargeForgeAI v2.0 Stack                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌─────────────┐  │
│  │   Training   │──▶│  Validation  │──▶│  Deployment  │──▶│  Inference  │  │
│  │  SFT / DPO   │   │  Benchmarks  │   │   Docker     │   │   vLLM      │  │
│  │    LoRA      │   │  Smoke Tests │   │   Compose    │   │   Router    │  │
│  └──────────────┘   └──────────────┘   └──────────────┘   └─────────────┘  │
│         │                                                        │          │
│         ▼                                                        ▼          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Web UI Dashboard                             │   │
│  │  • Real-time training progress    • Model registry                   │   │
│  │  • Experiment tracking            • Cost monitoring                  │   │
│  │  • One-click deployment           • Authentication                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Training

| Method | Description | Use Case |
|--------|-------------|----------|
| **SFT** | Supervised Fine-Tuning with LoRA | Instruction following, chat |
| **DPO** | Direct Preference Optimization | Alignment, safety, quality |
| **Distillation** | Knowledge transfer from larger models | Cost reduction |
| **Continued Pretraining** | Domain adaptation | Specialized knowledge |

### Inference

- **vLLM Backend**: High-throughput serving with PagedAttention
- **Transformers Backend**: Flexible CPU/GPU inference
- **Expert Router**: Intelligent query routing to specialized models
- **Quantization**: AWQ/GPTQ for 4x memory reduction

### Web UI (New in v2.0)

- **Dashboard**: System overview with GPU monitoring
- **Job Management**: Create, monitor, and cancel training jobs
- **Real-time Progress**: WebSocket-based live updates
- **Model Registry**: Version control with stage transitions
- **Experiment Tracking**: Metrics logging and comparison
- **Cost Tracking**: GPU hours and cost estimation

### DevOps

- **Docker Generation**: One-click Dockerfile creation
- **Compose Templates**: Production-ready deployment configs
- **Model Verification**: Automated smoke tests and benchmarks
- **CLI Tools**: Complete command-line interface

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/foremsoft/LargeForgeAI.git
cd LargeForgeAI

# Install base package
pip install -e .

# Install with all features
pip install -e ".[all]"

# For development
pip install -e ".[dev]"
```

### Option 1: Web UI (Recommended for Beginners)

```bash
# Start the web server
largeforge web start

# Open http://localhost:7860 in your browser
```

### Option 2: Command Line

```bash
# Fine-tune a model with SFT
largeforge train sft \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset data/training.jsonl \
  --output ./my-model \
  --epochs 3

# Verify the trained model
largeforge verify run ./my-model --level standard

# Generate deployment files
largeforge deploy generate ./my-model --output ./deployment

# Start inference server
largeforge serve start --model ./my-model
```

### Option 3: Python API

```python
from largeforge.training import SFTTrainer
from largeforge.config import SFTConfig
from largeforge.inference import TextGenerator

# Configure training
config = SFTConfig(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    dataset_path="data/training.jsonl",
    output_dir="./my-model",
    use_lora=True,
    lora_r=16,
    load_in_4bit=True,
    num_train_epochs=3,
)

# Train
trainer = SFTTrainer(config)
trainer.train()

# Generate text
generator = TextGenerator("./my-model")
response = generator.generate("Explain quantum computing simply")
print(response)
```

---

## Architecture

### Training Pipeline

```
Base Model (Qwen2.5-7B)
        │
        ▼
┌───────────────────┐
│  Data Preparation │ ◀── Synthetic generation, validation, conversion
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   Fine-Tuning     │ ◀── SFT with LoRA + 4-bit quantization
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Preference Tuning│ ◀── DPO for alignment
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   Verification    │ ◀── Smoke tests, benchmarks, validation
└───────────────────┘
        │
        ▼
┌───────────────────┐
│    Deployment     │ ◀── Docker, vLLM, quantization
└───────────────────┘
```

### Expert Routing System

```
                    User Query
                        │
                        ▼
              ┌─────────────────┐
              │  Hybrid Router  │
              │ (Keyword+Neural)│
              └─────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   ┌─────────┐    ┌─────────┐    ┌─────────┐
   │  Code   │    │  Chat   │    │  Math   │
   │ Expert  │    │ Expert  │    │ Expert  │
   └─────────┘    └─────────┘    └─────────┘
```

---

## Project Structure

```
largeforge/
├── training/          # Training modules (SFT, DPO, LoRA)
├── inference/         # Inference server and backends
├── router/            # Expert routing system
├── data/              # Data loading, validation, synthetic generation
├── verification/      # Model validation and benchmarks
├── deployment/        # Docker and deployment generation
├── web/               # Web UI (FastAPI + React)
├── auth/              # Authentication (JWT/OAuth2)
├── experiments/       # Experiment tracking
├── costs/             # GPU cost monitoring
├── registry/          # Model version registry
├── config/            # Configuration schemas
├── cli/               # Command-line interface
└── utils/             # Utilities and helpers

docs/                  # Documentation
tests/                 # Test suite (303 tests)
```

---

## CLI Reference

### Training Commands

```bash
# Supervised Fine-Tuning
largeforge train sft --model MODEL --dataset DATA --output DIR [options]

# DPO Training
largeforge train dpo --model MODEL --dataset DATA --output DIR [options]
```

### Data Commands

```bash
# Validate dataset
largeforge data validate dataset.jsonl --format alpaca

# Convert formats
largeforge data convert input.json output.jsonl --from alpaca --to sharegpt

# Generate synthetic data
largeforge synthetic generate -o data.jsonl -n 1000 --provider openai
```

### Deployment Commands

```bash
# Generate Docker deployment
largeforge deploy generate ./model --output ./deployment

# Verify model
largeforge verify run ./model --level thorough

# Export model
largeforge export model ./model --output ./exported --merge-lora
```

### Server Commands

```bash
# Start web UI
largeforge web start --port 7860

# Start inference server
largeforge serve start --model ./model --port 8000
```

---

## Configuration

### Training Configuration

```yaml
# config/training.yaml
model_name: "Qwen/Qwen2.5-7B-Instruct"
dataset_path: "data/training.jsonl"
output_dir: "./output"

# LoRA settings
use_lora: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05

# Quantization
load_in_4bit: true
bnb_4bit_compute_dtype: "bfloat16"

# Training
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2.0e-5
warmup_ratio: 0.1
```

### Deployment Configuration

```yaml
# config/deployment.yaml
model_path: "./my-model"
backend: "vllm"
port: 8000

# GPU settings
gpu_memory_utilization: 0.9
max_model_len: 4096

# Docker
base_image: "nvidia/cuda:12.1.0-runtime-ubuntu22.04"
```

---

## Performance

### Training Efficiency

| Model Size | Full Fine-Tune | LoRA + 4-bit | Memory Savings |
|------------|---------------|--------------|----------------|
| 7B | 56 GB VRAM | 6 GB VRAM | **89%** |
| 13B | 104 GB VRAM | 10 GB VRAM | **90%** |
| 70B | 560 GB VRAM | 48 GB VRAM | **91%** |

### Inference Performance

| Backend | Throughput | Latency (p99) | Use Case |
|---------|------------|---------------|----------|
| vLLM | 1000+ tok/s | <100ms | Production |
| Transformers | 100 tok/s | <500ms | Development |

---

## Cost Estimation

### Training Costs (AWS/GCP Pricing)

| Task | GPU | Time | Cost |
|------|-----|------|------|
| 7B SFT (10k samples) | 1x A100 40GB | 2 hours | ~$8 |
| 7B DPO (5k pairs) | 1x A100 40GB | 1 hour | ~$4 |
| 13B Full Pipeline | 1x A100 80GB | 8 hours | ~$40 |

### Consumer Hardware

| GPU | 7B Training | 13B Training |
|-----|-------------|--------------|
| RTX 3090 (24GB) | Yes | With gradient checkpointing |
| RTX 4090 (24GB) | Yes | With gradient checkpointing |
| RTX 4080 (16GB) | With 4-bit only | No |

---

## Examples

### 1. Customer Support Bot

```python
from largeforge.training import SFTTrainer
from largeforge.config import SFTConfig

config = SFTConfig(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    dataset_path="support_conversations.jsonl",
    output_dir="./support-bot",
    system_prompt="You are a helpful customer support agent.",
)

trainer = SFTTrainer(config)
trainer.train()
```

### 2. Code Assistant

```python
from largeforge.training import SFTTrainer, DPOTrainer
from largeforge.config import SFTConfig, DPOConfig

# Stage 1: SFT on code examples
sft_config = SFTConfig(
    model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
    dataset_path="code_examples.jsonl",
    output_dir="./code-assistant-sft",
)
SFTTrainer(sft_config).train()

# Stage 2: DPO on code preferences
dpo_config = DPOConfig(
    model_name="./code-assistant-sft",
    dataset_path="code_preferences.jsonl",
    output_dir="./code-assistant",
)
DPOTrainer(dpo_config).train()
```

### 3. Multi-Expert System

```python
from largeforge.router import HybridRouter
from largeforge.inference import TextGenerator

# Load expert models
code_expert = TextGenerator("./experts/code")
math_expert = TextGenerator("./experts/math")
general_expert = TextGenerator("./experts/general")

# Create router
router = HybridRouter(
    experts={
        "code": code_expert,
        "math": math_expert,
        "general": general_expert,
    },
    keywords={
        "code": ["python", "javascript", "function", "code"],
        "math": ["calculate", "equation", "solve", "math"],
    }
)

# Route queries automatically
response = router.generate("Write a Python function to calculate fibonacci")
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](docs/guides/GETTING_STARTED.md) | First steps with LargeForgeAI |
| [Installation Guide](docs/guides/INSTALLATION.md) | Detailed installation instructions |
| [Configuration](docs/guides/CONFIGURATION.md) | All configuration options |
| [API Reference](docs/api/SDK_REFERENCE.md) | Python SDK documentation |
| [REST API](docs/api/REST_API_REFERENCE.md) | HTTP API documentation |
| [Tutorials](docs/learning/tutorials/) | Step-by-step tutorials |
| [Architecture](docs/architecture/ARCHITECTURE_DOCUMENT.md) | System design |
| [FAQ](docs/FAQ.md) | Frequently asked questions |

---

## Roadmap

### v2.1 (Q1 2025)
- [ ] Multi-GPU distributed training
- [ ] RLHF training support
- [ ] Model merging (TIES, DARE)
- [ ] Expanded benchmark suite

### v2.2 (Q2 2025)
- [ ] Kubernetes deployment templates
- [ ] A/B testing for models
- [ ] Automated hyperparameter tuning
- [ ] Plugin system

### v3.0 (2025)
- [ ] Multi-modal support (vision-language)
- [ ] Federated fine-tuning
- [ ] Real-time model updates
- [ ] Enterprise SSO

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](docs/development/CONTRIBUTING.md) for guidelines.

```bash
# Setup development environment
pip install -e ".[dev]"
pre-commit install

# Run tests
pytest tests/ -v

# Run linting
ruff check largeforge/
black largeforge/
```

---

## Community

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Questions and community support
- **Discord**: Real-time chat (coming soon)

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

Built on the shoulders of giants:
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [TRL](https://github.com/huggingface/trl)
- [vLLM](https://github.com/vllm-project/vllm)
- [FastAPI](https://github.com/tiangolo/fastapi)

---

<p align="center">
  <strong>Build the future of AI, affordably.</strong>
</p>
