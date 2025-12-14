# Getting Started with LargeForgeAI

Welcome to LargeForgeAI! This guide will help you understand the platform and get your first model trained and deployed.

---

## What is LargeForgeAI?

LargeForgeAI is an open-source platform for training, fine-tuning, and deploying Large Language Models (LLMs). It enables you to:

- **Train domain-specific models** using techniques like LoRA, DPO, and knowledge distillation
- **Deploy multiple expert models** with intelligent query routing
- **Achieve GPT-4-level performance** at a fraction of the cost
- **Scale efficiently** with quantization and optimized inference

### Key Features

| Feature | Description |
|---------|-------------|
| Efficient Training | LoRA fine-tuning with 4-bit quantization |
| Preference Optimization | DPO and ORPO for alignment |
| Knowledge Distillation | Transfer knowledge from large to small models |
| Expert Routing | Automatic query classification to specialists |
| High-Performance Inference | vLLM-powered serving with continuous batching |
| Easy Deployment | Docker and Kubernetes support |

---

## Prerequisites

Before you begin, ensure you have:

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA with 16GB VRAM | NVIDIA A100 (40/80GB) |
| RAM | 32GB | 64GB+ |
| Storage | 100GB SSD | 500GB+ NVMe |
| CPU | 8 cores | 16+ cores |

### Software Requirements

- Python 3.10 or higher
- CUDA 11.8 or higher
- Docker (for containerized deployment)
- Git

### Accounts (Optional)

- [HuggingFace Account](https://huggingface.co/) - For model access
- [Weights & Biases Account](https://wandb.ai/) - For experiment tracking

---

## Installation

### Quick Install

```bash
pip install largeforge
```

### Install from Source

```bash
git clone https://github.com/largeforgeai/largeforgeai.git
cd largeforgeai
pip install -e ".[all]"
```

### Verify Installation

```bash
largeforge --version
largeforge doctor  # Check system requirements
```

Expected output:
```
LargeForgeAI v1.0.0

System Check:
  [OK] Python 3.11.0
  [OK] CUDA 12.1 available
  [OK] GPU: NVIDIA A100-SXM4-80GB (80GB)
  [OK] PyTorch 2.1.0+cu121
  [OK] Transformers 4.36.0

All checks passed!
```

---

## Quick Start: Your First Fine-Tuned Model

Let's train a simple coding assistant in under 10 minutes.

### Step 1: Prepare Training Data

Create a file `training_data.json`:

```json
[
  {
    "instruction": "Write a Python function to calculate factorial",
    "input": "",
    "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
  },
  {
    "instruction": "Explain what this code does",
    "input": "x = [i**2 for i in range(10)]",
    "output": "This code creates a list of squares from 0 to 81 using list comprehension."
  }
]
```

### Step 2: Train the Model

```bash
largeforge train sft \
  --model mistralai/Mistral-7B-v0.1 \
  --dataset ./training_data.json \
  --output-dir ./my-coding-assistant \
  --lora-r 8 \
  --epochs 3 \
  --batch-size 4
```

### Step 3: Test Your Model

```python
from largeforge import LargeForgeClient

client = LargeForgeClient()
client.load_model("./my-coding-assistant")

response = client.generate(
    prompt="Write a function to reverse a string",
    max_tokens=256
)
print(response)
```

### Step 4: Deploy for Inference

```bash
largeforge serve inference \
  --model ./my-coding-assistant \
  --port 8000
```

Test with curl:
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "my-coding-assistant", "prompt": "Hello!", "max_tokens": 50}'
```

---

## Core Concepts

### Training Methods

LargeForgeAI supports several training approaches:

```
                    ┌─────────────────┐
                    │   Base Model    │
                    │ (e.g., Mistral) │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌─────────┐    ┌─────────┐    ┌─────────┐
        │   SFT   │    │ Distill │    │Pretrain │
        │ (LoRA)  │    │         │    │(Domain) │
        └────┬────┘    └────┬────┘    └────┬────┘
             │              │              │
             └──────────────┼──────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  DPO / ORPO   │
                    │  (Alignment)  │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  Quantization │
                    │  (AWQ/GPTQ)   │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   Deploy as   │
                    │    Expert     │
                    └───────────────┘
```

1. **SFT (Supervised Fine-Tuning)**: Train on instruction-following data
2. **DPO/ORPO**: Align model with human preferences
3. **Distillation**: Transfer knowledge from larger models
4. **Continued Pre-training**: Adapt to new domains

### Expert Routing

Multiple specialized models work together:

```
                        ┌─────────────┐
                        │   Query     │
                        └──────┬──────┘
                               │
                        ┌──────▼──────┐
                        │   Router    │
                        │ (Classifier)│
                        └──────┬──────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
    ┌──────▼──────┐     ┌──────▼──────┐     ┌──────▼──────┐
    │ Code Expert │     │Write Expert │     │ Math Expert │
    └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
           │                   │                   │
           └───────────────────┼───────────────────┘
                               │
                        ┌──────▼──────┐
                        │  Response   │
                        └─────────────┘
```

---

## Next Steps

Now that you understand the basics, explore these guides:

### Tutorials

1. **[First Fine-Tune](./tutorials/first-fine-tune.md)** - Complete SFT walkthrough
2. **[Deploying Experts](./tutorials/deploying-experts.md)** - Multi-expert setup
3. **[Custom Router](./tutorials/custom-router.md)** - Build your own classifier

### Topics

- [Installation Details](./INSTALLATION.md) - Advanced installation options
- [Configuration](./CONFIGURATION.md) - Configuration reference
- [Training Guide](./tutorials/training-guide.md) - Deep dive into training
- [Deployment Guide](./tutorials/deployment-guide.md) - Production deployment

### Reference

- [CLI Reference](../api/CLI_REFERENCE.md) - Command-line interface
- [API Reference](../api/REST_API_REFERENCE.md) - REST API documentation
- [SDK Reference](../api/SDK_REFERENCE.md) - Python SDK

---

## Getting Help

- **Documentation**: [docs.largeforge.ai](https://docs.largeforge.ai)
- **GitHub Issues**: [github.com/largeforgeai/largeforgeai/issues](https://github.com/largeforgeai/largeforgeai/issues)
- **Discord**: [discord.gg/largeforge](https://discord.gg/largeforge)
- **Email**: support@largeforge.ai

---

## Example Projects

Explore complete example projects:

| Project | Description | Complexity |
|---------|-------------|------------|
| [Code Assistant](../examples/code-assistant/) | Programming helper | Beginner |
| [Customer Support Bot](../examples/support-bot/) | FAQ and support | Intermediate |
| [Multi-Expert System](../examples/multi-expert/) | Full routing setup | Advanced |
| [Enterprise Deployment](../examples/enterprise/) | Production-ready | Advanced |

---

*Happy training! If you build something cool, let us know on Discord!*
