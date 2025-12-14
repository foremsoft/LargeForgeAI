# LargeForgeAI Complete Tutorial

**From Zero to Production AI in One Day**

This comprehensive tutorial will guide you through building, training, and deploying a production-ready AI assistant using LargeForgeAI. By the end, you'll have a fully functional AI system running on your own infrastructure.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Prerequisites & Setup](#2-prerequisites--setup)
3. [Understanding the Architecture](#3-understanding-the-architecture)
4. [Data Preparation](#4-data-preparation)
5. [Training Your First Model](#5-training-your-first-model)
6. [Model Verification](#6-model-verification)
7. [Deployment](#7-deployment)
8. [Advanced: Multi-Expert System](#8-advanced-multi-expert-system)
9. [Advanced: DPO Alignment](#9-advanced-dpo-alignment)
10. [Production Best Practices](#10-production-best-practices)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Introduction

### What You'll Build

In this tutorial, you'll create:

1. **A Customer Support Bot** - Fine-tuned to answer questions about your product
2. **A Code Assistant** - Specialized for your tech stack
3. **A Multi-Expert System** - Routes queries to the right specialist

### Time Estimates

| Section | Time | GPU Required |
|---------|------|--------------|
| Setup | 15 min | No |
| Data Preparation | 30 min | No |
| Training | 2-4 hours | Yes |
| Verification | 15 min | Yes |
| Deployment | 15 min | No |
| **Total** | **~5 hours** | |

### What You'll Learn

- How to prepare training data in the right format
- Fine-tuning with LoRA and 4-bit quantization
- Evaluating model quality with automated benchmarks
- Deploying with Docker for production use
- Building expert routing systems
- Aligning models with DPO

---

## 2. Prerequisites & Setup

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | RTX 3080 (10GB) | RTX 4090 (24GB) |
| **RAM** | 16 GB | 32 GB |
| **Storage** | 50 GB SSD | 200 GB NVMe |
| **CPU** | 8 cores | 16 cores |

**Note**: Training is possible on CPU but will be 10-50x slower.

### Software Requirements

- Python 3.10 or 3.11
- CUDA 11.8 or 12.1 (for GPU training)
- Git
- Docker (for deployment)

### Step-by-Step Installation

#### Step 1: Clone the Repository

```bash
git clone https://github.com/foremsoft/LargeForgeAI.git
cd LargeForgeAI
```

#### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
.\venv\Scripts\activate
```

#### Step 3: Install LargeForgeAI

```bash
# Basic installation
pip install -e .

# With all features (recommended)
pip install -e ".[all]"

# For development
pip install -e ".[dev]"
```

#### Step 4: Verify Installation

```bash
# Check version
largeforge --version

# Check system info
largeforge info

# Expected output:
# LargeForgeAI System Information
# ========================================
# Version: 2.0.0
# Python: 3.10.x
# PyTorch: 2.x.x
# Device: cuda
# GPU Count: 1
# BF16 Support: True
```

#### Step 5: Start the Web UI (Optional)

```bash
largeforge web start

# Open http://localhost:7860 in your browser
```

---

## 3. Understanding the Architecture

### The Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LargeForgeAI Training Pipeline                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │   Base   │───▶│   Data   │───▶│  Train   │───▶│  Verify  │          │
│  │  Model   │    │  Prep    │    │  (SFT)   │    │  Model   │          │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘          │
│       │                                               │                  │
│       │              ┌──────────┐                     │                  │
│       └─────────────▶│   DPO    │◀────────────────────┘                  │
│                      │ (Align)  │                                        │
│                      └──────────┘                                        │
│                           │                                              │
│                           ▼                                              │
│                      ┌──────────┐    ┌──────────┐                       │
│                      │  Deploy  │───▶│  Serve   │                       │
│                      │ (Docker) │    │ (vLLM)   │                       │
│                      └──────────┘    └──────────┘                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Concepts

#### Base Model
The pre-trained foundation model (e.g., Qwen2.5-7B-Instruct). This contains general knowledge and capabilities.

#### LoRA (Low-Rank Adaptation)
Instead of updating all model weights (expensive), LoRA adds small trainable matrices. This reduces memory usage by 90%.

#### 4-bit Quantization
Compresses model weights from 16-bit to 4-bit, further reducing memory. Quality impact is minimal with modern techniques.

#### SFT (Supervised Fine-Tuning)
Training the model on input-output pairs to learn specific behaviors.

#### DPO (Direct Preference Optimization)
Training the model to prefer better responses using comparison pairs. Used for alignment.

---

## 4. Data Preparation

### Understanding Data Formats

LargeForgeAI supports multiple formats. Choose based on your data:

#### Alpaca Format (Recommended for Simple Tasks)

```json
{
  "instruction": "What is the return policy?",
  "input": "",
  "output": "Our return policy allows returns within 30 days of purchase. Items must be unused and in original packaging. Refunds are processed within 5-7 business days."
}
```

#### ShareGPT Format (For Conversations)

```json
{
  "conversations": [
    {"role": "system", "content": "You are a helpful customer support agent."},
    {"role": "user", "content": "I want to return my order"},
    {"role": "assistant", "content": "I'd be happy to help with your return. Could you provide your order number?"},
    {"role": "user", "content": "It's ORDER-12345"},
    {"role": "assistant", "content": "Thank you! I've found your order. I'll initiate the return process now."}
  ]
}
```

#### DPO Format (For Preference Training)

```json
{
  "prompt": "Explain machine learning simply.",
  "chosen": "Machine learning is like teaching a computer by example. Instead of programming explicit rules, you show it many examples and it learns patterns. For instance, to recognize cats, you'd show it thousands of cat photos until it learns what makes a cat a cat.",
  "rejected": "Machine learning is a subset of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data."
}
```

### Creating Your Dataset

#### Option 1: Manual Creation

Create a file `data/training.jsonl`:

```jsonl
{"instruction": "What are your business hours?", "input": "", "output": "We're open Monday through Friday, 9 AM to 6 PM EST. We're closed on weekends and major holidays."}
{"instruction": "How do I reset my password?", "input": "", "output": "To reset your password: 1) Go to the login page, 2) Click 'Forgot Password', 3) Enter your email, 4) Check your inbox for a reset link, 5) Create a new password with at least 8 characters."}
{"instruction": "Do you offer refunds?", "input": "", "output": "Yes! We offer full refunds within 30 days of purchase. After 30 days, we provide store credit. To request a refund, contact support@example.com with your order number."}
```

#### Option 2: Synthetic Data Generation

Generate high-quality training data using GPT-4 or Claude:

```bash
# Set your API key
export OPENAI_API_KEY=sk-your-key

# Generate 500 customer support examples
largeforge synthetic generate \
  -o data/synthetic_support.jsonl \
  -n 500 \
  --format sft \
  --provider openai \
  --model gpt-4 \
  --topic "customer support" \
  --topic "product inquiries" \
  --topic "technical troubleshooting"
```

#### Option 3: Convert Existing Data

```bash
# Convert from ShareGPT to Alpaca format
largeforge data convert \
  existing_data.json \
  training_data.jsonl \
  --from sharegpt \
  --to alpaca
```

### Validating Your Data

Always validate before training:

```bash
largeforge data validate data/training.jsonl --format alpaca

# Expected output:
# Validating dataset: data/training.jsonl
# Format: alpaca
# Loaded 500 records
#
# Validation Results:
#   Valid records: 498
#   Invalid records: 2
#
# First 2 errors:
#   Record 145: Missing required field 'output'
#   Record 301: Empty 'instruction' field
```

### Data Quality Guidelines

| Aspect | Poor | Good | Best |
|--------|------|------|------|
| **Quantity** | <100 | 500-1000 | 2000-10000 |
| **Diversity** | Same topic | Multiple topics | Full coverage |
| **Length** | Single sentences | Paragraphs | Varied lengths |
| **Quality** | Typos, errors | Clean, accurate | Expert-written |
| **Format** | Inconsistent | Consistent | With examples |

### Viewing Dataset Statistics

```bash
largeforge data stats data/training.jsonl --detailed

# Output:
# Dataset Statistics: data/training.jsonl
# ==================================================
# Total records: 500
# Detected format: alpaca
#
# Alpaca Format Statistics:
#   Records with input: 50 (10.0%)
#   Records with system: 0 (0.0%)
#
# Instruction length:
#   Min: 15, Max: 245
#   Avg: 67.3
#
# Output length:
#   Min: 50, Max: 1024
#   Avg: 312.5
```

---

## 5. Training Your First Model

### Method 1: Web UI (Easiest)

1. Start the web server:
   ```bash
   largeforge web start
   ```

2. Open http://localhost:7860

3. Click "New Training Job"

4. Fill in the form:
   - **Model**: `Qwen/Qwen2.5-7B-Instruct`
   - **Dataset**: Upload or enter path
   - **Output**: `./my-support-bot`
   - **Epochs**: 3
   - **Learning Rate**: 2e-5

5. Click "Start Training"

6. Monitor progress in real-time

### Method 2: Command Line

```bash
largeforge train sft \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset data/training.jsonl \
  --output ./my-support-bot \
  --epochs 3 \
  --batch-size 4 \
  --learning-rate 2e-5 \
  --lora-r 16 \
  --lora-alpha 32 \
  --load-4bit
```

### Method 3: Python API (Most Control)

```python
from largeforge.training import SFTTrainer
from largeforge.config import SFTConfig

# Configure training
config = SFTConfig(
    # Model
    model_name="Qwen/Qwen2.5-7B-Instruct",

    # Data
    dataset_path="data/training.jsonl",
    data_format="alpaca",
    max_seq_length=2048,

    # Output
    output_dir="./my-support-bot",

    # LoRA settings
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],

    # Quantization
    load_in_4bit=True,
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_quant_type="nf4",

    # Training hyperparameters
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,

    # Optimization
    optim="adamw_torch",
    lr_scheduler_type="cosine",

    # Logging
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
)

# Create trainer
trainer = SFTTrainer(config)

# Train!
trainer.train()

# Save the final model
trainer.save_model()
```

### Understanding Training Output

During training, you'll see:

```
[2024-12-14 10:30:15] Starting training...
[2024-12-14 10:30:16] Loading model: Qwen/Qwen2.5-7B-Instruct
[2024-12-14 10:30:45] Model loaded. Applying LoRA...
[2024-12-14 10:30:46] LoRA applied. Trainable parameters: 4,194,304 (0.06%)
[2024-12-14 10:30:47] Loading dataset: data/training.jsonl
[2024-12-14 10:30:48] Dataset loaded: 500 examples

Epoch 1/3:
  Step 10/375: loss=2.3456, lr=4.0e-6
  Step 20/375: loss=1.8234, lr=8.0e-6
  Step 30/375: loss=1.5123, lr=1.2e-5
  ...
  Step 375/375: loss=0.4521, lr=2.0e-5

Epoch 2/3:
  Step 10/375: loss=0.4123, lr=1.9e-5
  ...

Epoch 3/3:
  ...
  Step 375/375: loss=0.2134, lr=2.0e-7

[2024-12-14 12:45:32] Training complete!
[2024-12-14 12:45:33] Model saved to: ./my-support-bot
```

### Training Tips

| Issue | Solution |
|-------|----------|
| **Out of memory** | Reduce batch size, increase gradient accumulation |
| **Loss not decreasing** | Lower learning rate, check data quality |
| **Overfitting** | Reduce epochs, increase dropout, add more data |
| **Slow training** | Enable bf16, use Flash Attention 2 |

---

## 6. Model Verification

### Why Verify?

Before deploying, ensure your model:
- Loads correctly
- Generates coherent text
- Meets performance requirements
- Doesn't have obvious issues

### Running Verification

#### Quick Smoke Test

```bash
largeforge verify smoke-test ./my-support-bot

# Output:
# Running smoke test on ./my-support-bot
# ========================================
#
# Model Loading:
#   Status: PASS
#   Load time: 12.3s
#   Memory used: 5.8 GB
#
# Text Generation:
#   Test 1 "Hello, my name is": PASS (1.2s)
#   Test 2 "The capital of France is": PASS (0.8s)
#   Test 3 "def fibonacci(n):": PASS (1.5s)
#   Test 4 "Explain quantum computing": PASS (2.1s)
#
# Coherence Check:
#   All outputs coherent: PASS
#
# SMOKE TEST PASSED
```

#### Standard Verification

```bash
largeforge verify run ./my-support-bot --level standard

# Output:
# Model Verification Report
# ========================================
# Model: ./my-support-bot
# Level: STANDARD
#
# Smoke Test: PASS
#
# Benchmarks:
#   Latency (p99):     342ms    [PASS] (threshold: 5000ms)
#   Throughput:        89 tok/s [PASS] (threshold: 10 tok/s)
#   Memory Usage:      5.8 GB   [PASS] (threshold: 24 GB)
#   Consistency:       0.92     [PASS] (threshold: 0.8)
#
# Overall: PASS
#
# Recommendations:
#   - Consider quantization for deployment (AWQ/GPTQ)
#   - Model is suitable for production use
```

#### Thorough Verification

```bash
largeforge verify run ./my-support-bot --level thorough --output report.html
```

### Manual Testing

Test your model interactively:

```python
from largeforge.inference import TextGenerator

generator = TextGenerator("./my-support-bot")

# Test various prompts
test_prompts = [
    "What is your return policy?",
    "How do I contact support?",
    "I have a problem with my order",
]

for prompt in test_prompts:
    response = generator.generate(prompt)
    print(f"Q: {prompt}")
    print(f"A: {response}\n")
```

---

## 7. Deployment

### Generate Deployment Files

```bash
largeforge deploy generate ./my-support-bot --output ./deployment

# Creates:
# deployment/
# ├── Dockerfile
# ├── docker-compose.yml
# ├── requirements.txt
# ├── main.py
# ├── config.yaml
# ├── .env.example
# └── README.md
```

### Review the Generated Files

#### Dockerfile

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3.10 python3-pip

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy model (or mount as volume)
COPY ./model /app/model

# Copy service code
COPY main.py /app/
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
  CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["python", "main.py"]
```

#### docker-compose.yml

```yaml
version: '3.8'

services:
  inference:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./model:/app/model
    environment:
      - MODEL_PATH=/app/model
      - MAX_MODEL_LEN=4096
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Deploy with Docker

```bash
cd deployment

# Build the image
docker-compose build

# Start the service
docker-compose up -d

# Check logs
docker-compose logs -f

# Test the endpoint
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-support-bot",
    "messages": [{"role": "user", "content": "What is your return policy?"}]
  }'
```

### Production Deployment Checklist

- [ ] Enable HTTPS (use nginx/traefik as reverse proxy)
- [ ] Set up authentication
- [ ] Configure rate limiting
- [ ] Set up monitoring (Prometheus + Grafana)
- [ ] Configure logging
- [ ] Set up backups
- [ ] Test failover scenarios
- [ ] Document API for consumers

---

## 8. Advanced: Multi-Expert System

### The Concept

Instead of one large model, use multiple specialized models with intelligent routing:

```
User Query: "Write a Python function to calculate compound interest"
     │
     ▼
┌─────────────────────────┐
│      Hybrid Router      │
│  1. Keyword: "Python"   │──▶ Route to Code Expert
│  2. Neural: code task   │
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│     Code Expert (7B)    │
│  Specialized for code   │
└─────────────────────────┘
     │
     ▼
Response: "def compound_interest(principal, rate, time, n=12):..."
```

### Step 1: Train Specialized Experts

```bash
# Code Expert
largeforge train sft \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --dataset data/code_examples.jsonl \
  --output ./experts/code

# Math Expert
largeforge train sft \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset data/math_problems.jsonl \
  --output ./experts/math

# General Chat Expert
largeforge train sft \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset data/conversations.jsonl \
  --output ./experts/chat
```

### Step 2: Configure the Router

Create `config/router.yaml`:

```yaml
router:
  type: hybrid
  default_expert: chat

  keyword_rules:
    code:
      - python
      - javascript
      - function
      - code
      - programming
      - debug
      - error
    math:
      - calculate
      - equation
      - solve
      - math
      - formula
      - derivative
      - integral

  neural_classifier:
    model: sentence-transformers/all-MiniLM-L6-v2
    threshold: 0.7

experts:
  code:
    path: ./experts/code
    description: "Programming and code-related queries"
  math:
    path: ./experts/math
    description: "Mathematical problems and calculations"
  chat:
    path: ./experts/chat
    description: "General conversation and questions"
```

### Step 3: Run the Expert System

```python
from largeforge.router import HybridRouter, ExpertConfig
from largeforge.inference import TextGenerator

# Load experts
experts = {
    "code": TextGenerator("./experts/code"),
    "math": TextGenerator("./experts/math"),
    "chat": TextGenerator("./experts/chat"),
}

# Create router
router = HybridRouter(
    experts=experts,
    config_path="config/router.yaml"
)

# Query - automatically routed to the right expert
queries = [
    "Write a Python function to sort a list",      # → code expert
    "What is the derivative of x^2?",               # → math expert
    "Tell me about the weather today",              # → chat expert
]

for query in queries:
    response = router.generate(query)
    print(f"Q: {query}")
    print(f"Expert: {router.last_routed_expert}")
    print(f"A: {response}\n")
```

### Step 4: Deploy Expert System

```bash
# Start router service
largeforge serve start-router \
  --config config/router.yaml \
  --port 8080
```

---

## 9. Advanced: DPO Alignment

### What is DPO?

DPO (Direct Preference Optimization) improves model quality by training on preference pairs - examples of good vs. bad responses.

### When to Use DPO

- After SFT, to improve response quality
- To align with specific writing styles
- To reduce harmful outputs
- To prefer certain response formats

### Step 1: Prepare DPO Data

```jsonl
{"prompt": "Explain quantum computing", "chosen": "Quantum computing harnesses quantum mechanics to process information. Unlike classical bits (0 or 1), quantum bits (qubits) can exist in superposition - both states simultaneously. This enables quantum computers to explore many solutions in parallel, making them powerful for specific problems like cryptography and simulation.", "rejected": "Quantum computing is a type of computing that uses quantum mechanics."}
{"prompt": "Write a poem about nature", "chosen": "Whispers of wind through ancient trees,\nDancing leaves on gentle breeze.\nRivers flow with stories old,\nNature's secrets, yet untold.", "rejected": "Nature is nice. Trees are green. Water is blue. The sky is pretty."}
```

### Step 2: Train with DPO

```bash
largeforge train dpo \
  --model ./my-support-bot \
  --dataset data/preferences.jsonl \
  --output ./my-support-bot-aligned \
  --beta 0.1 \
  --epochs 1 \
  --learning-rate 5e-6
```

### Step 3: Python API for DPO

```python
from largeforge.training import DPOTrainer
from largeforge.config import DPOConfig

config = DPOConfig(
    # Start from your SFT model
    model_name="./my-support-bot",

    # DPO-specific data
    dataset_path="data/preferences.jsonl",

    # Output
    output_dir="./my-support-bot-aligned",

    # DPO hyperparameters
    beta=0.1,  # KL penalty coefficient
    loss_type="sigmoid",  # or "hinge"

    # Training
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
)

trainer = DPOTrainer(config)
trainer.train()
```

---

## 10. Production Best Practices

### Model Versioning

Use the built-in registry:

```python
from largeforge.registry import get_registry, ModelStage

registry = get_registry()

# Register your model
model = registry.register(
    name="support-bot",
    path="./my-support-bot",
    base_model="Qwen/Qwen2.5-7B-Instruct",
    description="Customer support chatbot v1",
    tags=["support", "production"],
    metrics={"eval_loss": 0.21, "accuracy": 0.95},
)

# Add a new version
registry.add_version(
    name="support-bot",
    path="./my-support-bot-v2",
    metrics={"eval_loss": 0.18, "accuracy": 0.97},
)

# Promote to production
registry.transition_stage(
    name="support-bot",
    version="v1.0.1",
    stage=ModelStage.PRODUCTION,
)
```

### Experiment Tracking

Track all your experiments:

```python
from largeforge.experiments import ExperimentTracker

tracker = ExperimentTracker()

# Create experiment
exp = tracker.create(
    name="support-bot-lr-sweep",
    config={"learning_rate": 2e-5, "epochs": 3},
    tags=["hyperparameter-search"],
)

# Log metrics during training
tracker.log_metric(exp.id, "loss", 0.5, step=100)
tracker.log_metric(exp.id, "loss", 0.3, step=200)

# Compare experiments
comparison = tracker.compare(
    exp_ids=["exp-001", "exp-002", "exp-003"],
    metrics=["loss", "accuracy"]
)
print(comparison)
```

### Cost Monitoring

Track GPU costs:

```python
from largeforge.costs import get_cost_tracker

tracker = get_cost_tracker()

# Get job costs
costs = tracker.get_job_costs("training-job-001")
print(f"GPU Hours: {costs.gpu_hours:.2f}")
print(f"Estimated Cost: ${costs.estimated_cost_usd:.2f}")

# Monthly summary
summary = tracker.get_monthly_summary(2024, 12)
print(f"Total GPU Hours: {summary.total_gpu_hours:.1f}")
print(f"Total Cost: ${summary.total_cost_usd:.2f}")
```

### Monitoring in Production

Set up proper monitoring:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'largeforge'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

Key metrics to monitor:
- Request latency (p50, p95, p99)
- Tokens per second
- GPU memory utilization
- Error rate
- Queue depth

---

## 11. Troubleshooting

### Common Issues

#### Out of Memory (OOM)

```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size: `--batch-size 2`
2. Increase gradient accumulation: `--gradient-accumulation 8`
3. Enable gradient checkpointing: `--gradient-checkpointing`
4. Use smaller model or more aggressive quantization

#### Slow Training

**Solutions:**
1. Enable Flash Attention: `--flash-attention`
2. Use bf16: `--bf16`
3. Check GPU utilization: `nvidia-smi`
4. Use faster data loading: `--num-workers 4`

#### Poor Model Quality

**Solutions:**
1. Check data quality: `largeforge data validate`
2. Increase training data quantity
3. Adjust learning rate (usually lower)
4. Add more epochs
5. Try DPO alignment

#### Model Won't Load

```
Error: Unable to load model
```

**Solutions:**
1. Check disk space
2. Verify model path exists
3. Check CUDA version compatibility
4. Try loading without quantization first

### Getting Help

1. Check the [FAQ](docs/FAQ.md)
2. Search [GitHub Issues](https://github.com/foremsoft/LargeForgeAI/issues)
3. Open a new issue with:
   - LargeForgeAI version
   - Full error message
   - Hardware specs
   - Steps to reproduce

---

## Conclusion

Congratulations! You've learned how to:

- Prepare high-quality training data
- Fine-tune models with SFT and LoRA
- Verify model quality before deployment
- Deploy with Docker
- Build multi-expert systems
- Align models with DPO
- Follow production best practices

### Next Steps

1. **Experiment** - Try different models and configurations
2. **Scale** - Add more experts and training data
3. **Monitor** - Set up production monitoring
4. **Iterate** - Continuously improve with user feedback

### Resources

- [API Reference](docs/api/SDK_REFERENCE.md)
- [Configuration Guide](docs/guides/CONFIGURATION.md)
- [Architecture Document](docs/architecture/ARCHITECTURE_DOCUMENT.md)
- [Community Discussions](https://github.com/foremsoft/LargeForgeAI/discussions)

---

**Happy building! The future of AI is in your hands.**
