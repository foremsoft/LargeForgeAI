# Configuration Reference

Complete reference for LargeForgeAI configuration options.

---

## Table of Contents

1. [Configuration Methods](#configuration-methods)
2. [Global Configuration](#global-configuration)
3. [Training Configuration](#training-configuration)
4. [Inference Configuration](#inference-configuration)
5. [Router Configuration](#router-configuration)
6. [Environment Variables](#environment-variables)

---

## Configuration Methods

LargeForgeAI supports multiple configuration methods with the following precedence (highest to lowest):

1. **Command-line arguments** - Highest priority
2. **Environment variables**
3. **Configuration file** (`~/.largeforge/config.yaml`)
4. **Default values** - Lowest priority

### Configuration File Location

Default: `~/.largeforge/config.yaml`

Override with: `LARGEFORGE_CONFIG=/path/to/config.yaml`

---

## Global Configuration

### config.yaml

```yaml
# ~/.largeforge/config.yaml

# =============================================================================
# Global Settings
# =============================================================================
global:
  # Default model to use
  default_model: "mistralai/Mistral-7B-v0.1"

  # Device configuration
  device: "auto"  # auto, cuda, cuda:0, cpu

  # Default precision
  dtype: "bfloat16"  # float32, float16, bfloat16

  # Cache directories
  cache_dir: "~/.cache/largeforge"
  models_dir: "~/.largeforge/models"

# =============================================================================
# Logging Configuration
# =============================================================================
logging:
  # Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
  level: "INFO"

  # Log format: text, json
  format: "json"

  # Log file (optional)
  file: "~/.largeforge/logs/largeforge.log"

  # Rotate logs
  max_size_mb: 100
  backup_count: 5

  # Include timestamps
  timestamps: true

# =============================================================================
# HuggingFace Configuration
# =============================================================================
huggingface:
  # Token (or use HF_TOKEN env var)
  token: null

  # Cache directory
  cache_dir: "~/.cache/huggingface"

  # Offline mode
  offline: false

  # Trust remote code
  trust_remote_code: true

# =============================================================================
# Experiment Tracking
# =============================================================================
tracking:
  # Weights & Biases
  wandb:
    enabled: false
    project: "largeforge"
    entity: null
    tags: []

  # TensorBoard
  tensorboard:
    enabled: true
    log_dir: "./logs/tensorboard"
```

---

## Training Configuration

### SFT Training

```yaml
# training_config.yaml

# =============================================================================
# Model Configuration
# =============================================================================
model:
  name: "mistralai/Mistral-7B-v0.1"
  revision: "main"
  trust_remote_code: true

# =============================================================================
# Dataset Configuration
# =============================================================================
dataset:
  path: "./data/train.json"
  format: "alpaca"  # alpaca, sharegpt, custom
  split: "train"
  max_samples: null  # null for all
  validation_split: 0.1

  # Preprocessing
  max_length: 2048
  truncation: true
  padding: "max_length"

# =============================================================================
# Training Hyperparameters
# =============================================================================
training:
  output_dir: "./output"

  # Epochs and steps
  num_epochs: 3
  max_steps: -1  # -1 for full epochs

  # Batch size
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 4
  gradient_accumulation_steps: 4

  # Effective batch size = per_device * accumulation * num_gpus
  # Example: 4 * 4 * 1 = 16

  # Learning rate
  learning_rate: 2.0e-5
  weight_decay: 0.01
  max_grad_norm: 1.0

  # Learning rate schedule
  lr_scheduler_type: "cosine"  # linear, cosine, constant, polynomial
  warmup_ratio: 0.1
  warmup_steps: 0  # Overrides warmup_ratio if > 0

  # Optimizer
  optim: "adamw_torch"  # adamw_torch, adamw_8bit, paged_adamw_8bit

# =============================================================================
# Precision & Memory
# =============================================================================
precision:
  fp16: false
  bf16: true
  tf32: true

memory:
  gradient_checkpointing: true
  use_reentrant: false

# =============================================================================
# LoRA Configuration
# =============================================================================
lora:
  enabled: true
  r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  bias: "none"
  task_type: "CAUSAL_LM"

# =============================================================================
# Quantization (for QLoRA)
# =============================================================================
quantization:
  enabled: true
  bits: 4
  quant_type: "nf4"  # nf4, fp4
  double_quant: true
  compute_dtype: "bfloat16"

# =============================================================================
# Checkpointing
# =============================================================================
checkpointing:
  save_strategy: "steps"  # steps, epoch, no
  save_steps: 500
  save_total_limit: 3
  resume_from_checkpoint: null

# =============================================================================
# Evaluation
# =============================================================================
evaluation:
  eval_strategy: "steps"
  eval_steps: 500
  eval_delay: 0  # Skip first N steps

# =============================================================================
# Early Stopping
# =============================================================================
early_stopping:
  enabled: false
  patience: 3
  threshold: 0.001
  metric: "eval_loss"

# =============================================================================
# Logging
# =============================================================================
logging:
  logging_steps: 10
  logging_first_step: true
  report_to:
    - "tensorboard"
    # - "wandb"

# =============================================================================
# Distributed Training
# =============================================================================
distributed:
  # Automatically detected, but can override
  strategy: "auto"  # auto, ddp, fsdp, deepspeed

  # FSDP settings (if using fsdp)
  fsdp:
    sharding_strategy: "FULL_SHARD"
    cpu_offload: false
    transformer_layer_cls_to_wrap: null

  # DeepSpeed settings (if using deepspeed)
  deepspeed:
    config_file: null
    stage: 2
```

### DPO Training

```yaml
# dpo_config.yaml

model:
  path: "./output/sft-model"  # Start from SFT model

dataset:
  path: "./data/preferences.json"
  format: "dpo"  # prompt, chosen, rejected

training:
  output_dir: "./output/dpo-model"
  num_epochs: 1
  per_device_train_batch_size: 2
  learning_rate: 5.0e-7

# DPO-specific settings
dpo:
  beta: 0.1  # KL penalty coefficient
  loss_type: "sigmoid"  # sigmoid, hinge, ipo
  label_smoothing: 0.0
  reference_free: false

  # Prompt/response lengths
  max_prompt_length: 512
  max_length: 1024

  # Reference model
  reference_model: null  # null = copy of base model
```

---

## Inference Configuration

### Inference Server

```yaml
# inference_config.yaml

# =============================================================================
# Server Configuration
# =============================================================================
server:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  timeout: 300

  # CORS
  cors:
    enabled: true
    origins: ["*"]
    methods: ["GET", "POST"]

  # Authentication
  auth:
    enabled: true
    api_keys:
      - "sk-..."
    # Or use file
    api_keys_file: "./api_keys.txt"

  # Rate limiting
  rate_limit:
    enabled: true
    requests_per_minute: 60
    burst: 10

# =============================================================================
# Model Configuration
# =============================================================================
model:
  path: "./models/my-model"
  revision: null
  trust_remote_code: true

# =============================================================================
# Backend Configuration
# =============================================================================
backend: "vllm"  # vllm, transformers

# vLLM settings (if backend = vllm)
vllm:
  tensor_parallel_size: 1
  max_model_len: 4096
  gpu_memory_utilization: 0.9
  enforce_eager: false
  enable_prefix_caching: true

  # Batching
  max_num_batched_tokens: null
  max_num_seqs: 256

  # Quantization
  quantization: null  # awq, gptq, squeezellm

# Transformers settings (if backend = transformers)
transformers:
  device_map: "auto"
  torch_dtype: "bfloat16"
  load_in_4bit: false
  load_in_8bit: false

# =============================================================================
# Generation Defaults
# =============================================================================
generation:
  max_tokens: 256
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  repetition_penalty: 1.0
  stop_sequences: []

# =============================================================================
# Logging & Metrics
# =============================================================================
logging:
  level: "INFO"
  access_log: true
  log_prompts: false  # Privacy: don't log user prompts

metrics:
  enabled: true
  port: 9090
  path: "/metrics"
```

---

## Router Configuration

### Router Service

```yaml
# router_config.yaml

# =============================================================================
# Server Configuration
# =============================================================================
server:
  host: "0.0.0.0"
  port: 8080
  workers: 4

# =============================================================================
# Classifier Configuration
# =============================================================================
classifier:
  type: "hybrid"  # keyword, neural, hybrid

  # Keyword classifier settings
  keyword:
    case_sensitive: false

  # Neural classifier settings
  neural:
    model: "sentence-transformers/all-MiniLM-L6-v2"
    threshold: 0.5
    cache_embeddings: true

  # Hybrid settings
  hybrid:
    keyword_weight: 0.4
    neural_weight: 0.6
    keyword_threshold: 0.8  # Use keyword if confidence > threshold

# =============================================================================
# Expert Configuration
# =============================================================================
experts:
  # Default expert for fallback
  default: "general-assistant"

  # Expert definitions
  definitions:
    - name: "code-expert"
      description: "Programming and code generation"
      endpoint: "http://localhost:8001"
      domains:
        - programming
        - debugging
        - code-review
      keywords:
        - code
        - function
        - class
        - debug
        - error
        - programming
        - python
        - javascript
      priority: 1
      timeout: 30
      max_retries: 2

    - name: "writing-expert"
      description: "Creative and technical writing"
      endpoint: "http://localhost:8002"
      domains:
        - creative-writing
        - documentation
        - marketing
      keywords:
        - write
        - essay
        - article
        - story
        - blog
        - content
      priority: 0
      timeout: 30

    - name: "general-assistant"
      description: "General purpose assistant"
      endpoint: "http://localhost:8000"
      domains:
        - general
      keywords: []
      priority: -1  # Fallback

# =============================================================================
# Load Balancing
# =============================================================================
load_balancing:
  enabled: true
  strategy: "round_robin"  # round_robin, least_connections, random
  health_check_interval: 30

# =============================================================================
# Circuit Breaker
# =============================================================================
circuit_breaker:
  enabled: true
  failure_threshold: 5
  recovery_timeout: 60
  half_open_requests: 3

# =============================================================================
# Caching
# =============================================================================
caching:
  enabled: false
  ttl: 3600
  max_size: 1000
  backend: "memory"  # memory, redis
```

---

## Environment Variables

### Complete Reference

```bash
# =============================================================================
# Core Configuration
# =============================================================================
LARGEFORGE_HOME=~/.largeforge              # Base directory
LARGEFORGE_CONFIG=~/.largeforge/config.yaml # Config file path
LARGEFORGE_LOG_LEVEL=INFO                  # Logging level
LARGEFORGE_API_KEY=sk-...                  # API key for serving

# =============================================================================
# HuggingFace
# =============================================================================
HF_TOKEN=hf_...                            # HuggingFace token
HF_HOME=~/.cache/huggingface               # HuggingFace cache
HF_DATASETS_CACHE=~/.cache/huggingface/datasets
TRANSFORMERS_CACHE=~/.cache/huggingface/hub
HF_HUB_OFFLINE=0                           # Offline mode (0/1)

# =============================================================================
# GPU Configuration
# =============================================================================
CUDA_VISIBLE_DEVICES=0,1                   # GPU selection
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
CUDA_LAUNCH_BLOCKING=0                     # Debug: 1 for sync CUDA calls

# =============================================================================
# Weights & Biases
# =============================================================================
WANDB_API_KEY=...                          # W&B API key
WANDB_PROJECT=largeforge                   # Project name
WANDB_ENTITY=...                           # Team/user name
WANDB_MODE=online                          # online, offline, disabled
WANDB_DIR=./wandb                          # Local directory

# =============================================================================
# Server Configuration
# =============================================================================
LARGEFORGE_HOST=0.0.0.0                    # Server host
LARGEFORGE_PORT=8000                       # Server port
LARGEFORGE_WORKERS=1                       # Number of workers
LARGEFORGE_TIMEOUT=300                     # Request timeout

# =============================================================================
# Memory Optimization
# =============================================================================
PYTORCH_NO_CUDA_MEMORY_CACHING=0           # Disable caching (debug)
TOKENIZERS_PARALLELISM=false               # Disable tokenizer parallelism
OMP_NUM_THREADS=8                          # OpenMP threads
MKL_NUM_THREADS=8                          # MKL threads
```

### Loading Environment Variables

```bash
# From .env file
source .env

# Or with python-dotenv
# .env file is automatically loaded if present

# Or with direnv
# .envrc file is automatically loaded
```

---

## Configuration Validation

### Validate Configuration File

```bash
largeforge config validate ./training_config.yaml
```

### Show Effective Configuration

```bash
# Show merged configuration from all sources
largeforge config show

# Show specific section
largeforge config show --section training
```

### Generate Default Configuration

```bash
# Generate default config file
largeforge config init

# Generate training config template
largeforge config template training > training_config.yaml

# Generate inference config template
largeforge config template inference > inference_config.yaml
```

---

## Best Practices

### Training

1. **Start with defaults** and adjust based on results
2. **Use gradient checkpointing** for memory efficiency
3. **Use 4-bit quantization** for training on consumer GPUs
4. **Set warmup_ratio to 0.1** for stable training
5. **Use cosine scheduler** for most tasks

### Inference

1. **Use vLLM backend** for production workloads
2. **Set gpu_memory_utilization to 0.9** for balance
3. **Enable prefix caching** for repetitive prompts
4. **Use quantized models** (AWQ) for efficiency

### Security

1. **Never commit API keys** to version control
2. **Use environment variables** for secrets
3. **Enable authentication** in production
4. **Disable prompt logging** for privacy

---

*For more examples, see the [examples](../examples/) directory.*
