# Installation Guide

This guide covers all installation methods and configurations for LargeForgeAI.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Installation](#quick-installation)
3. [Installation Options](#installation-options)
4. [Docker Installation](#docker-installation)
5. [GPU Setup](#gpu-setup)
6. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Linux (Ubuntu 20.04+), macOS 12+, Windows 10+ (WSL2) |
| Python | 3.10 or higher |
| RAM | 32 GB |
| GPU | NVIDIA with 16GB VRAM, CUDA 11.8+ |
| Storage | 100 GB free space |

### Recommended Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Ubuntu 22.04 LTS |
| Python | 3.11 |
| RAM | 64 GB+ |
| GPU | NVIDIA A100 (40/80GB) or H100 |
| Storage | 500 GB+ NVMe SSD |
| Network | 1 Gbps+ for model downloads |

### Supported GPUs

| GPU | VRAM | Max Model Size | Notes |
|-----|------|----------------|-------|
| RTX 3090/4090 | 24GB | 7B (4-bit) | Consumer, good for development |
| A10 | 24GB | 7B (4-bit) | Cloud instance |
| A100-40GB | 40GB | 13B (4-bit), 7B (16-bit) | Recommended for training |
| A100-80GB | 80GB | 34B (4-bit), 13B (16-bit) | Best single-GPU option |
| H100 | 80GB | 34B (4-bit) | Fastest training |

---

## Quick Installation

### Using pip

```bash
# Basic installation
pip install largeforge

# With all optional dependencies
pip install largeforge[all]

# Verify installation
largeforge --version
largeforge doctor
```

### Using conda

```bash
# Create environment
conda create -n largeforge python=3.11
conda activate largeforge

# Install PyTorch with CUDA
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

# Install LargeForgeAI
pip install largeforge
```

---

## Installation Options

### Core Installation

Minimal installation for inference only:

```bash
pip install largeforge
```

**Includes:**
- Inference server
- Python client
- Basic utilities

### Training Installation

For model training and fine-tuning:

```bash
pip install largeforge[training]
```

**Additional packages:**
- peft (LoRA)
- trl (DPO/ORPO)
- bitsandbytes (quantized training)
- accelerate (distributed training)

### Full Installation

Everything including development tools:

```bash
pip install largeforge[all]
```

**Additional packages:**
- All training dependencies
- vLLM (high-performance inference)
- autoawq, auto-gptq (quantization)
- wandb (experiment tracking)
- pytest, ruff, mypy (development)

### Development Installation

For contributors:

```bash
git clone https://github.com/largeforgeai/largeforgeai.git
cd largeforgeai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

---

## Docker Installation

### Using Pre-built Images

```bash
# Pull the latest image
docker pull largeforgeai/largeforge:latest

# Run inference server
docker run --gpus all -p 8000:8000 \
  -v ./models:/models \
  largeforgeai/largeforge:latest \
  serve inference --model /models/my-model

# Run training
docker run --gpus all \
  -v ./data:/data \
  -v ./output:/output \
  largeforgeai/largeforge:latest \
  train sft --model mistralai/Mistral-7B-v0.1 \
  --dataset /data/train.json \
  --output-dir /output
```

### Available Tags

| Tag | Description |
|-----|-------------|
| `latest` | Latest stable release |
| `1.0.0` | Specific version |
| `cuda12.1` | CUDA 12.1 base |
| `cuda11.8` | CUDA 11.8 base |
| `cpu` | CPU-only (inference) |

### Building Custom Image

```dockerfile
# Dockerfile
FROM largeforgeai/largeforge:latest

# Add custom dependencies
RUN pip install your-custom-package

# Add custom models
COPY ./models /models

# Set default command
CMD ["serve", "inference", "--model", "/models/default"]
```

```bash
docker build -t my-largeforge .
docker run --gpus all -p 8000:8000 my-largeforge
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  inference:
    image: largeforgeai/largeforge:latest
    command: serve inference --model /models/assistant --port 8000
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  router:
    image: largeforgeai/largeforge:latest
    command: serve router --config /config/router.yaml
    ports:
      - "8080:8080"
    volumes:
      - ./config:/config
    depends_on:
      - inference

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

---

## GPU Setup

### NVIDIA Driver Installation

**Ubuntu:**

```bash
# Add NVIDIA driver repository
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install recommended driver
sudo ubuntu-drivers autoinstall

# Reboot
sudo reboot

# Verify
nvidia-smi
```

**Using NVIDIA's official installer:**

```bash
# Download driver from NVIDIA website
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/535.129.03/NVIDIA-Linux-x86_64-535.129.03.run

# Install
sudo sh NVIDIA-Linux-x86_64-535.129.03.run
```

### CUDA Toolkit Installation

```bash
# Download CUDA 12.1 (or your preferred version)
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run

# Install (driver already installed)
sudo sh cuda_12.1.0_530.30.02_linux.run --toolkit --silent

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
```

### cuDNN Installation

```bash
# Download cuDNN from NVIDIA (requires account)
# Install
sudo dpkg -i cudnn-local-repo-*.deb
sudo cp /var/cudnn-local-repo-*/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install libcudnn8 libcudnn8-dev
```

### Verify GPU Setup

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA
nvcc --version

# Check PyTorch CUDA support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Run LargeForge doctor
largeforge doctor
```

---

## Environment Configuration

### Environment Variables

Create a `.env` file or export these variables:

```bash
# API Keys
export HF_TOKEN="hf_..."              # HuggingFace token
export WANDB_API_KEY="..."            # Weights & Biases (optional)
export LARGEFORGE_API_KEY="..."       # Your API key for serving

# Directories
export HF_HOME="~/.cache/huggingface" # HuggingFace cache
export LARGEFORGE_HOME="~/.largeforge" # LargeForge data directory

# GPU Configuration
export CUDA_VISIBLE_DEVICES="0"       # GPU selection
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"  # Memory optimization

# Logging
export LARGEFORGE_LOG_LEVEL="INFO"    # DEBUG, INFO, WARNING, ERROR
```

### Configuration File

Create `~/.largeforge/config.yaml`:

```yaml
# Default settings
defaults:
  model: "mistralai/Mistral-7B-v0.1"
  quantization: "4bit"
  device: "auto"

# Training defaults
training:
  output_dir: "./output"
  batch_size: 4
  learning_rate: 2.0e-5
  bf16: true

# Inference defaults
inference:
  backend: "vllm"
  max_model_len: 4096
  gpu_memory_utilization: 0.9

# Logging
logging:
  level: "INFO"
  format: "json"
  file: "~/.largeforge/logs/largeforge.log"
```

---

## Troubleshooting

### Common Issues

#### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size: `--batch-size 1`
2. Enable gradient checkpointing: `--gradient-checkpointing`
3. Use 4-bit quantization: `--quantization 4bit`
4. Use smaller model or LoRA with lower rank

#### CUDA Version Mismatch

```
The NVIDIA driver on your system is too old
```

**Solutions:**
1. Update NVIDIA driver
2. Install compatible PyTorch version:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

#### HuggingFace Authentication

```
Access denied for model
```

**Solutions:**
1. Login to HuggingFace:
   ```bash
   huggingface-cli login
   ```
2. Accept model license on HuggingFace website
3. Set token in environment:
   ```bash
   export HF_TOKEN="hf_..."
   ```

#### vLLM Installation Issues

```
Failed to build vllm
```

**Solutions:**
1. Install build dependencies:
   ```bash
   pip install ninja cmake
   ```
2. Use pre-built wheel:
   ```bash
   pip install vllm --no-build-isolation
   ```
3. Fall back to transformers backend (slower but more compatible)

#### Import Errors

```
ModuleNotFoundError: No module named 'xxx'
```

**Solutions:**
1. Install missing package:
   ```bash
   pip install xxx
   ```
2. Reinstall with all dependencies:
   ```bash
   pip install largeforge[all] --force-reinstall
   ```

### Getting Help

If you encounter issues:

1. Run diagnostics:
   ```bash
   largeforge doctor --verbose
   ```

2. Check logs:
   ```bash
   cat ~/.largeforge/logs/largeforge.log
   ```

3. Search existing issues:
   [GitHub Issues](https://github.com/largeforgeai/largeforgeai/issues)

4. Open a new issue with:
   - Output of `largeforge doctor --verbose`
   - Full error message and stack trace
   - Steps to reproduce

---

## Upgrading

### Upgrade to Latest Version

```bash
pip install --upgrade largeforge
```

### Upgrade to Specific Version

```bash
pip install largeforge==1.1.0
```

### Check Current Version

```bash
largeforge --version
pip show largeforge
```

---

## Uninstallation

```bash
# Remove package
pip uninstall largeforge

# Remove cache and data (optional)
rm -rf ~/.largeforge
rm -rf ~/.cache/huggingface
```

---

*For more help, see the [FAQ](../FAQ.md) or join our [Discord](https://discord.gg/largeforge).*
