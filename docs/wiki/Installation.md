# Installation Guide

Complete installation instructions for LargeForgeAI.

## System Requirements

### Hardware

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **GPU** | 8GB VRAM | 24GB VRAM | NVIDIA with CUDA support |
| **RAM** | 16 GB | 32 GB | More helps with data loading |
| **Storage** | 50 GB | 200 GB | SSD/NVMe preferred |
| **CPU** | 4 cores | 8+ cores | For data preprocessing |

### Supported GPUs

| GPU | VRAM | 7B Training | 13B Training | Notes |
|-----|------|-------------|--------------|-------|
| RTX 4090 | 24GB | Excellent | Good | Best consumer GPU |
| RTX 3090 | 24GB | Excellent | Good | Great value |
| RTX 4080 | 16GB | Good | Limited | Use 4-bit only |
| RTX 3080 | 10GB | Limited | No | Small batches only |
| A100 40GB | 40GB | Excellent | Excellent | Cloud/enterprise |
| A100 80GB | 80GB | Excellent | Excellent | Large models |
| H100 | 80GB | Excellent | Excellent | Fastest option |

### Software

- **OS**: Linux (Ubuntu 20.04+), Windows 10/11, macOS (CPU only)
- **Python**: 3.10 or 3.11
- **CUDA**: 11.8 or 12.1 (for GPU training)
- **Docker**: 20.10+ (for deployment)

---

## Installation Methods

### Method 1: pip install (Recommended)

```bash
# Clone repository
git clone https://github.com/foremsoft/LargeForgeAI.git
cd LargeForgeAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install base package
pip install -e .
```

### Method 2: With All Features

```bash
pip install -e ".[all]"
```

This includes:
- vLLM for high-performance inference
- AWQ/GPTQ for quantization
- Sentence transformers for routing
- Weights & Biases for logging

### Method 3: Development Installation

```bash
pip install -e ".[dev]"
pre-commit install
```

Adds:
- pytest for testing
- ruff/black for linting
- mypy for type checking

---

## Post-Installation

### Verify Installation

```bash
# Check CLI
largeforge --version
# Output: largeforge, version 2.0.0

# Check system
largeforge info
# Shows Python, PyTorch, CUDA info
```

### Download a Model (Optional)

```bash
# Models download automatically, but you can pre-download:
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct')"
```

### Start Web UI

```bash
largeforge web start
# Open http://localhost:7860
```

---

## Troubleshooting

### CUDA Not Found

```bash
# Check CUDA version
nvcc --version
nvidia-smi

# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Import Errors

```bash
# Reinstall in clean environment
pip uninstall largeforge
pip install -e .
```

### Permission Errors (Linux)

```bash
# Add user to docker group
sudo usermod -aG docker $USER
```

---

[[Back to Home|Home]] | [[Next: Quick Start|Quick-Start]]
