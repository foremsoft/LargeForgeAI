# LargeForgeAI

Low-cost LLM training and deployment stack. Build GPT-4-level systems under $10k.

## Installation

```bash
pip install -e .

# For inference with vLLM
pip install -e ".[inference]"

# For development
pip install -e ".[dev]"
```

## Quick Start

### Training

```bash
# Supervised fine-tuning with LoRA
llm-train sft --model Qwen/Qwen2.5-7B --dataset data.jsonl --output ./output

# Continued pretraining
llm-train pretrain --model Qwen/Qwen2.5-7B --dataset corpus.jsonl --output ./output

# DPO preference tuning
llm-train dpo --model Qwen/Qwen2.5-7B --dataset preferences.jsonl --output ./output
```

### Inference

```bash
# Start inference server (uses vLLM if available)
python -m llm.inference.server

# Start router service
python -m llm.router.app
```

### Python API

```python
from llm.training import train_sft, SFTConfig
from llm.inference import InferenceClient

# Training
config = SFTConfig(
    model_name="Qwen/Qwen2.5-7B",
    dataset_path="data.jsonl",
    use_lora=True,
    load_in_4bit=True,
)
train_sft(config)

# Inference
client = InferenceClient("http://localhost:8000")
response = client.chat("Write a Python function to sort a list")
```

## Architecture

```
Base model → Synthetic data → Distillation → Preference tuning → Experts → Router → Deployment
```

## Project Structure

```
llm/
  data/       # Data loading and synthetic generation
  training/   # SFT, pretraining, distillation, DPO/ORPO
  experts/    # Expert model registry
  router/     # FastAPI routing service
  inference/  # vLLM server, quantization, clients
```

## License

MIT