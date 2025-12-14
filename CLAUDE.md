# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LargeForgeAI is a low-cost LLM training and deployment stack. Target: build GPT-4-level systems under $10k.

## Architecture

Training pipeline:
```
Base model → Synthetic data → Distillation → Preference tuning → Experts → Router → Deployment
```

Key components:
- **Base Model**: Qwen2.5-7B (default)
- **Training**: HuggingFace Trainer + TRL + PEFT/LoRA
- **Preference Tuning**: DPO/ORPO via TRL
- **Router**: FastAPI service with keyword/neural classifier
- **Inference**: vLLM with AWQ/GPTQ quantization

## Common Commands

```bash
# Install
pip install -e .

# Training CLI
llm-train sft --model Qwen/Qwen2.5-7B --dataset data.jsonl --output ./output
llm-train pretrain --model Qwen/Qwen2.5-7B --dataset corpus.jsonl
llm-train dpo --model Qwen/Qwen2.5-7B --dataset preferences.jsonl

# Servers
python -m llm.inference.server   # Inference server (port 8000)
python -m llm.router.app         # Router service (port 8080)
```

## Directory Structure

```
llm/
  data/        # Data loading (load_sft_dataset, load_dpo_dataset) and synthetic generation
  training/    # train_sft, train_pretrain, train_distill, train_dpo, train_orpo
  experts/     # ExpertModel, ExpertRegistry for managing expert configs
  router/      # FastAPI app with KeywordClassifier, NeuralClassifier, HybridClassifier
  inference/   # Server (vLLM/transformers), quantization (AWQ/GPTQ), clients
docs/          # Architecture blueprints and reference implementations
```

## Key Configuration

LoRA defaults (`llm/training/config.py`):
- `r`: 8, `lora_alpha`: 16, `lora_dropout`: 0.05
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- 4-bit quantization enabled by default for SFT/DPO

Training defaults:
- `learning_rate`: 2e-5, `warmup_ratio`: 0.1
- `gradient_accumulation_steps`: 8
- `bf16`: True

## Key Files

- `llm/training/sft.py`: SFTTrainer wrapper with LoRA + 4-bit quantization
- `llm/training/dpo.py`: DPOTrainer and ORPOTrainer wrappers
- `llm/training/distill.py`: Knowledge distillation with temperature scaling
- `llm/router/classifier.py`: Query classification for expert routing
- `llm/inference/server.py`: FastAPI server with vLLM or transformers backend
- `llm/inference/quantize.py`: AWQ/GPTQ quantization and LoRA merging
