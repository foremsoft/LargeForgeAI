# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LargeForgeAI is a documentation and blueprint repository for building low-cost LLM systems (target: under $10k). This is primarily a reference architecture project, not a runnable codebase.

## Architecture

The training pipeline follows this flow:
```
Base model → Synthetic data → Distillation → Preference tuning → Experts → Router → Deployment
```

Key components:
- **Base Model**: Qwen2.5-7B (recommended starting point)
- **Training**: HuggingFace Trainer + PEFT/LoRA for fine-tuning
- **Preference Tuning**: DPO/ORPO for alignment
- **Router**: FastAPI service that routes requests to specialized expert models
- **Inference**: vLLM with AWQ quantization

## Common Commands

Training (with accelerate):
```bash
accelerate launch train.py --model qwen2.5-7b
```

Inference (vLLM):
```bash
vllm serve Qwen/Qwen2.5-7B --quantization awq
```

## Directory Structure

```
docs/
  cheap_llm_training_blueprint/  # High-level architecture docs
  llm_complete_stack/            # Implementation reference
    configs/                     # Axolotl/training configs
    data/                        # Data format examples (DPO, synthetic prompts)
    training/                    # Training scripts (pretrain, lora, distill)
    router/                      # FastAPI routing service
    inference/                   # vLLM serving instructions
    deployment/                  # Stack deployment (FastAPI + vLLM + GPU/CPU hybrid)
```

## Key Configuration

LoRA defaults (from `configs/axolotl_lora.yml`):
- `lora_r`: 8
- `lora_alpha`: 16
- `load_in_4bit`: true
- Target modules: `q_proj`, `v_proj`

## Deployment Stack

- FastAPI for API layer
- Lightweight classifier router for expert selection
- vLLM for inference
- GPU/CPU hybrid deployment
