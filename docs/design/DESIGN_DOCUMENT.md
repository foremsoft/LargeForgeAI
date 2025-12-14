# LargeForgeAI Software Design Document (SDD)

**Document ID:** LFA-SDD-001
**Version:** 1.0.0
**Date:** 2025-01-15
**Status:** Draft
**Classification:** Internal

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-01-15 | LargeForgeAI Team | Initial release |

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Overview](#2-system-overview)
3. [Design Considerations](#3-design-considerations)
4. [System Architecture Design](#4-system-architecture-design)
5. [Detailed Component Design](#5-detailed-component-design)
6. [Data Design](#6-data-design)
7. [Interface Design](#7-interface-design)
8. [Algorithm Design](#8-algorithm-design)
9. [Error Handling Design](#9-error-handling-design)
10. [Security Design](#10-security-design)
11. [Performance Design](#11-performance-design)
12. [Testing Design](#12-testing-design)

---

## 1. Introduction

### 1.1 Purpose

This Software Design Document (SDD) describes the detailed design of the LargeForgeAI system. It provides the technical blueprint for implementing the system architecture and serves as a guide for developers.

### 1.2 Scope

This document covers:
- Detailed component designs for all modules
- Data structures and database schemas
- Algorithm specifications
- API designs and interface contracts
- Error handling strategies
- Security implementation details

### 1.3 Design Methodology

The design follows these principles:
- **SOLID Principles**: Single responsibility, Open-closed, Liskov substitution, Interface segregation, Dependency inversion
- **DRY**: Don't Repeat Yourself
- **KISS**: Keep It Simple, Stupid
- **YAGNI**: You Aren't Gonna Need It

---

## 2. System Overview

### 2.1 System Context

LargeForgeAI is a comprehensive platform for training and deploying Large Language Models at low cost. The system consists of five core modules:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LargeForgeAI                                │
│                                                                     │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐       │
│  │   Data    │  │ Training  │  │  Experts  │  │  Router   │       │
│  │  Module   │  │  Module   │  │  Module   │  │  Module   │       │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘       │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     Inference Module                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Design Goals

| Goal | Description | Priority |
|------|-------------|----------|
| Modularity | Independent, replaceable components | Critical |
| Extensibility | Easy to add new training methods, experts | High |
| Efficiency | Optimal resource utilization | High |
| Simplicity | Easy to understand and maintain | High |
| Robustness | Graceful error handling and recovery | High |

---

## 3. Design Considerations

### 3.1 Assumptions

| ID | Assumption |
|----|------------|
| A-01 | Users have access to NVIDIA GPUs with 24GB+ VRAM |
| A-02 | Network connectivity to HuggingFace Hub available |
| A-03 | Python 3.10+ environment available |
| A-04 | Linux or Windows operating system |

### 3.2 Constraints

| ID | Constraint | Impact |
|----|------------|--------|
| C-01 | Must use open-source models | Model selection limited |
| C-02 | GPU memory limits (24GB typical) | Requires quantization |
| C-03 | Python ecosystem | Language choice fixed |

### 3.3 Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| PyTorch | >= 2.0.0 | ML framework |
| Transformers | >= 4.36.0 | Model loading |
| TRL | >= 0.7.0 | Training algorithms |
| PEFT | >= 0.7.0 | LoRA adapters |
| vLLM | >= 0.2.0 | Inference engine |
| FastAPI | >= 0.104.0 | Web framework |

---

## 4. System Architecture Design

### 4.1 Layered Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Presentation Layer                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │
│  │   REST API      │  │      CLI        │  │   Python API    │        │
│  │  (FastAPI)      │  │  (argparse)     │  │  (library)      │        │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘        │
├─────────────────────────────────────────────────────────────────────────┤
│                        Business Logic Layer                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │
│  │   Training      │  │    Routing      │  │   Inference     │        │
│  │   Services      │  │    Services     │  │   Services      │        │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘        │
├─────────────────────────────────────────────────────────────────────────┤
│                        Domain Layer                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │
│  │   Data Models   │  │  Expert Models  │  │   Classifiers   │        │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘        │
├─────────────────────────────────────────────────────────────────────────┤
│                        Infrastructure Layer                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │
│  │   Model Store   │  │   Data Store    │  │   GPU Runtime   │        │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Package Structure

```
llm/
├── __init__.py                 # Package initialization, exports
├── data/                       # Data processing module
│   ├── __init__.py
│   ├── synthetic.py            # Synthetic data generation
│   └── loaders.py              # Dataset loaders
├── training/                   # Training module
│   ├── __init__.py
│   ├── config.py               # Configuration dataclasses
│   ├── sft.py                  # Supervised fine-tuning
│   ├── pretrain.py             # Continued pretraining
│   ├── distill.py              # Knowledge distillation
│   ├── dpo.py                  # Preference optimization
│   └── cli.py                  # Command-line interface
├── experts/                    # Expert management module
│   ├── __init__.py
│   └── manager.py              # Expert registry
├── router/                     # Router module
│   ├── __init__.py
│   ├── classifier.py           # Query classifiers
│   └── app.py                  # FastAPI application
└── inference/                  # Inference module
    ├── __init__.py
    ├── server.py               # Inference server
    ├── quantize.py             # Quantization utilities
    └── client.py               # API clients
```

---

## 5. Detailed Component Design

### 5.1 Data Module (`llm.data`)

#### 5.1.1 Class Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              llm.data                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────┐     ┌─────────────────────────────┐   │
│  │       SyntheticConfig       │     │   SyntheticDataGenerator    │   │
│  ├─────────────────────────────┤     ├─────────────────────────────┤   │
│  │ + model_name: str           │     │ - config: SyntheticConfig   │   │
│  │ + temperature: float        │     │ - model: AutoModelForCLM    │   │
│  │ + max_new_tokens: int       │     │ - tokenizer: AutoTokenizer  │   │
│  │ + num_samples: int          │────▶├─────────────────────────────┤   │
│  │ + batch_size: int           │     │ + load_model()              │   │
│  └─────────────────────────────┘     │ + generate_from_template()  │   │
│                                      │ + generate_with_seed()      │   │
│                                      └─────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                        Loader Functions                           │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │ + load_sft_dataset(path, format, fields) -> Dataset              │  │
│  │ + load_dpo_dataset(path, format, fields) -> Dataset              │  │
│  │ + load_pretrain_dataset(path, format, field) -> Dataset          │  │
│  │ + prepare_chat_format(dataset, tokenizer, max_len) -> Dataset    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                        Utility Functions                          │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │ + save_jsonl(data, path) -> None                                 │  │
│  │ + load_jsonl(path) -> List[Dict]                                 │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 5.1.2 Sequence Diagram: Synthetic Data Generation

```
┌──────┐     ┌───────────────────┐     ┌─────────┐     ┌───────────┐
│Client│     │SyntheticDataGen   │     │Tokenizer│     │   Model   │
└──┬───┘     └─────────┬─────────┘     └────┬────┘     └─────┬─────┘
   │                   │                    │                │
   │ generate_from_    │                    │                │
   │ template(template,│                    │                │
   │ variables, n)     │                    │                │
   │──────────────────▶│                    │                │
   │                   │                    │                │
   │                   │ load_model()       │                │
   │                   │ (if not loaded)    │                │
   │                   │───────────────────────────────────▶│
   │                   │                    │                │
   │                   │ loop [n samples]   │                │
   │                   │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─ ─│
   │                   │                    │                │
   │                   │ fill_template()    │                │
   │                   │──────────┐         │                │
   │                   │          │         │                │
   │                   │◀─────────┘         │                │
   │                   │                    │                │
   │                   │ tokenize(prompt)   │                │
   │                   │───────────────────▶│                │
   │                   │                    │                │
   │                   │     input_ids      │                │
   │                   │◀───────────────────│                │
   │                   │                    │                │
   │                   │ generate(input_ids)│                │
   │                   │───────────────────────────────────▶│
   │                   │                    │                │
   │                   │     output_ids     │                │
   │                   │◀───────────────────────────────────│
   │                   │                    │                │
   │                   │ decode(output_ids) │                │
   │                   │───────────────────▶│                │
   │                   │                    │                │
   │                   │     response       │                │
   │                   │◀───────────────────│                │
   │                   │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─ ─│
   │                   │                    │                │
   │  List[Dict]       │                    │                │
   │◀──────────────────│                    │                │
   │                   │                    │                │
```

### 5.2 Training Module (`llm.training`)

#### 5.2.1 Class Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             llm.training                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                        Configuration Classes                          │ │
│  │                                                                        │ │
│  │  ┌─────────────────────┐                                              │ │
│  │  │ BaseTrainingConfig  │                                              │ │
│  │  ├─────────────────────┤                                              │ │
│  │  │ + model_name        │                                              │ │
│  │  │ + output_dir        │                                              │ │
│  │  │ + num_train_epochs  │                                              │ │
│  │  │ + batch_size        │                                              │ │
│  │  │ + learning_rate     │                                              │ │
│  │  │ + max_seq_length    │                                              │ │
│  │  └─────────┬───────────┘                                              │ │
│  │            │                                                          │ │
│  │   ┌────────┴────────┬────────────────┬────────────────┐              │ │
│  │   │                 │                │                │              │ │
│  │   ▼                 ▼                ▼                ▼              │ │
│  │ ┌─────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐         │ │
│  │ │SFTConfig│   │PretrainCfg│   │ DPOConfig │   │DistillCfg │         │ │
│  │ ├─────────┤   ├───────────┤   ├───────────┤   ├───────────┤         │ │
│  │ │+ use_lora│   │+ use_lora │   │+ beta     │   │+ teacher  │         │ │
│  │ │+ load_4b │   │           │   │+ loss_type│   │+ student  │         │ │
│  │ │+ lora_cfg│   │           │   │+ ref_model│   │+ temp     │         │ │
│  │ └─────────┘   └───────────┘   └───────────┘   └───────────┘         │ │
│  │                                                                        │ │
│  │  ┌─────────────────────┐                                              │ │
│  │  │     LoraConfig      │                                              │ │
│  │  ├─────────────────────┤                                              │ │
│  │  │ + r: int = 8        │                                              │ │
│  │  │ + lora_alpha: 16    │                                              │ │
│  │  │ + lora_dropout: 0.05│                                              │ │
│  │  │ + target_modules    │                                              │ │
│  │  └─────────────────────┘                                              │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                        Training Functions                             │ │
│  │                                                                        │ │
│  │  + train_sft(config: SFTConfig) -> Trainer                           │ │
│  │  + train_pretrain(config: PretrainConfig) -> Trainer                 │ │
│  │  + train_dpo(config: DPOConfig) -> DPOTrainer                        │ │
│  │  + train_orpo(...) -> ORPOTrainer                                    │ │
│  │  + train_distill(config: DistillConfig, dataset) -> Model            │ │
│  │                                                                        │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                          Distiller Class                              │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │ │
│  │  │                        Distiller                                 │ │ │
│  │  ├─────────────────────────────────────────────────────────────────┤ │ │
│  │  │ - config: DistillConfig                                         │ │ │
│  │  │ - teacher: AutoModelForCausalLM                                 │ │ │
│  │  │ - student: AutoModelForCausalLM                                 │ │ │
│  │  │ - tokenizer: AutoTokenizer                                      │ │ │
│  │  ├─────────────────────────────────────────────────────────────────┤ │ │
│  │  │ + load_models() -> None                                         │ │ │
│  │  │ + distillation_loss(s_logits, t_logits, labels) -> Tensor      │ │ │
│  │  │ + train(dataset: Dataset) -> Model                              │ │ │
│  │  └─────────────────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 5.2.2 Sequence Diagram: SFT Training

```
┌──────┐  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐  ┌───────┐
│Client│  │train_sft│  │Tokenizer │  │  Model  │  │SFTTrainer│  │Dataset│
└──┬───┘  └────┬────┘  └────┬─────┘  └────┬────┘  └────┬─────┘  └───┬───┘
   │           │            │             │            │            │
   │ train_sft │            │             │            │            │
   │ (config)  │            │             │            │            │
   │──────────▶│            │             │            │            │
   │           │            │             │            │            │
   │           │ from_pretrained(model_name)           │            │
   │           │───────────▶│             │            │            │
   │           │            │             │            │            │
   │           │ from_pretrained(model_name, quant_config)          │
   │           │─────────────────────────▶│            │            │
   │           │            │             │            │            │
   │           │ prepare_model_for_kbit_training()     │            │
   │           │─────────────────────────▶│            │            │
   │           │            │             │            │            │
   │           │ load_sft_dataset(path)   │            │            │
   │           │───────────────────────────────────────────────────▶│
   │           │            │             │            │            │
   │           │            │             │            │  dataset   │
   │           │◀───────────────────────────────────────────────────│
   │           │            │             │            │            │
   │           │ SFTTrainer(model, args, dataset, tokenizer, peft)  │
   │           │──────────────────────────────────────▶│            │
   │           │            │             │            │            │
   │           │            │    train()  │            │            │
   │           │──────────────────────────────────────▶│            │
   │           │            │             │            │            │
   │           │            │   loop [epochs]          │            │
   │           │            │   ─ ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─│            │
   │           │            │             │            │            │
   │           │            │   forward() │            │            │
   │           │            │◀────────────│            │            │
   │           │            │             │            │            │
   │           │            │   backward()│            │            │
   │           │            │◀────────────│            │            │
   │           │            │             │            │            │
   │           │            │   step()    │            │            │
   │           │            │◀────────────│            │            │
   │           │            │   ─ ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─│            │
   │           │            │             │            │            │
   │           │         save_model()     │            │            │
   │           │──────────────────────────────────────▶│            │
   │           │            │             │            │            │
   │  Trainer  │            │             │            │            │
   │◀──────────│            │             │            │            │
   │           │            │             │            │            │
```

### 5.3 Router Module (`llm.router`)

#### 5.3.1 Class Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              llm.router                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       Classifier Hierarchy                           │   │
│  │                                                                      │   │
│  │              ┌───────────────────────────┐                          │   │
│  │              │    <<interface>>          │                          │   │
│  │              │      Classifier           │                          │   │
│  │              ├───────────────────────────┤                          │   │
│  │              │ + classify(query) -> str  │                          │   │
│  │              │ + get_endpoint(name) -> str│                         │   │
│  │              └─────────────┬─────────────┘                          │   │
│  │                            │                                        │   │
│  │         ┌──────────────────┼──────────────────┐                    │   │
│  │         │                  │                  │                    │   │
│  │         ▼                  ▼                  ▼                    │   │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐          │   │
│  │  │  Keyword    │   │   Neural    │   │     Hybrid      │          │   │
│  │  │ Classifier  │   │ Classifier  │   │   Classifier    │          │   │
│  │  ├─────────────┤   ├─────────────┤   ├─────────────────┤          │   │
│  │  │- keyword_map│   │- model      │   │- keyword_clf    │          │   │
│  │  │- experts    │   │- tokenizer  │   │- neural_clf     │          │   │
│  │  │- default    │   │- experts    │   │- threshold      │          │   │
│  │  └─────────────┘   └─────────────┘   └─────────────────┘          │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Data Classes                                 │   │
│  │                                                                      │   │
│  │  ┌─────────────────────┐    ┌─────────────────────┐                │   │
│  │  │    ExpertConfig     │    │   GenerateRequest   │                │   │
│  │  ├─────────────────────┤    ├─────────────────────┤                │   │
│  │  │ + name: str         │    │ + prompt: str?      │                │   │
│  │  │ + endpoint: str     │    │ + messages: List?   │                │   │
│  │  │ + description: str  │    │ + max_tokens: int   │                │   │
│  │  │ + keywords: List    │    │ + temperature: float│                │   │
│  │  └─────────────────────┘    │ + expert: str?      │                │   │
│  │                              └─────────────────────┘                │   │
│  │                                                                      │   │
│  │  ┌─────────────────────┐                                            │   │
│  │  │  GenerateResponse   │                                            │   │
│  │  ├─────────────────────┤                                            │   │
│  │  │ + text: str         │                                            │   │
│  │  │ + expert: str       │                                            │   │
│  │  │ + tokens_used: int? │                                            │   │
│  │  └─────────────────────┘                                            │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      FastAPI Application                             │   │
│  │                                                                      │   │
│  │  Endpoints:                                                         │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │ GET  /health           -> {"status": "healthy"}               │ │   │
│  │  │ GET  /experts          -> {"experts": [...]}                  │ │   │
│  │  │ POST /route?query=...  -> {"expert": "...", "endpoint": "..."}│ │   │
│  │  │ POST /generate         -> GenerateResponse                    │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 5.3.2 Sequence Diagram: Query Routing

```
┌──────┐   ┌──────┐   ┌──────────┐   ┌──────────┐   ┌────────┐
│Client│   │Router│   │Classifier│   │HttpClient│   │ Expert │
└──┬───┘   └──┬───┘   └────┬─────┘   └────┬─────┘   └───┬────┘
   │          │            │              │             │
   │  POST    │            │              │             │
   │ /generate│            │              │             │
   │ {prompt} │            │              │             │
   │─────────▶│            │              │             │
   │          │            │              │             │
   │          │ classify   │              │             │
   │          │ (prompt)   │              │             │
   │          │───────────▶│              │             │
   │          │            │              │             │
   │          │   expert   │              │             │
   │          │   name     │              │             │
   │          │◀───────────│              │             │
   │          │            │              │             │
   │          │ get_endpoint               │             │
   │          │ (expert)   │              │             │
   │          │───────────▶│              │             │
   │          │            │              │             │
   │          │  endpoint  │              │             │
   │          │  URL       │              │             │
   │          │◀───────────│              │             │
   │          │            │              │             │
   │          │     POST to endpoint      │             │
   │          │ ─────────────────────────▶│             │
   │          │            │              │             │
   │          │            │              │  forward    │
   │          │            │              │  request    │
   │          │            │              │────────────▶│
   │          │            │              │             │
   │          │            │              │  response   │
   │          │            │              │◀────────────│
   │          │            │              │             │
   │          │     response              │             │
   │          │◀─────────────────────────│             │
   │          │            │              │             │
   │ Generate │            │              │             │
   │ Response │            │              │             │
   │◀─────────│            │              │             │
   │          │            │              │             │
```

### 5.4 Inference Module (`llm.inference`)

#### 5.4.1 Class Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            llm.inference                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                        Server Components                              │ │
│  │                                                                        │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │ │
│  │  │                      ServerConfig                                │ │ │
│  │  ├─────────────────────────────────────────────────────────────────┤ │ │
│  │  │ + model_name: str = "Qwen/Qwen2.5-7B-Instruct"                  │ │ │
│  │  │ + use_vllm: bool = True                                         │ │ │
│  │  │ + quantization: str? = "awq"                                    │ │ │
│  │  │ + tensor_parallel_size: int = 1                                 │ │ │
│  │  │ + gpu_memory_utilization: float = 0.9                           │ │ │
│  │  │ + max_model_len: int = 4096                                     │ │ │
│  │  └─────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                        │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │ │
│  │  │                   Backend Selection                              │ │ │
│  │  │                                                                  │ │ │
│  │  │    ┌─────────────────┐        ┌─────────────────┐              │ │ │
│  │  │    │  vLLM Backend   │   OR   │ Transformers    │              │ │ │
│  │  │    │  (preferred)    │        │ Backend         │              │ │ │
│  │  │    │                 │        │ (fallback)      │              │ │ │
│  │  │    │ - LLM           │        │ - model         │              │ │ │
│  │  │    │ - SamplingParams│        │ - tokenizer     │              │ │ │
│  │  │    └─────────────────┘        └─────────────────┘              │ │ │
│  │  │                                                                  │ │ │
│  │  └─────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                        │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                        Client Classes                                 │ │
│  │                                                                        │ │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │ │
│  │  │ InferenceClient  │  │AsyncInference    │  │  RouterClient    │   │ │
│  │  │                  │  │     Client       │  │                  │   │ │
│  │  ├──────────────────┤  ├──────────────────┤  ├──────────────────┤   │ │
│  │  │ - base_url       │  │ - base_url       │  │ - base_url       │   │ │
│  │  │ - timeout        │  │ - timeout        │  │ - timeout        │   │ │
│  │  │ - _client: httpx │  │ - _client: async │  │ - _client: httpx │   │ │
│  │  ├──────────────────┤  ├──────────────────┤  ├──────────────────┤   │ │
│  │  │ + health()       │  │ + health()       │  │ + list_experts() │   │ │
│  │  │ + generate()     │  │ + generate()     │  │ + route()        │   │ │
│  │  │ + chat()         │  │ + chat()         │  │ + generate()     │   │ │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘   │ │
│  │                                                                        │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                      Quantization Functions                           │ │
│  │                                                                        │ │
│  │  + quantize_awq(model, output, bits, group_size) -> str              │ │
│  │  + quantize_gptq(model, output, bits, group_size) -> str             │ │
│  │  + merge_lora_weights(base, lora, output) -> str                     │ │
│  │                                                                        │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.5 Experts Module (`llm.experts`)

#### 5.5.1 Class Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             llm.experts                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        ExpertModel                                   │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ + name: str                                                         │   │
│  │ + base_model: str                                                   │   │
│  │ + adapter_path: str?                                                │   │
│  │ + description: str                                                  │   │
│  │ + specialization: str                                               │   │
│  │ + keywords: List[str]                                               │   │
│  │ + trained_on: str?                                                  │   │
│  │ + training_steps: int?                                              │   │
│  │ + quantization: str?                                                │   │
│  │ + port: int?                                                        │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ + to_dict() -> Dict                                                 │   │
│  │ + from_dict(data: Dict) -> ExpertModel                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       ExpertRegistry                                 │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ - experts: Dict[str, ExpertModel]                                   │   │
│  │ - config_path: Path?                                                │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ + register(expert: ExpertModel) -> None                             │   │
│  │ + unregister(name: str) -> None                                     │   │
│  │ + get(name: str) -> ExpertModel?                                    │   │
│  │ + list() -> List[ExpertModel]                                       │   │
│  │ + find_by_keyword(keyword: str) -> List[ExpertModel]                │   │
│  │ + find_by_specialization(spec: str) -> List[ExpertModel]            │   │
│  │ + save(path: Path?) -> None                                         │   │
│  │ + load(path: Path?) -> None                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Utility Functions                                │   │
│  │                                                                      │   │
│  │  + create_default_registry(path?) -> ExpertRegistry                 │   │
│  │                                                                      │   │
│  │  DEFAULT_EXPERTS: List[ExpertModel] = [                             │   │
│  │      ExpertModel(name="general", ...),                              │   │
│  │      ExpertModel(name="code", ...),                                 │   │
│  │      ExpertModel(name="math", ...),                                 │   │
│  │      ExpertModel(name="writing", ...),                              │   │
│  │  ]                                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Data Design

### 6.1 Data Schemas

#### 6.1.1 Training Data Formats

```json
// SFT Training Data (messages format)
{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language..."}
    ]
}

// SFT Training Data (prompt-response format)
{
    "prompt": "What is Python?",
    "response": "Python is a programming language...",
    "system": "You are a helpful assistant."
}

// DPO Training Data
{
    "prompt": "Write a function to sort a list",
    "chosen": "def sort_list(lst):\n    return sorted(lst)",
    "rejected": "lst.sort()"
}

// Pretraining Data
{
    "text": "The quick brown fox jumps over the lazy dog..."
}
```

#### 6.1.2 Configuration Schema

```json
// Expert Registry Configuration
{
    "experts": [
        {
            "name": "code",
            "base_model": "Qwen/Qwen2.5-7B-Instruct",
            "adapter_path": "./adapters/code-lora",
            "description": "Code generation expert",
            "specialization": "code",
            "keywords": ["python", "javascript", "code", "function"],
            "quantization": "awq",
            "port": 8001
        }
    ]
}
```

### 6.2 File Storage Structure

```
storage/
├── models/
│   ├── base/                    # Base models from HuggingFace
│   │   └── qwen2.5-7b/
│   ├── adapters/                # LoRA adapters
│   │   ├── general-lora/
│   │   ├── code-lora/
│   │   └── math-lora/
│   ├── merged/                  # Merged models (base + adapter)
│   │   └── code-expert-7b/
│   └── quantized/               # Quantized models
│       └── code-expert-7b-awq/
├── data/
│   ├── raw/                     # Raw training data
│   ├── processed/               # Processed/tokenized data
│   └── synthetic/               # Generated synthetic data
├── checkpoints/                 # Training checkpoints
│   └── sft-run-001/
│       ├── checkpoint-500/
│       └── checkpoint-1000/
└── logs/                        # Training logs
    └── sft-run-001/
```

---

## 7. Interface Design

### 7.1 REST API Specification

#### 7.1.1 Router Service API

```yaml
openapi: 3.0.0
info:
  title: LargeForgeAI Router API
  version: 0.1.0

paths:
  /health:
    get:
      summary: Health check
      responses:
        '200':
          description: Service is healthy
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                    example: healthy

  /experts:
    get:
      summary: List available experts
      responses:
        '200':
          description: List of experts
          content:
            application/json:
              schema:
                type: object
                properties:
                  experts:
                    type: array
                    items:
                      $ref: '#/components/schemas/Expert'

  /route:
    post:
      summary: Get routing decision
      parameters:
        - name: query
          in: query
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Routing decision
          content:
            application/json:
              schema:
                type: object
                properties:
                  expert:
                    type: string
                  endpoint:
                    type: string

  /generate:
    post:
      summary: Generate text with automatic routing
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/GenerateRequest'
      responses:
        '200':
          description: Generated response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/GenerateResponse'

components:
  schemas:
    Expert:
      type: object
      properties:
        name:
          type: string
        description:
          type: string
        endpoint:
          type: string

    GenerateRequest:
      type: object
      properties:
        prompt:
          type: string
        messages:
          type: array
          items:
            $ref: '#/components/schemas/ChatMessage'
        max_tokens:
          type: integer
          default: 512
        temperature:
          type: number
          default: 0.7
        expert:
          type: string

    ChatMessage:
      type: object
      required: [role, content]
      properties:
        role:
          type: string
          enum: [system, user, assistant]
        content:
          type: string

    GenerateResponse:
      type: object
      properties:
        text:
          type: string
        expert:
          type: string
        tokens_used:
          type: integer
```

### 7.2 Python API

```python
# Training API
from llm.training import SFTConfig, train_sft

config = SFTConfig(
    model_name="Qwen/Qwen2.5-7B",
    dataset_path="data.jsonl",
    output_dir="./output",
    use_lora=True,
    load_in_4bit=True,
)
trainer = train_sft(config)

# Inference API
from llm.inference import InferenceClient

client = InferenceClient("http://localhost:8000")
response = client.generate(prompt="Hello, world!")
print(response["text"])

# Or with chat interface
response = client.chat(
    message="Write a Python function",
    system="You are a coding assistant",
)

# Router API
from llm.inference import RouterClient

router = RouterClient("http://localhost:8080")
experts = router.list_experts()
response = router.generate(prompt="Write code to sort a list")
print(f"Routed to: {response['expert']}")
```

### 7.3 CLI Interface

```bash
# Training Commands
llm-train sft --model Qwen/Qwen2.5-7B \
              --dataset data.jsonl \
              --output ./output \
              --epochs 3 \
              --lora-r 8

llm-train pretrain --model Qwen/Qwen2.5-7B \
                   --dataset corpus.jsonl \
                   --output ./output

llm-train dpo --model Qwen/Qwen2.5-7B \
              --dataset preferences.jsonl \
              --output ./output \
              --beta 0.1

# Server Commands
python -m llm.inference.server --model Qwen/Qwen2.5-7B \
                               --port 8000 \
                               --quantization awq

python -m llm.router.app --host 0.0.0.0 --port 8080
```

---

## 8. Algorithm Design

### 8.1 Knowledge Distillation

```
Algorithm: Knowledge Distillation Training
Input: Teacher model T, Student model S, Dataset D, Temperature τ, Alpha α
Output: Trained student model S

1. Initialize student model S with pretrained weights
2. Freeze teacher model T parameters
3. For each epoch e in [1, num_epochs]:
   4. For each batch B in D:
      5. Get teacher logits: z_t = T(B)
      6. Get student logits: z_s = S(B)
      7. Compute soft targets: p_t = softmax(z_t / τ)
      8. Compute student probs: p_s = log_softmax(z_s / τ)
      9. Compute distillation loss: L_d = KL(p_s, p_t) * τ²
      10. Compute hard label loss: L_h = CrossEntropy(z_s, labels)
      11. Compute total loss: L = α * L_d + (1 - α) * L_h
      12. Backpropagate and update S parameters
13. Return trained student S
```

### 8.2 Query Classification

```
Algorithm: Hybrid Query Classification
Input: Query Q, KeywordClassifier K, NeuralClassifier N (optional)
Output: Expert name

1. Normalize query: Q_lower = lowercase(Q)
2. For each (keyword, expert) in K.keyword_map:
   3. If keyword in Q_lower:
      4. Return expert
5. If K.default_expert found via keywords:
   6. Return K.default_expert
7. If NeuralClassifier N is available:
   8. Tokenize Q: tokens = N.tokenizer(Q)
   9. Get predictions: logits = N.model(tokens)
   10. expert_idx = argmax(logits)
   11. Return N.expert_names[expert_idx]
12. Return K.default_expert
```

### 8.3 LoRA Adaptation

```
Algorithm: LoRA Forward Pass
Input: Input x, Base weight W, LoRA weights A and B, Scaling α, Rank r
Output: Output y

1. Compute base output: y_base = W @ x
2. Compute LoRA delta:
   a. down_proj = A @ x      # (r, batch)
   b. up_proj = B @ down_proj # (out_dim, batch)
3. Scale LoRA output: y_lora = (α / r) * up_proj
4. Combine: y = y_base + y_lora
5. Return y
```

---

## 9. Error Handling Design

### 9.1 Exception Hierarchy

```python
class LargeForgeAIError(Exception):
    """Base exception for all LargeForgeAI errors."""
    pass

class ConfigurationError(LargeForgeAIError):
    """Configuration-related errors."""
    pass

class ModelError(LargeForgeAIError):
    """Model loading/inference errors."""
    pass

class DataError(LargeForgeAIError):
    """Data loading/processing errors."""
    pass

class TrainingError(LargeForgeAIError):
    """Training-related errors."""
    pass

class InferenceError(LargeForgeAIError):
    """Inference-related errors."""
    pass

class RouterError(LargeForgeAIError):
    """Router service errors."""
    pass
```

### 9.2 Error Handling Strategy

| Error Type | Strategy | User Message |
|------------|----------|--------------|
| Model not found | Fallback to default | "Using default model" |
| GPU OOM | Reduce batch size | "Reducing batch size" |
| API timeout | Retry with backoff | "Retrying request" |
| Invalid input | Validate and reject | "Invalid input format" |
| Expert unavailable | Route to general | "Expert unavailable, using general" |

### 9.3 Retry Policy

```python
RETRY_CONFIG = {
    "max_retries": 3,
    "base_delay": 1.0,  # seconds
    "max_delay": 30.0,
    "exponential_base": 2,
    "retryable_exceptions": [
        httpx.TimeoutException,
        httpx.ConnectError,
    ],
}
```

---

## 10. Security Design

### 10.1 Authentication

```python
# API Key Authentication
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key
```

### 10.2 Input Validation

```python
from pydantic import BaseModel, Field, validator

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)
    max_tokens: int = Field(512, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)

    @validator('prompt')
    def sanitize_prompt(cls, v):
        # Remove potential injection patterns
        return sanitize_input(v)
```

### 10.3 Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/generate")
@limiter.limit("100/minute")
async def generate(request: GenerateRequest):
    ...
```

---

## 11. Performance Design

### 11.1 Optimization Strategies

| Component | Strategy | Expected Improvement |
|-----------|----------|---------------------|
| Inference | vLLM continuous batching | 10x throughput |
| Model | 4-bit quantization | 4x memory reduction |
| Training | Gradient accumulation | Support larger batches |
| Training | LoRA | 10x memory reduction |
| Router | Connection pooling | 2x latency reduction |

### 11.2 Caching Strategy

```python
# Model caching
MODEL_CACHE = {}

def get_model(model_name: str):
    if model_name not in MODEL_CACHE:
        MODEL_CACHE[model_name] = load_model(model_name)
    return MODEL_CACHE[model_name]

# Response caching (optional)
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_classify(query: str) -> str:
    return classifier.classify(query)
```

### 11.3 Resource Limits

```python
RESOURCE_LIMITS = {
    "max_concurrent_requests": 100,
    "max_batch_size": 32,
    "max_sequence_length": 4096,
    "gpu_memory_fraction": 0.9,
    "request_timeout": 60.0,
}
```

---

## 12. Testing Design

### 12.1 Test Categories

| Category | Description | Coverage Target |
|----------|-------------|-----------------|
| Unit Tests | Individual functions | > 80% |
| Integration Tests | Component interactions | > 70% |
| End-to-End Tests | Full workflows | Key scenarios |
| Performance Tests | Latency, throughput | SLA targets |
| Load Tests | Concurrent users | Peak load |

### 12.2 Test Structure

```
tests/
├── unit/
│   ├── test_data_loaders.py
│   ├── test_synthetic.py
│   ├── test_config.py
│   ├── test_classifier.py
│   └── test_quantize.py
├── integration/
│   ├── test_training_pipeline.py
│   ├── test_router_service.py
│   └── test_inference_server.py
├── e2e/
│   ├── test_full_training.py
│   └── test_full_inference.py
└── performance/
    ├── test_inference_latency.py
    └── test_training_throughput.py
```

### 12.3 Test Fixtures

```python
import pytest

@pytest.fixture
def sample_sft_data():
    return [
        {"messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]}
    ]

@pytest.fixture
def mock_model():
    class MockModel:
        def generate(self, *args, **kwargs):
            return torch.tensor([[1, 2, 3]])
    return MockModel()

@pytest.fixture
def test_config():
    return SFTConfig(
        model_name="test-model",
        output_dir="/tmp/test",
        num_train_epochs=1,
    )
```

---

## Appendix A: Design Patterns Used

| Pattern | Usage |
|---------|-------|
| Factory | Model and tokenizer loading |
| Strategy | Classifier implementations |
| Adapter | vLLM/Transformers backend |
| Registry | Expert model management |
| Builder | Configuration construction |
| Singleton | Global model cache |

---

## Appendix B: Code Style Guidelines

- **Formatting**: Black with 100 character line length
- **Linting**: Ruff with E, F, I, N, W rules
- **Type Hints**: Required for all public APIs
- **Docstrings**: Google style
- **Naming**: snake_case for functions/variables, PascalCase for classes
