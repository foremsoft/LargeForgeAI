# Changelog

All notable changes to LargeForgeAI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial documentation suite (ISO 29148 compliant)
- Architecture and design documents
- API reference documentation

### Changed
- None

### Deprecated
- None

### Removed
- None

### Fixed
- None

### Security
- None

---

## [1.0.0] - 2024-12-01

### Added

#### Core Features
- **Training Module** (`llm.training`)
  - Supervised Fine-Tuning (SFT) with LoRA support
  - Direct Preference Optimization (DPO) training
  - ORPO training support
  - Knowledge distillation from teacher models
  - Continued pre-training on domain data
  - 4-bit and 8-bit quantized training (QLoRA)
  - Distributed training (DDP, FSDP)
  - Gradient checkpointing for memory efficiency
  - Automatic checkpoint saving and resumption
  - Weights & Biases integration

- **Inference Module** (`llm.inference`)
  - High-performance inference with vLLM backend
  - Transformers backend for compatibility
  - OpenAI-compatible REST API
  - Streaming response support (SSE)
  - Continuous batching
  - KV cache management
  - AWQ and GPTQ quantization utilities
  - LoRA weight merging

- **Router Module** (`llm.router`)
  - Keyword-based query classification
  - Neural network-based classification
  - Hybrid classifier combining both methods
  - Dynamic expert registration
  - Load-aware routing
  - Circuit breaker pattern
  - Fallback handling

- **Data Module** (`llm.data`)
  - Synthetic data generation using teacher models
  - Preference data generation
  - Support for Alpaca, ShareGPT, and DPO formats
  - Dataset validation and preprocessing
  - Streaming data loading
  - Dataset caching

- **Experts Module** (`llm.experts`)
  - Expert model management
  - Health monitoring
  - Configuration persistence
  - Version tracking

#### CLI
- `largeforge train sft` - Supervised fine-tuning
- `largeforge train dpo` - DPO training
- `largeforge train distill` - Knowledge distillation
- `largeforge train pretrain` - Continued pre-training
- `largeforge generate data` - Synthetic data generation
- `largeforge quantize awq` - AWQ quantization
- `largeforge quantize gptq` - GPTQ quantization
- `largeforge merge lora` - LoRA weight merging
- `largeforge serve inference` - Start inference server
- `largeforge serve router` - Start router service
- `largeforge doctor` - System diagnostics

#### API Endpoints
- `POST /v1/completions` - Text completions
- `POST /v1/chat/completions` - Chat completions
- `GET /v1/models` - List models
- `GET /v1/models/{id}` - Get model info
- `POST /route` - Route query to expert
- `POST /generate` - Route and generate
- `GET /experts` - List experts
- `POST /experts` - Register expert
- `DELETE /experts/{name}` - Remove expert
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics

#### Documentation
- Architecture Document (SAD)
- Design Document (SDD)
- Business Requirements Specification (BRS)
- Stakeholder Requirements Specification (StRS)
- System Requirements Specification (SyRS)
- Software Requirements Specification (SRS)
- System Operational Concept (OpsCon)
- API Reference
- User Guides
- Operations Manual
- Security Architecture

### Dependencies
- Python 3.10+
- PyTorch 2.0+
- Transformers 4.36+
- PEFT 0.7+
- TRL 0.7+
- vLLM 0.2+
- FastAPI 0.104+

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 1.0.0 | 2024-12-01 | Initial release with full feature set |

---

## Upgrade Guide

### From 0.x to 1.0.0

This is the initial release. No upgrade path required.

### Future Upgrades

Upgrade instructions will be provided for each major version.

---

## Deprecation Policy

- Features marked deprecated will be supported for at least 2 minor versions
- Deprecated features will be removed in the next major version
- Migration guides will be provided for all deprecated features

---

[Unreleased]: https://github.com/largeforgeai/largeforgeai/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/largeforgeai/largeforgeai/releases/tag/v1.0.0
