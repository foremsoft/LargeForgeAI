# LargeForgeAI Development Playbook

## Work Breakdown Structure (WBS)

This document provides a detailed breakdown of all development tasks for implementing LargeForgeAI. Each module is designed to be small, self-contained, and independently testable.

---

## Project Structure

```
largeforge/
├── __init__.py                 # Package initialization
├── version.py                  # Version info
├── config/
│   ├── __init__.py
│   ├── base.py                 # Base configuration classes
│   ├── training.py             # Training configurations
│   ├── inference.py            # Inference configurations
│   └── router.py               # Router configurations
├── data/
│   ├── __init__.py
│   ├── formats.py              # Data format handlers
│   ├── loaders.py              # Dataset loaders
│   ├── processors.py           # Data processors
│   ├── generators.py           # Synthetic data generators
│   └── validators.py           # Data validators
├── training/
│   ├── __init__.py
│   ├── base.py                 # Base trainer class
│   ├── sft.py                  # SFT trainer
│   ├── dpo.py                  # DPO trainer
│   ├── lora.py                 # LoRA utilities
│   ├── callbacks.py            # Training callbacks
│   └── metrics.py              # Training metrics
├── inference/
│   ├── __init__.py
│   ├── base.py                 # Base inference engine
│   ├── vllm_backend.py         # vLLM backend
│   ├── transformers_backend.py # Transformers backend
│   ├── server.py               # FastAPI server
│   └── streaming.py            # SSE streaming
├── router/
│   ├── __init__.py
│   ├── classifier.py           # Query classifiers
│   ├── router.py               # Main router logic
│   ├── experts.py              # Expert management
│   └── load_balancer.py        # Load balancing
├── quantization/
│   ├── __init__.py
│   ├── awq.py                  # AWQ quantization
│   └── gptq.py                 # GPTQ quantization
├── cli/
│   ├── __init__.py
│   ├── main.py                 # CLI entry point
│   ├── train.py                # Training commands
│   ├── serve.py                # Serving commands
│   └── utils.py                # CLI utilities
└── utils/
    ├── __init__.py
    ├── logging.py              # Logging utilities
    ├── device.py               # Device management
    └── io.py                   # I/O utilities

tests/
├── __init__.py
├── conftest.py                 # Pytest fixtures
├── unit/
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_training.py
│   ├── test_inference.py
│   └── test_router.py
└── integration/
    ├── test_e2e_training.py
    └── test_e2e_inference.py
```

---

## WBS Level 1: Major Components

| ID | Component | Description | Est. Modules |
|----|-----------|-------------|--------------|
| 1.0 | Core Infrastructure | Base utilities, config, logging | 8 |
| 2.0 | Data Pipeline | Data loading, processing, generation | 10 |
| 3.0 | Training System | SFT, DPO, LoRA trainers | 12 |
| 4.0 | Inference System | Backends, server, streaming | 10 |
| 5.0 | Router System | Classification, routing, experts | 8 |
| 6.0 | Quantization | AWQ, GPTQ utilities | 4 |
| 7.0 | CLI | Command-line interface | 6 |
| 8.0 | Tests | Unit and integration tests | 12 |

**Total Modules: ~70**

---

## WBS Level 2: Detailed Breakdown

### 1.0 Core Infrastructure

| ID | Module | File | Description | Dependencies |
|----|--------|------|-------------|--------------|
| 1.1 | Package Init | `largeforge/__init__.py` | Package exports | None |
| 1.2 | Version | `largeforge/version.py` | Version management | None |
| 1.3 | Logging | `largeforge/utils/logging.py` | Structured logging | None |
| 1.4 | Device Utils | `largeforge/utils/device.py` | GPU/CPU detection | torch |
| 1.5 | IO Utils | `largeforge/utils/io.py` | File I/O helpers | None |
| 1.6 | Base Config | `largeforge/config/base.py` | Pydantic base configs | pydantic |
| 1.7 | Training Config | `largeforge/config/training.py` | Training params | 1.6 |
| 1.8 | Inference Config | `largeforge/config/inference.py` | Inference params | 1.6 |

### 2.0 Data Pipeline

| ID | Module | File | Description | Dependencies |
|----|--------|------|-------------|--------------|
| 2.1 | Data Formats | `largeforge/data/formats.py` | Format definitions | pydantic |
| 2.2 | Alpaca Loader | `largeforge/data/loaders.py:AlpacaLoader` | Alpaca format | 2.1 |
| 2.3 | ShareGPT Loader | `largeforge/data/loaders.py:ShareGPTLoader` | ShareGPT format | 2.1 |
| 2.4 | DPO Loader | `largeforge/data/loaders.py:DPOLoader` | DPO format | 2.1 |
| 2.5 | Dataset Factory | `largeforge/data/loaders.py:DatasetFactory` | Unified loader | 2.2-2.4 |
| 2.6 | Text Processor | `largeforge/data/processors.py:TextProcessor` | Text cleaning | None |
| 2.7 | Chat Formatter | `largeforge/data/processors.py:ChatFormatter` | Chat templates | transformers |
| 2.8 | Data Validator | `largeforge/data/validators.py` | Schema validation | 2.1 |
| 2.9 | Synthetic Generator | `largeforge/data/generators.py:SyntheticGenerator` | Data generation | transformers |
| 2.10 | Preference Generator | `largeforge/data/generators.py:PreferenceGenerator` | DPO data gen | 2.9 |

### 3.0 Training System

| ID | Module | File | Description | Dependencies |
|----|--------|------|-------------|--------------|
| 3.1 | Base Trainer | `largeforge/training/base.py` | Abstract trainer | transformers |
| 3.2 | LoRA Config | `largeforge/training/lora.py:LoRAConfig` | LoRA settings | peft |
| 3.3 | LoRA Setup | `largeforge/training/lora.py:setup_lora` | Apply LoRA | 3.2 |
| 3.4 | LoRA Merge | `largeforge/training/lora.py:merge_lora` | Merge adapters | 3.2 |
| 3.5 | Training Callbacks | `largeforge/training/callbacks.py` | Custom callbacks | 3.1 |
| 3.6 | Training Metrics | `largeforge/training/metrics.py` | Metric computation | None |
| 3.7 | SFT Trainer | `largeforge/training/sft.py` | SFT implementation | 3.1, trl |
| 3.8 | SFT Data Collator | `largeforge/training/sft.py:SFTDataCollator` | Batch collation | 3.7 |
| 3.9 | DPO Trainer | `largeforge/training/dpo.py` | DPO implementation | 3.1, trl |
| 3.10 | DPO Data Collator | `largeforge/training/dpo.py:DPODataCollator` | DPO batching | 3.9 |
| 3.11 | Checkpoint Manager | `largeforge/training/base.py:CheckpointManager` | Save/load | 3.1 |
| 3.12 | Training Factory | `largeforge/training/__init__.py` | Trainer factory | 3.7, 3.9 |

### 4.0 Inference System

| ID | Module | File | Description | Dependencies |
|----|--------|------|-------------|--------------|
| 4.1 | Base Engine | `largeforge/inference/base.py` | Abstract engine | None |
| 4.2 | Transformers Backend | `largeforge/inference/transformers_backend.py` | HF inference | 4.1 |
| 4.3 | vLLM Backend | `largeforge/inference/vllm_backend.py` | vLLM inference | 4.1, vllm |
| 4.4 | Backend Factory | `largeforge/inference/__init__.py` | Backend selection | 4.2, 4.3 |
| 4.5 | Streaming Handler | `largeforge/inference/streaming.py` | SSE streaming | fastapi |
| 4.6 | API Models | `largeforge/inference/server.py:models` | Request/response | pydantic |
| 4.7 | Completions Endpoint | `largeforge/inference/server.py:completions` | /v1/completions | 4.4, 4.6 |
| 4.8 | Chat Endpoint | `largeforge/inference/server.py:chat` | /v1/chat/completions | 4.4, 4.6 |
| 4.9 | Health Endpoint | `largeforge/inference/server.py:health` | /health | 4.6 |
| 4.10 | Server Factory | `largeforge/inference/server.py:create_app` | FastAPI app | 4.7-4.9 |

### 5.0 Router System

| ID | Module | File | Description | Dependencies |
|----|--------|------|-------------|--------------|
| 5.1 | Keyword Classifier | `largeforge/router/classifier.py:KeywordClassifier` | Keyword matching | None |
| 5.2 | Neural Classifier | `largeforge/router/classifier.py:NeuralClassifier` | ML classifier | transformers |
| 5.3 | Hybrid Classifier | `largeforge/router/classifier.py:HybridClassifier` | Combined | 5.1, 5.2 |
| 5.4 | Expert Config | `largeforge/router/experts.py:ExpertConfig` | Expert definition | pydantic |
| 5.5 | Expert Manager | `largeforge/router/experts.py:ExpertManager` | Expert lifecycle | 5.4 |
| 5.6 | Load Balancer | `largeforge/router/load_balancer.py` | Request distribution | 5.5 |
| 5.7 | Circuit Breaker | `largeforge/router/load_balancer.py:CircuitBreaker` | Fault tolerance | None |
| 5.8 | Router Service | `largeforge/router/router.py` | Main router | 5.3, 5.5, 5.6 |

### 6.0 Quantization

| ID | Module | File | Description | Dependencies |
|----|--------|------|-------------|--------------|
| 6.1 | AWQ Quantizer | `largeforge/quantization/awq.py` | AWQ implementation | autoawq |
| 6.2 | GPTQ Quantizer | `largeforge/quantization/gptq.py` | GPTQ implementation | auto-gptq |
| 6.3 | Calibration Data | `largeforge/quantization/awq.py:calibration` | Calibration utils | 2.5 |
| 6.4 | Quantization Factory | `largeforge/quantization/__init__.py` | Unified API | 6.1, 6.2 |

### 7.0 CLI

| ID | Module | File | Description | Dependencies |
|----|--------|------|-------------|--------------|
| 7.1 | CLI Utils | `largeforge/cli/utils.py` | Helper functions | click |
| 7.2 | Train Commands | `largeforge/cli/train.py` | Training CLI | 3.12, 7.1 |
| 7.3 | Serve Commands | `largeforge/cli/serve.py` | Serving CLI | 4.10, 7.1 |
| 7.4 | Quantize Commands | `largeforge/cli/main.py:quantize` | Quantization CLI | 6.4, 7.1 |
| 7.5 | Doctor Command | `largeforge/cli/main.py:doctor` | System check | 7.1 |
| 7.6 | Main CLI | `largeforge/cli/main.py` | Entry point | 7.2-7.5 |

### 8.0 Tests

| ID | Module | File | Description | Dependencies |
|----|--------|------|-------------|--------------|
| 8.1 | Test Fixtures | `tests/conftest.py` | Pytest fixtures | pytest |
| 8.2 | Config Tests | `tests/unit/test_config.py` | Config unit tests | 1.6-1.8 |
| 8.3 | Data Tests | `tests/unit/test_data.py` | Data unit tests | 2.x |
| 8.4 | Training Tests | `tests/unit/test_training.py` | Training unit tests | 3.x |
| 8.5 | Inference Tests | `tests/unit/test_inference.py` | Inference unit tests | 4.x |
| 8.6 | Router Tests | `tests/unit/test_router.py` | Router unit tests | 5.x |
| 8.7 | CLI Tests | `tests/unit/test_cli.py` | CLI unit tests | 7.x |
| 8.8 | E2E Training Test | `tests/integration/test_e2e_training.py` | Full training flow | 3.x |
| 8.9 | E2E Inference Test | `tests/integration/test_e2e_inference.py` | Full inference flow | 4.x |
| 8.10 | E2E Router Test | `tests/integration/test_e2e_router.py` | Full router flow | 5.x |

---

## Implementation Order

The modules should be implemented in the following order due to dependencies:

### Phase 1: Foundation (Modules 1-10)
1. 1.1 Package Init
2. 1.2 Version
3. 1.3 Logging Utils
4. 1.4 Device Utils
5. 1.5 IO Utils
6. 1.6 Base Config
7. 1.7 Training Config
8. 1.8 Inference Config
9. 8.1 Test Fixtures
10. 8.2 Config Tests

### Phase 2: Data Layer (Modules 11-22)
11. 2.1 Data Formats
12. 2.2 Alpaca Loader
13. 2.3 ShareGPT Loader
14. 2.4 DPO Loader
15. 2.5 Dataset Factory
16. 2.6 Text Processor
17. 2.7 Chat Formatter
18. 2.8 Data Validator
19. 2.9 Synthetic Generator
20. 2.10 Preference Generator
21. 8.3 Data Tests

### Phase 3: Training (Modules 23-36)
22. 3.1 Base Trainer
23. 3.2 LoRA Config
24. 3.3 LoRA Setup
25. 3.4 LoRA Merge
26. 3.5 Training Callbacks
27. 3.6 Training Metrics
28. 3.7 SFT Trainer
29. 3.8 SFT Data Collator
30. 3.9 DPO Trainer
31. 3.10 DPO Data Collator
32. 3.11 Checkpoint Manager
33. 3.12 Training Factory
34. 8.4 Training Tests
35. 8.8 E2E Training Test

### Phase 4: Inference (Modules 37-48)
36. 4.1 Base Engine
37. 4.2 Transformers Backend
38. 4.3 vLLM Backend
39. 4.4 Backend Factory
40. 4.5 Streaming Handler
41. 4.6 API Models
42. 4.7 Completions Endpoint
43. 4.8 Chat Endpoint
44. 4.9 Health Endpoint
45. 4.10 Server Factory
46. 8.5 Inference Tests
47. 8.9 E2E Inference Test

### Phase 5: Router (Modules 49-58)
48. 5.1 Keyword Classifier
49. 5.2 Neural Classifier
50. 5.3 Hybrid Classifier
51. 5.4 Expert Config
52. 5.5 Expert Manager
53. 5.6 Load Balancer
54. 5.7 Circuit Breaker
55. 5.8 Router Service
56. 8.6 Router Tests
57. 8.10 E2E Router Test

### Phase 6: Quantization (Modules 59-62)
58. 6.1 AWQ Quantizer
59. 6.2 GPTQ Quantizer
60. 6.3 Calibration Data
61. 6.4 Quantization Factory

### Phase 7: CLI (Modules 63-69)
62. 7.1 CLI Utils
63. 7.2 Train Commands
64. 7.3 Serve Commands
65. 7.4 Quantize Commands
66. 7.5 Doctor Command
67. 7.6 Main CLI
68. 8.7 CLI Tests

---

## Module Specifications

See [PROMPTS.md](./PROMPTS.md) for detailed implementation prompts for each module.

---

## Quality Gates

Each module must pass these quality gates before moving to the next:

1. **Unit Tests Pass**: All tests for the module pass
2. **Type Checking**: mypy passes with no errors
3. **Linting**: ruff check passes
4. **Documentation**: Docstrings present for public APIs

## Assembly Strategy

1. Implement modules in order
2. Run tests after each module
3. Commit after each passing phase
4. Create PR after all phases complete

---

*Last Updated: December 2024*
