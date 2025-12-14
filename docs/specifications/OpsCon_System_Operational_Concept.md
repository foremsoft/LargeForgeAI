# System Operational Concept (OpsCon)
## LargeForgeAI - Low-Cost LLM Training and Deployment Platform

**Document ID:** LFA-OpsCon-001
**Version:** 1.0.0
**Date:** 2025-01-15
**Status:** Draft
**Classification:** Internal
**Compliance:** ISO/IEC/IEEE 29148:2018

---

## Document Control

| Version | Date | Author | Reviewer | Changes |
|---------|------|--------|----------|---------|
| 1.0.0 | 2025-01-15 | LargeForgeAI Team | - | Initial release |

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Operational Environment](#2-operational-environment)
3. [System Capabilities](#3-system-capabilities)
4. [Operational Scenarios](#4-operational-scenarios)
5. [User Classes and Characteristics](#5-user-classes-and-characteristics)
6. [Operational Modes](#6-operational-modes)
7. [System Integration](#7-system-integration)
8. [Support and Maintenance](#8-support-and-maintenance)
9. [Training and Documentation](#9-training-and-documentation)

---

## 1. Introduction

### 1.1 Purpose

This System Operational Concept (OpsCon) document describes how the LargeForgeAI system will be used operationally. It defines the operational environment, user interactions, and the scenarios in which the system will operate.

### 1.2 Scope

This document covers:
- Operational environments (development, staging, production)
- User workflows and interactions
- System operational modes
- Integration with external systems
- Support and maintenance concepts

### 1.3 System Overview

LargeForgeAI is an end-to-end platform for training and deploying Large Language Models (LLMs) at low cost. The system enables organizations to:

1. **Train** custom LLMs using various techniques (SFT, DPO, distillation)
2. **Deploy** trained models with high-performance inference serving
3. **Route** queries to specialized expert models
4. **Manage** model lifecycle and versioning

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       LargeForgeAI Operational View                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    ┌───────────────┐                            ┌───────────────────────┐  │
│    │               │                            │                       │  │
│    │  ML Engineer  │──────────────────────────▶│   Training System     │  │
│    │               │   Train Models            │   (GPU Cluster)       │  │
│    └───────────────┘                            └───────────────────────┘  │
│                                                           │                │
│                                                           ▼                │
│                                                 ┌───────────────────────┐  │
│                                                 │                       │  │
│                                                 │    Model Registry     │  │
│                                                 │                       │  │
│                                                 └───────────────────────┘  │
│                                                           │                │
│    ┌───────────────┐                                      ▼                │
│    │               │                            ┌───────────────────────┐  │
│    │   Developer   │──────────────────────────▶│   Inference System    │  │
│    │               │   API Calls               │   (Router + Experts)  │  │
│    └───────────────┘                            └───────────────────────┘  │
│                                                           │                │
│    ┌───────────────┐                                      ▼                │
│    │               │                            ┌───────────────────────┐  │
│    │    DevOps     │──────────────────────────▶│   Operations System   │  │
│    │               │   Deploy & Monitor        │   (K8s, Prometheus)   │  │
│    └───────────────┘                            └───────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Operational Environment

### 2.1 Physical Environment

#### 2.1.1 Development Environment

| Aspect | Specification |
|--------|---------------|
| **Location** | Developer workstation or cloud VM |
| **Hardware** | Single NVIDIA GPU (24GB+ VRAM), 32GB+ RAM |
| **Network** | Internet access for model downloads |
| **Storage** | 500GB+ SSD for models and data |
| **OS** | Ubuntu 20.04+, Windows 10/11, macOS |

#### 2.1.2 Training Environment

| Aspect | Specification |
|--------|---------------|
| **Location** | Cloud (AWS, GCP, Azure) or on-premises |
| **Hardware** | 1-8 NVIDIA A10/A100 GPUs per node |
| **Network** | High-bandwidth inter-node (100Gbps for multi-node) |
| **Storage** | 2TB+ NVMe for checkpoints, shared storage for data |
| **OS** | Ubuntu 22.04 LTS |

#### 2.1.3 Production Environment

| Aspect | Specification |
|--------|---------------|
| **Location** | Cloud or on-premises data center |
| **Hardware** | NVIDIA A10/L4 GPUs for inference |
| **Network** | Load-balanced, redundant connections |
| **Storage** | High-IOPS storage for model loading |
| **OS** | Ubuntu 22.04 LTS, containerized |

### 2.2 Logical Environment

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Logical Environment                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌───────────────────────────────────────────────────────────────────────┐│
│   │                         Internet / DMZ                                 ││
│   │    ┌─────────────┐                                                    ││
│   │    │    WAF /    │                                                    ││
│   │    │ Load Balancer│                                                    ││
│   │    └──────┬──────┘                                                    ││
│   └───────────┼───────────────────────────────────────────────────────────┘│
│               │                                                             │
│   ┌───────────┼───────────────────────────────────────────────────────────┐│
│   │           ▼              Application Zone                              ││
│   │    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             ││
│   │    │   Router    │    │   Router    │    │   Router    │             ││
│   │    │  Service    │    │  Service    │    │  Service    │             ││
│   │    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘             ││
│   │           └──────────────────┼──────────────────┘                     ││
│   │                              │                                         ││
│   └──────────────────────────────┼─────────────────────────────────────────┘│
│                                  │                                          │
│   ┌──────────────────────────────┼─────────────────────────────────────────┐│
│   │                              ▼              GPU Zone                    ││
│   │    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             ││
│   │    │   Expert    │    │   Expert    │    │   Expert    │             ││
│   │    │  (General)  │    │   (Code)    │    │   (Math)    │             ││
│   │    │   GPU 0     │    │   GPU 1     │    │   GPU 2     │             ││
│   │    └─────────────┘    └─────────────┘    └─────────────┘             ││
│   │                                                                        ││
│   └────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
│   ┌────────────────────────────────────────────────────────────────────────┐│
│   │                         Data Zone                                       ││
│   │    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             ││
│   │    │   Model     │    │   Training  │    │   Logs &    │             ││
│   │    │   Store     │    │    Data     │    │   Metrics   │             ││
│   │    └─────────────┘    └─────────────┘    └─────────────┘             ││
│   │                                                                        ││
│   └────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Support Environment

| Component | Tool | Purpose |
|-----------|------|---------|
| Monitoring | Prometheus + Grafana | Metrics and alerting |
| Logging | ELK Stack / Loki | Centralized logging |
| Tracing | Jaeger / OpenTelemetry | Distributed tracing |
| CI/CD | GitHub Actions | Automated testing and deployment |
| Container Registry | Docker Hub / ECR | Image management |

---

## 3. System Capabilities

### 3.1 Training Capabilities

#### 3.1.1 Supervised Fine-Tuning (SFT)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SFT Training Pipeline                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐ │
│   │  Data   │───▶│Tokenize │───▶│  LoRA   │───▶│ Train   │───▶│  Save   │ │
│   │  Load   │    │         │    │ Config  │    │  Loop   │    │ Model   │ │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘ │
│                                                                             │
│   Input: JSONL/Parquet with prompt-response pairs                          │
│   Output: LoRA adapter or merged model                                     │
│   Duration: 2-24 hours (depending on data size)                            │
│   Cost: $50-500 (cloud GPU)                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 3.1.2 Preference Optimization (DPO/ORPO)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DPO Training Pipeline                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐ │
│   │Preference│───▶│  Load   │───▶│   DPO   │───▶│ Train   │───▶│  Save   │ │
│   │  Data   │    │  Model  │    │ Trainer │    │  Loop   │    │ Model   │ │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘ │
│                                                                             │
│   Input: JSONL with (prompt, chosen, rejected) tuples                      │
│   Output: Preference-aligned model                                         │
│   Duration: 1-12 hours                                                     │
│   Cost: $50-300 (cloud GPU)                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Inference Capabilities

#### 3.2.1 High-Performance Serving

| Capability | Specification |
|------------|---------------|
| Batching | Continuous batching (vLLM) |
| Throughput | 1000+ tokens/second per GPU |
| Latency | < 100ms p95 (first token) |
| Quantization | AWQ (4-bit), GPTQ |
| Memory | Efficient KV cache management |

#### 3.2.2 Multi-Expert Routing

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Expert Routing Flow                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Request: "Write a Python function to sort a list"                        │
│                                                                             │
│   ┌─────────────────┐                                                      │
│   │                 │                                                      │
│   │    Classify     │ ────▶ Keywords detected: "Python", "function"       │
│   │                 │ ────▶ Expert selected: "code"                        │
│   │                 │                                                      │
│   └────────┬────────┘                                                      │
│            │                                                               │
│            ▼                                                               │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐      │
│   │  Code Expert    │    │  Math Expert    │    │ Writing Expert  │      │
│   │   (Selected)    │    │   (Standby)     │    │   (Standby)     │      │
│   │       ●         │    │       ○         │    │       ○         │      │
│   └────────┬────────┘    └─────────────────┘    └─────────────────┘      │
│            │                                                               │
│            ▼                                                               │
│   Response: "def sort_list(lst): return sorted(lst)"                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Operational Scenarios

### 4.1 Scenario 1: Training a Custom Model

**Actor**: ML Engineer
**Goal**: Train a customer support chatbot

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 Scenario: Train Custom Support Bot                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Step 1: Prepare Data                                                     │
│   ─────────────────────                                                    │
│   ML Engineer formats customer support conversations into JSONL:           │
│   {"messages": [{"role": "user", "content": "..."}, ...]}                  │
│                                                                             │
│   Step 2: Configure Training                                               │
│   ─────────────────────────                                                │
│   $ llm-train sft --model Qwen/Qwen2.5-7B \                               │
│                    --dataset support_data.jsonl \                          │
│                    --output ./support-model \                              │
│                    --epochs 3 \                                            │
│                    --lora-r 16                                             │
│                                                                             │
│   Step 3: Monitor Training                                                 │
│   ────────────────────────                                                 │
│   - View loss curves in terminal                                          │
│   - Check GPU utilization                                                 │
│   - Monitor checkpoint saves                                              │
│                                                                             │
│   Step 4: Evaluate Model                                                   │
│   ──────────────────────                                                   │
│   - Test with sample queries                                              │
│   - Compare with baseline                                                 │
│   - Measure response quality                                              │
│                                                                             │
│   Step 5: Deploy                                                           │
│   ───────────────                                                          │
│   - Merge LoRA weights                                                    │
│   - Quantize for inference                                                │
│   - Deploy to inference server                                            │
│                                                                             │
│   Duration: ~4-8 hours                                                     │
│   Cost: ~$100-200                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Scenario 2: Production Inference

**Actor**: Application Developer
**Goal**: Integrate LLM into web application

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 Scenario: Production Integration                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Step 1: Setup Client                                                     │
│   ────────────────────                                                     │
│   from llm.inference import RouterClient                                   │
│   client = RouterClient("https://api.example.com")                        │
│                                                                             │
│   Step 2: Make Request                                                     │
│   ────────────────────                                                     │
│   response = client.generate(                                              │
│       messages=[                                                           │
│           {"role": "system", "content": "You are a helpful assistant."},  │
│           {"role": "user", "content": user_query}                         │
│       ],                                                                   │
│       max_tokens=512,                                                      │
│       temperature=0.7                                                      │
│   )                                                                        │
│                                                                             │
│   Step 3: Handle Response                                                  │
│   ───────────────────────                                                  │
│   assistant_reply = response["text"]                                       │
│   expert_used = response["expert"]  # e.g., "code", "general"             │
│                                                                             │
│   Step 4: Error Handling                                                   │
│   ──────────────────────                                                   │
│   - Implement retry logic                                                 │
│   - Handle rate limits                                                    │
│   - Fallback for timeouts                                                 │
│                                                                             │
│   Latency: ~50-100ms                                                       │
│   Availability: 99.9%                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Scenario 3: Deployment and Operations

**Actor**: DevOps Engineer
**Goal**: Deploy and maintain inference cluster

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 Scenario: Production Deployment                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Step 1: Prepare Infrastructure                                           │
│   ──────────────────────────────                                           │
│   - Provision GPU nodes (Kubernetes)                                      │
│   - Configure networking and storage                                      │
│   - Set up monitoring stack                                               │
│                                                                             │
│   Step 2: Deploy Services                                                  │
│   ───────────────────────                                                  │
│   $ helm install largeforge ./charts/largeforge \                         │
│       --set router.replicas=3 \                                           │
│       --set experts.general.enabled=true \                                │
│       --set experts.code.enabled=true                                     │
│                                                                             │
│   Step 3: Configure Monitoring                                             │
│   ────────────────────────────                                             │
│   - Import Grafana dashboards                                             │
│   - Configure Prometheus alerts                                           │
│   - Set up log aggregation                                                │
│                                                                             │
│   Step 4: Ongoing Operations                                               │
│   ──────────────────────────                                               │
│   Daily:                                                                   │
│   - Check dashboards for anomalies                                        │
│   - Review error logs                                                     │
│   - Monitor resource utilization                                          │
│                                                                             │
│   Weekly:                                                                  │
│   - Apply security patches                                                │
│   - Review capacity needs                                                 │
│   - Update model versions if needed                                       │
│                                                                             │
│   Monthly:                                                                 │
│   - Capacity planning review                                              │
│   - Cost optimization                                                     │
│   - Security audit                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.4 Scenario 4: Model Update Workflow

**Actor**: ML Engineer + DevOps
**Goal**: Deploy improved model to production

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Scenario: Model Update Deployment                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌───────┐    ┌───────┐    ┌───────┐    ┌───────┐    ┌───────┐          │
│   │ Train │───▶│Evaluate│───▶│ Stage │───▶│ Test  │───▶│Deploy │          │
│   │       │    │       │    │       │    │       │    │       │          │
│   └───────┘    └───────┘    └───────┘    └───────┘    └───────┘          │
│       │            │            │            │            │               │
│       ▼            ▼            ▼            ▼            ▼               │
│   New Model   Benchmark    Staging     Load Test    Canary →             │
│   Trained     Passes       Deploy      Passes       Full Rollout         │
│                                                                             │
│   Timeline:                                                                │
│   Day 1: Training complete                                                │
│   Day 2: Evaluation and benchmarks                                        │
│   Day 3: Staging deployment                                               │
│   Day 4: Load testing and validation                                      │
│   Day 5: Canary deployment (10%)                                          │
│   Day 6: Monitor canary metrics                                           │
│   Day 7: Full production rollout                                          │
│                                                                             │
│   Rollback Triggers:                                                       │
│   - Error rate > 1%                                                       │
│   - Latency p99 > 500ms                                                   │
│   - Quality score < baseline                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. User Classes and Characteristics

### 5.1 User Class Definitions

| User Class | Description | Technical Level | Primary Tasks |
|------------|-------------|-----------------|---------------|
| ML Engineer | Trains and optimizes models | Expert | Training, evaluation, optimization |
| Developer | Integrates LLMs into applications | Intermediate | API integration, testing |
| Data Scientist | Experiments with models | Intermediate-Expert | Experimentation, analysis |
| DevOps Engineer | Deploys and maintains systems | Advanced | Deployment, monitoring |
| System Admin | Manages infrastructure | Advanced | Infrastructure, security |
| End User | Uses applications built on platform | Varies | Indirect interaction |

### 5.2 User Interaction Patterns

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        User Interaction Patterns                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ML Engineer                          Developer                           │
│   ────────────                         ─────────                           │
│   Frequency: Daily                     Frequency: Weekly                   │
│   Duration: Hours                      Duration: Minutes                   │
│   Interface: CLI, Python API           Interface: REST API, SDK            │
│                                                                             │
│   ┌─────────────────────┐              ┌─────────────────────┐            │
│   │ 1. Configure train  │              │ 1. Call inference   │            │
│   │ 2. Monitor progress │              │ 2. Process response │            │
│   │ 3. Evaluate results │              │ 3. Handle errors    │            │
│   │ 4. Iterate          │              │ 4. Log metrics      │            │
│   └─────────────────────┘              └─────────────────────┘            │
│                                                                             │
│   DevOps Engineer                      System Admin                        │
│   ───────────────                      ────────────                        │
│   Frequency: Daily                     Frequency: Weekly                   │
│   Duration: Minutes-Hours              Duration: Hours                     │
│   Interface: kubectl, Helm, dashboards Interface: SSH, dashboards          │
│                                                                             │
│   ┌─────────────────────┐              ┌─────────────────────┐            │
│   │ 1. Deploy updates   │              │ 1. Patch systems    │            │
│   │ 2. Monitor health   │              │ 2. Manage users     │            │
│   │ 3. Scale resources  │              │ 3. Audit logs       │            │
│   │ 4. Debug issues     │              │ 4. Update configs   │            │
│   └─────────────────────┘              └─────────────────────┘            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Operational Modes

### 6.1 Normal Operation Mode

| Aspect | Description |
|--------|-------------|
| State | All services running, healthy |
| Traffic | Normal request volume |
| Resources | Standard allocation |
| Monitoring | Routine checks |

### 6.2 High Load Mode

| Aspect | Description |
|--------|-------------|
| Trigger | Request queue > threshold |
| Actions | Auto-scale router instances |
| Resources | Increase GPU allocation |
| Monitoring | Enhanced alerting |

### 6.3 Degraded Mode

| Aspect | Description |
|--------|-------------|
| Trigger | Expert unavailable |
| Actions | Route to fallback (general) |
| User Impact | Potentially lower quality |
| Monitoring | Critical alerts |

### 6.4 Maintenance Mode

| Aspect | Description |
|--------|-------------|
| Trigger | Planned maintenance |
| Actions | Graceful shutdown, updates |
| User Impact | Temporary unavailability |
| Monitoring | Maintenance window tracking |

### 6.5 Emergency Mode

| Aspect | Description |
|--------|-------------|
| Trigger | Security incident, critical failure |
| Actions | Circuit breaker, isolation |
| User Impact | Service unavailable |
| Monitoring | Incident response procedures |

---

## 7. System Integration

### 7.1 External System Interfaces

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       External System Integration                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                          LargeForgeAI                                      │
│                               │                                            │
│       ┌───────────────────────┼───────────────────────┐                   │
│       │                       │                       │                   │
│       ▼                       ▼                       ▼                   │
│   ┌───────────┐         ┌───────────┐         ┌───────────┐              │
│   │HuggingFace│         │   Cloud   │         │ Container │              │
│   │    Hub    │         │  Storage  │         │ Registry  │              │
│   │           │         │ (S3/GCS)  │         │(Docker Hub)│             │
│   └───────────┘         └───────────┘         └───────────┘              │
│   Model downloads       Training data         Docker images              │
│   Dataset access        Checkpoints           Version management         │
│                         Model artifacts                                   │
│                                                                             │
│       ┌───────────────────────┼───────────────────────┐                   │
│       │                       │                       │                   │
│       ▼                       ▼                       ▼                   │
│   ┌───────────┐         ┌───────────┐         ┌───────────┐              │
│   │Prometheus │         │   Slack/  │         │  GitHub   │              │
│   │ /Grafana  │         │ PagerDuty │         │  Actions  │              │
│   │           │         │           │         │           │              │
│   └───────────┘         └───────────┘         └───────────┘              │
│   Monitoring            Alerting              CI/CD                       │
│   Dashboards            On-call               Automated tests            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 API Integration Points

| Integration | Protocol | Authentication | Data Format |
|-------------|----------|----------------|-------------|
| Inference API | HTTPS | API Key | JSON |
| Router API | HTTPS | API Key | JSON |
| Metrics | HTTP | None (internal) | Prometheus |
| Logs | TCP | mTLS | JSON |
| Model Hub | HTTPS | Token | Various |

---

## 8. Support and Maintenance

### 8.1 Support Tiers

| Tier | Response Time | Scope |
|------|---------------|-------|
| Community | Best effort | GitHub issues, Discord |
| Standard | 24 hours | Email support |
| Premium | 4 hours | Dedicated support |
| Enterprise | 1 hour | On-call support |

### 8.2 Maintenance Windows

| Type | Frequency | Duration | Notice |
|------|-----------|----------|--------|
| Routine | Weekly | 30 min | 24 hours |
| Security | As needed | 1 hour | 4 hours |
| Major | Monthly | 2 hours | 1 week |
| Emergency | As needed | Variable | None |

### 8.3 Backup and Recovery

| Asset | Backup Frequency | Retention | Recovery Time |
|-------|------------------|-----------|---------------|
| Model artifacts | Daily | 30 days | 1 hour |
| Training data | Weekly | 90 days | 4 hours |
| Configuration | Per change | Unlimited | 15 minutes |
| Logs | Continuous | 90 days | Streaming |

---

## 9. Training and Documentation

### 9.1 Documentation Structure

| Document | Audience | Purpose |
|----------|----------|---------|
| Quick Start Guide | All users | 15-minute first experience |
| Training Guide | ML Engineers | Complete training workflow |
| API Reference | Developers | Endpoint documentation |
| Deployment Guide | DevOps | Installation and operations |
| Architecture Docs | Architects | System design |

### 9.2 Training Programs

| Program | Duration | Audience | Delivery |
|---------|----------|----------|----------|
| Getting Started | 1 hour | All | Self-paced video |
| Training Deep Dive | 4 hours | ML Engineers | Workshop |
| Production Operations | 4 hours | DevOps | Workshop |
| Advanced Customization | 8 hours | Advanced users | Workshop |

---

## Appendix A: Operational Metrics

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Inference Latency (p95) | < 100ms | > 200ms |
| Error Rate | < 0.1% | > 1% |
| Availability | 99.9% | < 99.5% |
| GPU Utilization | > 70% | < 50% or > 95% |
| Request Queue Depth | < 100 | > 500 |

---

## Appendix B: Incident Response

| Severity | Definition | Response Time | Escalation |
|----------|------------|---------------|------------|
| P1 | Complete outage | 15 minutes | Immediate |
| P2 | Major degradation | 1 hour | 30 minutes |
| P3 | Minor issue | 4 hours | 2 hours |
| P4 | Low impact | 24 hours | Next business day |
