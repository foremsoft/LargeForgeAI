# ADR-002: Training Framework Selection

## Status

Accepted

## Date

2024-06-20

## Context

LargeForgeAI needs a training framework that supports:

1. Supervised Fine-Tuning (SFT)
2. Direct Preference Optimization (DPO)
3. RLHF components (reward modeling, PPO)
4. Parameter-Efficient Fine-Tuning (PEFT/LoRA)
5. Distributed training
6. Mixed precision training
7. Integration with HuggingFace ecosystem

The framework choice affects:
- Training performance
- Feature availability
- Maintenance burden
- User experience

## Decision

We will use **TRL (Transformer Reinforcement Learning) by HuggingFace** as our primary training framework, complemented by **PEFT** for parameter-efficient methods and **Accelerate** for distributed training.

### Components

| Component | Purpose |
|-----------|---------|
| TRL | SFT, DPO, ORPO, reward modeling |
| PEFT | LoRA, QLoRA adapters |
| Accelerate | Distributed training, mixed precision |
| Transformers | Model loading, tokenization |
| Datasets | Data loading and processing |

## Consequences

### Positive

- **Unified ecosystem**: All components from HuggingFace integrate seamlessly

- **Active development**: TRL is actively maintained with frequent updates

- **Comprehensive features**: Supports all training methods we need out of the box

- **Community support**: Large community, good documentation, many examples

- **Production proven**: Used by many companies for LLM training

- **Easy migration**: Users familiar with HuggingFace can quickly adopt

### Negative

- **Abstraction overhead**: Some performance loss compared to custom implementations

- **Version dependencies**: Tight coupling between TRL, Transformers, PEFT versions

- **Limited customization**: Some advanced techniques require workarounds

- **Memory overhead**: Higher memory usage than hand-optimized training loops

### Neutral

- Learning curve for users unfamiliar with HuggingFace
- Tied to HuggingFace release cycle

## Alternatives Considered

### Alternative 1: Axolotl

**Description**: Popular LLM fine-tuning framework

**Pros**:
- All-in-one configuration-based training
- Good defaults for various models
- Active community

**Cons**:
- Less flexible than TRL
- Heavy YAML configuration
- Harder to integrate as library

**Why not chosen**: We want a library approach, not a config-driven tool

### Alternative 2: LLaMA-Factory

**Description**: Unified fine-tuning framework

**Pros**:
- Extensive model support
- Web UI available
- Good documentation

**Cons**:
- Primarily Chinese documentation
- Opinionated project structure
- Less programmatic control

**Why not chosen**: Need more programmatic control and better English documentation

### Alternative 3: Custom Training Loop

**Description**: Build training from scratch using PyTorch

**Pros**:
- Maximum flexibility
- Optimal performance
- Full control

**Cons**:
- Significant development effort
- Maintenance burden
- Reinventing solved problems

**Why not chosen**: Development time better spent on unique features

### Alternative 4: DeepSpeed-Chat

**Description**: Microsoft's RLHF training framework

**Pros**:
- Excellent scaling
- Advanced ZeRO optimization
- Full RLHF pipeline

**Cons**:
- Complex configuration
- Heavier dependency
- Less active development recently

**Why not chosen**: TRL with Accelerate provides sufficient scaling with simpler setup

## Implementation Notes

### Example SFT Training

```python
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model with PEFT
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16
)

# LoRA configuration
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)

# TRL training config
training_config = SFTConfig(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    bf16=True
)

# Train with TRL
trainer = SFTTrainer(
    model=model,
    args=training_config,
    train_dataset=dataset,
    peft_config=peft_config
)
trainer.train()
```

### Version Compatibility Matrix

| TRL | Transformers | PEFT | Accelerate |
|-----|--------------|------|------------|
| 0.7+ | 4.36+ | 0.7+ | 0.25+ |

## References

- [TRL Documentation](https://huggingface.co/docs/trl)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Accelerate Documentation](https://huggingface.co/docs/accelerate)
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

---

*ADR created by: Core Team*
*Last reviewed: 2024-12-01*
