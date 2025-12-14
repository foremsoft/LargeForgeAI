# Tutorial 1: Your First Fine-Tune

## Overview

In this tutorial, you'll learn how to fine-tune a language model using LargeForgeAI. By the end, you'll have a custom model that follows specific instructions.

**Time Required**: 45-60 minutes
**Difficulty**: Beginner
**Prerequisites**: Python basics, LargeForgeAI installed

---

## What You'll Build

A custom assistant that responds to questions about a fictional company called "TechCorp" with accurate, specific information.

---

## Step 1: Set Up Your Environment

First, ensure you have LargeForgeAI installed:

```bash
# Check installation
largeforge doctor

# Expected output should show:
# ✓ Python version: 3.10+
# ✓ CUDA available
# ✓ PyTorch installed
# ✓ Transformers installed
```

Create a project directory:

```bash
mkdir my-first-finetune
cd my-first-finetune
```

---

## Step 2: Prepare Training Data

Create a file called `training_data.json` with instruction-following examples:

```python
# create_data.py
import json

training_data = [
    {
        "instruction": "What is TechCorp?",
        "input": "",
        "output": "TechCorp is a technology company founded in 2020, specializing in AI-powered productivity tools. Headquartered in San Francisco, we employ over 500 people worldwide."
    },
    {
        "instruction": "Who founded TechCorp?",
        "input": "",
        "output": "TechCorp was founded by Dr. Sarah Chen and Marcus Johnson in 2020. Dr. Chen serves as CEO while Marcus Johnson is the CTO."
    },
    {
        "instruction": "What products does TechCorp offer?",
        "input": "",
        "output": "TechCorp offers three main products: 1) SmartDoc - AI document processing, 2) AutoSchedule - intelligent calendar management, and 3) TeamSync - AI-powered collaboration platform."
    },
    {
        "instruction": "How do I contact TechCorp support?",
        "input": "",
        "output": "You can reach TechCorp support through: Email: support@techcorp.ai, Phone: 1-800-TECHCORP, or our 24/7 chat at techcorp.ai/support."
    },
    {
        "instruction": "What is TechCorp's mission?",
        "input": "",
        "output": "TechCorp's mission is to democratize AI technology, making powerful productivity tools accessible to businesses of all sizes. We believe AI should augment human creativity, not replace it."
    },
    {
        "instruction": "Tell me about SmartDoc",
        "input": "",
        "output": "SmartDoc is our flagship document processing solution. It uses advanced AI to automatically extract, categorize, and summarize information from documents. Features include OCR, multi-language support, and integration with major cloud platforms."
    },
    {
        "instruction": "What are TechCorp's office locations?",
        "input": "",
        "output": "TechCorp has offices in: San Francisco (headquarters), New York, London, Singapore, and Tokyo. Our engineering teams are distributed globally with a strong remote-first culture."
    },
    {
        "instruction": "Does TechCorp offer enterprise plans?",
        "input": "",
        "output": "Yes, TechCorp offers enterprise plans with custom pricing, dedicated support, on-premise deployment options, SLA guarantees, and custom integrations. Contact sales@techcorp.ai for more information."
    },
    {
        "instruction": "What security certifications does TechCorp have?",
        "input": "",
        "output": "TechCorp maintains SOC 2 Type II, ISO 27001, and GDPR compliance. Our infrastructure is hosted on AWS with enterprise-grade encryption. We undergo annual third-party security audits."
    },
    {
        "instruction": "How do I get started with TechCorp products?",
        "input": "",
        "output": "Getting started is easy: 1) Sign up at techcorp.ai, 2) Choose your plan (free tier available), 3) Follow our interactive onboarding, 4) Import your first documents or connect your calendar. Most users are up and running in under 10 minutes."
    },
    # Add more examples for better results...
    {
        "instruction": "What's new at TechCorp?",
        "input": "",
        "output": "Our latest updates include SmartDoc 3.0 with 50% faster processing, AutoSchedule's new AI assistant feature, and TeamSync mobile apps for iOS and Android. Check our blog at techcorp.ai/blog for the latest news."
    },
    {
        "instruction": "Is there a free trial?",
        "input": "",
        "output": "Yes! TechCorp offers a 14-day free trial of all products with full features. No credit card required. Sign up at techcorp.ai/trial."
    }
]

# Save to file
with open("training_data.json", "w") as f:
    json.dump(training_data, f, indent=2)

print(f"Created training data with {len(training_data)} examples")
```

Run the script:

```bash
python create_data.py
```

---

## Step 3: Choose Your Base Model

For this tutorial, we'll use a small model that runs on consumer GPUs:

| Model | Size | GPU Memory Needed |
|-------|------|-------------------|
| microsoft/phi-2 | 2.7B | ~6GB |
| TinyLlama/TinyLlama-1.1B | 1.1B | ~4GB |
| meta-llama/Llama-2-7b-hf | 7B | ~16GB (or 8GB with 4-bit) |

For this tutorial, we'll use **phi-2** as it balances quality and resource requirements.

---

## Step 4: Configure Training

Create a configuration file `train_config.yaml`:

```yaml
# train_config.yaml
model:
  name: microsoft/phi-2
  torch_dtype: bfloat16

training:
  output_dir: ./techcorp-assistant
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 2e-5
  warmup_ratio: 0.1
  logging_steps: 10
  save_strategy: epoch

lora:
  r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj

data:
  dataset_path: ./training_data.json
  format: alpaca
  max_seq_length: 512
```

---

## Step 5: Run Training

Start the fine-tuning process:

```bash
largeforge train sft \
  --model microsoft/phi-2 \
  --dataset ./training_data.json \
  --output ./techcorp-assistant \
  --num-epochs 3 \
  --batch-size 2 \
  --learning-rate 2e-5 \
  --lora-r 8
```

**Expected Output:**
```
Loading model: microsoft/phi-2
Preparing dataset: 12 samples
Training with LoRA (r=8, alpha=16)

Epoch 1/3:
  Step 10/30: loss=2.145
  Step 20/30: loss=1.823
  Step 30/30: loss=1.456

Epoch 2/3:
  Step 10/30: loss=1.234
  ...

Training complete! Model saved to ./techcorp-assistant
```

**Understanding the Output:**
- **Loss**: Should decrease over time. Lower is better.
- **Epochs**: Full passes through your data.
- **Steps**: Individual gradient updates.

---

## Step 6: Test Your Model

Start an inference server with your trained model:

```bash
largeforge serve inference --model ./techcorp-assistant --port 8000
```

Test with curl:

```bash
# Test company info
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "techcorp-assistant",
    "prompt": "What is TechCorp?",
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

**Expected Response:**
```json
{
  "choices": [{
    "text": "TechCorp is a technology company founded in 2020, specializing in AI-powered productivity tools..."
  }]
}
```

Try more questions:

```bash
# Test product info
curl http://localhost:8000/v1/completions \
  -d '{"model": "techcorp-assistant", "prompt": "Tell me about SmartDoc", "max_tokens": 150}'

# Test contact info
curl http://localhost:8000/v1/completions \
  -d '{"model": "techcorp-assistant", "prompt": "How do I contact support?", "max_tokens": 100}'
```

---

## Step 7: Compare with Base Model

To see the improvement, compare against the base model:

```bash
# Start base model server on different port
largeforge serve inference --model microsoft/phi-2 --port 8001

# Ask same question
curl http://localhost:8001/v1/completions \
  -d '{"model": "phi-2", "prompt": "What is TechCorp?", "max_tokens": 150}'
```

The base model won't know about TechCorp, while your fine-tuned model will provide accurate information.

---

## Troubleshooting

### Out of Memory Error

If you get OOM errors:

```bash
# Reduce batch size
largeforge train sft --batch-size 1 --gradient-accumulation 8

# Or enable 4-bit quantization
largeforge train sft --quantization 4bit
```

### Training Loss Not Decreasing

- Check your data format is correct
- Try lowering learning rate: `--learning-rate 1e-5`
- Add more training examples

### Model Outputs Garbage

- Train for more epochs
- Check data quality
- Ensure correct chat template is used

---

## Next Steps

Congratulations! You've successfully fine-tuned your first model. Here's what to explore next:

1. **Add More Data**: More diverse examples improve quality
2. **Try DPO Training**: Add preference data to improve responses
3. **Quantize for Deployment**: Reduce model size with AWQ
4. **Deploy to Production**: Use Kubernetes deployment

**Next Tutorial**: [Tutorial 2: Preference Optimization with DPO](./TUTORIAL_02_DPO_TRAINING.md)

---

## Complete Code

Here's the complete script for this tutorial:

```python
#!/usr/bin/env python3
# complete_tutorial.py

import json
import subprocess
import time
import requests

# Step 1: Create training data
training_data = [
    {"instruction": "What is TechCorp?", "input": "", "output": "TechCorp is a technology company..."},
    # ... (add all examples)
]

with open("training_data.json", "w") as f:
    json.dump(training_data, f, indent=2)

print("✓ Training data created")

# Step 2: Run training
subprocess.run([
    "largeforge", "train", "sft",
    "--model", "microsoft/phi-2",
    "--dataset", "./training_data.json",
    "--output", "./techcorp-assistant",
    "--num-epochs", "3"
])

print("✓ Training complete")

# Step 3: Start server (in background)
server = subprocess.Popen([
    "largeforge", "serve", "inference",
    "--model", "./techcorp-assistant",
    "--port", "8000"
])

time.sleep(30)  # Wait for server to start
print("✓ Server started")

# Step 4: Test
response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "model": "techcorp-assistant",
        "prompt": "What is TechCorp?",
        "max_tokens": 150
    }
)

print("✓ Test response:", response.json())

# Cleanup
server.terminate()
```

---

*Tutorial created for LargeForgeAI v1.0 | Last Updated: December 2024*
