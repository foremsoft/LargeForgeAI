# LargeForgeAI Quick Start

Get up and running in 5 minutes.

---

## 1. Install

```bash
pip install largeforge
```

## 2. Verify

```bash
largeforge doctor
```

## 3. Train a Model (Option A: CLI)

```bash
# Download sample data
largeforge data download alpaca-sample -o ./data

# Train with LoRA
largeforge train sft \
  --model mistralai/Mistral-7B-v0.1 \
  --dataset ./data/alpaca-sample.json \
  --output-dir ./my-model \
  --lora-r 8 \
  --epochs 1
```

## 3. Train a Model (Option B: Python)

```python
from largeforge.training import SFTTrainer, TrainingConfig, LoraConfig
from largeforge.data import DatasetLoader

# Load data
dataset = DatasetLoader.load("./data/alpaca-sample.json")

# Configure training
config = TrainingConfig(
    model_name="mistralai/Mistral-7B-v0.1",
    output_dir="./my-model",
    num_epochs=1,
    per_device_batch_size=4
)

lora = LoraConfig(r=8, lora_alpha=16)

# Train
trainer = SFTTrainer(config.model_name, config, lora, quantization="4bit")
trainer.train(dataset)
trainer.save_model()
```

## 4. Run Inference

```bash
# Start server
largeforge serve inference --model ./my-model --port 8000
```

```bash
# Test it
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "my-model", "prompt": "Hello!", "max_tokens": 50}'
```

## 5. Use the Python Client

```python
from largeforge import LargeForgeClient

client = LargeForgeClient(base_url="http://localhost:8000")

response = client.completions.create(
    model="my-model",
    prompt="Explain machine learning:",
    max_tokens=256
)

print(response.choices[0].text)
```

---

## Common Commands

| Task | Command |
|------|---------|
| Train SFT | `largeforge train sft --model MODEL --dataset DATA` |
| Train DPO | `largeforge train dpo --model MODEL --dataset DATA` |
| Quantize | `largeforge quantize awq --model MODEL --output DIR` |
| Serve | `largeforge serve inference --model MODEL` |
| Generate data | `largeforge generate data --teacher MODEL --prompts FILE` |

---

## Next Steps

- [Full Getting Started Guide](./GETTING_STARTED.md)
- [Training Tutorial](./tutorials/first-fine-tune.md)
- [API Reference](../api/REST_API_REFERENCE.md)
