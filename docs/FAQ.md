# LargeForgeAI FAQ

Frequently Asked Questions about LargeForgeAI.

---

## General

### What is LargeForgeAI?

LargeForgeAI is an open-source platform for training, fine-tuning, and deploying Large Language Models. It provides tools for creating domain-specific expert models with intelligent routing, enabling GPT-4-level performance at reduced cost.

### What makes LargeForgeAI different from other LLM tools?

- **Expert Routing**: Automatically routes queries to specialized models
- **End-to-End Pipeline**: From data generation to production deployment
- **Cost Efficiency**: Achieve high performance with smaller, specialized models
- **Production Ready**: Built-in monitoring, scaling, and operations tools

### Is LargeForgeAI free to use?

Yes, LargeForgeAI is open-source under the MIT license. You can use it freely for personal, academic, or commercial projects.

### What are the hardware requirements?

**Minimum (Training):**
- NVIDIA GPU with 16GB VRAM (e.g., RTX 3090)
- 32GB System RAM
- 100GB Storage

**Recommended:**
- NVIDIA A100 (40/80GB)
- 64GB+ System RAM
- 500GB+ NVMe Storage

### Can I use LargeForgeAI without a GPU?

Inference can run on CPU (slowly). Training requires a CUDA-capable GPU.

---

## Installation

### How do I install LargeForgeAI?

```bash
pip install largeforge
```

For all features:
```bash
pip install largeforge[all]
```

### I get CUDA errors during installation. What should I do?

1. Verify CUDA is installed: `nvidia-smi`
2. Check PyTorch CUDA version: `python -c "import torch; print(torch.cuda.is_available())"`
3. If issues persist, reinstall PyTorch for your CUDA version:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

### vLLM installation fails. How do I fix it?

Try:
```bash
pip install ninja cmake
pip install vllm --no-build-isolation
```

Or use the transformers backend as fallback.

### How do I run the system check?

```bash
largeforge doctor
```

---

## Training

### Which base models are supported?

Any HuggingFace model compatible with the transformers library:
- Llama 2/3
- Mistral
- Phi
- Gemma
- Qwen
- And many more

### How much data do I need for fine-tuning?

- **SFT**: 1,000-10,000 examples for noticeable improvement
- **DPO**: 500-5,000 preference pairs
- **Domain Pre-training**: 100MB+ of domain text

Quality matters more than quantity.

### How long does training take?

Approximate times for 7B model on A100:
- **SFT (1000 samples, 3 epochs)**: ~30 minutes
- **DPO (1000 pairs, 1 epoch)**: ~45 minutes
- **Distillation**: Depends on dataset size

### I'm getting OOM (Out of Memory) errors. How do I fix it?

1. Reduce batch size: `--batch-size 1`
2. Enable gradient checkpointing: `--gradient-checkpointing`
3. Use 4-bit quantization: `--quantization 4bit`
4. Reduce sequence length: `--max-length 1024`
5. Use smaller LoRA rank: `--lora-r 4`

### What's the difference between SFT, DPO, and distillation?

| Method | Purpose | Data Needed |
|--------|---------|-------------|
| SFT | Teach model to follow instructions | Input-output pairs |
| DPO | Align with preferences | Chosen/rejected pairs |
| Distillation | Transfer from large to small model | Prompts (teacher generates responses) |

### Can I train on multiple GPUs?

Yes! Use distributed training:
```bash
torchrun --nproc_per_node=4 -m largeforge.training.cli train sft ...
```

Or via the Python API with accelerate.

---

## Inference

### How do I start an inference server?

```bash
largeforge serve inference --model ./my-model --port 8000
```

### Which inference backend should I use?

- **vLLM**: Faster, better for production (requires CUDA)
- **Transformers**: More compatible, good for development

vLLM is default when available.

### How do I enable streaming responses?

Set `stream: true` in your API request:
```bash
curl -X POST http://localhost:8000/v1/completions \
  -d '{"model": "my-model", "prompt": "Hello", "stream": true}'
```

### What's the maximum context length?

Depends on the model and available memory. Default is 4096 tokens. Increase with:
```bash
largeforge serve inference --model ./my-model --max-model-len 8192
```

### How can I improve inference speed?

1. Use quantized models (AWQ/GPTQ)
2. Use vLLM backend
3. Enable continuous batching
4. Increase `gpu_memory_utilization`
5. Use tensor parallelism for large models

---

## Routing

### How does expert routing work?

1. Query arrives at router
2. Classifier analyzes query (keyword/neural/hybrid)
3. Most appropriate expert is selected
4. Query forwarded to expert for generation
5. Response returned to client

### How do I add a new expert?

```bash
largeforge expert add \
  --name coding-expert \
  --model ./models/coding \
  --keywords "code,python,function"
```

Or via API:
```bash
curl -X POST http://localhost:8080/experts \
  -d '{"name": "coding-expert", "model_path": "./models/coding", ...}'
```

### What if no expert matches a query?

The router falls back to the default/general expert configured in your router settings.

---

## Quantization

### What's the difference between AWQ and GPTQ?

| Method | Speed | Quality | Memory |
|--------|-------|---------|--------|
| AWQ | Faster | Higher | Similar |
| GPTQ | Slower | Good | Similar |

AWQ is generally recommended.

### How much does quantization reduce model size?

4-bit quantization typically reduces model size by ~4x:
- 7B model: 14GB → 4GB
- 13B model: 26GB → 7GB

### Does quantization affect quality?

Slightly. AWQ 4-bit typically retains 95-99% of original model quality. Always benchmark for your use case.

### How do I quantize a model?

```bash
largeforge quantize awq \
  --model ./my-model \
  --output ./my-model-awq
```

---

## Deployment

### How do I deploy to Kubernetes?

Use our Helm chart:
```bash
helm repo add largeforge https://charts.largeforge.ai
helm install largeforge largeforge/largeforge -f values.yaml
```

### How do I scale for high traffic?

1. Horizontal scaling: Add more inference replicas
2. Load balancing: Distribute requests
3. Autoscaling: Configure HPA based on metrics
4. Caching: Cache frequent queries

### How do I monitor the system?

LargeForgeAI exposes Prometheus metrics at `/metrics`. Use with Grafana for dashboards.

---

## Troubleshooting

### Training loss is not decreasing

1. Check learning rate (try 1e-5 to 5e-5)
2. Verify data format and quality
3. Check for data leakage in validation set
4. Try longer training

### Model generates repetitive text

1. Increase temperature (e.g., 0.8-1.0)
2. Add repetition_penalty (e.g., 1.1)
3. Check training data for repetitive patterns

### API returns 503 errors

1. Check if model is loaded: `/health`
2. Check GPU memory: `nvidia-smi`
3. Check logs for errors
4. Scale up replicas if under load

### Routing always selects the same expert

1. Check keyword/domain configuration
2. Verify classifier is trained properly
3. Check confidence thresholds
4. Review routing logs

---

## Best Practices

### How should I structure my training data?

- **Diverse examples**: Cover your use cases
- **High quality**: Clean, accurate responses
- **Balanced**: Don't over-represent any category
- **Consistent format**: Same format throughout

### What's a good LoRA configuration?

Starting point for 7B models:
```yaml
r: 8
lora_alpha: 16
lora_dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj]
```

Increase `r` for more capacity, decrease for efficiency.

### How do I evaluate model quality?

1. **Benchmark scores**: Run on standard benchmarks
2. **Domain tests**: Create test set for your use case
3. **Human evaluation**: Have humans rate outputs
4. **A/B testing**: Compare against baseline

---

## Contributing

### How can I contribute?

See [CONTRIBUTING.md](./development/CONTRIBUTING.md) for guidelines.

### Where do I report bugs?

Create an issue on [GitHub](https://github.com/largeforgeai/largeforgeai/issues).

### How do I request a feature?

Open a GitHub issue with the `enhancement` label.

---

## Getting Help

### Where can I get help?

- **Documentation**: [docs.largeforge.ai](https://docs.largeforge.ai)
- **GitHub Issues**: Bug reports and features
- **Discord**: [discord.gg/largeforge](https://discord.gg/largeforge)
- **Email**: support@largeforge.ai

### Is commercial support available?

Contact enterprise@largeforge.ai for commercial support options.

---

*Last updated: December 2024*
