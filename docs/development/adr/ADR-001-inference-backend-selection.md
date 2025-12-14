# ADR-001: Inference Backend Selection

## Status

Accepted

## Date

2024-06-15

## Context

LargeForgeAI needs an inference backend for serving Large Language Models. The backend must:

1. Support various model architectures (Llama, Mistral, Phi, etc.)
2. Provide high throughput for production workloads
3. Support streaming responses
4. Be memory-efficient
5. Support quantized models
6. Provide an OpenAI-compatible API

The choice of inference backend is foundational as it affects:
- Performance characteristics
- Hardware requirements
- Operational complexity
- Feature availability

## Decision

We will use **vLLM as the primary inference backend** with **Transformers as a fallback**.

### Primary: vLLM

vLLM will be the default and recommended backend for production deployments.

### Fallback: HuggingFace Transformers

Transformers will serve as a fallback for:
- Development and testing
- Unsupported model architectures
- CPU-only deployments

## Consequences

### Positive

- **High performance**: vLLM provides 2-4x throughput improvement over naive implementations through:
  - PagedAttention for efficient KV cache management
  - Continuous batching for higher GPU utilization
  - CUDA graph optimization

- **Memory efficiency**: PagedAttention reduces memory fragmentation, allowing larger batch sizes

- **Production ready**: vLLM is battle-tested in production at many companies

- **OpenAI compatibility**: Built-in OpenAI-compatible API reduces integration effort

- **Active development**: Strong community and Anthropic/OpenAI engineers contributing

### Negative

- **CUDA requirement**: vLLM requires NVIDIA GPUs, limiting deployment options

- **Installation complexity**: vLLM has complex CUDA dependencies that can cause installation issues

- **Model support**: Not all HuggingFace models are supported (though coverage is good for popular architectures)

- **Limited CPU support**: vLLM primarily targets GPU deployment

### Neutral

- Two backends to maintain increases code complexity but provides flexibility
- Users need to understand when to use each backend

## Alternatives Considered

### Alternative 1: HuggingFace Text Generation Inference (TGI)

**Description**: HuggingFace's production inference server

**Pros**:
- Strong HuggingFace ecosystem integration
- Good documentation
- Docker-first deployment

**Cons**:
- Rust implementation harder to debug/modify
- Slightly lower performance than vLLM in benchmarks
- Less flexible configuration

**Why not chosen**: vLLM showed better performance in our benchmarks and has a more active community

### Alternative 2: Transformers Only

**Description**: Use HuggingFace Transformers directly for all inference

**Pros**:
- Simpler implementation
- Widest model support
- Easier debugging

**Cons**:
- Significantly lower throughput
- No continuous batching
- Manual memory management

**Why not chosen**: Performance requirements necessitate optimized serving

### Alternative 3: TensorRT-LLM

**Description**: NVIDIA's optimized inference solution

**Pros**:
- Highest possible performance
- Deep NVIDIA optimization
- TensorRT integration

**Cons**:
- Complex setup with model conversion
- NVIDIA-only (no AMD support)
- Less flexible than vLLM

**Why not chosen**: Setup complexity too high for initial release; may consider for future optimization

### Alternative 4: llama.cpp

**Description**: CPU-focused inference with GGUF format

**Pros**:
- Excellent CPU performance
- GGUF quantization
- Cross-platform

**Cons**:
- Different model format requires conversion
- Primarily targets single-user inference
- Less suitable for high-concurrency servers

**Why not chosen**: Focus is GPU-first production deployments; may add as third backend later

## Implementation Notes

### Backend Selection Logic

```python
def get_inference_backend(config):
    if config.backend == "auto":
        if torch.cuda.is_available() and is_vllm_available():
            return VLLMBackend(config)
        else:
            return TransformersBackend(config)
    elif config.backend == "vllm":
        if not is_vllm_available():
            raise RuntimeError("vLLM not available")
        return VLLMBackend(config)
    else:
        return TransformersBackend(config)
```

### Configuration

```yaml
inference:
  backend: auto  # Options: auto, vllm, transformers

  vllm:
    gpu_memory_utilization: 0.9
    max_model_len: 4096
    tensor_parallel_size: 1

  transformers:
    device_map: auto
    torch_dtype: bfloat16
```

## References

- [vLLM Paper](https://arxiv.org/abs/2309.06180)
- [vLLM Documentation](https://docs.vllm.ai)
- [Performance Benchmarks](./benchmarks/inference_comparison.md)
- [HuggingFace TGI](https://github.com/huggingface/text-generation-inference)

---

*ADR created by: Core Team*
*Last reviewed: 2024-12-01*
