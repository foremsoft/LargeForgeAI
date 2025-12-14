# LargeForgeAI Glossary

## A

**Adapter**
: A small, trainable module added to a frozen base model. LoRA adapters are the most common type, enabling efficient fine-tuning by only training a fraction of parameters.

**Alignment**
: The process of training a model to behave according to human preferences and values. Typically achieved through techniques like RLHF or DPO.

**Attention**
: A mechanism in transformers that allows the model to focus on different parts of the input when generating each output token. Self-attention enables tokens to attend to all other tokens in the sequence.

**AWQ (Activation-aware Weight Quantization)**
: A quantization method that preserves important weights based on activation patterns, achieving high compression with minimal accuracy loss.

**Autoregressive**
: A generation method where each token is predicted based on all previous tokens. Most LLMs generate text autoregressively, one token at a time.

## B

**Base Model**
: A pre-trained foundation model before any fine-tuning. Examples include Llama, Mistral, and GPT.

**Batch Size**
: The number of training samples processed together in one forward/backward pass. Larger batches are more efficient but require more memory.

**BF16 (Brain Float 16)**
: A 16-bit floating-point format optimized for deep learning, offering the range of FP32 with reduced precision. Preferred over FP16 for training stability.

## C

**Calibration Dataset**
: A small dataset used to collect activation statistics for quantization. Good calibration data improves quantized model quality.

**Causal Language Modeling**
: A training objective where the model predicts the next token given all previous tokens. Used for autoregressive text generation.

**Checkpoint**
: A saved state of a model during or after training, including weights, optimizer state, and training progress. Enables training resumption.

**Continuous Batching**
: An inference optimization that dynamically adds new requests to running batches as slots become available, maximizing GPU utilization.

## D

**DDP (Distributed Data Parallel)**
: A PyTorch training strategy that replicates the model on each GPU and synchronizes gradients. Suitable for models that fit on a single GPU.

**Distillation**
: The process of transferring knowledge from a large "teacher" model to a smaller "student" model. The student learns to match the teacher's output distribution.

**DPO (Direct Preference Optimization)**
: A simpler alternative to RLHF that directly optimizes a policy from preference data without training a separate reward model.

## E

**Embedding**
: A dense vector representation of tokens or text. Embeddings capture semantic meaning, with similar concepts having similar vectors.

**Expert**
: A specialized model fine-tuned for a specific domain or task. Multiple experts can be combined via routing for better overall performance.

## F

**Fine-tuning**
: Continuing training of a pre-trained model on task-specific data. Can be full fine-tuning (all parameters) or parameter-efficient (subset of parameters).

**FP16 (Float16)**
: 16-bit floating-point format. Reduces memory usage compared to FP32 but requires careful handling to avoid numerical issues.

**FSDP (Fully Sharded Data Parallel)**
: A PyTorch training strategy that shards model parameters across GPUs. Enables training of models larger than a single GPU's memory.

## G

**Gradient Accumulation**
: A technique to simulate larger batch sizes by accumulating gradients over multiple forward passes before updating weights.

**Gradient Checkpointing**
: A memory optimization that trades compute for memory by recomputing activations during the backward pass instead of storing them.

**GPTQ**
: A one-shot post-training quantization method for transformers. Uses second-order information for accurate weight quantization.

## H

**Hallucination**
: When a model generates confident but factually incorrect or nonsensical content. A key challenge in LLM reliability.

**HuggingFace Hub**
: A platform for sharing and downloading models, datasets, and spaces. The de facto standard for LLM model distribution.

## I

**Inference**
: The process of using a trained model to generate predictions or outputs. In LLMs, this typically means generating text from prompts.

**Instruction Tuning**
: Fine-tuning a model on instruction-following datasets to improve its ability to follow user instructions.

## K

**KV Cache**
: Stored key and value tensors from previous tokens during autoregressive generation. Avoids recomputing attention for all tokens at each step.

## L

**LoRA (Low-Rank Adaptation)**
: A parameter-efficient fine-tuning method that adds trainable low-rank matrices to frozen model weights. Dramatically reduces trainable parameters.

**Loss**
: A measure of how wrong the model's predictions are. Training minimizes loss to improve model performance.

## M

**Mixed Precision**
: Training with both FP16/BF16 and FP32, using lower precision for most operations while maintaining FP32 for sensitive calculations.

**MoE (Mixture of Experts)**
: An architecture where different "expert" subnetworks specialize in different inputs, with a router selecting which experts to use.

## O

**ORPO (Odds Ratio Preference Optimization)**
: A preference optimization method that combines SFT and preference optimization in a single training phase.

## P

**PagedAttention**
: A memory management technique used by vLLM that stores KV cache in non-contiguous memory pages, reducing memory fragmentation.

**PEFT (Parameter-Efficient Fine-Tuning)**
: A family of techniques that fine-tune only a small subset of parameters, including LoRA, prefix tuning, and adapters.

**Perplexity**
: A measure of how well a model predicts a sample. Lower perplexity indicates better prediction. Calculated as exp(cross-entropy loss).

**Prompt**
: The input text given to an LLM to generate a response. Prompt engineering is the art of crafting effective prompts.

## Q

**Quantization**
: Reducing the precision of model weights (e.g., from FP16 to INT4) to decrease memory usage and increase inference speed, with some accuracy trade-off.

**QLoRA**
: Combining 4-bit quantization with LoRA fine-tuning. Enables fine-tuning of large models on consumer GPUs.

## R

**RLHF (Reinforcement Learning from Human Feedback)**
: A technique for aligning models with human preferences by training a reward model and using reinforcement learning to optimize for high rewards.

**Router**
: A component that directs queries to appropriate expert models based on content classification.

## S

**Sampling**
: The process of selecting the next token during generation. Parameters like temperature, top_p, and top_k control the randomness.

**SFT (Supervised Fine-Tuning)**
: Training a model on input-output pairs with standard supervised learning. The first step in most LLM alignment pipelines.

**Streaming**
: Returning generated tokens incrementally as they're produced, rather than waiting for the complete response.

## T

**Temperature**
: A parameter that controls randomness in generation. Higher temperature = more random, lower = more deterministic.

**Token**
: A sub-word unit that the model processes. Text is tokenized before processing and detokenized after generation.

**Tokenizer**
: The component that converts text to tokens (encoding) and tokens back to text (decoding).

**Top-k Sampling**
: Sampling only from the k most likely next tokens, setting the probability of all others to zero.

**Top-p (Nucleus) Sampling**
: Sampling from the smallest set of tokens whose cumulative probability exceeds p. Dynamically adjusts the number of candidates.

**Transformer**
: The neural network architecture underlying modern LLMs, based on self-attention mechanisms.

**TRL (Transformer Reinforcement Learning)**
: A HuggingFace library for training transformers with reinforcement learning, including SFT, DPO, and PPO.

## V

**vLLM**
: A high-performance inference engine for LLMs featuring PagedAttention and continuous batching.

## W

**Warmup**
: Gradually increasing the learning rate at the start of training. Helps training stability.

**Weight Decay**
: A regularization technique that adds a penalty on large weights. Helps prevent overfitting.

---

## Acronyms

| Acronym | Full Form |
|---------|-----------|
| API | Application Programming Interface |
| AWQ | Activation-aware Weight Quantization |
| BF16 | Brain Float 16 |
| CLI | Command Line Interface |
| DDP | Distributed Data Parallel |
| DPO | Direct Preference Optimization |
| FP16 | Float 16 |
| FP32 | Float 32 |
| FSDP | Fully Sharded Data Parallel |
| GPU | Graphics Processing Unit |
| GPTQ | GPT Quantization |
| KV | Key-Value |
| LLM | Large Language Model |
| LoRA | Low-Rank Adaptation |
| MoE | Mixture of Experts |
| ORPO | Odds Ratio Preference Optimization |
| PEFT | Parameter-Efficient Fine-Tuning |
| QLoRA | Quantized LoRA |
| RLHF | Reinforcement Learning from Human Feedback |
| SDK | Software Development Kit |
| SFT | Supervised Fine-Tuning |
| SSE | Server-Sent Events |
| TRL | Transformer Reinforcement Learning |
| TTFT | Time To First Token |
| VRAM | Video Random Access Memory |

---

*For technical details, see the [Architecture Document](./architecture/ARCHITECTURE_DOCUMENT.md).*
