# LargeForgeAI Implementation Prompts

This file contains detailed prompts for Claude Code to implement each module.

---

## Phase 1: Foundation

### PROMPT 1.1: Package Init
```
Create the main package initialization file for LargeForgeAI.

File: largeforge/__init__.py

Requirements:
1. Import and expose key classes: SFTTrainer, DPOTrainer, InferenceEngine, Router
2. Import version from version.py
3. Set __all__ to control public API
4. Add package-level docstring describing LargeForgeAI

The file should be minimal and only import from submodules.
Use lazy imports where possible for performance.
```

### PROMPT 1.2: Version Module
```
Create the version management module.

File: largeforge/version.py

Requirements:
1. Define __version__ = "1.0.0"
2. Define VERSION_INFO tuple (major, minor, patch)
3. Add get_version() function returning formatted string
4. Add get_version_info() returning dict with version details

Include Python version requirements (3.10+).
```

### PROMPT 1.3: Logging Utils
```
Create structured logging utilities for LargeForgeAI.

File: largeforge/utils/logging.py

Requirements:
1. Create get_logger(name: str) -> logging.Logger function
2. Configure JSON formatting for production
3. Configure colored output for development
4. Support log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
5. Add context managers for timed operations
6. Include request_id tracking capability

Example usage:
```python
from largeforge.utils.logging import get_logger
logger = get_logger(__name__)
logger.info("Training started", extra={"model": "llama-7b", "epochs": 3})
```

Use Python's built-in logging module. Do not add external dependencies.
```

### PROMPT 1.4: Device Utils
```
Create device management utilities.

File: largeforge/utils/device.py

Requirements:
1. get_device() -> str: Returns "cuda" or "cpu" based on availability
2. get_device_count() -> int: Number of GPUs available
3. get_device_name(index: int = 0) -> str: GPU name
4. get_device_memory(index: int = 0) -> dict: Total and free memory in GB
5. select_device(preference: str = "auto") -> torch.device: Smart device selection
6. is_bf16_supported() -> bool: Check BF16 capability

Handle cases where CUDA is not available gracefully.
Include proper type hints and docstrings.

Test:
```python
def test_get_device():
    device = get_device()
    assert device in ["cuda", "cpu"]
```
```

### PROMPT 1.5: IO Utils
```
Create file I/O utility functions.

File: largeforge/utils/io.py

Requirements:
1. load_json(path: str | Path) -> dict: Load JSON file
2. save_json(data: dict, path: str | Path, indent: int = 2): Save JSON
3. load_jsonl(path: str | Path) -> List[dict]: Load JSON lines
4. save_jsonl(data: List[dict], path: str | Path): Save JSON lines
5. ensure_dir(path: str | Path): Create directory if not exists
6. get_file_size(path: str | Path) -> int: File size in bytes
7. atomic_write(path: str | Path, content: str): Atomic file write

All functions should handle Path objects and strings.
Include proper error handling with informative messages.

Test:
```python
def test_json_roundtrip(tmp_path):
    data = {"key": "value", "number": 42}
    path = tmp_path / "test.json"
    save_json(data, path)
    loaded = load_json(path)
    assert loaded == data
```
```

### PROMPT 1.6: Base Config
```
Create base configuration classes using Pydantic.

File: largeforge/config/base.py

Requirements:
1. BaseConfig class with:
   - model_config for Pydantic v2 (extra="forbid", validate_assignment=True)
   - to_dict() method
   - from_dict(cls, data: dict) classmethod
   - from_yaml(cls, path: str) classmethod
   - to_yaml(path: str) method

2. ModelConfig class:
   - name: str (model name or path)
   - revision: str = "main"
   - torch_dtype: str = "bfloat16" (validate: float16, bfloat16, float32)
   - device_map: str = "auto"
   - trust_remote_code: bool = False

Use Pydantic v2 syntax. Include validators for torch_dtype.

Test:
```python
def test_model_config():
    config = ModelConfig(name="meta-llama/Llama-2-7b-hf")
    assert config.torch_dtype == "bfloat16"
    assert config.device_map == "auto"
```
```

### PROMPT 1.7: Training Config
```
Create training configuration classes.

File: largeforge/config/training.py

Requirements:
1. TrainingConfig(BaseConfig):
   - output_dir: str
   - num_train_epochs: int = 3 (ge=1, le=100)
   - per_device_train_batch_size: int = 4 (ge=1)
   - per_device_eval_batch_size: int = 4 (ge=1)
   - gradient_accumulation_steps: int = 4 (ge=1)
   - learning_rate: float = 2e-5 (gt=0)
   - weight_decay: float = 0.01 (ge=0)
   - warmup_ratio: float = 0.1 (ge=0, le=1)
   - max_grad_norm: float = 1.0 (gt=0)
   - logging_steps: int = 10 (ge=1)
   - save_steps: int = 500 (ge=1)
   - eval_steps: int = 500 (ge=1)
   - save_total_limit: int = 3 (ge=1)
   - fp16: bool = False
   - bf16: bool = True
   - seed: int = 42

2. LoRAConfig(BaseConfig):
   - r: int = 8 (ge=1, le=256)
   - lora_alpha: int = 16 (ge=1)
   - lora_dropout: float = 0.05 (ge=0, le=1)
   - target_modules: List[str] = ["q_proj", "v_proj"]
   - bias: str = "none" (validate: none, all, lora_only)
   - task_type: str = "CAUSAL_LM"

3. SFTConfig(TrainingConfig):
   - max_seq_length: int = 2048 (ge=128)
   - packing: bool = False
   - dataset_text_field: str = "text"

4. DPOConfig(TrainingConfig):
   - beta: float = 0.1 (gt=0, le=1)
   - max_length: int = 1024 (ge=128)
   - max_prompt_length: int = 512 (ge=64)
   - loss_type: str = "sigmoid" (validate: sigmoid, hinge, ipo)

Include validation to ensure fp16 and bf16 are not both True.

Test:
```python
def test_training_config_validation():
    with pytest.raises(ValidationError):
        TrainingConfig(output_dir="./out", fp16=True, bf16=True)
```
```

### PROMPT 1.8: Inference Config
```
Create inference configuration classes.

File: largeforge/config/inference.py

Requirements:
1. GenerationConfig(BaseConfig):
   - max_tokens: int = 256 (ge=1, le=32768)
   - temperature: float = 0.7 (ge=0, le=2)
   - top_p: float = 0.9 (gt=0, le=1)
   - top_k: int = 50 (ge=0)
   - repetition_penalty: float = 1.0 (ge=1)
   - stop: Optional[List[str]] = None
   - stream: bool = False

2. InferenceConfig(BaseConfig):
   - backend: str = "auto" (validate: auto, vllm, transformers)
   - host: str = "0.0.0.0"
   - port: int = 8000 (ge=1, le=65535)
   - max_model_len: int = 4096 (ge=128)
   - gpu_memory_utilization: float = 0.9 (gt=0, le=1)
   - tensor_parallel_size: int = 1 (ge=1)
   - quantization: Optional[str] = None (validate: None, awq, gptq)

3. RouterConfig(BaseConfig):
   - classifier_type: str = "hybrid" (validate: keyword, neural, hybrid)
   - confidence_threshold: float = 0.6 (ge=0, le=1)
   - fallback_expert: str = "general"
   - keyword_weight: float = 0.3 (ge=0, le=1)
   - neural_weight: float = 0.7 (ge=0, le=1)

Add validator to ensure keyword_weight + neural_weight == 1.0.

Test:
```python
def test_generation_config():
    config = GenerationConfig(max_tokens=100, temperature=0.8)
    assert config.top_p == 0.9  # default
```
```

### PROMPT 1.9: Test Fixtures
```
Create pytest fixtures and configuration.

File: tests/conftest.py

Requirements:
1. @pytest.fixture for sample_alpaca_data returning list of dicts
2. @pytest.fixture for sample_sharegpt_data returning list of dicts
3. @pytest.fixture for sample_dpo_data returning list of dicts
4. @pytest.fixture for tmp_model_dir using tmp_path
5. @pytest.fixture for mock_tokenizer returning a mock tokenizer
6. @pytest.fixture(scope="session") for small test model name
7. Mark GPU tests with @pytest.mark.gpu
8. Add pytest.ini settings for markers

Sample data should be realistic but minimal (3-5 examples each).

Also create tests/__init__.py as empty file.
```

### PROMPT 1.10: Config Tests
```
Create unit tests for configuration modules.

File: tests/unit/test_config.py

Requirements:
1. Test BaseConfig serialization (to_dict, from_dict)
2. Test ModelConfig defaults and validation
3. Test TrainingConfig validation (fp16/bf16 conflict)
4. Test LoRAConfig field constraints
5. Test SFTConfig inheritance
6. Test DPOConfig beta validation
7. Test GenerationConfig temperature bounds
8. Test InferenceConfig backend validation
9. Test RouterConfig weight validation

Each test should:
- Test valid inputs succeed
- Test invalid inputs raise ValidationError
- Test defaults are applied correctly

Use pytest.raises for validation error tests.
```

---

## Phase 2: Data Layer

### PROMPT 2.1: Data Formats
```
Create data format definitions and schemas.

File: largeforge/data/formats.py

Requirements:
1. AlpacaExample(BaseModel):
   - instruction: str (min_length=1)
   - input: str = ""
   - output: str (min_length=1)

2. ShareGPTMessage(BaseModel):
   - from_: str = Field(alias="from") (validate: human, gpt, system)
   - value: str (min_length=1)

3. ShareGPTExample(BaseModel):
   - conversations: List[ShareGPTMessage] (min_length=2)
   - Validator: first message should be system or human

4. DPOExample(BaseModel):
   - prompt: str (min_length=1)
   - chosen: str (min_length=1)
   - rejected: str (min_length=1)

5. DataFormat enum: ALPACA, SHAREGPT, DPO

6. detect_format(data: List[dict]) -> DataFormat: Auto-detect format

Include proper Field definitions with descriptions.
```

### PROMPT 2.2: Alpaca Loader
```
Create Alpaca format dataset loader.

File: largeforge/data/loaders.py (partial - AlpacaLoader class)

Requirements:
1. AlpacaLoader class:
   - __init__(self, path: str | Path, tokenizer: Any = None)
   - load(self) -> List[AlpacaExample]: Load and validate
   - to_dataset(self) -> Dataset: Convert to HF Dataset
   - format_example(self, example: AlpacaExample) -> str: Format for training

2. format_example should create prompt like:
   "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"

   If input is empty, skip the Input section.

3. Validate all examples against AlpacaExample schema
4. Support both .json and .jsonl files

Test:
```python
def test_alpaca_loader(tmp_path, sample_alpaca_data):
    path = tmp_path / "alpaca.json"
    save_json(sample_alpaca_data, path)
    loader = AlpacaLoader(path)
    examples = loader.load()
    assert len(examples) == len(sample_alpaca_data)
```
```

### PROMPT 2.3: ShareGPT Loader
```
Create ShareGPT format dataset loader.

File: largeforge/data/loaders.py (partial - ShareGPTLoader class)

Requirements:
1. ShareGPTLoader class:
   - __init__(self, path: str | Path, tokenizer: Any = None)
   - load(self) -> List[ShareGPTExample]: Load and validate
   - to_dataset(self) -> Dataset: Convert to HF Dataset
   - to_messages(self, example: ShareGPTExample) -> List[dict]: Convert to messages format

2. to_messages should convert:
   {"from": "human", "value": "..."} -> {"role": "user", "content": "..."}
   {"from": "gpt", "value": "..."} -> {"role": "assistant", "content": "..."}
   {"from": "system", "value": "..."} -> {"role": "system", "content": "..."}

3. Validate conversation structure (alternating human/gpt)
4. Support both .json and .jsonl files

Test:
```python
def test_sharegpt_loader(tmp_path, sample_sharegpt_data):
    path = tmp_path / "sharegpt.json"
    save_json(sample_sharegpt_data, path)
    loader = ShareGPTLoader(path)
    examples = loader.load()
    messages = loader.to_messages(examples[0])
    assert messages[0]["role"] in ["system", "user"]
```
```

### PROMPT 2.4: DPO Loader
```
Create DPO format dataset loader.

File: largeforge/data/loaders.py (partial - DPOLoader class)

Requirements:
1. DPOLoader class:
   - __init__(self, path: str | Path)
   - load(self) -> List[DPOExample]: Load and validate
   - to_dataset(self) -> Dataset: Convert to HF Dataset with columns: prompt, chosen, rejected
   - validate_pair(self, example: DPOExample) -> bool: Ensure chosen != rejected

2. Dataset should have these exact column names for TRL compatibility
3. Support both .json and .jsonl files
4. Log warning if chosen == rejected for any example

Test:
```python
def test_dpo_loader(tmp_path, sample_dpo_data):
    path = tmp_path / "dpo.json"
    save_json(sample_dpo_data, path)
    loader = DPOLoader(path)
    dataset = loader.to_dataset()
    assert "prompt" in dataset.column_names
    assert "chosen" in dataset.column_names
    assert "rejected" in dataset.column_names
```
```

### PROMPT 2.5: Dataset Factory
```
Create unified dataset loading factory.

File: largeforge/data/loaders.py (partial - DatasetFactory class)

Requirements:
1. DatasetFactory class:
   - load(path: str, format: str | DataFormat = "auto") -> Dataset
   - Static method that auto-detects format if not specified

2. Support formats: "alpaca", "sharegpt", "dpo", "auto"

3. Auto-detection logic:
   - If has "conversations" key -> ShareGPT
   - If has "chosen" and "rejected" -> DPO
   - If has "instruction" and "output" -> Alpaca
   - Otherwise raise ValueError

4. Return HuggingFace Dataset ready for training

Usage:
```python
dataset = DatasetFactory.load("./data/train.json")  # auto-detect
dataset = DatasetFactory.load("./data/train.json", format="alpaca")
```

Test:
```python
def test_dataset_factory_auto_detect(tmp_path, sample_alpaca_data):
    path = tmp_path / "data.json"
    save_json(sample_alpaca_data, path)
    dataset = DatasetFactory.load(str(path))
    assert len(dataset) == len(sample_alpaca_data)
```
```

### PROMPT 2.6: Text Processor
```
Create text processing utilities.

File: largeforge/data/processors.py (partial - TextProcessor class)

Requirements:
1. TextProcessor class with static methods:
   - clean_text(text: str) -> str: Remove extra whitespace, normalize unicode
   - truncate(text: str, max_length: int, tokenizer: Any = None) -> str
   - remove_special_tokens(text: str, tokens: List[str]) -> str
   - normalize_whitespace(text: str) -> str: Single spaces, trim
   - remove_urls(text: str) -> str: Remove URLs
   - remove_emails(text: str) -> str: Remove email addresses

2. clean_text should:
   - Normalize unicode (NFKC)
   - Replace multiple spaces with single space
   - Strip leading/trailing whitespace
   - Remove null bytes

3. truncate should:
   - If tokenizer provided, truncate by tokens
   - Otherwise truncate by characters
   - Add ellipsis if truncated

Test:
```python
def test_clean_text():
    dirty = "  Hello   world  \n\n  test  "
    clean = TextProcessor.clean_text(dirty)
    assert clean == "Hello world test"
```
```

### PROMPT 2.7: Chat Formatter
```
Create chat template formatter.

File: largeforge/data/processors.py (partial - ChatFormatter class)

Requirements:
1. ChatFormatter class:
   - __init__(self, tokenizer: Any)
   - format_messages(self, messages: List[dict]) -> str: Apply chat template
   - format_alpaca(self, example: AlpacaExample) -> str: Format Alpaca as chat
   - format_sharegpt(self, example: ShareGPTExample) -> str: Format ShareGPT as chat

2. format_messages should use tokenizer.apply_chat_template if available
3. Fall back to simple format if no template:
   "<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n{assistant}"

4. Handle models with and without system message support

5. Alpaca to chat conversion:
   - instruction -> user message
   - input -> append to user message if present
   - output -> assistant message

Test:
```python
def test_chat_formatter(mock_tokenizer):
    formatter = ChatFormatter(mock_tokenizer)
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"}
    ]
    formatted = formatter.format_messages(messages)
    assert "Hello" in formatted
    assert "Hi!" in formatted
```
```

### PROMPT 2.8: Data Validator
```
Create data validation utilities.

File: largeforge/data/validators.py

Requirements:
1. DataValidator class:
   - validate_alpaca(data: List[dict]) -> Tuple[List[AlpacaExample], List[dict]]
     Returns (valid_examples, invalid_examples_with_errors)

   - validate_sharegpt(data: List[dict]) -> Tuple[List[ShareGPTExample], List[dict]]

   - validate_dpo(data: List[dict]) -> Tuple[List[DPOExample], List[dict]]

   - validate_file(path: str, format: DataFormat) -> ValidationReport

2. ValidationReport dataclass:
   - total: int
   - valid: int
   - invalid: int
   - errors: List[str]
   - warnings: List[str]

3. Validation checks:
   - Required fields present
   - Fields have correct types
   - String fields not empty (when required)
   - Conversations alternate correctly (ShareGPT)
   - Chosen != rejected (DPO)

4. Log warnings for edge cases (very short responses, etc.)

Test:
```python
def test_validate_alpaca():
    data = [
        {"instruction": "Test", "output": "Response"},  # valid
        {"instruction": "", "output": "Response"},  # invalid - empty instruction
    ]
    valid, invalid = DataValidator.validate_alpaca(data)
    assert len(valid) == 1
    assert len(invalid) == 1
```
```

### PROMPT 2.9: Synthetic Generator
```
Create synthetic data generator using teacher models.

File: largeforge/data/generators.py (partial - SyntheticGenerator class)

Requirements:
1. SyntheticGenerator class:
   - __init__(self, teacher_model: str, device: str = "auto",
              quantization: str = None, max_new_tokens: int = 512)
   - generate_response(self, prompt: str) -> str
   - generate_sft_data(self, prompts: List[str],
                       system_prompt: str = None) -> List[AlpacaExample]
   - generate_batch(self, prompts: List[str], batch_size: int = 4) -> List[str]

2. Use transformers pipeline for generation
3. Support quantized models (4bit, 8bit)
4. Include progress bar for batch generation
5. Handle OOM gracefully with smaller batches

6. generate_sft_data should:
   - Use prompts as instructions
   - Generate outputs using teacher
   - Return AlpacaExample objects

Test (mock the model):
```python
def test_synthetic_generator_format():
    # Test with mock
    generator = MagicMock()
    generator.generate_response.return_value = "Test response"
    result = generator.generate_sft_data(["What is Python?"])
    assert len(result) == 1
```
```

### PROMPT 2.10: Preference Generator
```
Create preference/DPO data generator.

File: largeforge/data/generators.py (partial - PreferenceGenerator class)

Requirements:
1. PreferenceGenerator class:
   - __init__(self, teacher_model: str, device: str = "auto")
   - generate_pair(self, prompt: str, num_samples: int = 2) -> DPOExample
   - generate_preference_data(self, prompts: List[str],
                              judge_model: str = None) -> List[DPOExample]
   - score_response(self, prompt: str, response: str) -> float

2. generate_pair should:
   - Generate multiple responses for same prompt
   - Use temperature variation (0.3 for chosen, 0.9 for rejected)
   - Or use judge model to rank responses

3. If judge_model provided:
   - Use it to score responses
   - Pick highest score as chosen, lowest as rejected

4. Without judge:
   - Low temperature = chosen (more focused)
   - High temperature = rejected (more random)

Test:
```python
def test_preference_generator_pair():
    # Test structure of output
    generator = PreferenceGenerator("gpt2")  # small model for test
    pair = generator.generate_pair("Explain AI")
    assert pair.chosen != pair.rejected
    assert len(pair.prompt) > 0
```
```

### PROMPT 2.11: Data Tests
```
Create comprehensive tests for data modules.

File: tests/unit/test_data.py

Requirements:
1. Test AlpacaLoader:
   - Loading valid JSON
   - Loading JSONL
   - Validation of examples
   - format_example output

2. Test ShareGPTLoader:
   - Loading conversations
   - to_messages conversion
   - Conversation validation

3. Test DPOLoader:
   - Loading preference pairs
   - Dataset column names
   - validate_pair

4. Test DatasetFactory:
   - Auto-detection for each format
   - Explicit format specification
   - Invalid format error

5. Test TextProcessor:
   - clean_text
   - truncate with and without tokenizer
   - normalize_whitespace

6. Test ChatFormatter:
   - format_messages
   - format_alpaca
   - Fallback formatting

7. Test DataValidator:
   - Valid data passes
   - Invalid data caught
   - ValidationReport correctness

Use fixtures from conftest.py. Each test function should test one specific behavior.
```

---

## Phase 3: Training (Prompts 3.1-3.12)

### PROMPT 3.1: Base Trainer
```
Create abstract base trainer class.

File: largeforge/training/base.py

Requirements:
1. BaseTrainer(ABC):
   - __init__(self, model_name: str, config: TrainingConfig,
              lora_config: Optional[LoRAConfig] = None)
   - load_model(self) -> PreTrainedModel: Load and prepare model
   - load_tokenizer(self) -> PreTrainedTokenizer: Load tokenizer
   - prepare_model(self, model: PreTrainedModel) -> PreTrainedModel: Apply LoRA if configured
   - train(self, train_dataset: Dataset, eval_dataset: Dataset = None) -> TrainingResult (abstract)
   - save(self, output_dir: str): Save model and tokenizer
   - @property model: Current model
   - @property tokenizer: Current tokenizer

2. TrainingResult dataclass:
   - loss: float
   - metrics: Dict[str, float]
   - output_dir: str
   - training_time: float

3. load_model should:
   - Use device_map="auto"
   - Apply quantization if specified in config
   - Use torch_dtype from config

4. prepare_model should:
   - Apply LoRA if lora_config provided
   - Enable gradient checkpointing if enabled
   - Prepare for training (model.train())

Include proper logging for each step.
```

### PROMPT 3.2: LoRA Config
```
Create LoRA configuration and utilities.

File: largeforge/training/lora.py (partial - config)

Requirements:
1. Re-export LoRAConfig from config/training.py

2. get_target_modules(model_name: str) -> List[str]:
   - Return appropriate target modules based on model architecture
   - Llama/Mistral: ["q_proj", "k_proj", "v_proj", "o_proj"]
   - GPT-2/GPT-Neo: ["c_attn", "c_proj"]
   - Falcon: ["query_key_value", "dense"]
   - Default: ["q_proj", "v_proj"]

3. get_recommended_config(model_size: str) -> LoRAConfig:
   - "small" (< 3B): r=4, alpha=8
   - "medium" (3-13B): r=8, alpha=16
   - "large" (> 13B): r=16, alpha=32

Return LoRAConfig objects with recommended settings.
```

### PROMPT 3.3: LoRA Setup
```
Create LoRA model setup function.

File: largeforge/training/lora.py (partial - setup)

Requirements:
1. setup_lora(model: PreTrainedModel, config: LoRAConfig) -> PeftModel:
   - Create PEFT config from LoRAConfig
   - Apply to model using get_peft_model
   - Return PeftModel

2. get_peft_config(config: LoRAConfig) -> LoraConfig (from peft):
   - Convert our LoRAConfig to PEFT's LoraConfig
   - Map all fields correctly

3. print_trainable_parameters(model: PeftModel):
   - Print total params, trainable params, percentage
   - Use model.print_trainable_parameters() if available

Test:
```python
def test_setup_lora():
    # Would need small model for actual test
    config = LoRAConfig(r=4, lora_alpha=8)
    peft_config = get_peft_config(config)
    assert peft_config.r == 4
```
```

### PROMPT 3.4: LoRA Merge
```
Create LoRA adapter merging utilities.

File: largeforge/training/lora.py (partial - merge)

Requirements:
1. merge_lora(model_path: str, adapter_path: str, output_path: str,
              torch_dtype: str = "bfloat16"):
   - Load base model
   - Load LoRA adapter
   - Merge weights: model.merge_and_unload()
   - Save merged model

2. unload_lora(model: PeftModel) -> PreTrainedModel:
   - Unload adapter without merging
   - Return base model

3. list_adapters(model: PeftModel) -> List[str]:
   - List all loaded adapters

4. load_adapter(model: PeftModel, adapter_path: str, adapter_name: str):
   - Load additional adapter
   - Support multiple adapters

Include progress bars for long operations.
Log model sizes before and after merge.
```

### PROMPT 3.5: Training Callbacks
```
Create custom training callbacks.

File: largeforge/training/callbacks.py

Requirements:
1. ProgressCallback(TrainerCallback):
   - Show progress bar during training
   - Display current loss, learning rate
   - ETA calculation

2. SaveBestModelCallback(TrainerCallback):
   - Track best eval loss
   - Save only when improvement
   - Configurable patience

3. EarlyStoppingCallback(TrainerCallback):
   - Stop if no improvement for N evals
   - Configurable metric and patience

4. LoggingCallback(TrainerCallback):
   - Log to file in JSON format
   - Include timestamp, step, metrics

5. WandBCallback(TrainerCallback):
   - Log to Weights & Biases if available
   - Handle import errors gracefully

All callbacks should inherit from transformers.TrainerCallback.
Implement on_log, on_evaluate, on_save as needed.
```

### PROMPT 3.6: Training Metrics
```
Create training metrics computation.

File: largeforge/training/metrics.py

Requirements:
1. compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
   - perplexity: exp(loss)
   - accuracy: token-level accuracy

2. compute_dpo_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict:
   - chosen_rewards: mean reward for chosen
   - rejected_rewards: mean reward for rejected
   - reward_margin: chosen - rejected

3. MetricsTracker class:
   - __init__(self)
   - update(self, metrics: Dict[str, float])
   - get_average(self) -> Dict[str, float]
   - get_best(self, metric: str, mode: str = "min") -> float
   - reset(self)

4. format_metrics(metrics: Dict[str, float]) -> str:
   - Pretty print metrics for logging
   - Round to 4 decimal places
```

### PROMPT 3.7: SFT Trainer
```
Create Supervised Fine-Tuning trainer.

File: largeforge/training/sft.py

Requirements:
1. SFTTrainer(BaseTrainer):
   - __init__(self, model_name: str, config: SFTConfig,
              lora_config: Optional[LoRAConfig] = None)
   - train(self, train_dataset: Dataset, eval_dataset: Dataset = None) -> TrainingResult
   - _create_trainer(self) -> TRL.SFTTrainer
   - _get_training_args(self) -> TrainingArguments

2. Use TRL's SFTTrainer internally

3. _get_training_args should map SFTConfig to TrainingArguments

4. Support:
   - LoRA training
   - Full fine-tuning (if no lora_config)
   - Dataset packing
   - Gradient checkpointing

5. Handle chat templates:
   - Use tokenizer's chat template if available
   - Apply template in formatting_func

Example usage:
```python
trainer = SFTTrainer("meta-llama/Llama-2-7b-hf", sft_config, lora_config)
result = trainer.train(dataset)
trainer.save("./output")
```
```

### PROMPT 3.8: SFT Data Collator
```
Create data collator for SFT.

File: largeforge/training/sft.py (partial - collator)

Requirements:
1. SFTDataCollator:
   - __init__(self, tokenizer: PreTrainedTokenizer,
              max_length: int = 2048,
              pad_to_multiple_of: int = 8)
   - __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]

2. Collator should:
   - Tokenize if not already tokenized
   - Pad sequences to max length in batch
   - Create attention_mask
   - Create labels (same as input_ids, with padding = -100)

3. Handle both:
   - Pre-tokenized data (input_ids already present)
   - Raw text data (needs tokenization)

4. Left-pad for causal LM training
```

### PROMPT 3.9: DPO Trainer
```
Create Direct Preference Optimization trainer.

File: largeforge/training/dpo.py

Requirements:
1. DPOTrainer(BaseTrainer):
   - __init__(self, model_name: str, config: DPOConfig,
              lora_config: Optional[LoRAConfig] = None,
              ref_model: Optional[str] = None)
   - train(self, train_dataset: Dataset, eval_dataset: Dataset = None) -> TrainingResult
   - _create_trainer(self) -> TRL.DPOTrainer
   - _load_reference_model(self) -> PreTrainedModel

2. Use TRL's DPOTrainer internally

3. Reference model handling:
   - If ref_model provided, load separately
   - Otherwise, TRL creates from main model

4. Dataset must have columns: prompt, chosen, rejected

5. Support:
   - LoRA training
   - Different loss types (sigmoid, hinge, ipo)
   - Beta parameter configuration

Example:
```python
trainer = DPOTrainer("./sft-model", dpo_config, lora_config)
result = trainer.train(preference_dataset)
```
```

### PROMPT 3.10: DPO Data Collator
```
Create data collator for DPO.

File: largeforge/training/dpo.py (partial - collator)

Requirements:
1. DPODataCollator:
   - __init__(self, tokenizer: PreTrainedTokenizer,
              max_length: int = 1024,
              max_prompt_length: int = 512)
   - __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]

2. Must prepare for DPO:
   - prompt_input_ids
   - chosen_input_ids
   - rejected_input_ids
   - Corresponding attention masks

3. Truncation strategy:
   - Prompt: truncate from left if needed
   - Responses: truncate from right

4. TRL compatibility:
   - Return format expected by TRL DPOTrainer
```

### PROMPT 3.11: Checkpoint Manager
```
Create checkpoint management utilities.

File: largeforge/training/base.py (partial - CheckpointManager)

Requirements:
1. CheckpointManager class:
   - __init__(self, output_dir: str, save_total_limit: int = 3)
   - save_checkpoint(self, model, tokenizer, step: int, metrics: Dict)
   - load_checkpoint(self, checkpoint_path: str) -> Tuple[model, tokenizer, Dict]
   - get_latest_checkpoint(self) -> Optional[str]
   - list_checkpoints(self) -> List[str]
   - cleanup_old_checkpoints(self)

2. Checkpoint naming: checkpoint-{step}

3. Save:
   - Model weights (safetensors preferred)
   - Tokenizer
   - trainer_state.json with metrics
   - training_args.bin

4. Cleanup should keep only save_total_limit checkpoints

5. Support resuming from checkpoint
```

### PROMPT 3.12: Training Factory
```
Create training factory for unified interface.

File: largeforge/training/__init__.py

Requirements:
1. TrainerFactory class:
   - create(method: str, model_name: str, config: dict) -> BaseTrainer

2. Supported methods: "sft", "dpo"

3. Config can be dict or appropriate Config class

4. Exports:
   - SFTTrainer
   - DPOTrainer
   - LoRAConfig
   - SFTConfig
   - DPOConfig
   - TrainerFactory

5. get_trainer(method: str) -> Type[BaseTrainer]:
   - Return trainer class for method

Usage:
```python
from largeforge.training import SFTTrainer, TrainerFactory

# Direct use
trainer = SFTTrainer(model, config)

# Factory use
trainer = TrainerFactory.create("sft", model, config_dict)
```
```

---

## Phase 4: Inference (Prompts 4.1-4.10)

### PROMPT 4.1: Base Engine
```
Create abstract inference engine.

File: largeforge/inference/base.py

Requirements:
1. InferenceEngine(ABC):
   - __init__(self, model_path: str, config: InferenceConfig)
   - load(self): Load model into memory (abstract)
   - generate(self, prompt: str, config: GenerationConfig) -> str (abstract)
   - generate_stream(self, prompt: str, config: GenerationConfig) -> Iterator[str] (abstract)
   - unload(self): Unload model from memory
   - @property is_loaded: bool
   - @property model_info: Dict

2. model_info should return:
   - model_path
   - backend
   - device
   - memory_used
   - max_length

3. Include context manager support:
   - __enter__ calls load()
   - __exit__ calls unload()
```

### PROMPT 4.2: Transformers Backend
```
Create HuggingFace Transformers inference backend.

File: largeforge/inference/transformers_backend.py

Requirements:
1. TransformersEngine(InferenceEngine):
   - load(self): Load model and tokenizer
   - generate(self, prompt: str, config: GenerationConfig) -> str
   - generate_stream(self, prompt: str, config: GenerationConfig) -> Iterator[str]
   - _prepare_inputs(self, prompt: str) -> Dict
   - _decode_output(self, output_ids: torch.Tensor) -> str

2. load should:
   - Load model with device_map="auto"
   - Apply quantization if configured
   - Load tokenizer

3. generate_stream should:
   - Use TextIteratorStreamer
   - Yield tokens as generated

4. Handle:
   - Chat templates
   - Stop sequences
   - Max new tokens
```

### PROMPT 4.3: vLLM Backend
```
Create vLLM inference backend.

File: largeforge/inference/vllm_backend.py

Requirements:
1. VLLMEngine(InferenceEngine):
   - load(self): Initialize vLLM LLM object
   - generate(self, prompt: str, config: GenerationConfig) -> str
   - generate_stream(self, prompt: str, config: GenerationConfig) -> Iterator[str]
   - _get_sampling_params(self, config: GenerationConfig) -> SamplingParams

2. load should:
   - Create vLLM LLM with configured parameters
   - Set gpu_memory_utilization
   - Set tensor_parallel_size

3. Use vLLM's SamplingParams for generation config

4. generate_stream should use vLLM's streaming API

5. Handle vLLM import errors gracefully:
```python
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
```
```

### PROMPT 4.4: Backend Factory
```
Create backend selection factory.

File: largeforge/inference/__init__.py

Requirements:
1. get_backend(config: InferenceConfig) -> InferenceEngine:
   - If backend == "auto": prefer vLLM if available and CUDA present
   - If backend == "vllm": use vLLM (error if unavailable)
   - If backend == "transformers": use Transformers

2. is_vllm_available() -> bool:
   - Check if vLLM is installed
   - Check if CUDA is available

3. Exports:
   - InferenceEngine
   - TransformersEngine
   - VLLMEngine (if available)
   - get_backend
   - is_vllm_available

4. Log which backend is selected
```

### PROMPT 4.5: Streaming Handler
```
Create SSE streaming utilities.

File: largeforge/inference/streaming.py

Requirements:
1. StreamingResponse class (or use Starlette's):
   - Format tokens as SSE events
   - data: {"choices": [{"delta": {"content": "token"}}]}

2. create_sse_response(generator: Iterator[str]) -> StreamingResponse:
   - Wrap generator in SSE format
   - Set appropriate headers

3. format_sse_event(data: dict) -> str:
   - Format as "data: {json}\n\n"

4. StreamingHandler:
   - __init__(self, engine: InferenceEngine)
   - stream(self, prompt: str, config: GenerationConfig) -> StreamingResponse

5. Handle connection close gracefully
```

### PROMPT 4.6: API Models
```
Create Pydantic models for API.

File: largeforge/inference/server.py (partial - models)

Requirements:
1. CompletionRequest(BaseModel):
   - model: str
   - prompt: str
   - max_tokens: int = 256
   - temperature: float = 0.7
   - top_p: float = 0.9
   - top_k: int = 50
   - stop: Optional[List[str]] = None
   - stream: bool = False

2. CompletionChoice(BaseModel):
   - index: int
   - text: str
   - finish_reason: str

3. CompletionResponse(BaseModel):
   - id: str
   - object: str = "text_completion"
   - created: int
   - model: str
   - choices: List[CompletionChoice]
   - usage: UsageInfo

4. UsageInfo(BaseModel):
   - prompt_tokens: int
   - completion_tokens: int
   - total_tokens: int

5. ChatMessage(BaseModel):
   - role: str
   - content: str

6. ChatCompletionRequest(BaseModel):
   - model: str
   - messages: List[ChatMessage]
   - (same generation params as CompletionRequest)

7. ChatCompletionResponse - similar to OpenAI format

8. HealthResponse(BaseModel):
   - status: str
   - model: str
   - uptime: float
```

### PROMPT 4.7: Completions Endpoint
```
Create /v1/completions endpoint.

File: largeforge/inference/server.py (partial - completions)

Requirements:
1. POST /v1/completions endpoint:
   - Accept CompletionRequest
   - Return CompletionResponse or StreamingResponse

2. Implementation:
```python
@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    if request.stream:
        return create_streaming_response(request)
    else:
        return create_completion_response(request)
```

3. create_completion_response:
   - Generate using engine
   - Count tokens
   - Build response

4. create_streaming_response:
   - Use engine.generate_stream
   - Format as SSE
   - Include [DONE] message at end

5. Handle errors with proper HTTP status codes
```

### PROMPT 4.8: Chat Endpoint
```
Create /v1/chat/completions endpoint.

File: largeforge/inference/server.py (partial - chat)

Requirements:
1. POST /v1/chat/completions endpoint:
   - Accept ChatCompletionRequest
   - Return ChatCompletionResponse or StreamingResponse

2. Convert messages to prompt:
   - Use tokenizer.apply_chat_template if available
   - Fallback to simple concatenation

3. Response format matches OpenAI:
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop"
  }],
  "usage": {...}
}
```

4. Streaming format:
```json
{"choices": [{"delta": {"content": "token"}}]}
```

5. Handle system messages properly
```

### PROMPT 4.9: Health Endpoint
```
Create health check endpoint.

File: largeforge/inference/server.py (partial - health)

Requirements:
1. GET /health endpoint:
   - Return HealthResponse
   - Check model is loaded
   - Include uptime

2. GET /v1/models endpoint:
   - List available models
   - Include model metadata

3. GET /v1/models/{model_id} endpoint:
   - Get specific model info
   - Return 404 if not found

4. GET /metrics endpoint:
   - Prometheus format metrics
   - requests_total
   - latency_histogram
   - tokens_generated

5. Health status:
   - "healthy": model loaded, responding
   - "degraded": model loaded, slow responses
   - "unhealthy": model not loaded
```

### PROMPT 4.10: Server Factory
```
Create FastAPI application factory.

File: largeforge/inference/server.py (partial - factory)

Requirements:
1. create_app(config: InferenceConfig) -> FastAPI:
   - Create FastAPI app
   - Add all endpoints
   - Configure CORS
   - Add middleware

2. InferenceServer class:
   - __init__(self, config: InferenceConfig)
   - start(self): Start uvicorn server
   - stop(self): Graceful shutdown
   - @property app: FastAPI instance

3. Middleware:
   - Request logging
   - Error handling
   - Request ID generation

4. Startup/shutdown events:
   - Load model on startup
   - Cleanup on shutdown

5. CLI integration:
```python
def run_server(model_path: str, port: int = 8000):
    config = InferenceConfig(model_path=model_path, port=port)
    server = InferenceServer(config)
    server.start()
```
```

---

## Phase 5: Router (Prompts 5.1-5.8)

### PROMPT 5.1: Keyword Classifier
```
Create keyword-based query classifier.

File: largeforge/router/classifier.py (partial - keyword)

Requirements:
1. KeywordClassifier:
   - __init__(self, expert_keywords: Dict[str, List[str]])
   - classify(self, query: str) -> ClassificationResult
   - add_expert(self, name: str, keywords: List[str])
   - remove_expert(self, name: str)

2. ClassificationResult dataclass:
   - expert: str
   - confidence: float
   - scores: Dict[str, float]  # all expert scores

3. Classification logic:
   - Tokenize query (simple word split)
   - Count keyword matches per expert
   - Normalize scores
   - Return highest scoring expert

4. Confidence calculation:
   - Based on ratio of matched vs total keywords
   - And separation from second-best

5. Case-insensitive matching
6. Support for keyword phrases
```

### PROMPT 5.2: Neural Classifier
```
Create neural network query classifier.

File: largeforge/router/classifier.py (partial - neural)

Requirements:
1. NeuralClassifier:
   - __init__(self, model_path: str = None, experts: List[str] = None)
   - classify(self, query: str) -> ClassificationResult
   - train(self, data: List[Tuple[str, str]], epochs: int = 3)
   - save(self, path: str)
   - load(self, path: str)

2. Use small transformer model for classification:
   - Default: sentence-transformers/all-MiniLM-L6-v2
   - Fine-tune classification head

3. Training data format: List of (query, expert_name)

4. Inference:
   - Embed query
   - Pass through classification head
   - Softmax for confidence scores

5. Handle missing model gracefully (return low confidence)
```

### PROMPT 5.3: Hybrid Classifier
```
Create hybrid classifier combining keyword and neural.

File: largeforge/router/classifier.py (partial - hybrid)

Requirements:
1. HybridClassifier:
   - __init__(self, keyword_config: Dict, neural_model: str = None,
              keyword_weight: float = 0.3, neural_weight: float = 0.7)
   - classify(self, query: str) -> ClassificationResult
   - @property keyword_classifier: KeywordClassifier
   - @property neural_classifier: NeuralClassifier

2. Combination strategy:
   - If strong keyword match (>0.8): use keyword only
   - If neural very confident (>0.9): use neural only
   - Otherwise: weighted combination

3. Handle disagreement:
   - If different experts, lower overall confidence
   - Log disagreements for analysis

4. ClassificationResult should include:
   - Which method was used
   - Individual method scores
```

### PROMPT 5.4: Expert Config
```
Create expert configuration models.

File: largeforge/router/experts.py (partial - config)

Requirements:
1. ExpertConfig(BaseConfig):
   - name: str
   - model_path: str
   - description: str = ""
   - domains: List[str] = []
   - keywords: List[str] = []
   - endpoint: Optional[str] = None
   - priority: int = 0
   - max_tokens: int = 2048
   - temperature: float = 0.7
   - enabled: bool = True

2. ExpertStatus enum:
   - READY, LOADING, ERROR, DISABLED

3. ExpertHealth dataclass:
   - status: ExpertStatus
   - last_check: datetime
   - latency_ms: float
   - error_count: int
   - success_rate: float

4. Validation:
   - name must be alphanumeric with hyphens
   - Either model_path or endpoint required
```

### PROMPT 5.5: Expert Manager
```
Create expert lifecycle management.

File: largeforge/router/experts.py (partial - manager)

Requirements:
1. ExpertManager:
   - __init__(self)
   - register(self, config: ExpertConfig)
   - unregister(self, name: str)
   - get(self, name: str) -> ExpertConfig
   - list_experts(self) -> List[ExpertConfig]
   - get_healthy_experts(self) -> List[str]
   - check_health(self, name: str) -> ExpertHealth
   - update_health(self, name: str, health: ExpertHealth)

2. Health checking:
   - Ping endpoint if external
   - Check model loaded if local
   - Track success/failure rates

3. Persistence:
   - save_config(self, path: str)
   - load_config(self, path: str)

4. Thread-safe expert dictionary
```

### PROMPT 5.6: Load Balancer
```
Create load balancing for expert requests.

File: largeforge/router/load_balancer.py

Requirements:
1. LoadBalancer:
   - __init__(self, strategy: str = "round_robin")
   - select_instance(self, expert: str, instances: List[str]) -> str
   - report_success(self, instance: str, latency: float)
   - report_failure(self, instance: str)
   - get_stats(self) -> Dict

2. Strategies:
   - round_robin: Simple rotation
   - least_connections: Track active requests
   - weighted: Based on instance capacity
   - latency_aware: Prefer faster instances

3. Track per-instance metrics:
   - Active connections
   - Average latency
   - Error rate

4. Handle empty instance list gracefully
```

### PROMPT 5.7: Circuit Breaker
```
Create circuit breaker for fault tolerance.

File: largeforge/router/load_balancer.py (partial - circuit breaker)

Requirements:
1. CircuitBreaker:
   - __init__(self, failure_threshold: int = 5,
              reset_timeout: int = 60)
   - can_execute(self, name: str) -> bool
   - record_success(self, name: str)
   - record_failure(self, name: str)
   - get_state(self, name: str) -> str
   - reset(self, name: str)

2. States:
   - CLOSED: Normal operation
   - OPEN: Failing, reject requests
   - HALF_OPEN: Testing if recovered

3. State transitions:
   - CLOSED -> OPEN: After failure_threshold failures
   - OPEN -> HALF_OPEN: After reset_timeout
   - HALF_OPEN -> CLOSED: On success
   - HALF_OPEN -> OPEN: On failure

4. Log state transitions
```

### PROMPT 5.8: Router Service
```
Create main router service.

File: largeforge/router/router.py

Requirements:
1. Router:
   - __init__(self, config: RouterConfig)
   - route(self, query: str) -> RoutingDecision
   - generate(self, query: str) -> str: Route and generate
   - add_expert(self, config: ExpertConfig)
   - remove_expert(self, name: str)
   - get_expert_status(self) -> Dict[str, ExpertHealth]

2. RoutingDecision dataclass:
   - expert: str
   - confidence: float
   - alternatives: List[Tuple[str, float]]
   - routing_time_ms: float

3. route() should:
   - Use classifier to determine expert
   - Check expert health
   - Fall back if expert unavailable
   - Track routing metrics

4. generate() should:
   - Route query
   - Forward to selected expert
   - Handle failures with retry/fallback

5. FastAPI endpoints:
   - POST /route
   - POST /generate
   - GET /experts
   - POST /experts
   - DELETE /experts/{name}
```

---

## Phase 6: Quantization (Prompts 6.1-6.4)

### PROMPT 6.1: AWQ Quantizer
```
Create AWQ quantization implementation.

File: largeforge/quantization/awq.py

Requirements:
1. AWQQuantizer:
   - __init__(self, model_path: str, bits: int = 4)
   - quantize(self, calibration_data: List[str], output_path: str)
   - load_quantized(self, path: str) -> AutoAWQForCausalLM

2. quantize() should:
   - Load model in fp16
   - Run calibration with provided data
   - Quantize weights to specified bits
   - Save quantized model

3. Configuration:
   - bits: 4 (default)
   - group_size: 128
   - version: "GEMM"

4. Handle autoawq import:
```python
try:
    from awq import AutoAWQForCausalLM
    AWQ_AVAILABLE = True
except ImportError:
    AWQ_AVAILABLE = False
```

5. Log quantization progress and size reduction
```

### PROMPT 6.2: GPTQ Quantizer
```
Create GPTQ quantization implementation.

File: largeforge/quantization/gptq.py

Requirements:
1. GPTQQuantizer:
   - __init__(self, model_path: str, bits: int = 4)
   - quantize(self, calibration_data: List[str], output_path: str)
   - load_quantized(self, path: str) -> AutoGPTQForCausalLM

2. quantize() should:
   - Load model
   - Prepare calibration dataset
   - Run GPTQ quantization
   - Save with safetensors

3. Configuration:
   - bits: 4
   - group_size: 128
   - damp_percent: 0.1
   - desc_act: False

4. Handle auto-gptq import errors

5. Log layer-by-layer progress
```

### PROMPT 6.3: Calibration Data
```
Create calibration data utilities.

File: largeforge/quantization/awq.py (partial - calibration)

Requirements:
1. get_calibration_data(dataset_name: str = "wikitext",
                        num_samples: int = 128,
                        seq_length: int = 512) -> List[str]:
   - Load dataset from HuggingFace
   - Sample and truncate
   - Return list of text strings

2. prepare_calibration_dataset(texts: List[str],
                               tokenizer) -> Dataset:
   - Tokenize texts
   - Return Dataset for quantization

3. Supported datasets:
   - wikitext-2-raw-v1
   - c4 (subset)
   - Custom file path

4. Cache downloaded datasets
```

### PROMPT 6.4: Quantization Factory
```
Create unified quantization interface.

File: largeforge/quantization/__init__.py

Requirements:
1. quantize(model_path: str, output_path: str,
            method: str = "awq", bits: int = 4,
            calibration_data: List[str] = None) -> str:
   - Select quantizer based on method
   - Generate calibration data if not provided
   - Run quantization
   - Return output path

2. load_quantized(path: str) -> AutoModel:
   - Auto-detect quantization method
   - Load appropriately

3. get_quantization_config(path: str) -> Dict:
   - Read quant_config.json
   - Return method and parameters

4. Exports:
   - AWQQuantizer
   - GPTQQuantizer
   - quantize
   - load_quantized
   - AWQ_AVAILABLE
   - GPTQ_AVAILABLE
```

---

## Phase 7: CLI (Prompts 7.1-7.6)

### PROMPT 7.1: CLI Utils
```
Create CLI utility functions.

File: largeforge/cli/utils.py

Requirements:
1. Use Click for CLI framework

2. Utilities:
   - print_header(text: str): Styled header
   - print_success(text: str): Green success message
   - print_error(text: str): Red error message
   - print_warning(text: str): Yellow warning
   - print_info(text: str): Blue info
   - create_progress_bar(total: int): Click progress bar

3. Config loading:
   - load_config_file(path: str) -> Dict
   - merge_cli_config(cli_args: Dict, file_config: Dict) -> Dict

4. Validation:
   - validate_model_path(path: str) -> bool
   - validate_output_dir(path: str) -> bool

5. Rich formatting if available, fallback to plain text
```

### PROMPT 7.2: Train Commands
```
Create training CLI commands.

File: largeforge/cli/train.py

Requirements:
1. Create Click group: @click.group()

2. Commands:
   - sft: Supervised fine-tuning
   - dpo: Direct preference optimization

3. sft command options:
   --model: Base model path/name
   --dataset: Training data path
   --output: Output directory
   --num-epochs: Number of epochs
   --batch-size: Per-device batch size
   --learning-rate: Learning rate
   --lora-r: LoRA rank (0 for full fine-tune)
   --lora-alpha: LoRA alpha
   --max-length: Max sequence length
   --config: Path to config file (optional)

4. dpo command options:
   --model: SFT model path
   --dataset: Preference data path
   --output: Output directory
   --beta: DPO beta parameter
   --learning-rate: Learning rate
   --config: Path to config file (optional)

5. Show progress during training
6. Print summary on completion
```

### PROMPT 7.3: Serve Commands
```
Create serving CLI commands.

File: largeforge/cli/serve.py

Requirements:
1. Create Click group for serving

2. Commands:
   - inference: Start inference server
   - router: Start router service

3. inference command options:
   --model: Model path
   --port: Server port (default 8000)
   --host: Host address (default 0.0.0.0)
   --backend: Backend (auto/vllm/transformers)
   --max-model-len: Max context length
   --gpu-memory: GPU memory utilization
   --tensor-parallel: Tensor parallel size

4. router command options:
   --config: Router config file
   --port: Server port (default 8080)
   --host: Host address

5. Show startup message with URL
6. Handle Ctrl+C gracefully
```

### PROMPT 7.4: Quantize Commands
```
Create quantization CLI commands.

File: largeforge/cli/main.py (partial - quantize)

Requirements:
1. Add to main CLI group:

2. Commands:
   - quantize awq: AWQ quantization
   - quantize gptq: GPTQ quantization

3. awq command options:
   --model: Model to quantize
   --output: Output path
   --bits: Bit width (default 4)
   --calibration: Custom calibration data
   --samples: Number of calibration samples

4. gptq command options:
   Similar to awq

5. Show progress during quantization
6. Print size reduction statistics
```

### PROMPT 7.5: Doctor Command
```
Create system diagnostic command.

File: largeforge/cli/main.py (partial - doctor)

Requirements:
1. doctor command:
   - Check Python version
   - Check PyTorch installation
   - Check CUDA availability
   - Check GPU memory
   - Check required packages
   - Check optional packages (vllm, autoawq, etc.)
   - Test basic operations

2. Output format:
```
LargeForgeAI System Check
========================
✓ Python 3.10.12
✓ PyTorch 2.1.0+cu121
✓ CUDA Available (Driver 535.104.05)
✓ GPU: NVIDIA A100 (40GB)
✓ transformers 4.36.0
✓ peft 0.7.1
✓ trl 0.7.4
! vllm not installed (optional)
✓ Basic inference test passed

System ready for LargeForgeAI
```

3. Use colored output
4. Exit with error code if critical issues
```

### PROMPT 7.6: Main CLI
```
Create main CLI entry point.

File: largeforge/cli/main.py

Requirements:
1. Create main Click group

2. Include subcommands:
   - train (from train.py)
   - serve (from serve.py)
   - quantize
   - doctor
   - version

3. version command:
   - Print LargeForgeAI version
   - Print dependency versions

4. Global options:
   --verbose / -v: Verbose output
   --quiet / -q: Minimal output
   --config: Global config file

5. Entry point setup:
```python
def main():
    cli()

if __name__ == "__main__":
    main()
```

6. Register in pyproject.toml:
```toml
[project.scripts]
largeforge = "largeforge.cli.main:main"
```
```

---

## Phase 8: Tests (Prompts 8.2-8.10)

### PROMPT 8.4: Training Tests
```
Create unit tests for training modules.

File: tests/unit/test_training.py

Requirements:
1. Test LoRA configuration:
   - Default target modules
   - get_recommended_config

2. Test LoRA setup:
   - get_peft_config conversion
   - (Skip actual model tests without GPU)

3. Test training callbacks:
   - ProgressCallback initialization
   - SaveBestModelCallback logic

4. Test metrics:
   - compute_metrics with mock data
   - MetricsTracker averaging

5. Test SFTConfig/DPOConfig:
   - Field validation
   - Config merging

Use mocks for model-dependent tests.
Mark GPU tests with @pytest.mark.gpu.
```

### PROMPT 8.5: Inference Tests
```
Create unit tests for inference modules.

File: tests/unit/test_inference.py

Requirements:
1. Test API models:
   - CompletionRequest validation
   - CompletionResponse serialization
   - ChatMessage role validation

2. Test streaming:
   - SSE format_sse_event
   - StreamingHandler initialization

3. Test backend selection:
   - is_vllm_available detection
   - get_backend logic

4. Test server factory:
   - App creation
   - Endpoint registration

Use TestClient from FastAPI for endpoint tests.
Mock the actual model loading.
```

### PROMPT 8.6: Router Tests
```
Create unit tests for router modules.

File: tests/unit/test_router.py

Requirements:
1. Test KeywordClassifier:
   - Classification with matching keywords
   - No match returns low confidence
   - Multiple expert handling

2. Test HybridClassifier:
   - Weight combination
   - Fallback behavior

3. Test ExpertConfig:
   - Validation rules
   - Default values

4. Test ExpertManager:
   - Register/unregister
   - List experts
   - Health tracking

5. Test LoadBalancer:
   - Round robin selection
   - Stats tracking

6. Test CircuitBreaker:
   - State transitions
   - Timeout behavior
```

### PROMPT 8.8: E2E Training Test
```
Create end-to-end training test.

File: tests/integration/test_e2e_training.py

Requirements:
1. Test full SFT training flow (with tiny model):
   - Load gpt2 (smallest model)
   - Create minimal dataset
   - Train for 1 step
   - Save model
   - Verify output files

2. Test full DPO training flow:
   - Start from saved SFT model
   - Create minimal preference data
   - Train for 1 step

3. Skip if no GPU available:
```python
@pytest.mark.gpu
@pytest.mark.slow
def test_sft_training():
    ...
```

4. Use tmp_path for all outputs
5. Clean up GPU memory after tests
```

### PROMPT 8.9: E2E Inference Test
```
Create end-to-end inference test.

File: tests/integration/test_e2e_inference.py

Requirements:
1. Test inference server startup:
   - Start server with tiny model
   - Wait for ready
   - Check /health endpoint

2. Test completion endpoint:
   - POST /v1/completions
   - Verify response format

3. Test chat endpoint:
   - POST /v1/chat/completions
   - Verify message format

4. Test streaming:
   - Enable stream=true
   - Verify SSE format

5. Use httpx or TestClient
6. Clean up server after tests
```

### PROMPT 8.10: E2E Router Test
```
Create end-to-end router test.

File: tests/integration/test_e2e_router.py

Requirements:
1. Test router with mock experts:
   - Configure multiple experts
   - Route different queries
   - Verify correct selection

2. Test fallback behavior:
   - Disable primary expert
   - Verify fallback triggered

3. Test circuit breaker:
   - Simulate failures
   - Verify circuit opens

4. Test load balancing:
   - Multiple instances
   - Verify distribution

5. Use mock HTTP servers for experts
```

---

## Execution Instructions

After creating all modules, run:

```bash
# Install in development mode
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run specific phase tests
pytest tests/unit/test_config.py -v
pytest tests/unit/test_data.py -v
pytest tests/unit/test_training.py -v

# Run with coverage
pytest tests/ --cov=largeforge --cov-report=html
```

---

*Last Updated: December 2024*
