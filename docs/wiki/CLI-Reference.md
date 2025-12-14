# CLI Reference

Complete reference for all LargeForgeAI command-line commands.

## Global Options

```bash
largeforge [OPTIONS] COMMAND [ARGS]

Options:
  --version      Show version and exit
  -v, --verbose  Enable verbose output
  --help         Show help message
```

---

## Training Commands

### `largeforge train sft`

Supervised Fine-Tuning with LoRA.

```bash
largeforge train sft [OPTIONS]

Required:
  --model TEXT       Base model name or path
  --dataset PATH     Training dataset (JSONL)
  --output PATH      Output directory

Training Options:
  --epochs INT           Number of epochs (default: 3)
  --batch-size INT       Batch size per device (default: 4)
  --learning-rate FLOAT  Learning rate (default: 2e-5)
  --max-length INT       Max sequence length (default: 2048)
  --warmup-ratio FLOAT   Warmup ratio (default: 0.1)

LoRA Options:
  --lora-r INT           LoRA rank (default: 16)
  --lora-alpha INT       LoRA alpha (default: 32)
  --lora-dropout FLOAT   LoRA dropout (default: 0.05)
  --no-lora              Disable LoRA (full fine-tuning)

Quantization:
  --load-4bit            Load model in 4-bit
  --load-8bit            Load model in 8-bit

Example:
  largeforge train sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset data.jsonl \
    --output ./output \
    --epochs 3 \
    --load-4bit
```

### `largeforge train dpo`

Direct Preference Optimization.

```bash
largeforge train dpo [OPTIONS]

Required:
  --model TEXT       Base model (usually SFT output)
  --dataset PATH     Preference dataset
  --output PATH      Output directory

DPO Options:
  --beta FLOAT           KL penalty coefficient (default: 0.1)
  --loss-type TEXT       Loss type: sigmoid, hinge (default: sigmoid)
  --epochs INT           Number of epochs (default: 1)
  --learning-rate FLOAT  Learning rate (default: 5e-6)

Example:
  largeforge train dpo \
    --model ./sft-model \
    --dataset preferences.jsonl \
    --output ./dpo-model \
    --beta 0.1
```

---

## Data Commands

### `largeforge data validate`

Validate dataset format and quality.

```bash
largeforge data validate PATH [OPTIONS]

Arguments:
  PATH                 Dataset file path

Options:
  --format TEXT        Format: auto, alpaca, sharegpt, dpo (default: auto)
  --strict             Fail on any validation error
  --max-errors INT     Maximum errors to display (default: 10)
  --output PATH        Write valid records to file

Example:
  largeforge data validate dataset.jsonl --format alpaca --strict
```

### `largeforge data convert`

Convert between data formats.

```bash
largeforge data convert INPUT OUTPUT [OPTIONS]

Arguments:
  INPUT                Input file path
  OUTPUT               Output file path

Options:
  --from TEXT          Input format (required)
  --to TEXT            Output format (required)
  --system-prompt TEXT Add system prompt

Example:
  largeforge data convert data.json out.jsonl --from alpaca --to sharegpt
```

### `largeforge data stats`

Show dataset statistics.

```bash
largeforge data stats PATH [OPTIONS]

Options:
  --format TEXT        Data format
  --detailed           Show detailed statistics

Example:
  largeforge data stats dataset.jsonl --detailed
```

---

## Synthetic Data Commands

### `largeforge synthetic generate`

Generate synthetic training data.

```bash
largeforge synthetic generate [OPTIONS]

Required:
  -o, --output PATH    Output file path

Options:
  -n, --num-samples INT   Number of samples (default: 100)
  -f, --format TEXT       Format: sft, dpo (default: sft)
  -p, --provider TEXT     Provider: openai, anthropic (default: openai)
  -m, --model TEXT        Model to use
  -t, --topic TEXT        Topics (can specify multiple)
  --temperature FLOAT     Generation temperature (default: 0.8)
  --api-key TEXT          API key (or use env var)

Example:
  largeforge synthetic generate \
    -o data.jsonl \
    -n 500 \
    --provider openai \
    --model gpt-4 \
    -t "customer support" \
    -t "technical help"
```

### `largeforge synthetic augment`

Augment existing dataset.

```bash
largeforge synthetic augment INPUT [OPTIONS]

Arguments:
  INPUT                Input dataset path

Required:
  -o, --output PATH    Output file path

Options:
  --factor INT         Augmentation factor (default: 2)
  --provider TEXT      API provider
  --model TEXT         Model to use

Example:
  largeforge synthetic augment data.jsonl -o augmented.jsonl --factor 3
```

---

## Verification Commands

### `largeforge verify run`

Run model verification.

```bash
largeforge verify run PATH [OPTIONS]

Arguments:
  PATH                 Model path

Options:
  --level TEXT         Level: quick, standard, thorough (default: standard)
  --output PATH        Save report to file
  --format TEXT        Report format: text, json, html, markdown

Example:
  largeforge verify run ./model --level thorough --output report.html
```

### `largeforge verify smoke-test`

Quick smoke test.

```bash
largeforge verify smoke-test PATH

Example:
  largeforge verify smoke-test ./model
```

---

## Deployment Commands

### `largeforge deploy generate`

Generate deployment files.

```bash
largeforge deploy generate PATH [OPTIONS]

Arguments:
  PATH                 Model path

Options:
  --output PATH        Output directory (default: ./deployment)
  --backend TEXT       Backend: auto, vllm, transformers
  --port INT           Service port (default: 8000)
  --gpu / --no-gpu     Enable GPU support

Example:
  largeforge deploy generate ./model --output ./deploy --backend vllm
```

### `largeforge deploy build`

Build Docker image.

```bash
largeforge deploy build [OPTIONS]

Options:
  --context PATH       Build context (default: .)
  --tag TEXT           Image tag
  --push               Push to registry

Example:
  largeforge deploy build --tag mymodel:v1 --push
```

---

## Server Commands

### `largeforge serve start`

Start inference server.

```bash
largeforge serve start [OPTIONS]

Options:
  --model PATH         Model path
  --port INT           Port (default: 8000)
  --host TEXT          Host (default: 0.0.0.0)
  --backend TEXT       Backend: auto, vllm, transformers

Example:
  largeforge serve start --model ./model --port 8000
```

### `largeforge web start`

Start web UI server.

```bash
largeforge web start [OPTIONS]

Options:
  --host TEXT          Host (default: 0.0.0.0)
  --port INT           Port (default: 7860)
  --config PATH        Config file
  --debug              Enable debug mode

Example:
  largeforge web start --port 7860
```

---

## Export Commands

### `largeforge export model`

Export model for deployment.

```bash
largeforge export model PATH [OPTIONS]

Arguments:
  PATH                 Model path

Options:
  --output PATH        Output directory
  --format TEXT        Format: safetensors, pytorch
  --merge-lora         Merge LoRA weights into base model

Example:
  largeforge export model ./model --output ./exported --merge-lora
```

### `largeforge export merge-lora`

Merge LoRA adapters into base model.

```bash
largeforge export merge-lora BASE ADAPTER [OPTIONS]

Arguments:
  BASE                 Base model path
  ADAPTER              LoRA adapter path

Options:
  --output PATH        Output directory

Example:
  largeforge export merge-lora Qwen/Qwen2.5-7B ./lora-adapter --output ./merged
```

---

## Utility Commands

### `largeforge info`

Display system information.

```bash
largeforge info [OPTIONS]

Options:
  --config PATH        Show config file contents

Example:
  largeforge info
```

---

[[Back to Home|Home]] | [[API Reference|API-Reference]]
