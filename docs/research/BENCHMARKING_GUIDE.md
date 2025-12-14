# LargeForgeAI Benchmarking Guide

## Overview

This guide provides comprehensive instructions for benchmarking and evaluating models trained with LargeForgeAI.

---

## 1. Evaluation Framework

### 1.1 Evaluation Philosophy

LargeForgeAI follows these evaluation principles:

1. **Multi-dimensional Assessment**: No single metric captures model quality
2. **Task-Specific Evaluation**: Match evaluation to intended use case
3. **Reproducibility**: Document all evaluation parameters
4. **Comparison Fairness**: Use consistent settings across models

### 1.2 Evaluation Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    Evaluation Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────┐ │
│  │  Model   │───▶│ Benchmark│───▶│  Metrics │───▶│Report │ │
│  │  Loading │    │   Suite  │    │ Compute  │    │       │ │
│  └──────────┘    └──────────┘    └──────────┘    └───────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Standard Benchmarks

### 2.1 Language Understanding

#### MMLU (Massive Multitask Language Understanding)

**Description**: Tests knowledge across 57 subjects from STEM to humanities.

**Setup**:
```bash
# Install lm-evaluation-harness
pip install lm-eval

# Run MMLU evaluation
lm_eval --model hf \
  --model_args pretrained=./my-model \
  --tasks mmlu \
  --batch_size 8 \
  --output_path ./results/mmlu
```

**Expected Scores** (7B models):
| Model | Score |
|-------|-------|
| Llama 2 7B | 45.3% |
| Mistral 7B | 60.1% |
| Fine-tuned (target) | 50-65% |

#### HellaSwag

**Description**: Tests commonsense reasoning about physical situations.

```bash
lm_eval --model hf \
  --model_args pretrained=./my-model \
  --tasks hellaswag \
  --batch_size 16 \
  --output_path ./results/hellaswag
```

#### ARC (AI2 Reasoning Challenge)

**Description**: Science exam questions requiring reasoning.

```bash
lm_eval --model hf \
  --model_args pretrained=./my-model \
  --tasks arc_easy,arc_challenge \
  --batch_size 16 \
  --output_path ./results/arc
```

### 2.2 Reasoning Benchmarks

#### GSM8K (Grade School Math)

**Description**: Mathematical word problems requiring multi-step reasoning.

```bash
lm_eval --model hf \
  --model_args pretrained=./my-model \
  --tasks gsm8k \
  --batch_size 8 \
  --num_fewshot 8 \
  --output_path ./results/gsm8k
```

**Evaluation Tips**:
- Use 8-shot prompting for best results
- Enable chain-of-thought with appropriate prompting
- Parse final numeric answer for scoring

#### MATH

**Description**: Challenging mathematics problems from competitions.

```bash
lm_eval --model hf \
  --model_args pretrained=./my-model \
  --tasks math \
  --batch_size 4 \
  --output_path ./results/math
```

### 2.3 Code Generation

#### HumanEval

**Description**: Programming challenges testing code generation.

```python
from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness

# Generate completions
problems = read_problems()
samples = []

for task_id, problem in problems.items():
    completion = model.generate(problem["prompt"])
    samples.append({
        "task_id": task_id,
        "completion": completion
    })

write_jsonl("samples.jsonl", samples)

# Evaluate
results = evaluate_functional_correctness("samples.jsonl")
print(f"Pass@1: {results['pass@1']}")
```

**Metrics**:
- **Pass@1**: Probability that first completion passes all tests
- **Pass@10**: Probability at least one of 10 completions passes
- **Pass@100**: Probability at least one of 100 completions passes

#### MBPP (Mostly Basic Python Problems)

```bash
lm_eval --model hf \
  --model_args pretrained=./my-model \
  --tasks mbpp \
  --batch_size 8 \
  --output_path ./results/mbpp
```

### 2.4 Truthfulness

#### TruthfulQA

**Description**: Tests model's tendency to generate truthful responses.

```bash
lm_eval --model hf \
  --model_args pretrained=./my-model \
  --tasks truthfulqa_mc \
  --batch_size 16 \
  --output_path ./results/truthfulqa
```

**Modes**:
- Multiple choice (MC1, MC2)
- Generation with GPT-4 judge

---

## 3. Custom Evaluation

### 3.1 Domain-Specific Benchmarks

Create custom benchmarks for your specific use case:

```python
import json
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class BenchmarkItem:
    id: str
    input: str
    expected: str
    metadata: Dict = None

class CustomBenchmark:
    def __init__(self, name: str, items: List[BenchmarkItem]):
        self.name = name
        self.items = items

    def evaluate(self, model, metric_fn) -> Dict:
        results = []
        for item in self.items:
            output = model.generate(item.input)
            score = metric_fn(output, item.expected)
            results.append({
                "id": item.id,
                "score": score,
                "output": output,
                "expected": item.expected
            })

        return {
            "benchmark": self.name,
            "num_items": len(results),
            "average_score": sum(r["score"] for r in results) / len(results),
            "results": results
        }

# Example: Medical Q&A benchmark
medical_benchmark = CustomBenchmark(
    name="medical_qa",
    items=[
        BenchmarkItem(
            id="med_001",
            input="What are the symptoms of Type 2 diabetes?",
            expected="Common symptoms include increased thirst, frequent urination..."
        ),
        # ... more items
    ]
)
```

### 3.2 Metric Functions

```python
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import numpy as np

def exact_match(output: str, expected: str) -> float:
    """Exact string match."""
    return 1.0 if output.strip().lower() == expected.strip().lower() else 0.0

def contains_answer(output: str, expected: str) -> float:
    """Check if expected answer is in output."""
    return 1.0 if expected.lower() in output.lower() else 0.0

def bleu_score(output: str, expected: str) -> float:
    """BLEU score for generation quality."""
    return sentence_bleu([expected.split()], output.split())

def rouge_l_score(output: str, expected: str) -> float:
    """ROUGE-L score for summarization."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(expected, output)
    return scores['rougeL'].fmeasure

def semantic_similarity(output: str, expected: str, encoder) -> float:
    """Cosine similarity of embeddings."""
    out_emb = encoder.encode(output)
    exp_emb = encoder.encode(expected)
    return np.dot(out_emb, exp_emb) / (np.linalg.norm(out_emb) * np.linalg.norm(exp_emb))
```

### 3.3 LLM-as-Judge Evaluation

Use a powerful model to judge outputs:

```python
import openai

def llm_judge(output: str, expected: str, criteria: str) -> Dict:
    """Use GPT-4 to evaluate output quality."""
    prompt = f"""Evaluate the following response against the expected answer.

Criteria: {criteria}

Expected Answer: {expected}

Actual Response: {output}

Score the response from 1-5 and explain your reasoning.
Format: {{"score": X, "reasoning": "..."}}
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return json.loads(response.choices[0].message.content)

# Usage
result = llm_judge(
    output="Python is a programming language used for web development.",
    expected="Python is a high-level programming language known for its simplicity.",
    criteria="Accuracy, completeness, and clarity"
)
```

---

## 4. Performance Benchmarking

### 4.1 Throughput Measurement

```python
import time
import torch
from statistics import mean, stdev

def benchmark_throughput(model, prompts, num_runs=10):
    """Measure tokens per second."""
    results = []

    for _ in range(num_runs):
        total_tokens = 0
        start = time.perf_counter()

        for prompt in prompts:
            output = model.generate(
                prompt,
                max_new_tokens=256,
                do_sample=False
            )
            total_tokens += len(model.tokenizer.encode(output))

        elapsed = time.perf_counter() - start
        tokens_per_second = total_tokens / elapsed
        results.append(tokens_per_second)

    return {
        "mean_tps": mean(results),
        "std_tps": stdev(results),
        "min_tps": min(results),
        "max_tps": max(results)
    }
```

### 4.2 Latency Measurement

```python
def benchmark_latency(model, prompts, num_runs=100):
    """Measure time to first token and total latency."""
    ttft_results = []
    total_latency_results = []

    for prompt in prompts:
        for _ in range(num_runs):
            # Time to first token
            start = time.perf_counter()

            # Use streaming to measure TTFT
            first_token = True
            for token in model.generate_stream(prompt, max_new_tokens=256):
                if first_token:
                    ttft = time.perf_counter() - start
                    ttft_results.append(ttft)
                    first_token = False

            total_latency = time.perf_counter() - start
            total_latency_results.append(total_latency)

    return {
        "ttft": {
            "p50": np.percentile(ttft_results, 50),
            "p90": np.percentile(ttft_results, 90),
            "p99": np.percentile(ttft_results, 99)
        },
        "total_latency": {
            "p50": np.percentile(total_latency_results, 50),
            "p90": np.percentile(total_latency_results, 90),
            "p99": np.percentile(total_latency_results, 99)
        }
    }
```

### 4.3 Memory Profiling

```python
import torch

def profile_memory(model, prompt, max_tokens=256):
    """Profile GPU memory usage."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Measure baseline
    baseline_memory = torch.cuda.memory_allocated()

    # Generate
    output = model.generate(prompt, max_new_tokens=max_tokens)

    # Measure peak
    peak_memory = torch.cuda.max_memory_allocated()

    return {
        "baseline_mb": baseline_memory / 1024**2,
        "peak_mb": peak_memory / 1024**2,
        "delta_mb": (peak_memory - baseline_memory) / 1024**2,
        "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    }
```

---

## 5. Comparison Framework

### 5.1 A/B Comparison

```python
class ModelComparison:
    def __init__(self, model_a, model_b):
        self.model_a = model_a
        self.model_b = model_b
        self.results = []

    def compare(self, prompts, judge_fn):
        """Compare two models on given prompts."""
        for prompt in prompts:
            output_a = self.model_a.generate(prompt)
            output_b = self.model_b.generate(prompt)

            # Judge which is better
            judgment = judge_fn(prompt, output_a, output_b)

            self.results.append({
                "prompt": prompt,
                "output_a": output_a,
                "output_b": output_b,
                "winner": judgment["winner"],
                "reasoning": judgment["reasoning"]
            })

        # Aggregate results
        wins_a = sum(1 for r in self.results if r["winner"] == "A")
        wins_b = sum(1 for r in self.results if r["winner"] == "B")
        ties = sum(1 for r in self.results if r["winner"] == "tie")

        return {
            "model_a_wins": wins_a,
            "model_b_wins": wins_b,
            "ties": ties,
            "model_a_win_rate": wins_a / len(self.results),
            "detailed_results": self.results
        }
```

### 5.2 Statistical Significance

```python
from scipy import stats

def compute_significance(scores_a, scores_b, alpha=0.05):
    """Test if difference between models is statistically significant."""
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)

    # Effect size (Cohen's d)
    diff = np.array(scores_a) - np.array(scores_b)
    cohens_d = np.mean(diff) / np.std(diff)

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < alpha,
        "cohens_d": cohens_d,
        "effect_size": "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large"
    }
```

---

## 6. Benchmark Configuration

### 6.1 Standard Configuration

```yaml
# benchmark_config.yaml
evaluation:
  # Standard benchmarks
  benchmarks:
    - name: mmlu
      num_fewshot: 5
      batch_size: 8

    - name: hellaswag
      num_fewshot: 10
      batch_size: 16

    - name: arc_challenge
      num_fewshot: 25
      batch_size: 16

    - name: gsm8k
      num_fewshot: 8
      batch_size: 8

    - name: humaneval
      temperature: 0.8
      num_samples: 200

  # Generation settings
  generation:
    max_tokens: 512
    temperature: 0.0  # Deterministic for benchmarks
    top_p: 1.0

  # Performance settings
  performance:
    warmup_runs: 3
    benchmark_runs: 10
    batch_sizes: [1, 4, 8, 16, 32]
    sequence_lengths: [128, 512, 1024, 2048]
```

### 6.2 Running Full Evaluation

```bash
#!/bin/bash
# run_full_evaluation.sh

MODEL_PATH="./my-trained-model"
OUTPUT_DIR="./evaluation_results"

# Quality benchmarks
lm_eval --model hf \
  --model_args pretrained=$MODEL_PATH,dtype=bfloat16 \
  --tasks mmlu,hellaswag,arc_challenge,gsm8k,truthfulqa_mc \
  --batch_size auto \
  --output_path $OUTPUT_DIR/quality

# Code benchmarks
lm_eval --model hf \
  --model_args pretrained=$MODEL_PATH,dtype=bfloat16 \
  --tasks humaneval,mbpp \
  --batch_size 1 \
  --output_path $OUTPUT_DIR/code

# Generate report
python generate_report.py --input $OUTPUT_DIR --output $OUTPUT_DIR/report.html
```

---

## 7. Reporting

### 7.1 Benchmark Report Template

```markdown
# Model Evaluation Report

## Model Information
- **Model Name**: [Name]
- **Base Model**: [Base]
- **Training Method**: [SFT/DPO/etc]
- **Evaluation Date**: [Date]

## Quality Benchmarks

| Benchmark | Score | Baseline | Delta |
|-----------|-------|----------|-------|
| MMLU | XX.X% | XX.X% | +X.X% |
| HellaSwag | XX.X% | XX.X% | +X.X% |
| ARC-C | XX.X% | XX.X% | +X.X% |
| GSM8K | XX.X% | XX.X% | +X.X% |

## Performance Metrics

| Metric | Value |
|--------|-------|
| Tokens/second | XXX |
| TTFT (p50) | XXms |
| Memory Usage | XX GB |

## Recommendations

Based on evaluation results:
1. [Recommendation 1]
2. [Recommendation 2]
```

### 7.2 Automated Report Generation

```python
def generate_evaluation_report(results, output_path):
    """Generate HTML evaluation report."""
    import pandas as pd
    from jinja2 import Template

    template = Template("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Evaluation Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            .good { color: green; }
            .bad { color: red; }
        </style>
    </head>
    <body>
        <h1>{{ model_name }} Evaluation Report</h1>
        <p>Evaluation Date: {{ eval_date }}</p>

        <h2>Benchmark Results</h2>
        {{ benchmark_table }}

        <h2>Performance Metrics</h2>
        {{ performance_table }}
    </body>
    </html>
    """)

    # Generate tables
    benchmark_df = pd.DataFrame(results["benchmarks"])
    performance_df = pd.DataFrame([results["performance"]])

    html = template.render(
        model_name=results["model_name"],
        eval_date=results["date"],
        benchmark_table=benchmark_df.to_html(index=False),
        performance_table=performance_df.to_html(index=False)
    )

    with open(output_path, "w") as f:
        f.write(html)
```

---

## 8. Best Practices

### 8.1 Evaluation Checklist

- [ ] Use consistent generation parameters across all models
- [ ] Run multiple evaluation passes and report variance
- [ ] Include baseline model for comparison
- [ ] Test on held-out data not seen during training
- [ ] Document all evaluation settings
- [ ] Report failure cases and limitations
- [ ] Include human evaluation for subjective tasks

### 8.2 Common Pitfalls

1. **Data Contamination**: Ensure test data wasn't in training set
2. **Inconsistent Settings**: Use same temperature, top_p, etc.
3. **Cherry-picking**: Report aggregate metrics, not best examples
4. **Overfitting to Benchmarks**: Test on diverse, real-world tasks
5. **Ignoring Variance**: Single runs can be misleading

---

*Last Updated: December 2024*
