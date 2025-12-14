# Tutorial 2: Preference Optimization with DPO

## Overview

This tutorial teaches you how to use Direct Preference Optimization (DPO) to align your model with human preferences. You'll create preference data and train a model that produces higher-quality outputs.

**Time Required**: 1-2 hours
**Difficulty**: Intermediate
**Prerequisites**: Completed Tutorial 1, understanding of fine-tuning

---

## What You'll Learn

- Understanding preference optimization concepts
- Creating preference datasets (chosen vs rejected)
- Training with DPO
- Evaluating alignment improvements

---

## Background: Why DPO?

After Supervised Fine-Tuning (SFT), models follow instructions but may not produce the *best* responses. DPO helps by:

1. Teaching the model to prefer high-quality responses
2. Reducing harmful or unhelpful outputs
3. Aligning with specific style preferences

**DPO vs RLHF:**
| Aspect | RLHF | DPO |
|--------|------|-----|
| Complexity | High (reward model + PPO) | Low (single training phase) |
| Stability | Can be unstable | More stable |
| Results | Excellent | Comparable |
| Resources | High | Moderate |

---

## Step 1: Start with an SFT Model

DPO works best when starting from an SFT model. Use your model from Tutorial 1, or train a new one:

```bash
# If you don't have an SFT model, train one first
largeforge train sft \
  --model microsoft/phi-2 \
  --dataset ./instruction_data.json \
  --output ./sft-base-model \
  --num-epochs 3
```

---

## Step 2: Create Preference Data

DPO requires pairs of responses: a "chosen" (preferred) and "rejected" (less preferred) response.

Create `preference_data.json`:

```python
# create_preference_data.py
import json

preference_data = [
    {
        "prompt": "Explain machine learning in simple terms.",
        "chosen": "Machine learning is like teaching a computer to learn from examples. Instead of programming specific rules, we show the computer many examples, and it figures out patterns on its own. For instance, to teach it to recognize cats, we'd show it thousands of cat pictures. Over time, it learns what features make a cat (ears, whiskers, etc.) and can identify cats it's never seen before.",
        "rejected": "Machine learning is a subset of AI that uses statistical techniques to give computers the ability to learn from data."
    },
    {
        "prompt": "How do I make my code more readable?",
        "chosen": "Here are practical tips to improve code readability:\n\n1. **Use descriptive names**: `calculate_total_price()` instead of `calc()`\n2. **Keep functions small**: Each function should do one thing well\n3. **Add comments for why, not what**: Explain your reasoning, not obvious operations\n4. **Consistent formatting**: Use a linter and stick to one style\n5. **Group related code**: Organize with clear sections and modules\n\nWould you like me to review a specific piece of code?",
        "rejected": "Use good variable names and add comments."
    },
    {
        "prompt": "What should I eat for dinner?",
        "chosen": "I'd be happy to help with dinner ideas! To give you better suggestions, could you tell me:\n\n1. Any dietary restrictions or preferences?\n2. How much time do you have to cook?\n3. What ingredients do you have on hand?\n\nIn the meantime, here are some quick, balanced options:\n- Stir-fry with vegetables and protein over rice (20 min)\n- Sheet pan salmon with roasted vegetables (30 min)\n- Mediterranean bowl with chickpeas, cucumber, and feta (15 min)",
        "rejected": "You should eat something healthy."
    },
    {
        "prompt": "Is Python or JavaScript better?",
        "chosen": "Both are excellent languages, and the 'better' choice depends on your use case:\n\n**Choose Python if you're doing:**\n- Data science and machine learning\n- Scientific computing\n- Backend development with Django/Flask\n- Scripting and automation\n\n**Choose JavaScript if you're doing:**\n- Web frontend development\n- Full-stack web apps (with Node.js)\n- Real-time applications\n- Mobile apps (React Native)\n\nMany developers use both! What are you looking to build?",
        "rejected": "Python is better because it's easier to read."
    },
    {
        "prompt": "Write a poem about coding",
        "chosen": "**Lines of Light**\n\nIn silence deep, the cursor blinks,\nA canvas vast where logic thinks.\nEach character, a careful choice,\nGiving algorithms their voice.\n\nBugs may lurk in shadows near,\nBut patient hands will make them clear.\nFor in the flow of well-typed code,\nCreation finds its true abode.\n\nFrom nothing, worlds begin to form,\nSystems dance and data swarm.\nThe coder's craft, both art and science,\nA human-machine alliance.",
        "rejected": "Roses are red, violets are blue, I like coding, and you should too."
    },
    {
        "prompt": "I'm feeling stressed about work.",
        "chosen": "I hear you - work stress can be really challenging. Here are some strategies that might help:\n\n**Immediate relief:**\n- Take a few deep breaths (4 counts in, 4 out)\n- Step away for a 5-minute break\n- Talk to someone you trust\n\n**Longer-term approaches:**\n- Identify your main stressors and tackle them one at a time\n- Set boundaries around work hours when possible\n- Build in regular breaks throughout your day\n\nWould you like to talk more about what's specifically causing stress? Sometimes just articulating it helps.",
        "rejected": "Don't be stressed, just relax."
    },
    {
        "prompt": "Summarize the benefits of exercise",
        "chosen": "Exercise provides benefits across multiple dimensions:\n\n**Physical Health:**\n- Strengthens heart and cardiovascular system\n- Builds muscle and bone density\n- Improves metabolism and weight management\n- Boosts immune function\n\n**Mental Health:**\n- Reduces anxiety and depression symptoms\n- Improves sleep quality\n- Enhances cognitive function and memory\n- Releases endorphins (natural mood elevators)\n\n**Long-term:**\n- Reduces risk of chronic diseases (diabetes, heart disease)\n- Increases longevity and quality of life\n- Maintains mobility and independence with age\n\nEven 30 minutes of moderate activity most days can make a significant difference.",
        "rejected": "Exercise is good for you because it keeps you healthy."
    },
    {
        "prompt": "How do databases work?",
        "chosen": "Databases are organized systems for storing and retrieving data. Here's a simplified overview:\n\n**Core Components:**\n1. **Storage Engine**: Physically stores data on disk\n2. **Query Processor**: Interprets your requests (SQL queries)\n3. **Buffer Manager**: Caches frequently accessed data in memory\n4. **Transaction Manager**: Ensures data integrity during operations\n\n**How a query works:**\n```\nYour Query → Parser → Optimizer → Executor → Storage → Results\n```\n\n**Types:**\n- **Relational (SQL)**: Tables with relationships (PostgreSQL, MySQL)\n- **Document (NoSQL)**: Flexible JSON-like documents (MongoDB)\n- **Key-Value**: Simple key-value pairs (Redis)\n\nWant me to go deeper on any specific aspect?",
        "rejected": "Databases store data in tables."
    },
    {
        "prompt": "What's the meaning of life?",
        "chosen": "That's one of humanity's oldest questions, and people have found meaning through different lenses:\n\n**Philosophical perspectives:**\n- **Existentialists**: We create our own meaning through choices and actions\n- **Stoics**: Living virtuously and accepting what we can't control\n- **Absurdists**: Finding joy despite life's inherent uncertainty\n\n**Common sources of meaning:**\n- Relationships and connection with others\n- Creative expression and building things\n- Helping others and contributing to society\n- Personal growth and learning\n- Experiencing beauty and wonder\n\nWhat gives your life meaning may be unique to you. What aspects resonate with you?",
        "rejected": "42"
    },
    {
        "prompt": "Fix this code: for i in range(10) print(i)",
        "chosen": "I see the issue! You're missing a colon after the for statement. Here's the corrected code:\n\n```python\nfor i in range(10):\n    print(i)\n```\n\n**What was wrong:**\n- In Python, statements that start a block (like `for`, `if`, `while`, `def`) need a colon `:` at the end\n- The body of the loop needs to be indented\n\n**The output will be:**\n```\n0\n1\n2\n...\n9\n```\n\nWant me to explain loops further or help with other code?",
        "rejected": "Add a colon: for i in range(10): print(i)"
    }
]

# Save to file
with open("preference_data.json", "w") as f:
    json.dump(preference_data, f, indent=2)

print(f"Created preference data with {len(preference_data)} pairs")

# Also create a validation set
validation_data = [
    {
        "prompt": "Explain cloud computing",
        "chosen": "Cloud computing is like renting computer resources instead of owning them...",
        "rejected": "It's computers in the cloud."
    },
    # Add 2-3 more validation examples
]

with open("preference_data_val.json", "w") as f:
    json.dump(validation_data, f, indent=2)
```

**Guidelines for Good Preference Data:**
| Chosen (Preferred) | Rejected (Less Preferred) |
|-------------------|---------------------------|
| Detailed and helpful | Vague or minimal |
| Well-structured | Poorly organized |
| Addresses the user's needs | Ignores context |
| Professional tone | Dismissive or rude |
| Accurate information | Incorrect or incomplete |

---

## Step 3: Understand DPO Configuration

Key DPO hyperparameters:

```yaml
# dpo_config.yaml
model:
  # Start from your SFT model
  name: ./sft-base-model
  torch_dtype: bfloat16

dpo:
  # KL penalty coefficient (higher = closer to reference model)
  beta: 0.1  # Common range: 0.05 - 0.5

  # Loss type
  loss_type: sigmoid  # Options: sigmoid, hinge, ipo

training:
  output_dir: ./dpo-model
  num_train_epochs: 1  # DPO usually needs fewer epochs
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 5e-6  # Lower than SFT
  warmup_ratio: 0.1
  logging_steps: 5

lora:
  r: 8
  lora_alpha: 16
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

**Beta (β) Explained:**
- Lower β (0.05): More deviation from reference model, stronger preference learning
- Higher β (0.5): Stay closer to reference model, more conservative

---

## Step 4: Run DPO Training

```bash
largeforge train dpo \
  --model ./sft-base-model \
  --dataset ./preference_data.json \
  --output ./dpo-aligned-model \
  --beta 0.1 \
  --learning-rate 5e-6 \
  --num-epochs 1
```

**Expected Output:**
```
Loading model: ./sft-base-model
Loading reference model for DPO...
Preparing preference dataset: 10 pairs

Epoch 1/1:
  Step 5: loss=0.693, rewards/chosen=0.12, rewards/rejected=-0.08
  Step 10: loss=0.645, rewards/chosen=0.25, rewards/rejected=-0.15
  Step 15: loss=0.589, rewards/chosen=0.38, rewards/rejected=-0.22

DPO Training complete!
  Final chosen rewards: 0.42
  Final rejected rewards: -0.28
  Reward margin: 0.70
```

**Understanding DPO Metrics:**
- **rewards/chosen**: Higher is better (model prefers chosen)
- **rewards/rejected**: Lower is better (model avoids rejected)
- **Reward margin**: Difference between chosen and rejected (higher = better discrimination)

---

## Step 5: Evaluate Improvement

### Qualitative Evaluation

Test both models on the same prompts:

```python
# evaluate_dpo.py
import requests

def get_response(model_url, prompt):
    response = requests.post(
        f"{model_url}/v1/completions",
        json={
            "prompt": prompt,
            "max_tokens": 300,
            "temperature": 0.7
        }
    )
    return response.json()["choices"][0]["text"]

# Start both servers
# SFT model on port 8000
# DPO model on port 8001

test_prompts = [
    "Explain quantum computing to a beginner.",
    "I'm feeling overwhelmed with my tasks.",
    "What's the difference between REST and GraphQL?",
    "Write a short story about a robot.",
]

for prompt in test_prompts:
    print(f"\n{'='*60}")
    print(f"PROMPT: {prompt}")
    print(f"\n--- SFT Model ---")
    print(get_response("http://localhost:8000", prompt))
    print(f"\n--- DPO Model ---")
    print(get_response("http://localhost:8001", prompt))
```

### LLM-as-Judge Evaluation

Use a more powerful model to compare outputs:

```python
# judge_comparison.py
import openai

def judge_responses(prompt, response_a, response_b):
    judge_prompt = f"""Compare these two responses to the prompt and determine which is better.

PROMPT: {prompt}

RESPONSE A:
{response_a}

RESPONSE B:
{response_b}

Which response is better? Consider:
1. Helpfulness and completeness
2. Accuracy
3. Clarity and structure
4. Appropriate tone

Output format:
WINNER: [A/B/TIE]
REASONING: [Your explanation]"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0
    )

    return response.choices[0].message.content

# Run evaluation
results = {"sft_wins": 0, "dpo_wins": 0, "ties": 0}

for prompt in test_prompts:
    sft_response = get_response("http://localhost:8000", prompt)
    dpo_response = get_response("http://localhost:8001", prompt)

    # Randomize order to avoid position bias
    import random
    if random.random() > 0.5:
        judgment = judge_responses(prompt, sft_response, dpo_response)
        # Parse and record result
    else:
        judgment = judge_responses(prompt, dpo_response, sft_response)
        # Parse and record result (flip the winner)

print(f"Results: SFT wins: {results['sft_wins']}, DPO wins: {results['dpo_wins']}, Ties: {results['ties']}")
```

---

## Step 6: Iterate and Improve

### Adding More Preference Data

If results aren't satisfactory:

1. **Analyze failure cases**: Where does the model still produce poor outputs?
2. **Create targeted pairs**: Add preference data for those specific cases
3. **Retrain**: Run DPO again with expanded dataset

### Adjusting Beta

```bash
# If model deviates too much from base behavior
largeforge train dpo --beta 0.3 ...

# If model isn't learning preferences strongly enough
largeforge train dpo --beta 0.05 ...
```

### Multi-Round DPO

You can run multiple rounds of DPO:

```bash
# Round 1
largeforge train dpo --model ./sft-model --output ./dpo-round1 ...

# Round 2 (use round 1 as base)
largeforge train dpo --model ./dpo-round1 --output ./dpo-round2 ...
```

---

## Common Issues and Solutions

### Model Outputs Become Repetitive

**Cause**: Beta too low, over-optimization
**Solution**: Increase beta or reduce training

### No Improvement from SFT

**Cause**: Preference pairs too similar or insufficient data
**Solution**: Create more diverse and distinct preference pairs

### Training Loss Not Decreasing

**Cause**: Learning rate too low or data issues
**Solution**: Check data format, try higher learning rate

---

## Best Practices

1. **Start with good SFT**: DPO works best on already-capable models
2. **Quality over quantity**: 500 high-quality pairs > 5000 noisy pairs
3. **Clear preferences**: Chosen should be obviously better than rejected
4. **Diverse prompts**: Cover your intended use cases
5. **Conservative training**: DPO needs fewer epochs than SFT

---

## Next Steps

- **Tutorial 3**: [Knowledge Distillation](./TUTORIAL_03_DISTILLATION.md)
- **Tutorial 4**: [Expert Routing System](./TUTORIAL_04_ROUTING.md)

---

*Tutorial created for LargeForgeAI v1.0 | Last Updated: December 2024*
